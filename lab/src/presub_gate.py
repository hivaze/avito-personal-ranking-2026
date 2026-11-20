"""Pre-submission gate: leak / overfit / validity checks that every submission must pass.

Usage (example, v12):
  python src/presub_gate.py --sub subs/cand_G_F10F7xF7x70_memeq.parquet \
     --val  "0.5*nn_final5v3_val_oof+0.5*nn_final5_trainonly_val" \
     --test "0.5*nn_final5v3_test+0.5*nn_final5_trainonly_test" \
     --ref_val "0.5*nn_final5_val_oof+0.5*nn_final5_trainonly_val" \
     --ref_test "0.5*nn_final5_test+0.5*nn_final5_trainonly_test" \
     --members v12_all_e10,v12_all_s81,v12_all_s82,v12_allprew_s83,v12_allprew_s84,v12_allprew_s85 --ft_suffix n2cpv3

Specs are sums of w*name over preds/<name>.npy, each term z-scored within feed (same as evalq / make_sub blends).
Checks (FAIL blocks the submission; WARN needs a written justification in CURRENT_STATUS):
  A  file validity: key set == sample_submission, no duplicate keys, finite, no within-feed ties
  B  reproduction: the submission's within-feed order equals the order of the --test spec
  C  inputs audit: train-stage and fine-tune feature lists of every member (from nn/logs) vs banned / warn lists,
     and no member input hard-linked to a quarantined file
  D  val: paired delta vs reference with SE and slices (M<50, M>=50, Tue, Thu, user0); LEAK PROBE = the gain on
     feeds whose user has a LATER feed (val/test) minus the gain on feeds without one (a later-snapshot leak
     concentrates gains on has-later feeds)
  E  calibration: predicted public / private (src/calib.py, +-0.003 bands)
  F  test/val parity: per-feed rank correlation new-vs-ref on test vs on val (a construction break on test shows
     up as much lower agreement on test than on val)
"""
import argparse, re, sys, os
from pathlib import Path
import numpy as np, polars as pl

sys.path.insert(0, "<repo>/src")
from common import feed_meta, FLAT
from metrics import ndcg_per_feed
from stack_val import feed_rank_z
import calib

ROOT = Path("<repo>"); P = ROOT / "preds"; LOGS = ROOT / "nn/logs"
# never allowed anywhere (same-user later snapshots / quarantined constructions)
BANNED_ANY = [r"^it_hist_(all|et\d+)(_rk|_z|_gap)?$", r"^lh_", r"later"]
# never allowed as TRAIN-STAGE inputs (train-row construction leak or parity break)
BANNED_TRAIN = [r"^pre_(fe|new|hmax|idmax|fe_noid|id_contrib)", r"_id_contrib"]
# quality issues as train-stage inputs (raw prew/pre2 hit the +-10 clip in attach_xt; use the *_s copies)
WARN_TRAIN = [r"^prew_fe$", r"^prew_new$", r"^pre2_fe$", r"^pre2_new$"]
# drift- or leak-prone (train-candidate stats, ALS, hour): allowed only with a written justification
WARN_ANY = [r"^als_", r"^hour$", r"^it_tr_", r"^u_tr_", r"^it_cand_cnt_train$"]
ST = {"FAIL": 0, "WARN": 0}


def say(level, msg):
    if level in ST:
        ST[level] += 1
    print(f"[{level}] {msg}", flush=True)


def spec_score(spec, off, M):
    s = 0
    for term in spec.split("+"):
        w, n = term.split("*") if "*" in term else ("1", term)
        x = np.load(P / f"{n.strip()}.npy").astype(np.float32)
        if not np.isfinite(x).all():
            say("FAIL", f"{n}: non-finite predictions")
        s = s + float(w) * feed_rank_z(off, M, np.nan_to_num(x))[:, 1]
    return s


def within_feed_rank(off, M, x):
    fid = np.repeat(np.arange(len(M)), M)
    o = np.lexsort((x, fid))
    r = np.empty(len(x), np.int64); r[o] = np.arange(len(x)) - np.repeat(off, M)
    return r


def per_feed_spearman(off, M, a, b):
    ra, rb = within_feed_rank(off, M, a).astype(np.float64), within_feed_rank(off, M, b).astype(np.float64)
    fid = np.repeat(np.arange(len(M)), M)
    def c(x):
        mu = np.bincount(fid, x) / M
        return x - mu[fid]
    ca, cb = c(ra), c(rb)
    num = np.bincount(fid, ca * cb); den = np.sqrt(np.bincount(fid, ca * ca) * np.bincount(fid, cb * cb))
    return np.where(den > 0, num / np.maximum(den, 1e-12), 1.0)


def feat_list(path, key):
    if not path.exists():
        return None
    for line in path.read_text(errors="ignore").splitlines():
        if line.startswith(key):
            return re.findall(r"'([^']+)'", line.split(":", 1)[1])
    return None


def train_feat_list(v):
    """Train-stage input list of variant v: the 'train-stage extra features' line that follows the args line of v in any
    nn/logs/*.out or *.log (pipeline logs may hold several variants, e.g. lanes)."""
    tag = f"'variant': '{v}'"
    for fp in sorted(LOGS.glob("*.out")) + sorted(LOGS.glob("*.log")):
        cur = False
        for line in fp.read_text(errors="ignore").splitlines():
            if "args: {" in line:
                cur = tag in line
            elif cur and line.startswith("train-stage extra features"):
                return re.findall(r"'([^']+)'", line.split(":", 1)[1])
    return None


def audit_members(members, ft_suffix):
    qino = set()
    for q in (ROOT / "data/feats_quarantine").rglob("*.npy"):
        qino.add(os.stat(q).st_ino)
    for v in members:
        base = v.split("@")[0]  # snapshot variants share the run's inputs
        tr = train_feat_list(base)
        if tr is None:
            say("WARN", f"{v}: no train-stage feature list found in nn/logs for variant {base}")
            tr = []
        for f in tr:
            g = f[4:] if f.startswith("has:") else f
            if any(re.search(p, g) for p in BANNED_ANY + BANNED_TRAIN):
                say("FAIL", f"{v}: banned train-stage input {f}")
            elif any(re.search(p, g) for p in WARN_ANY) or (not f.startswith("has:") and any(re.search(p, g) for p in WARN_TRAIN)):
                say("WARN", f"{v}: train-stage input {f} is on the warn list")
            for split in ["train", "val", "test"]:
                for d in [ROOT / "nn/feat_cache" / split, ROOT / "data/feats" / split]:
                    fp = d / f"{g}.npy"
                    if fp.exists() and os.stat(fp).st_ino in qino:
                        say("FAIL", f"{v}: input {g} ({split}) is hard-linked to a quarantined file")
                        break
        if ft_suffix:
            ft = feat_list(LOGS / f"{v}_{ft_suffix}_ft.log", "extra features")
            if ft is None:
                say("WARN", f"{v}: no fine-tune feature list in nn/logs/{v}_{ft_suffix}_ft.log")
                continue
            for f in ft:
                if any(re.search(p, f) for p in BANNED_ANY):
                    say("FAIL", f"{v}: banned fine-tune input {f}")
                elif any(re.search(p, f) for p in WARN_ANY):
                    say("WARN", f"{v}: fine-tune input {f} is on the warn list")
        print(f"  {v}: {len(tr)} train-stage inputs audited" + (f", fine-tune {ft_suffix} audited" if ft_suffix else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", required=True); ap.add_argument("--val", required=True); ap.add_argument("--test", required=True)
    ap.add_argument("--ref_val", required=True); ap.add_argument("--ref_test", required=True)
    ap.add_argument("--members", default=""); ap.add_argument("--ft_suffix", default="")
    a = ap.parse_args()

    fv, Mv, ov = feed_meta("val"); ft, Mt, ot = feed_meta("test")
    Mv = Mv.astype(np.int64); Mt = Mt.astype(np.int64)
    y = np.load(FLAT / "val/cand_target.npy")

    # A: file validity
    s = pl.read_parquet(a.sub); ss = pl.read_parquet(ROOT / "data/competition/sample_submission.parquet")
    j = ss.select("feed_id", "item_id").join(s, on=["feed_id", "item_id"], how="left")
    if s.height != ss.height or j["score"].null_count() > 0:
        say("FAIL", f"A: key set differs from sample_submission ({s.height} rows, {j['score'].null_count()} missing)")
    if s.select("feed_id", "item_id").is_duplicated().any():
        say("FAIL", "A: duplicate keys")
    if not np.isfinite(s["score"].to_numpy()).all():
        say("FAIL", "A: non-finite scores")
    ties = s.group_by("feed_id").agg((pl.col("score").n_unique() != pl.len()).alias("t"))["t"].sum()
    if ties:
        say("FAIL", f"A: {ties} feeds with tied scores")
    print("A file validity checked")

    # B: reproduction from the --test spec (align submission rows to flat test order)
    t_new = spec_score(a.test, ot, Mt)
    flat = pl.DataFrame({"feed_id": np.repeat(ft["feed_id"].to_numpy(), Mt),
                         "item_id": np.load(FLAT / "test/cand_item_id.npy"), "row": np.arange(len(t_new))})
    sub_sc = flat.join(s, on=["feed_id", "item_id"], how="left").sort("row")["score"].to_numpy().astype(np.float64)
    rho = per_feed_spearman(ot, Mt, sub_sc, t_new)
    if rho.min() < 0.9999:
        say("FAIL", f"B: submission order does not match the --test spec (min per-feed Spearman {rho.min():.6f})")
    print(f"B reproduction: min per-feed Spearman {rho.min():.6f}")

    # C: inputs audit
    if a.members:
        audit_members([m for m in a.members.split(",") if m], a.ft_suffix)
    else:
        say("WARN", "C: no --members given, inputs not audited")

    # D: val deltas, slices, leak probe
    v_new = spec_score(a.val, ov, Mv); v_ref = spec_score(a.ref_val, ov, Mv)
    pn, pr = ndcg_per_feed(v_new, y, ov, Mv), ndcg_per_feed(v_ref, y, ov, Mv)
    d = pn - pr
    se = lambda x: x.std() / np.sqrt(len(x))
    print(f"D val new {pn.mean():.5f} ref {pr.mean():.5f} delta {d.mean():+.5f} +- {se(d):.5f}")
    uv = fv["user_id"].to_numpy(); tv = fv["timestamp"].to_numpy()
    allu = np.concatenate([uv, ft["user_id"].to_numpy()]); allt = np.concatenate([tv, ft["timestamp"].to_numpy()])
    last = pl.DataFrame({"u": allu, "t": allt}).group_by("u").agg(pl.col("t").max())
    lastd = dict(zip(last["u"].to_list(), last["t"].to_list()))
    has_later = np.array([(u != 0) and (lastd[u] > t) for u, t in zip(uv, tv)])
    tue = (tv + 10800) // 86400 < 20720  # Moscow-time day: Tue Sep 22 vs Thu Sep 24
    slices = {"M<50": Mv < 50, "M>=50": Mv >= 50, "Tue": tue, "Thu": ~tue, "user0": uv == 0,
              "has_later": has_later, "no_later": ~has_later}
    for k, m in slices.items():
        if m.sum() < 50:
            continue
        z = d[m].mean() / max(se(d[m]), 1e-12)
        print(f"   {k:9s} n={m.sum():6d} delta {d[m].mean():+.5f} +- {se(d[m]):.5f}")
        if z < -2:
            say("WARN", f"D slice {k} below -2 SE")
    gap = d[has_later].mean() - d[~has_later].mean()
    gse = np.sqrt(se(d[has_later]) ** 2 + se(d[~has_later]) ** 2)
    print(f"   LEAK PROBE has_later - no_later gain: {gap:+.5f} +- {gse:.5f} (low power on val: few has-later feeds)")
    # only ~117 val feeds have a later snapshot (SE ~0.006): same-recipe comparisons swing +-0.015 (z +-2) both ways
    # (2026-11-11: F5cd vs F5 -0.016, F7 vs F5 +0.011), so FAIL needs z > 3; 2 < z <= 3 is a WARN that requires a
    # controlled check (a paired run differing only in the suspect input, or the suspect feature alone on has-later feeds).
    if gap > 3 * gse:
        say("FAIL", "D leak probe: the gain concentrates on feeds whose user has a later snapshot (z > 3)")
    elif gap > 2 * gse:
        say("WARN", "D leak probe: gain larger on has-later feeds (2 < z <= 3): run the controlled check")

    # E: calibration
    pnp, prp = calib.predict(pn), calib.predict(pr)
    print(f"E predicted public {pnp['public']:.4f} private {pnp['private']:.4f} (+-0.003); ref {prp['public']:.4f} / {prp['private']:.4f}")

    # F: test/val parity of the change
    t_ref = spec_score(a.ref_test, ot, Mt)
    rv = per_feed_spearman(ov, Mv, v_new, v_ref); rt = per_feed_spearman(ot, Mt, t_new, t_ref)
    print(f"F new-vs-ref per-feed Spearman: val mean {rv.mean():.4f} p5 {np.percentile(rv, 5):.4f} | test mean {rt.mean():.4f} p5 {np.percentile(rt, 5):.4f}")
    if rt.mean() < rv.mean() - 0.02:
        say("WARN", "F the change moves test rankings much more than val rankings (possible test-side construction break)")

    verdict = "FAIL" if ST["FAIL"] else ("PASS with WARN" if ST["WARN"] else "PASS")
    print(f"\nGATE: {verdict}  (FAIL {ST['FAIL']}, WARN {ST['WARN']})")
    sys.exit(1 if ST["FAIL"] else 0)


if __name__ == "__main__":
    main()
