"""End-to-end checks of the CPU smoke runs (runs/<tag>/ written by run_smoke.sh / cpu_run.py). CPU only, light.

  .venv/bin/python check_e2e.py identity <tag orig> <tag copy flag-off>      PREDECLARED.txt R1
  .venv/bin/python check_e2e.py flagon   <tag flag-on> [<tag flag-off of the same configuration>]   PREDECLARED.txt R2
Appends to logs/check_e2e.txt; exit code 1 on any FAIL.
"""
import glob
import json
import math
import os
import re
import sys

assert os.environ.get("CUDA_VISIBLE_DEVICES", "x") == ""
sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(12)
HERE = os.path.dirname(os.path.abspath(__file__))
R = "<repo>"
LOG = open(f"{HERE}/logs/check_e2e.txt", "a")
FAIL = []


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + "\n")
    LOG.flush()


def check(name, ok, info=""):
    P(f"  [{'PASS' if ok else 'FAIL'}] {name} {info}")
    if not ok:
        FAIL.append(name)


def run(tag):
    d = f"{HERE}/runs/{tag}"
    lg = [p for p in glob.glob(f"{d}/logs/*.log")]
    assert len(lg) == 1, lg
    v = os.path.basename(lg[0])[:-4]
    return d, v


def norm_log(path):
    out = []
    for l in open(path).read().splitlines():
        l = re.sub(r"^==== .*? args: ", "==== args: ", l)
        l = re.sub(r"\d+\.\d+ min", "X min", l)
        l = re.sub(r"\b\d+s\b", "Xs", l)
        out.append(l)
    return out


def state_equal(a, b):
    ca = torch.load(a, map_location="cpu", weights_only=False)
    cb = torch.load(b, map_location="cpu", weights_only=False)
    sa, sb = ca["state_dict"], cb["state_dict"]
    same = list(sa.keys()) == list(sb.keys()) and all(torch.equal(sa[k], sb[k]) for k in sa)
    return same, ca, cb


def identity(ta, tb):
    P(f"\n== IDENTITY (R1): {ta} (original) vs {tb} (copy, flag off)   [{os.popen('date -u +%FT%TZ').read().strip()}]")
    (da, v), (db, vb) = run(ta), run(tb)
    assert v == vb
    P(f"   {open(da + '/cmd.txt').read().strip()[-330:]}")
    ra, rb = np.load(f"{da}/record.npz"), np.load(f"{db}/record.npz")
    check("both runs finished (record ok)", bool(ra["ok"]) and bool(rb["ok"]), f"minutes {float(ra['minutes']):.1f} / {float(rb['minutes']):.1f}; eval cap {ra['cap']}")
    la, lb = norm_log(f"{da}/logs/{v}.log"), norm_log(f"{db}/logs/{v}.log")
    # the copy's args dict has one more key
    la0 = la[0]
    lb0 = lb[0].replace(", 'holdout_in_train': False", "")
    diff = [(i, x, y) for i, (x, y) in enumerate(zip(la[1:], lb[1:])) if x != y]
    check("(a) logs identical line for line (wall-clock fields removed; args line: copy has the extra key holdout_in_train False)",
          la0 == lb0 and len(la) == len(lb) and not diff, f"{len(la)} lines; first diff {diff[:1]}")
    for k in ("fidx", "off", "shapes", "ci", "y", "loss", "lgsum", "calls"):
        check(f"(b) record.{k} identical", np.array_equal(ra[k], rb[k]), f"{ra[k].shape}")
    P(f"       training steps {len(ra['loss'])}, per-step main loss first/last {ra['loss'][0]:.6f} / {ra['loss'][-1]:.6f}")
    ca, cb = np.load(f"{da}/logs/{v}_cliplog.npz"), np.load(f"{db}/logs/{v}_cliplog.npz")
    for k in ("gn", "is_pool", "mpad", "split"):
        check(f"(c) cliplog.{k} identical (pre-clip grad norm of every step)", np.array_equal(ca[k], cb[k], equal_nan=True), f"{ca[k].shape}")
    for nm in (f"{v}.pt", f"{v}@last.pt"):
        same, xa, xb = state_equal(f"{da}/ckpt/{nm}", f"{db}/ckpt/{nm}")
        ab = dict(xb["args"])
        hit = ab.pop("holdout_in_train", None)
        check(f"(d) ckpt/{nm}: every state_dict tensor bit-identical; cfg equal; args equal up to holdout_in_train=False",
              same and xa["cfg"] == xb["cfg"] and xa["args"] == ab and hit is False, f"{len(xa['state_dict'])} tensors")
    fa = sorted(os.path.relpath(p, da) for p in glob.glob(f"{da}/preds/**/*.npy", recursive=True))
    fb = sorted(os.path.relpath(p, db) for p in glob.glob(f"{db}/preds/**/*.npy", recursive=True))
    check("(e) same set of prediction files", fa == fb, fa)
    for p in fa:
        x, y = np.load(f"{da}/{p}"), np.load(f"{db}/{p}")
        check(f"(e) {p} identical", np.array_equal(x, y), f"{x.shape} nonzero {int(np.count_nonzero(x))}")
    ja, jb = json.load(open(f"{da}/logs/{v}.json")), json.load(open(f"{db}/logs/{v}.json"))
    for j in (ja, jb):
        j.pop("train_minutes", None)
        j["args"].pop("holdout_in_train", None)
        for h in j["history"]:
            h.pop("minutes", None)
    check("summary json identical (minutes removed)", json.dumps(ja, sort_keys=True) == json.dumps(jb, sort_keys=True))


# ---------------------------------------------------------------------------------------------- flag on
def train_meta():
    f = pl.read_parquet(f"{R}/data/flat/train/feeds.parquet", columns=["M", "cand_off"])
    M = f["M"].to_numpy().astype(np.int64)
    off = f["cand_off"].to_numpy().astype(np.int64)
    pr = np.load(f"{R}/eda/labels_position/out/train_prandom.npz")
    p = np.nan_to_num(pr["p_random"].astype(np.float32), nan=0.0)
    p[M < 50] = 0.0
    cand = np.where((p > 0.9) & (M >= 50))[0]
    hold = np.sort(np.random.default_rng(12345).choice(cand, 30000, replace=False))
    return M, off, p, hold


def n_batches(M, feed_ids, cand_budget, max_feeds, mpad=16):
    Mp = (M[feed_ids] + mpad - 1) // mpad * mpad
    return sum(-(-int(c) // int(max(1, min(max_feeds, cand_budget // mp)))) for mp, c in zip(*np.unique(Mp, return_counts=True)))


def kd_target_check(rec, M, off, hold, kd_path, kd_w, kd_T):
    """Recompute the listwise target of every recorded training batch from the teacher FILE and the label FILE."""
    kd = np.load(kd_path, mmap_mode="r")
    yl = np.load(f"{R}/data/flat/train/cand_target.npy", mmap_mode="r")
    ish = np.zeros(len(M), bool)
    ish[hold] = True
    o, sh, ysh = rec["off"], rec["shapes"], rec["yshapes"]
    assert np.array_equal(sh, ysh)
    cio = np.concatenate([[0], np.cumsum(sh[:, 0] * sh[:, 1])])
    mx = {True: 0.0, False: 0.0}
    cnt = {True: 0, False: 0}
    tmass = {True: [], False: []}
    for s in range(len(sh)):
        B, Mx = sh[s]
        f = rec["fidx"][o[s]:o[s + 1]]
        ci = rec["ci"][cio[s]:cio[s + 1]].reshape(B, Mx)
        yr = rec["y"][cio[s]:cio[s + 1]].reshape(B, Mx).astype(np.float64)
        m = ci >= 0
        assert (m.sum(1) == M[f]).all() and (ci[:, 0] == off[f]).all(), "recorded rows are not the feeds' own candidate rows"
        rows = np.where(m, ci, 0)
        lab = np.where(m, np.asarray(yl[rows.ravel()]).reshape(B, Mx).astype(np.float64), 0.0)
        t = np.where(m, np.asarray(kd[rows.ravel()]).reshape(B, Mx).astype(np.float64) / kd_T, -1e4)
        t = t - t.max(1, keepdims=True)
        q = np.exp(t)
        q = q / q.sum(1, keepdims=True) * m
        ys = lab.sum(1, keepdims=True)
        exp = (1 - kd_w) * lab / np.maximum(ys, 1) + kd_w * q * (ys > 0)
        d = np.abs(exp - yr).max(1)
        for h in (True, False):
            sel = ish[f] == h
            if sel.any():
                mx[h] = max(mx[h], float(d[sel].max()))
                cnt[h] += int(sel.sum())
                tmass[h].append((kd_w * q * (ys > 0))[sel].sum(1))
    return mx, cnt, {h: (float(np.concatenate(v).mean()) if v else float("nan")) for h, v in tmass.items()}


def flagon(tag, ref=None):
    P(f"\n== FLAG ON (R2): {tag}" + (f" (flag-off run of the same configuration: {ref})" if ref else "") + f"   [{os.popen('date -u +%FT%TZ').read().strip()}]")
    d, v = run(tag)
    P(f"   {open(d + '/cmd.txt').read().strip()[-360:]}")
    rec = np.load(f"{d}/record.npz")
    check("run finished", bool(rec["ok"]), f"{float(rec['minutes']):.1f} min; eval cap {rec['cap']}")
    ck = torch.load(f"{d}/ckpt/{v}.pt", map_location="cpu", weights_only=False)
    a = ck["args"]
    check("checkpoint args: holdout_in_train True, keep 'last'", a["holdout_in_train"] is True and a["keep"] == "last")
    M, off, p, hold = train_meta()
    n = len(M)
    sub = np.sort(np.random.default_rng(a["seed"]).choice(n, a["train_feeds"], replace=False)) if a["train_feeds"] else np.arange(n)
    in_h = np.isin(sub, hold)
    cl = np.load(f"{d}/logs/{v}_cliplog.npz")
    is_pool = cl["is_pool"].astype(bool)
    o = rec["off"]
    steps = len(o) - 1
    lines = open(f"{d}/logs/{v}.log").read().splitlines()
    m = re.search(r"train feeds (\d+), steps/epoch (\d+), total steps (\d+)", "\n".join(lines))
    n_feeds, spe, total = (int(x) for x in m.groups())
    reg = [rec["fidx"][o[s]:o[s + 1]] for s in range(steps) if not is_pool[s]]
    pool_b = [rec["fidx"][o[s]:o[s + 1]] for s in range(steps) if is_pool[s]]
    n_ep = math.ceil(a["epochs"])
    regf = np.concatenate(reg)
    # (a) feeds of the regular batches
    if not pool_b and steps == spe * n_ep:
        u, c = np.unique(regf, return_counts=True)
        check(f"(a) regular batches = every feed of the {len(sub)}-feed subsample BEFORE the holdout removal, each {n_ep}x",
              np.array_equal(u, sub) and (c == n_ep).all(), f"{len(u)} feeds in {steps} steps")
    else:  # pool batches replace some regular batches of the stream (train.py: fb = expl[...]), so some regular feeds are skipped
        check("(a) regular-batch feeds are a subset of the subsample BEFORE the holdout removal, none twice per epoch",
              np.isin(regf, sub).all() and np.unique(regf, return_counts=True)[1].max() <= n_ep,
              f"{len(np.unique(regf))} of {len(sub)} feeds seen in {len(reg)} regular steps ({len(pool_b)} steps replaced by pool batches)")
    nh_reg = int(np.isin(regf, hold).sum())
    check(f"(a) holdout feeds IN the regular batches: {nh_reg} (subsample holds {int(in_h.sum())} holdout feeds; log says training feeds {n_feeds})",
          n_feeds == len(sub) and nh_reg > 0 and (bool(pool_b) or nh_reg == int(in_h.sum()) * n_ep),
          f"holdout candidates in the regular batches {int(M[regf[np.isin(regf, hold)]].sum())}")
    if ref:
        dr, vr = run(ref)
        rr = np.load(f"{dr}/record.npz")
        clr = np.load(f"{dr}/logs/{vr}_cliplog.npz")
        refreg = np.concatenate([rr["fidx"][rr["off"][s]:rr["off"][s + 1]] for s in range(len(rr["off"]) - 1) if not clr["is_pool"][s]])
        allr = rr["fidx"]
        check("(a) the flag-off run of the same configuration has 0 holdout feeds in ANY batch", int(np.isin(allr, hold).sum()) == 0,
              f"{len(np.unique(allr))} distinct feeds")
        if not pool_b and not clr["is_pool"].any():
            check("(a)/(d) same feeds otherwise: flag-on feeds minus holdout == flag-off feeds",
                  np.array_equal(np.setdiff1d(regf, hold), np.unique(refreg)), f"{len(np.unique(refreg))} feeds")
        mr = re.search(r"train feeds (\d+), steps/epoch (\d+), total steps (\d+)", open(f"{dr}/logs/{vr}.log").read())
        P(f"       flag off: train feeds {mr.group(1)}, steps/epoch {mr.group(2)}, total {mr.group(3)}  |  flag on: {n_feeds}, {spe}, {total}")
    # (b) distillation target
    if a.get("kd_file"):
        mx, cnt, tm = kd_target_check(rec, M, off, hold, a["kd_file"], a["kd_w"], a["kd_T"])
        check(f"(b) target of every HOLDOUT feed in a training batch == (1-kd_w) y/sum(y) + kd_w softmax(teacher FILE rows / T)",
              cnt[True] > 0 and mx[True] < 2e-6, f"{cnt[True]} holdout feed-occurrences, max abs diff {mx[True]:.2e}, mean teacher mass per feed {tm[True]:.4f}")
        check("(b) the same identity on the non-holdout feeds", mx[False] < 2e-6, f"{cnt[False]} feed-occurrences, max abs diff {mx[False]:.2e}, mean teacher mass {tm[False]:.4f}")
    # (c) steps
    s_with = n_batches(M, sub, a["cand_budget"], a["max_feeds"])
    s_without = n_batches(M, sub[~in_h], a["cand_budget"], a["max_feeds"])
    t_cool = int(round((1 - a["cooldown_frac"]) * total))
    check(f"(c) steps/epoch {spe} == batches of the feed list WITH the holdout ({s_with}; without: {s_without}); total == int(steps/epoch x epochs)",
          spe == s_with and total == int(spe * a["epochs"]) and steps == total, f"total {total}, t_cool {t_cool}, recorded steps {steps}")
    fl = [l for l in lines if l.startswith("FULLDATA")]
    check("(c) FULLDATA log lines: put back / steps note agree with the independent count",
          any(f"training feeds {len(sub) - int(in_h.sum())} -> {len(sub)}" in l for l in fl)
          and any(f"steps/epoch {s_with} (the same feeds without the holdout: {s_without})" in l for l in fl), f"{len(fl)} FULLDATA lines")
    # (d) logs / files
    ev = [l for l in lines if re.match(r"step \d+/\d+ loss", l)]
    ins = [l for l in lines if re.match(r"FULLDATA step \d+: holdout IN-SAMPLE ndcg", l)]
    check("(d) every eval line shows 'holdout_ndcg nan'; one IN-SAMPLE line per eval",
          ev and all("holdout_ndcg nan" in l for l in ev) and len(ins) == len(ev), f"{len(ev)} evals; e.g. '{ins[-1][:60]}...'")
    best = int(re.search(r"BEST step (\d+)", "\n".join(lines)).group(1))
    same, _, _ = state_equal(f"{d}/ckpt/{v}.pt", f"{d}/ckpt/{v}@last.pt")
    check("(d) BEST step == total steps; kept checkpoint bit-identical to @last", best == total and same, f"BEST step {best}")
    hd = sorted(os.listdir(f"{d}/preds/holdout")) if os.path.isdir(f"{d}/preds/holdout") else []
    want = {f"nn_{v}_holdoutINSAMPLE.npy", f"nn_{v}@last_holdoutINSAMPLE.npy"} | ({f"nn_{v}_holdoutINSAMPLE_hz.npy"} if a.get("hzcd") else set())
    check("(d) holdout files: only *_holdoutINSAMPLE*.npy (no *_holdout.npy, no *_holdout_hz.npy, no holdout_meta / holdout_y)",
          set(hd) == want, hd)
    pf = sorted(os.path.basename(x) for x in glob.glob(f"{d}/preds/*.npy"))
    check("(d) val / test predictions of the kept and the @last checkpoint exist and are finite",
          pf == sorted([f"nn_{v}_trainonly_val.npy", f"nn_{v}_trainonly_test.npy", f"nn_{v}@last_trainonly_val.npy", f"nn_{v}@last_trainonly_test.npy"])
          and all(np.isfinite(np.load(f"{d}/preds/{x}")).all() for x in pf), pf)
    x1, x2 = np.load(f"{d}/preds/holdout/nn_{v}_holdoutINSAMPLE.npy"), np.load(f"{d}/preds/holdout/nn_{v}@last_holdoutINSAMPLE.npy")
    check("(d) in-sample holdout files: 4,630,966 rows, kept == @last", x1.shape == (4630966,) and np.array_equal(x1, x2))
    j = json.load(open(f"{d}/logs/{v}.json"))
    fj = json.load(open(f"{d}/logs/{v}_fulldata.json"))
    check("(d) summary json: every history holdout_ndcg is NaN, best_step == total; <v>_fulldata.json has the in-sample history",
          all(h["holdout_ndcg"] != h["holdout_ndcg"] for h in j["history"]) and j["best_step"] == total
          and len(fj["insample_history"]) == len(ev) and fj["n_holdout_feeds_in_training"] == int(in_h.sum()), f"in-sample history {[round(h['holdout_insample_ndcg'], 5) for h in fj['insample_history']]}")
    # (e) pool
    pl_line = [l for l in lines if l.startswith("cooldown mix:")]
    pool_n = int(re.search(r"exploration pool (\d+) feeds", pl_line[0]).group(1)) if pl_line else None
    exp_pool = int((p[sub] > a["pool_p"]).sum())
    exp_pool_h = int(((p[sub] > a["pool_p"]) & in_h).sum())
    check(f"(e) pool size {pool_n} == subsample feeds with p_random > {a['pool_p']} INCLUDING the {exp_pool_h} holdout feeds (without them: {exp_pool - exp_pool_h})",
          pool_n == exp_pool and any(f"of which {exp_pool_h} holdout feeds" in l for l in fl))
    if pool_b:
        pf_ = np.concatenate(pool_b)
        check(f"(e) pool batches drawn: {len(pool_b)}; they contain holdout feeds; every pool-batch feed has p_random > {a['pool_p']}",
              int(np.isin(pf_, hold).sum()) > 0 and (p[pf_] > a["pool_p"]).all(), f"{int(np.isin(pf_, hold).sum())} holdout of {len(pf_)} pool-batch feeds")
    else:
        P(f"       (e) no pool batch was drawn in this run ({total - t_cool} cooldown steps at p {a['cooldown_mix']}): pool content is checked in the run with --cooldown_mix 0.3")

    # (f) schedule
    def lr_at(s):
        if s < a["warmup"]:
            return a["lr"] * (s + 1) / a["warmup"]
        if s < t_cool:
            return a["lr"]
        return a["lr"] * (0.03 + 0.97 * (1 - math.sqrt((s - t_cool) / max(1, total - t_cool))))
    ok = all(f"lr {lr_at(int(re.match(r'step (\d+)/', l).group(1)) - 1):.2e}" in l for l in ev)
    check("(f) logged lr of every eval == the unchanged schedule formula at the new total", ok,
          [re.search(r"lr \S+", l).group(0) for l in ev])


if __name__ == "__main__":
    if sys.argv[1] == "identity":
        identity(sys.argv[2], sys.argv[3])
    else:
        flagon(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
    P(f"   -> {'ALL PASS' if not FAIL else 'FAILED: ' + str(FAIL)}")
    sys.exit(1 if FAIL else 0)
