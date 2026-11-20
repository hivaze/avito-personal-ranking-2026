# Acceptance check for members trained on all data. Such a member has no out-of-sample holdout score, so it
# is compared with its held-out twin (same command and seed): A1 structure of the run (feed counts, steps,
# last step kept, only IN-SAMPLE holdout files), A2 the in-sample holdout NDCG must exceed the twin's
# out-of-sample one by a calibrated bar, A3 the pool of twins must match the pool of originals on val
# within a band of three times the pool-level seed noise. The rules were written before the first run.
# Run on CPU as: python nn_x/w12/stale/fulldata/accept_twins.py check F   (or check P); 21 s for 16 twins.
# Produced logs/accept_F.txt and logs/accept_P.txt (cut versions in ../results/).
# Kept as it ran: paths and names refer to the lab tree (see lab/ and docs/glossary.md).
# Original: lab/nn_x/w12/stale/fulldata/accept_twins.py
#
"""Acceptance check of the FULL-DATA twins = PREDECLARED.txt R4 (+ R6) made executable. CPU only, read-only, 2-4 threads.

  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python nn_x/w12/stale/fulldata/accept_twins.py <mode>
    null                 A/A band of the re-cool harness (ref2 - ref pairs that exist NOW) and "ref re-cool - original cooldown";
                         run once BEFORE any twin exists -> logs/accept_null.txt / .json (the bands the check modes read)
    selftest             the script checked without any twin (R6): A1 parser on the CPU smoke runs, A3 / A4 numerics with the
                         originals' selected checkpoints standing in for twins -> logs/accept_selftest.txt
    check <stage> [arm]  stage F | FH | P | R | RH | PR (lane_full.sh); every twin of that stage whose logs/<v>.json exists
                         -> appended to logs/accept_<stage>.txt; exit code 1 on any DEFECT, 2 if no twin exists yet

  A1 structure (per twin; any failure = DEFECT)   A2 positive control (IN-SAMPLE holdout lift)   A3 pool level on val (paired
  per feed, band of R4 / R6)   A4 member level (information).  Val is the only clean witness of a twin: every holdout number
  of a twin is IN-SAMPLE and is used here ONLY as the positive control "the holdout feeds reached the batches".
"""
import glob
import json
import os
import re
import sys

assert os.environ.get("CUDA_VISIBLE_DEVICES", "x") == "", "start with CUDA_VISIBLE_DEVICES=''"
sys.dont_write_bytecode = True
sys.path.insert(0, "<repo>/src")
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from metrics import ndcg_per_feed  # noqa: E402

R = "<repo>"
HERE = os.path.dirname(os.path.abspath(__file__))
W5, ABL = f"{R}/nn_x/w5", f"{R}/nn_x/abl"
N_VAL_ROWS, N_TEST_ROWS, N_HOLD_ROWS = 1389971, 2904806, 4630966
S16 = json.load(open(f"{HERE}/logs/facts.json"))["null"]["s16"]          # 0.00030, frozen by facts.py before any twin existed
MEMBER_SD = json.load(open(f"{HERE}/logs/facts.json"))["member_val_last"]["sd"]  # 0.00048
FULL_REAL = dict(before=3121208, after=3151208, spe=9984, total10=99840, total_cd=19968, pool=164118, pool_hold=30000, n_hold=30000)
W16_ARMS, POP_ARMS = ("kd05w192", "kd05t25w192"), ("pop1kd05t25", "pop1kd05w192")
LOG, FAIL = None, []


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    if LOG:
        LOG.write(s + "\n")
        LOG.flush()


def check(name, ok, info=""):
    P(f"    [{'PASS' if ok else 'DEFECT'}] {name} {info}")
    if not ok:
        FAIL.append(name)
    return ok


# ------------------------------------------------------------------------------------------------ naming (lane_full.sh)
def stage_table(stage, arm=None, W5=W5, ABL=ABL):
    """-> list of (twin variant, harness dir, reference variant, reference harness dir, re-cool form?)."""
    out = []
    if stage in ("F", "FH", "R", "RH"):
        for a in W16_ARMS:
            if arm and a != arm:
                continue
            kl = "k" if a == "kd05w192" else "l"
            for s in range(951, 959):
                tw = {"F": f"w5_fd10_{a}_s{s}", "FH": f"w5_fd10hz_{a}_s{s}", "R": f"w5_fdcd_{a}_s{s}", "RH": f"w5_fdcdhz_{a}_s{s}"}[stage]
                ref = {"F": f"w5_e10_{a}_s{s}", "FH": f"w5_{kl}cd_hz_s{s}", "R": f"w5_{kl}cd_ref_s{s}", "RH": f"w5_{kl}cd_hz_s{s}"}[stage]
                out.append((tw, W5, ref, W5, stage in ("R", "RH")))
    elif stage in ("P", "PR"):
        for a in POP_ARMS:
            if arm and a != arm:
                continue
            for s in range(951, 955):
                out.append((f"abl_{'fd10' if stage == 'P' else 'fdcd'}_{a}_s{s}", ABL, f"abl_e10_{a}_s{s}", ABL, stage == "PR"))
    else:
        raise SystemExit(f"unknown stage {stage}")
    return out


# ------------------------------------------------------------------------------------------------ val data + pooling
_V = {}


def val():
    if not _V:
        f = pl.read_parquet(f"{R}/data/flat/val/feeds.parquet", columns=["M", "cand_off"])
        _V["M"] = f["M"].to_numpy().astype(np.int64)
        _V["off"] = f["cand_off"].to_numpy().astype(np.int64)
        _V["y"] = np.load(f"{R}/data/flat/val/cand_target.npy").astype(np.float64)
        _V["fid"] = np.repeat(np.arange(len(_V["M"])), _V["M"])
    return _V


def zf(s):  # per-feed z (facts.py zf)
    v = val()
    s = np.asarray(s, np.float64)
    mu = np.bincount(v["fid"], s, len(v["M"])) / v["M"]
    sd = np.sqrt(np.maximum(np.bincount(v["fid"], s * s, len(v["M"])) / v["M"] - mu ** 2, 0)) + 1e-9
    return (s - mu[v["fid"]]) / sd[v["fid"]]


def nd(s):
    v = val()
    return ndcg_per_feed(np.asarray(s, np.float64), v["y"], v["off"], v["M"])


def vfile(h, v, last=False):
    return f"{h}/preds/nn_{v}{'@last' if last else ''}_trainonly_val.npy"


def ms(d):
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))


def fmt(t):
    return f"{t[0]:+.5f} +- {t[1]:.5f}"


def pool(files):
    acc = 0.0
    for f in files:
        acc = acc + zf(np.load(f))
    return nd(acc / len(files))


# ------------------------------------------------------------------------------------------------ A1
_PD = {}


def parser_defaults(v):
    """Defaults of the copy's own parser (AST-exec of parse() only, as check_cmds.py does); v = the twin's variant name."""
    key = "abl" if v.startswith("abl_") else "w5"
    if key not in _PD:
        sys.path.insert(0, HERE)
        from test_full import load_parse
        _PD[key] = load_parse(f"{HERE}/train_full{'_abl' if key == 'abl' else ''}.py", 0.5)([])
    return _PD[key]


def a1(h, v, exp, total, hz=False, ref_json=None, quiet_rows=False, allow=()):
    """Structure of ONE twin run. exp = dict(before, after, spe, pool, pool_hold, n_hold); total = expected total steps."""
    ok = True
    lp = f"{h}/logs/{v}.log"
    if not os.path.exists(lp):
        return check(f"{v}: log exists", False, lp)
    txt = open(lp).read()
    lines = txt.splitlines()
    fl = [l for l in lines if l.startswith("FULLDATA")]
    ok &= check(f"{v}: 'training feeds {exp['before']} -> {exp['after']}'", any(f"training feeds {exp['before']} -> {exp['after']}" in l for l in fl))
    m = re.search(r"train feeds (\d+), steps/epoch (\d+), total steps (\d+)", txt)
    got = tuple(int(x) for x in m.groups()) if m else None
    ok &= check(f"{v}: train feeds {exp['after']}, steps/epoch {exp['spe']}, total steps {total}", got == (exp["after"], exp["spe"], total), f"log: {got}")
    ok &= check(f"{v}: pool {exp['pool']} feeds, of which {exp['pool_hold']} holdout feeds",
                any(f"exploration pool {exp['pool']} feeds, of which {exp['pool_hold']} holdout feeds" in l for l in fl))
    ev = [l for l in lines if re.match(r"step \d+/\d+ loss", l)]
    ins = [l for l in lines if re.match(r"FULLDATA step \d+: holdout IN-SAMPLE ndcg", l)]
    ok &= check(f"{v}: every eval line 'holdout_ndcg nan' ({len(ev)} evals), one IN-SAMPLE line per eval",
                bool(ev) and all("holdout_ndcg nan" in l for l in ev) and len(ins) == len(ev))
    b = re.search(r"BEST step (\d+)", txt)
    ok &= check(f"{v}: BEST step == total", bool(b) and int(b.group(1)) == total, f"BEST step {b.group(1) if b else None}")
    jp, fp = f"{h}/logs/{v}.json", f"{h}/logs/{v}_fulldata.json"
    if check(f"{v}: logs/<v>.json and logs/<v>_fulldata.json exist", os.path.exists(jp) and os.path.exists(fp)):
        j, fj = json.load(open(jp)), json.load(open(fp))
        ok &= check(f"{v}: json args holdout_in_train True, keep 'last', holdout 30000; best_step == total; history holdout_ndcg all NaN",
                    j["args"].get("holdout_in_train") is True and j["args"]["keep"] == "last" and j["args"]["holdout"] == 30000
                    and j["best_step"] == total and all(x["holdout_ndcg"] != x["holdout_ndcg"] for x in j["history"]))
        ok &= check(f"{v}: the run's own counts: {exp['n_hold']} holdout feeds in training, in every epoch's batches, {exp['pool_hold']} in the pool; kept step == total",
                    fj["n_holdout_feeds_in_training"] == exp["n_hold"] and fj["holdout_feeds_per_epoch"] == exp["n_hold"]
                    and fj["pool_holdout_feeds"] == exp["pool_hold"] and fj["kept_step"] == total == fj["total_steps"],
                    f"(json: {fj['n_holdout_feeds_in_training']}, {fj['holdout_feeds_per_epoch']}, {fj['pool_holdout_feeds']}, kept {fj['kept_step']})")
        if ref_json is not None and os.path.exists(ref_json):
            ra, ta = json.load(open(ref_json))["args"], dict(j["args"])
            skip = {"variant", "holdout_in_train", "keep"}
            diff = sorted(k for k in set(ra) | set(ta) if k not in skip and ra.get(k, "<absent>") != ta.get(k, "<absent>"))
            # a key absent from the reference's json = a flag added to train.py after that run: fine iff the twin holds the parser default
            dflt = parser_defaults(v)
            late = [k for k in diff if k not in ra and k in dflt and ta[k] == dflt[k]]
            diff = [k for k in diff if k not in late and k not in allow]
            ok &= check(f"{v}: arguments == the reference run's arguments except variant / holdout_in_train / keep", not diff,
                        f"differing keys: {dict((k, (ra.get(k, '<absent>'), ta.get(k, '<absent>'))) for k in diff)}"
                        + (f"; flags newer than the reference's log, at their defaults: {late}" if late else ""))
    else:
        ok = False
    hd = f"{h}/preds/holdout"
    bad = [x for x in (f"nn_{v}_holdout.npy", f"nn_{v}@last_holdout.npy", f"nn_{v}_holdout_hz.npy") if os.path.exists(f"{hd}/{x}")]
    ok &= check(f"{v}: NO out-of-sample-named holdout file", not bad, f"found {bad}" if bad else "")
    want = [f"nn_{v}_holdoutINSAMPLE.npy", f"nn_{v}@last_holdoutINSAMPLE.npy"] + ([f"nn_{v}_holdoutINSAMPLE_hz.npy"] if hz else [])
    have = [x for x in want if os.path.exists(f"{hd}/{x}")]
    ok &= check(f"{v}: in-sample holdout files present ({len(want)}), {N_HOLD_ROWS} rows",
                have == want and all(np.load(f"{hd}/{x}", mmap_mode="r").shape == (N_HOLD_ROWS,) for x in have), f"{have}")
    files = {"val": (vfile(h, v), vfile(h, v, True), N_VAL_ROWS),
             "test": (f"{h}/preds/nn_{v}_trainonly_test.npy", f"{h}/preds/nn_{v}@last_trainonly_test.npy", N_TEST_ROWS)}
    for nm, (fk, fl_, nrow) in files.items():
        if not (os.path.exists(fk) and os.path.exists(fl_)):
            ok &= check(f"{v}: {nm} predictions of the kept and the @last checkpoint exist", False)
            continue
        xk, xl = np.load(fk), np.load(fl_)
        fin = bool(np.isfinite(xk).all() and np.isfinite(xl).all())
        rows_ok = quiet_rows or (np.count_nonzero(xk) > 0.99 * nrow)
        ok &= check(f"{v}: {nm} files finite, {nrow} rows, not constant; kept vs @last max abs diff {float(np.abs(xk - xl).max()):.2e} (same weights)",
                    xk.shape == xl.shape == (nrow,) and fin and rows_ok and float(np.corrcoef(xk, xl)[0, 1]) > 0.9999)
    return ok


def last_insample(h, v):
    fj = json.load(open(f"{h}/logs/{v}_fulldata.json"))
    return float(fj["insample_history"][-1]["holdout_insample_ndcg"]), fj


def ref_last_holdout(h, v):
    m = re.search(r"LAST checkpoint .*? holdout_ndcg ([\d.]+)", open(f"{h}/logs/{v}.log").read())
    return float(m.group(1)) if m else float("nan")


# ------------------------------------------------------------------------------------------------ modes
def mode_null():
    global LOG
    LOG = open(f"{HERE}/logs/accept_null.txt", "a")
    stamp = os.popen('date -u +%FT%TZ').read().strip()
    P(f"\n== accept_twins.py null   [{stamp}]   (R6c amended: pool-level label-swap statistic)")
    P(f"   s16 = {S16:.5f} (facts.py); bands 3 s16 sqrt(16 / k): 16-pool {3 * S16:.5f}, 8-pool {3 * S16 * np.sqrt(2):.5f}, 4-pool {3 * S16 * 2:.5f}; member SD {MEMBER_SD:.5f}")
    res = dict(s16=S16, pairs=[], ref_vs_orig=[])
    P("   A/A of the re-cool harness: ref2 (cooldown seed 2000 + s) minus ref (1000 + s), same snapshot, val, @last checkpoints")
    for kl, a in (("k", "kd05w192"), ("l", "kd05t25w192")):
        for s in range(951, 959):
            r1, r2, o = f"w5_{kl}cd_ref_s{s}", f"w5_{kl}cd_ref2_s{s}", f"w5_e10_{a}_s{s}"
            done = lambda v: os.path.exists(f"{W5}/logs/{v}.json") and os.path.exists(vfile(W5, v, True))  # noqa: E731
            if done(r1):
                d = nd(zf(np.load(vfile(W5, r1, True)))) - nd(zf(np.load(vfile(W5, o, True))))
                res["ref_vs_orig"].append(dict(ref=r1, orig=o, d=ms(d)))
            if done(r1) and done(r2):
                d = nd(zf(np.load(vfile(W5, r2, True)))) - nd(zf(np.load(vfile(W5, r1, True))))
                res["pairs"].append(dict(ref=r1, ref2=r2, d=ms(d)))
                P(f"     {r2} - {r1}: {fmt(ms(d))}")
    k = len(res["pairs"])
    if k:
        dm = np.array([p["d"][0] for p in res["pairs"]])
        rms = float(np.sqrt((dm ** 2).mean()))
        d = pool([vfile(W5, p["ref2"], True) for p in res["pairs"]]) - pool([vfile(W5, p["ref"], True) for p in res["pairs"]])
        res.update(n_pairs=k, aa_member_rms=rms, aa_pool_direct=ms(d))
        P(f"   {k} A/A pairs: member-level RMS {rms:.5f} (mean {dm.mean():+.5f}); direct {k}-pool(ref2) - {k}-pool(ref) {fmt(ms(d))} (n {len(d)})")
        P(f"   [superseded first version of R6(c)] member RMS / sqrt(k): 8-pool {rms / np.sqrt(8):.5f}, 16-pool {rms / 4:.5f}")
        # R6(c) AMENDED: pool-level A/A noise by within-pair label swaps (the analogue of s8)
        za = [zf(np.load(vfile(W5, p["ref"], True))) for p in res["pairs"]]
        zb = [zf(np.load(vfile(W5, p["ref2"], True))) for p in res["pairs"]]
        if k <= 10:
            pats = [[(m >> i) & 1 for i in range(k - 1)] + [0] for m in range(2 ** (k - 1))]
        else:
            r_ = np.random.default_rng(20261115)
            pats = [list(r_.integers(0, 2, k - 1)) + [0] for _ in range(256)]
        ds = []
        for pat in pats:
            A = sum(zb[i] if b else za[i] for i, b in enumerate(pat)) / k
            B = sum(za[i] if b else zb[i] for i, b in enumerate(pat)) / k
            ds.append(float((nd(A) - nd(B)).mean()))
        ds = np.array(ds)
        s_aa = float(np.sqrt((ds ** 2).mean()))
        res.update(s_aa=s_aa, s_aa_k0=k, s_aa_patterns=len(pats), s_aa_max_abs=float(np.abs(ds).max()), stamp=stamp)
        P(f"   R6(c) amended: s_AA({k}) = {s_aa:.5f} (RMS of the {k}-pool vs {k}-pool val difference over {len(pats)} within-pair label swaps; max |d| {np.abs(ds).max():.5f})")
        for kk in (4, 8, 16):
            b_aa, b_a = 3 * s_aa * np.sqrt(k / kk), 3 * S16 * np.sqrt(16 / kk)
            P(f"     band for a {kk}-pool of re-cool twins: 3 s_AA sqrt({k} / {kk}) = {b_aa:.5f}; R6(a) cap {b_a:.5f} -> in force {min(b_aa, b_a) if k >= 4 else b_a:.5f}"
              + ("" if k >= 4 else "   [fewer than 4 pairs: R6(a)]"))
    else:
        res.update(n_pairs=0)
        P("   no finished (ref, ref2) pair: R6(c) falls back to 3 s16 sqrt(16 / k)")
    ro = res["ref_vs_orig"]
    if ro:
        dm = np.array([x["d"][0] for x in ro])
        d = pool([vfile(W5, x["ref"], True) for x in ro]) - pool([vfile(W5, x["orig"], True) for x in ro])
        res.update(ref_vs_orig_n=len(ro), ref_vs_orig_member_mean=float(dm.mean()), ref_vs_orig_member_sd=float(dm.std(ddof=1)), ref_vs_orig_pool=ms(d))
        P(f"   information: ref re-cool - original cooldown of the same snapshot (@last, val), {len(ro)} pairs: member mean {dm.mean():+.5f}, "
          f"SD {dm.std(ddof=1):.5f}; {len(ro)}-pool difference {fmt(ms(d))}")
    json.dump(res, open(f"{HERE}/logs/accept_null.json", "w"), indent=1)
    os.makedirs(f"{HERE}/logs/accept_null_history", exist_ok=True)
    json.dump(res, open(f"{HERE}/logs/accept_null_history/accept_null_{stamp.replace(':', '')}.json", "w"), indent=1)


def band_for(k, recool):
    b = 3 * S16 * np.sqrt(16 / k)
    src = f"3 s16 sqrt(16 / {k})"
    if recool and os.path.exists(f"{HERE}/logs/accept_null.json"):
        nj = json.load(open(f"{HERE}/logs/accept_null.json"))
        if nj.get("s_aa_k0", 0) >= 4:  # R6(c) amended: 3 s_AA(k0) sqrt(k0 / k), never wider than R6(a)
            b_aa = 3 * nj["s_aa"] * np.sqrt(nj["s_aa_k0"] / k)
            if b_aa < b:
                b, src = b_aa, f"3 s_AA({nj['s_aa_k0']}) sqrt({nj['s_aa_k0']} / {k}), s_AA = {nj['s_aa']:.5f} (accept_null of {nj.get('stamp')})"
    return float(b), src


def a3_a4(pairs, complete, recool, what):
    """pairs = [(twin val file, reference @last val file, reference selected val file or None, name)]."""
    k = len(pairs)
    pt, pr = pool([p[0] for p in pairs]), pool([p[1] for p in pairs])
    d = pt - pr
    band, src = band_for(k, recool)
    P(f"  A3 pool level ({what}): val NDCG {k}-pool twins {pt.mean():.5f} vs {k}-pool references @last {pr.mean():.5f}: D = {fmt(ms(d))} (n {len(d)}); band {band:.5f} = {src}")
    if all(p[2] and os.path.exists(p[2]) for p in pairs):
        ds = pt - pool([p[2] for p in pairs])
        P(f"     information: against the references' SELECTED checkpoints D = {fmt(ms(ds))}")
    verdict = None
    if complete:
        D = float(d.mean())
        verdict = "ACCEPT" if abs(D) <= band else ("DEFECT SUSPECTED (D below the band: inspect before any use)" if D < 0
                                                   else "UNEXPECTED (D above the band: inspect, do not claim it as a gain)")
        P(f"     A3 verdict: {verdict}")
        if verdict != "ACCEPT":
            FAIL.append(f"A3 {what}")
    else:
        P("     A3: incomplete set -> information only, no verdict (R6a)")
    P("  A4 member level (information): twin - reference @last, val")
    for p in pairs:
        dm = nd(zf(np.load(p[0]))) - nd(zf(np.load(p[1])))
        flag = "   <-- below -4 member SDs" if dm.mean() < -4 * MEMBER_SD else ""
        P(f"     {p[3]}: {fmt(ms(dm))}{flag}")
    return float(d.mean()), verdict


def mode_check(stage, arm=None, w5=None, abl=None, full=None, allow=(), logname=None):
    """w5 / abl / full / allow / logname are set only by the self-test (a mock harness tree with smoke expectations)."""
    global LOG
    LOG = open(f"{HERE}/logs/{logname or 'accept_' + stage}.txt", "a")
    P(f"\n== accept_twins.py check {stage} {arm or ''}   [{os.popen('date -u +%FT%TZ').read().strip()}]" + ("   [SELF-TEST MOCK]" if w5 else ""))
    FULL = full or FULL_REAL
    tab = stage_table(stage, arm, w5 or W5, abl or ABL)
    have = [t for t in tab if os.path.exists(f"{t[1]}/logs/{t[0]}.json")]
    P(f"   stage {stage}: {len(have)} of {len(tab)} twins finished")
    if not have:
        P("   no twin yet")
        return 2
    recool = tab[0][4]
    total = FULL["total_cd"] if recool else FULL["total10"]
    exp = dict(before=FULL["before"], after=FULL["after"], spe=FULL["spe"], pool=FULL["pool"], pool_hold=FULL["pool_hold"], n_hold=FULL["n_hold"])
    bar_r4 = 0.002 if recool else 0.005
    bar, bar_src = bar_r4, "R4"
    if os.path.exists(f"{HERE}/logs/a2_calib.json"):  # PREDECLARED R7: bars calibrated on the originals' measured in-sample gap
        cj = json.load(open(f"{HERE}/logs/a2_calib.json"))
        if not cj["void"]:
            bar, bar_src = cj["bar_recool" if recool else "bar_full"], f"R7 (G_s {cj['G_s']:+.5f}, floor {cj['floor']:.5f}; R4 bar {bar_r4})"
    P(f"   A2 bar in force: +{bar:.5f} [{bar_src}]")
    pairs = []
    for tw, h, ref, hr, _ in have:
        P(f"  A1 {tw} (reference {ref})")
        ok = a1(h, tw, exp, total, hz=stage in ("RH", "FH"), ref_json=f"{hr}/logs/{ref}.json" if stage in ("F", "P", "R", "RH") else None,
                quiet_rows=bool(w5), allow=allow)
        if os.path.exists(f"{h}/logs/{tw}_fulldata.json") and os.path.exists(f"{hr}/logs/{ref}.log"):
            ins, fj = last_insample(h, tw)
            oos = ref_last_holdout(hr, ref)
            check(f"A2 {tw}: IN-SAMPLE holdout NDCG at the last eval {ins:.5f} - reference's out-of-sample @last {oos:.5f} = {ins - oos:+.5f} >= +{bar:.5f}",
                  ins - oos >= bar, f"(the run's own count of holdout feeds per epoch: {fj['holdout_feeds_per_epoch']}; "
                  f"if that is 30000 and only A2 fails, the bar may be too high for this form: see PREDECLARED R6)")
        if ok and os.path.exists(vfile(h, tw)) and os.path.exists(vfile(hr, ref, True)):
            pairs.append((vfile(h, tw), vfile(hr, ref, True), vfile(hr, ref), f"{tw} vs {ref}"))
        elif ok:
            P(f"     (no @last val file of the reference {ref} yet: not in A3 / A4)")
    arms = sorted({t[0].split("_s9")[0] for t in tab})
    for a in arms:  # one pool per arm (R6a band for its size) ...
        sub = [p for p in pairs if p[3].startswith(a + "_s9")]
        if sub:
            a3_a4(sub, len(sub) == sum(t[0].startswith(a + "_s9") for t in tab), recool, f"arm {a}")
    if len(arms) > 1 and pairs:  # ... and the whole stage (for stage F: the 16-pool of R4)
        a3_a4(pairs, len(pairs) == len(tab), recool, f"all {len(tab)} twins of stage {stage}")
    P(f"   -> {'NO DEFECT FOUND' if not FAIL else 'DEFECT / INSPECT: ' + str(FAIL)}")
    return 1 if FAIL else 0


def mode_selftest():
    global LOG
    LOG = open(f"{HERE}/logs/accept_selftest.txt", "w")
    P(f"== accept_twins.py selftest   [{os.popen('date -u +%FT%TZ').read().strip()}]")
    P(" S1 the A1 parser on the CPU smoke runs (flag ON): must PASS with the smoke expectations (eval-capped predictions: row-count test off)")
    smoke = {"C_copy_E_on": dict(before=19806, after=20000, spe=73, pool=1004, pool_hold=194, n_hold=194, total=73, hz=False),
             "I_copy_P_on": dict(before=19806, after=20000, spe=73, pool=1004, pool_hold=194, n_hold=194, total=73, hz=False),
             "D_copy_K_on_mix03": dict(before=19814, after=20000, spe=74, pool=998, pool_hold=186, n_hold=186, total=74, hz=False),
             "J_copy_K_hz_on": dict(before=19814, after=20000, spe=74, pool=998, pool_hold=186, n_hold=186, total=74, hz=True)}
    SM = ("cliplog", "epochs", "eval_every", "select_min_step", "train_feeds")  # the lane's smoke settings
    smoke["C_copy_E_on"].update(ref=f"{W5}/logs/w5_e10_kd05w192_s951.json", allow=SM)
    smoke["I_copy_P_on"].update(ref=f"{ABL}/logs/abl_e10_pop1kd05w192_s951.json", allow=SM)
    smoke["D_copy_K_on_mix03"].update(ref=f"{W5}/logs/w5_kcd_ref_s951.json", allow=SM + ("cooldown_mix",))
    smoke["J_copy_K_hz_on"].update(ref=f"{W5}/logs/w5_kcd_ref_s951.json", allow=SM + ("cooldown_mix", "hzcd"))
    for tag, e in smoke.items():
        h = f"{HERE}/runs/{tag}"
        (lg,) = [p for p in glob.glob(f"{h}/logs/*.log")]
        v = os.path.basename(lg)[:-4]
        P(f"  {tag} ({v})")
        a1(h, v, e, e["total"], hz=e["hz"], quiet_rows=True, ref_json=e["ref"], allow=e["allow"])
    n_ok = len(FAIL)
    P(f" S1 result: {'ALL PASS' if not n_ok else 'FAILED ' + str(FAIL)}")
    P(" S2 the A1 parser on flag-OFF smoke runs: every one must be REJECTED (a twin trained without the flag must not pass)")
    for tag, e in (("B_copy_E_off", smoke["C_copy_E_on"]), ("F_copy_K_off_mix03", smoke["D_copy_K_on_mix03"])):
        h = f"{HERE}/runs/{tag}"
        (lg,) = [p for p in glob.glob(f"{h}/logs/*.log")]
        before = len(FAIL)
        a1(h, os.path.basename(lg)[:-4], e, e["total"], quiet_rows=True)
        rejected = len(FAIL) > before
        del FAIL[before:]
        P(f"  {tag}: {'REJECTED as it must be' if rejected else 'NOT REJECTED: the parser is too weak'}")
        if not rejected:
            FAIL.append(f"S2 {tag}")
    P(" S3 A3 / A4 numerics: the 16 originals' SELECTED checkpoints stand in for twins of the @last checkpoints (facts.py: -0.00013 +- 0.00016)")
    names = [f"w5_e10_{a}_s{s}" for a in W16_ARMS for s in range(951, 959)]
    D, verdict = a3_a4([(vfile(W5, v), vfile(W5, v, True), None, f"{v} selected vs @last") for v in names], True, False, "self-test, 16 W16 members")
    fj = json.load(open(f"{HERE}/logs/facts.json"))["w16_pool"]["d_val"][0]
    check("S3 pool difference == facts.py's number", abs(D - fj) < 2e-6, f"{D:+.6f} vs {fj:+.6f}")
    check("S3 verdict ACCEPT (|D| <= 0.00090)", verdict == "ACCEPT")
    P(" S4 the band logic on a deliberately broken 'twin' pool: the 16 @last members with 8 of them replaced by a 2-epoch screen model is not available, "
      "so the break is synthetic: per-feed z of each member + N(0, 1) noise (seed 0) -> must be DEFECT SUSPECTED")
    rng = np.random.default_rng(0)
    v0 = val()
    acc_t, acc_r = 0.0, 0.0
    for v in names:
        z = zf(np.load(vfile(W5, v, True)))
        acc_r = acc_r + z
        acc_t = acc_t + zf(z + rng.normal(0, 1.0, len(z)))
    d = nd(acc_t / 16) - nd(acc_r / 16)
    band, _ = band_for(16, False)
    check(f"S4 noisy pool D = {fmt(ms(d))} is below -band ({band:.5f})", d.mean() < -band)
    del v0
    P(" S5 names: every twin variant this script looks for == a '--variant' of the lane_full.sh DRY output of that stage, and nothing else")
    import subprocess
    for st, arms, seeds in (("F", W16_ARMS, range(951, 959)), ("FH", W16_ARMS, range(951, 959)), ("R", W16_ARMS, range(951, 959)),
                            ("RH", W16_ARMS, range(951, 959)), ("P", POP_ARMS, range(951, 955)), ("PR", POP_ARMS, range(951, 955))):
        jobs = " ".join(f"{st}:{a}:{x}" for a in arms for x in seeds)
        out = subprocess.run(["bash", f"{HERE}/lane_full.sh", "0", jobs], env=dict(os.environ, DRY="1"), capture_output=True, text=True).stdout
        dry = sorted(re.findall(r"--variant (\S+)", out))
        mine = sorted(t[0] for t in stage_table(st))
        refs_exist = sum(os.path.exists(f"{t[3]}/logs/{t[2]}.log") for t in stage_table(st))
        check(f"S5 stage {st}: {len(mine)} variants == lane_full.sh DRY", dry == mine, f"(reference logs present now: {refs_exist} of {len(mine)})")
    P(" S6 the check mode end to end on a MOCK harness tree (runs/_mock_accept: 8 'twins' of arm kd05w192 = symlinks to the flag-on smoke "
      "run C, references = the real originals): expected A1 PASS x 8, A2 DEFECT x 8 (a 73-step smoke model is no twin), A3 verdict below the band")
    mock = f"{HERE}/runs/_mock_accept"
    src = f"{HERE}/runs/C_copy_E_on"
    sv = "w5_smoke_e10_kd05w192_s951"
    for d in ("logs", "preds/holdout"):
        os.makedirs(f"{mock}/{d}", exist_ok=True)

    def link(a, b):
        if not os.path.lexists(b):
            os.symlink(a, b)
    for x in range(951, 959):
        tw, ref = f"w5_fd10_kd05w192_s{x}", f"w5_e10_kd05w192_s{x}"
        for suf in (".log", ".json", "_fulldata.json"):
            link(f"{src}/logs/{sv}{suf}", f"{mock}/logs/{tw}{suf}")
        for suf in ("_trainonly_val.npy", "@last_trainonly_val.npy", "_trainonly_test.npy", "@last_trainonly_test.npy"):
            link(f"{src}/preds/nn_{sv}{suf}", f"{mock}/preds/nn_{tw}{suf}")
        for suf in ("_holdoutINSAMPLE.npy", "@last_holdoutINSAMPLE.npy"):
            link(f"{src}/preds/holdout/nn_{sv}{suf}", f"{mock}/preds/holdout/nn_{tw}{suf}")
        for suf in (".log", ".json"):
            link(f"{W5}/logs/{ref}{suf}", f"{mock}/logs/{ref}{suf}")
        for suf in ("_trainonly_val.npy", "@last_trainonly_val.npy"):
            link(f"{W5}/preds/nn_{ref}{suf}", f"{mock}/preds/nn_{ref}{suf}")
    keep_log, before = LOG, len(FAIL)
    rc = mode_check("F", "kd05w192", w5=mock, full=dict(before=19806, after=20000, spe=73, total10=73, total_cd=73, pool=1004, pool_hold=194, n_hold=194),
                    allow=("cliplog", "epochs", "eval_every", "select_min_step", "train_feeds", "seed"), logname="accept_selftest_mock")
    got = FAIL[before:]
    del FAIL[before:]
    LOG = keep_log
    n_a2 = sum(g.startswith("A2 ") for g in got)
    n_a3 = sum(g.startswith("A3 ") for g in got)
    check("S6 mock: exit code 1; the only failures are 8 x A2 and the A3 verdict of the arm (A1 passed for all 8, so A3 / A4 ran)",
          rc == 1 and n_a2 == 8 and n_a3 == 1 and len(got) == 9, f"failures: {len(got)} = {n_a2} A2 + {n_a3} A3 + {len(got) - n_a2 - n_a3} other; log logs/accept_selftest_mock.txt")
    P(f"   -> {'SELFTEST ALL PASS' if not FAIL else 'SELFTEST FAILED: ' + str(FAIL)}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "null":
        mode_null()
        rc = 0
    elif mode == "selftest":
        rc = mode_selftest()
    elif mode == "check":
        rc = mode_check(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
    else:
        raise SystemExit(__doc__)
    sys.exit(rc)
