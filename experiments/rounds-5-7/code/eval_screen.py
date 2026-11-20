# Round 7: read-out of the screen. For each arm it pairs the two seeds with the baseline of the same seed on val
# and on the holdout, adds a run-to-run noise term to the standard error, and applies the promotion rule that
# was frozen before the first launch (docstring).
# Run as (docstring): CUDA_VISIBLE_DEVICES="" python nn_x/r7/eval_screen.py screen [arm ...]
# It wrote nn_x/r7/logs/eval_screen.{txt,json}; the text is ../results/round7_screen_readout.txt.
# Result: 0 of 13 arms promoted.
# Kept as it ran: paths and names refer to the lab tree (see lab/ and docs/glossary.md).
# Original: lab/nn_x/r7/eval_screen.py
"""Paired evaluation of the r7 screen (reports/r7_build.md §4: rules PRE-DECLARED before any r7 screen run).
CPU only, light (<= 16 threads), read-only on every prediction file.

  CUDA_VISIBLE_DEVICES="" python nn_x/r7/eval_screen.py screen [arm ...]   # stage 1: r7_<arm>_s{23,24} vs r7_base_s{23,24}
  CUDA_VISIBLE_DEVICES="" python nn_x/r7/eval_screen.py e10 [arm ...]      # stage 2+3: r7_<arm>_e10_s* vs r7_base_e10_s*
  CUDA_VISIBLE_DEVICES="" python nn_x/r7/eval_screen.py aa                 # A/A: r7_base_s23r vs r7_base_s23 (same seed)
  CUDA_VISIBLE_DEVICES="" python nn_x/r7/eval_screen.py pair <A> <B>       # one ad-hoc pair (B = treatment), info only
Options: --seeds 23,24  --base base  --sigma_val X --sigma_hold Y (run-noise SD of a paired delta; stage defaults below).
Arms with '+' (combinations) map to variant tags with '-', as in screen.sh. Names may carry a source prefix
("r7:" default, "sx:" nn_x/solvers, "r6:" nn_x/r6, "nn:" preds/ + nn/logs) for read-only cross-checks.

Metric: full-list NDCG per feed (src/metrics.ndcg_per_feed) of the train-only predictions of the holdout-selected
checkpoint; val = 14,423 val feeds; holdout = the fixed 30K random-like M>=50 train holdout.
Paired delta per seed = per-feed NDCG(arm_s) - NDCG(base_s); pooled = per-feed mean over seeds; SE_feed = sd / sqrt(n).
CALIBRATED SE = sqrt(SE_feed^2 + sigma_run^2 / n_seeds): sigma_run = the run-to-run SD of a same-config delta (GPU
non-determinism), stage 1 (2 ep): val 0.00035 (same-seed reruns differ by 0.0002-0.0005), holdout 0.00015; stage 2
(10 ep): val 0.0007, holdout 0.0003 (r6_recap §4.5, r7_research §6.3). Slices use SE_feed (stricter for WARN).

STAGE 1 (2-epoch screen) PROMOTE iff  pooled val delta > 1.5 SE_cal  AND  pooled holdout delta >= 0  AND
    (both seeds' val deltas > 0  OR  both seeds' holdout deltas > 0).
  batch2s also: COST-PASS iff pooled val delta >= -0.0005 AND holdout delta >= -0.0003 AND train minutes <= 0.75 x the
    paired baseline run on the same GPU in the same wave (start times within 60 min), for every seed with such a pair
    (no comparable pair -> COST undetermined). A COST-PASS goes to stage 2 as a cost lever.
  Information (not part of the rule): WARN slices (val M<50 / M>=50 / Tue / Thu, holdout late 20%) < -2 SE_feed;
    SCALE-SUSPECT (r7_research §2.4) iff the arm's growth |final - init| exceeds 1.25 x the paired baseline's growth on
    >= 2 of {top sim-bias scale, c_ln gain, retr.a, cross0 W_O RMS} (seed means; such a promotion carries the tag);
    diversity: rho_bb = corr(base_s23, base_s24), rho_x = mean cross-seed corr(arm, base), within-feed z, all val
    candidates; FAMILY-CANDIDATE iff not promoted, val delta >= -0.0010, holdout delta >= -0.0005, rho_x <= rho_bb - 0.005
    and the mixed 2-member ensemble {base_sA, arm_sB} >= the 2x-base ensemble on the holdout.
STAGE 2 (10 epochs, unified rule) ACCEPT iff pooled val delta > 2 SE_cal AND pooled holdout delta >= 0 AND no slice
    < -2 SE_feed.  batch2s adoption also needs the 2-member TO ensemble (arm pair) >= the base pair on the holdout AND
    corr(arm_s23, arm_s24) <= corr(base_s23, base_s24) + 0.005.
STAGE 3 (report only; the lead decides): z-mean of the 2 treated members vs the 2 baseline members (val, holdout, slices),
    correlations, and the end-minus-selected decline of every run (late-cooldown memorisation readout).
Writes nn_x/r7/logs/eval_<mode>[_<arms>].{txt,json}.
"""
import argparse
import json
import os
import sys

assert os.environ.get("CUDA_VISIBLE_DEVICES", "x") == "", "CPU only: run with CUDA_VISIBLE_DEVICES=''"
os.environ.setdefault("OMP_NUM_THREADS", "12")
os.environ.setdefault("POLARS_MAX_THREADS", "12")
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

sys.path.insert(0, "<repo>/src")
from metrics import ndcg_per_feed  # noqa: E402

ROOT = "<repo>"
R7 = f"{ROOT}/nn_x/r7"
DIRS = {"r7": (f"{R7}/preds", f"{R7}/logs"), "sx": (f"{ROOT}/nn_x/solvers/preds", f"{ROOT}/nn_x/solvers/logs"),
        "r6": (f"{ROOT}/nn_x/r6/preds", f"{ROOT}/nn_x/r6/logs"), "nn": (f"{ROOT}/preds", f"{ROOT}/nn/logs")}
# default arm list = the r7 batch (priority order; postln = reserve). The r6-owned arms (a4u, wd01, cset, typed, adrop0,
# a4w) are NOT in the r7 batch: nn_x/r6/queue_morning.sh launched them at 04:51 UTC (read nn_x/r6/logs/r6_eval_screen.txt)
ARMS = ["muon_in", "batch2s", "resdrop0", "pbld", "hc4", "dwa", "diff2", "swiglu", "derf", "hl2", "periln", "ftok",
        "postln"]
SIGMA = {"s1": (0.00035, 0.00015), "e10": (0.0007, 0.0003)}
SCALE_INIT = {"top_sim_bias": 8.0, "c_ln_gain": 1.0, "retr_a": 4.0, "cross0_WO_rms": None}  # None: from diag_init
OUT = []


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    OUT.append(s)


def split_name(n):
    src, v = n.split(":", 1) if ":" in n else ("r7", n)
    return DIRS[src], v


def fz(s, fid, M):
    s = s.astype(np.float64)
    mu = np.bincount(fid, s, len(M)) / M
    sd = np.sqrt(np.maximum(np.bincount(fid, s * s, len(M)) / M - mu ** 2, 0)) + 1e-6
    return (s - mu[fid]) / sd[fid]


def setup(kind):
    if kind == "val":
        f = pl.read_parquet(f"{ROOT}/data/flat/val/feeds.parquet")
        M = f["M"].to_numpy().astype(np.int64)
        off = f["cand_off"].to_numpy().astype(np.int64)
        y = np.load(f"{ROOT}/data/flat/val/cand_target.npy").astype(np.float64)
        day = (f["timestamp"].to_numpy() + 3 * 3600) // 86400  # MSK calendar day; val = Tue + Thu
        day = np.where(day <= day.min() + 1, 0, 1)
        fid = np.repeat(np.arange(len(M)), M)
        luo = np.load(f"{R7}/feat_cache/val/it_hist_all_luo.npy", mmap_mode="r")
        cold = np.nan_to_num(np.asarray(luo, dtype=np.float64), nan=0.0) == 0
        npos = np.bincount(fid, y, len(M))
        ncold = np.bincount(fid, y * cold, len(M))
        slices = {"all": np.ones(len(M), bool), "M<50": M < 50, "M>=50": M >= 50, "Tue": day == 0, "Thu": day == 1,
                  "coldpos": (npos > 0) & (ncold == npos), "warmpos": (npos > 0) & (ncold == 0)}
        rule_slices = ("M<50", "M>=50", "Tue", "Thu")
        path = lambda n: f"{split_name(n)[0][0]}/nn_{split_name(n)[1]}_trainonly_val.npy"  # noqa: E731
    else:
        metas = [f"{d[0]}/holdout/holdout_meta.npy" for d in DIRS.values() if os.path.exists(f"{d[0]}/holdout/holdout_meta.npy")]
        M, off = np.load(metas[0])
        for m2 in metas[1:]:
            assert (np.load(m2) == np.stack([M, off])).all(), f"holdout meta mismatch {m2}"
        y = np.load(metas[0].replace("holdout_meta", "holdout_y")).astype(np.float64)
        tr = pl.read_parquet(f"{ROOT}/data/flat/train/feeds.parquet", columns=["timestamp", "M"])
        pr = np.nan_to_num(np.load(f"{ROOT}/eda/labels_position/out/train_prandom.npz")["p_random"].astype(np.float32),
                           nan=0.0)
        trM = tr["M"].to_numpy()
        pr[trM < 50] = 0.0
        cand = np.where((pr > 0.9) & (trM >= 50))[0]
        hold = np.sort(np.random.default_rng(12345).choice(cand, 30000, replace=False))
        assert (trM[hold] == M).all(), "holdout feed selection mismatch"
        ts = tr["timestamp"].to_numpy()[hold]
        fid = np.repeat(np.arange(len(M)), M)
        slices = {"all": np.ones(len(M), bool), "late20%": ts >= np.quantile(ts, 0.8)}
        rule_slices = ("late20%",)
        path = lambda n: f"{split_name(n)[0][0]}/holdout/nn_{split_name(n)[1]}_holdout.npy"  # noqa: E731
    return dict(M=M, off=off, y=y, fid=fid, path=path, slices=slices, rule_slices=rule_slices, cache={}, pf={})


def load_z(ctx, n):
    if n not in ctx["cache"]:
        s = np.load(ctx["path"](n))
        assert len(s) == len(ctx["y"]), f"{n}: {len(s)} scores vs {len(ctx['y'])} candidates"
        ctx["cache"][n] = fz(s, ctx["fid"], ctx["M"])
    return ctx["cache"][n]


def per_feed(ctx, spec):
    """spec = 'a+b+...' -> per-feed NDCG of the equal-weight per-feed z-mean ensemble (a single name = the member)."""
    if spec not in ctx["pf"]:
        sc = np.mean([load_z(ctx, n) for n in spec.split("+")], 0)
        ctx["pf"][spec] = ndcg_per_feed(sc, ctx["y"], ctx["off"], ctx["M"])
    return ctx["pf"][spec]


def stats(d, slices, sigma=0.0, n_seeds=1):
    out = {}
    for k, m in slices.items():
        se = float(d[m].std() / np.sqrt(m.sum()))
        se_cal = float(np.sqrt(se ** 2 + sigma ** 2 / n_seeds)) if k == "all" else se
        out[k] = dict(delta=float(d[m].mean()), se=se, se_cal=se_cal, n=int(m.sum()),
                      z=float(d[m].mean() / (se_cal + 1e-12)))
    return out


def fmt(st, keys=None):
    return "  ".join(f"{k} {v['delta']:+.5f}+-{v['se_cal']:.5f} (z {v['z']:+.2f})" for k, v in st.items()
                     if keys is None or k in keys)


def compare(ctx, pairs, sigma):
    ds, out = [], {"pairs": []}
    for a, b in pairs:
        pa, pb = per_feed(ctx, a), per_feed(ctx, b)
        ds.append(pb - pa)
        out["pairs"].append(dict(A=a, B=b, A_mean=float(pa.mean()), B_mean=float(pb.mean()),
                                 slices=stats(pb - pa, ctx["slices"], sigma, 1)))
    out["pooled"] = stats(np.mean(ds, 0), ctx["slices"], sigma, len(pairs))
    return out


def corr(ctx, a, b):
    return float(np.corrcoef(load_z(ctx, a), load_z(ctx, b))[0, 1])


def run_json(n):
    (_, ldir), v = split_name(n)
    return json.load(open(f"{ldir}/{v}.json"))


def readout(n):
    j = run_json(n)
    h = j["history"]
    sel = [r for r in h if r["step"] == j["best_step"]][0]
    return dict(best_step=j["best_step"], total=h[-1]["step"], sel_val=sel["val_ndcg"], sel_hold=sel["holdout_ndcg"],
                final_val=h[-1]["val_ndcg"], final_hold=h[-1]["holdout_ndcg"], minutes=j["train_minutes"],
                gpu=j.get("gpu"), start=j.get("start_unix"), peak=j.get("peak_reserved_gb"), diag=j.get("diag"),
                diag_init=j.get("diag_init"))


def scale_q(diag):
    if not diag:
        return {}
    bs = [max(v) for k, v in diag.items() if k.endswith("_bias_scale") and k.startswith("cross")]
    return {"top_sim_bias": max(bs) if bs else None, "c_ln_gain": diag.get("c_ln_gain"), "retr_a": diag.get("retr_a"),
            "cross0_WO_rms": diag.get("cross0_WO_rms")}


def growth(r):
    q, q0 = scale_q(r["diag"]), scale_q(r["diag_init"])
    out = {}
    for k, init in SCALE_INIT.items():
        i0 = q0.get(k) if q0.get(k) is not None else init  # the run's own init (postln scales W_O) else the constant
        if q.get(k) is not None and i0 is not None:
            out[k] = abs(q[k] - i0)
    return out


def scale_suspect(base_runs, arm_runs):
    gb = [growth(r) for r in base_runs]
    ga = [growth(r) for r in arm_runs]
    keys = [k for k in SCALE_INIT if all(k in g for g in gb + ga)]
    flags = {}
    for k in keys:
        b, a = np.mean([g[k] for g in gb]), np.mean([g[k] for g in ga])
        flags[k] = (round(float(a), 4), round(float(b), 4), bool(a > 1.25 * b))
    return sum(v[2] for v in flags.values()) >= 2, flags


def runs_table(names):
    P("== runs (selected step / final step; train minutes; GPU; peak reserved GB)")
    rr = {}
    for n in names:
        try:
            r = readout(n)
            rr[n] = r
            P(f"  {n:28s} best {r['best_step']:>6}/{r['total']} val {r['sel_val']:.5f} hold {r['sel_hold']:.5f} | "
              f"final val {r['final_val']:.5f} hold {r['final_hold']:.5f} (end-sel {r['final_val'] - r['sel_val']:+.5f} / "
              f"{r['final_hold'] - r['sel_hold']:+.5f}) | {r['minutes']:.0f} min | GPU {r['gpu']} | "
              f"{(r['peak'] or 0):.0f} GB")
        except FileNotFoundError:
            P(f"  {n:28s} (missing)")
    return rr


def arm_block(vctx, hctx, base, arm, seeds, sig, stage, rr, armname):
    pairs = [(base(s), arm(s)) for s in seeds]
    rv, rh = compare(vctx, pairs, sig[0]), compare(hctx, pairs, sig[1])
    for p in rv["pairs"]:
        P(f"  VAL  {p['B']} {p['B_mean']:.5f} vs {p['A']} {p['A_mean']:.5f}: {fmt(p['slices'])}")
    P(f"  VAL  pooled: {fmt(rv['pooled'])}")
    for p in rh["pairs"]:
        P(f"  HOLD {p['B']} {p['B_mean']:.5f} vs {p['A']} {p['A_mean']:.5f}: {fmt(p['slices'])}")
    P(f"  HOLD pooled: {fmt(rh['pooled'])}")
    res = dict(val=rv, hold=rh)
    dv, dh = rv["pooled"]["all"], rh["pooled"]["all"]
    warn = [f"val {k}" for k in vctx["rule_slices"] if rv["pooled"][k]["z"] < -2] + \
           [f"hold {k}" for k in hctx["rule_slices"] if rh["pooled"][k]["z"] < -2]
    seed_pos_v = all(p["slices"]["all"]["delta"] > 0 for p in rv["pairs"])
    seed_pos_h = all(p["slices"]["all"]["delta"] > 0 for p in rh["pairs"])
    if len(seeds) >= 2:
        s1, s2 = seeds[:2]
        rho_bb = corr(vctx, base(s1), base(s2))
        rho_x = 0.5 * (corr(vctx, arm(s1), base(s2)) + corr(vctx, arm(s2), base(s1)))
        rho_aa = corr(vctx, arm(s1), arm(s2))
        rho_same = 0.5 * (corr(vctx, arm(s1), base(s1)) + corr(vctx, arm(s2), base(s2)))
        P(f"  CORR val (within-feed z): base-base {rho_bb:.4f} | arm-base cross-seed {rho_x:.4f} | arm-arm {rho_aa:.4f} "
          f"| arm-base same-seed {rho_same:.4f}")
        A = f"{base(s1)}+{base(s2)}"
        mix = [(A, f"{base(s1)}+{arm(s2)}"), (A, f"{base(s2)}+{arm(s1)}")]
        ev, eh = compare(vctx, mix, 0.0), compare(hctx, mix, 0.0)
        P(f"  ENS mixed {{base_sA, arm_sB}} vs 2x base: VAL {fmt(ev['pooled'], ('all',))}  HOLD {fmt(eh['pooled'], ('all',))}")
        Z = f"{arm(s1)}+{arm(s2)}"
        zv, zh = compare(vctx, [(A, Z)], sig[0] / np.sqrt(2)), compare(hctx, [(A, Z)], sig[1] / np.sqrt(2))
        P(f"  ENS 2 arm members vs 2 base members (z-mean): VAL {zv['pairs'][0]['B_mean']:.5f} vs "
          f"{zv['pairs'][0]['A_mean']:.5f}: {fmt(zv['pooled'])}")
        P(f"                                                HOLD {zh['pairs'][0]['B_mean']:.5f} vs "
          f"{zh['pairs'][0]['A_mean']:.5f}: {fmt(zh['pooled'])}")
        res.update(rho_bb=rho_bb, rho_x=rho_x, rho_aa=rho_aa, rho_same=rho_same, ens_mixed_val=ev, ens_mixed_hold=eh,
                   ens2_val=zv, ens2_hold=zh)
    try:
        sus, flags = scale_suspect([rr[base(s)] for s in seeds], [rr[arm(s)] for s in seeds])
        P(f"  SCALE growth |final-init| arm vs base (seed means): {flags}  -> SCALE-SUSPECT {sus}")
    except (KeyError, TypeError):
        sus, flags = False, {}
    res.update(scale_suspect=sus, scale_flags=flags, warn=warn)
    if stage == "s1":
        promote = dv["delta"] > 1.5 * dv["se_cal"] and dh["delta"] >= 0 and (seed_pos_v or seed_pos_h)
        verdict = "PROMOTE -> stage 2 (10-epoch pair)" if promote else "FAIL"
        if armname.split("+")[0] == "batch2s" or "batch2s" in armname.split("+"):
            cmp_ = []
            for s in seeds:
                ra, rb = rr.get(arm(s)), rr.get(base(s))
                if ra and rb and ra["gpu"] == rb["gpu"] and ra["start"] and rb["start"] and abs(ra["start"] - rb["start"]) <= 3600:
                    cmp_.append(ra["minutes"] / rb["minutes"])
            P(f"  COST: train-minute ratios on comparable (same GPU, same wave) pairs: "
              f"{[round(x, 3) for x in cmp_] or 'none comparable'}")
            cost_ok = dv["delta"] >= -0.0005 and dh["delta"] >= -0.0003 and cmp_ and all(x <= 0.75 for x in cmp_)
            res["cost_ratios"] = cmp_
            if not promote:
                verdict = ("COST-PASS -> stage 2 as a cost lever" if cost_ok else
                           "COST undetermined (no comparable pair)" if not cmp_ and dv["delta"] >= -0.0005
                           and dh["delta"] >= -0.0003 else "FAIL")
        if not promote and len(seeds) >= 2 and verdict == "FAIL" and dv["delta"] >= -0.0010 and dh["delta"] >= -0.0005 \
                and res["rho_x"] <= res["rho_bb"] - 0.005 and res["ens_mixed_hold"]["pooled"]["all"]["delta"] >= 0:
            verdict = "FAIL as a stronger member; FAMILY-CANDIDATE (info for the lead)"
        if promote and sus:
            verdict += " [SCALE-SUSPECT]"
    else:
        accept = dv["delta"] > 2 * dv["se_cal"] and dh["delta"] >= 0 and not warn
        verdict = "ACCEPT (unified rule)" if accept else "REJECT"
        if "batch2s" in armname.split("+") and len(seeds) >= 2:
            ok_b = res["ens2_hold"]["pooled"]["all"]["delta"] >= 0 and res["rho_aa"] <= res["rho_bb"] + 0.005
            P(f"  batch2s adoption extras: 2-member TO ensemble hold delta "
              f"{res['ens2_hold']['pooled']['all']['delta']:+.5f}, rho_aa {res['rho_aa']:.4f} vs rho_bb + 0.005 "
              f"{res['rho_bb'] + 0.005:.4f} -> {'OK' if ok_b else 'NOT OK'}")
            if accept and not ok_b:
                verdict = "REJECT (batch2s adoption extras)"
    res.update(verdict=verdict, seed_pos_val=seed_pos_v, seed_pos_hold=seed_pos_h)
    P(f"  ==> {verdict}" + (f"   WARN slices < -2 SE: {warn}" if warn else ""))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["screen", "e10", "aa", "pair"])
    ap.add_argument("names", nargs="*")
    ap.add_argument("--seeds", default="23,24")
    ap.add_argument("--base", default="base")
    ap.add_argument("--sigma_val", type=float, default=None)
    ap.add_argument("--sigma_hold", type=float, default=None)
    ap.add_argument("--prefix", default="r7:r7_", help="variant prefix (e.g. sx:sx_ to re-score the nn_x/solvers arms)")
    a = ap.parse_args()
    seeds = tuple(int(x) for x in a.seeds.split(","))
    stage = "e10" if a.mode == "e10" else "s1"
    sig = (a.sigma_val if a.sigma_val is not None else SIGMA[stage][0],
           a.sigma_hold if a.sigma_hold is not None else SIGMA[stage][1])
    vctx, hctx = setup("val"), setup("hold")
    res, tag = {}, a.mode
    P(f"== r7 eval_screen {a.mode}; sigma_run val {sig[0]} holdout {sig[1]}; seeds {seeds}")
    if a.mode in ("screen", "e10"):
        arms = a.names or ARMS
        tag = f"{a.mode}" + ("_" + "_".join(x.replace("+", "-") for x in a.names) if a.names else "") + \
            ("" if a.prefix == "r7:r7_" else "_" + a.prefix.split(":")[0])
        suf = "_e10" if a.mode == "e10" else ""
        vn = lambda arm, s: f"{a.prefix}{arm.replace('+', '-')}{suf}_s{s}"  # noqa: E731
        base = lambda s: vn(a.base, s)  # noqa: E731
        rr = runs_table([base(s) for s in seeds] + [vn(x, s) for x in arms for s in seeds])
        for x in arms:
            ok = all(os.path.exists(vctx["path"](vn(x, s))) and os.path.exists(hctx["path"](vn(x, s))) for s in seeds) \
                and all(os.path.exists(vctx["path"](base(s))) and os.path.exists(hctx["path"](base(s))) for s in seeds)
            if not ok:
                P(f"\n== {x}: incomplete (preds missing), skipped")
                continue
            P(f"\n== {x} vs {a.base} (B = treatment)")
            res[x] = arm_block(vctx, hctx, base, lambda s, x=x: vn(x, s), seeds, sig, stage, rr, x)
        P("\n== summary")
        for x, r in res.items():
            P(f"  {x:10s} val {r['val']['pooled']['all']['delta']:+.5f} (z {r['val']['pooled']['all']['z']:+.2f})  "
              f"hold {r['hold']['pooled']['all']['delta']:+.5f} (z {r['hold']['pooled']['all']['z']:+.2f})  "
              + (f"rho_x {r['rho_x']:.4f} vs rho_bb {r['rho_bb']:.4f}  " if "rho_x" in r else "") + f"-> {r['verdict']}")
    elif a.mode == "aa":
        A, B = "r7:r7_base_s23", "r7:r7_base_s23r"
        runs_table([A, B])
        rv, rh = compare(vctx, [(A, B)], 0.0), compare(hctx, [(A, B)], 0.0)
        dv, dh = rv["pooled"]["all"], rh["pooled"]["all"]
        cv = corr(vctx, A, B)
        P(f"  A/A same seed: val {dv['delta']:+.5f} (SE_feed {dv['se']:.5f}) hold {dh['delta']:+.5f} (SE_feed {dh['se']:.5f})"
          f"; same-seed corr {cv:.4f}")
        ok = abs(dv["delta"]) <= 0.0010 and abs(dh["delta"]) <= 0.0005
        within = abs(dv["delta"]) <= 2 * np.sqrt(dv["se"] ** 2 + sig[0] ** 2) and \
            abs(dh["delta"]) <= 2 * np.sqrt(dh["se"] ** 2 + sig[1] ** 2)
        P(f"  ==> fork check {'OK' if ok else 'STOP: audit the copy'}; calibration "
          f"{'consistent' if within else 'EXCEEDED: raise --sigma_val/--sigma_hold to the A/A |delta| before reading arms'}")
        res = dict(val=rv, hold=rh, corr=cv, ok=ok, calib_ok=within)
    else:
        A, B = a.names
        rr = runs_table([A, B])
        rv, rh = compare(vctx, [(A, B)], sig[0]), compare(hctx, [(A, B)], sig[1])
        P(f"  VAL  {fmt(rv['pooled'])}\n  HOLD {fmt(rh['pooled'])}\n  corr val {corr(vctx, A, B):.4f}")
        res = dict(val=rv, hold=rh)
        tag = "pair_" + "_vs_".join(x.split(":")[-1] for x in (A, B))
    json.dump(res, open(f"{R7}/logs/eval_{tag}.json", "w"), indent=1, default=float)
    open(f"{R7}/logs/eval_{tag}.txt", "w").write("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
