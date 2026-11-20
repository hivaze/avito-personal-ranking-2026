"""Paired evaluation of the r6 screen (reports/r6_design.md section 4). CPU only, read-only on every prediction file.

  CUDA_VISIBLE_DEVICES="" python nn_x/r6/r6_eval.py screen [arm ...]     # 2-epoch screen vs sx_base_s23/s24
  CUDA_VISIBLE_DEVICES="" python nn_x/r6/r6_eval.py aa                   # A/A: r6_base_s23 vs sx_base_s23 (fork check)
  CUDA_VISIBLE_DEVICES="" python nn_x/r6/r6_eval.py e10 <arm>            # 10-epoch pair vs F7 v17_pre2s_s101/s102
  CUDA_VISIBLE_DEVICES="" python nn_x/r6/r6_eval.py calib                # diversity calibration on the old sx arms

Names: "sx:<v>" = nn_x/solvers/{preds,logs}, "r6:<v>" = nn_x/r6/{preds,logs}, "nn:<v>" = preds/ + nn/logs (live, read-only).
Screen variants: r6_<arm>_s<seed>; control sx_base_s<seed> (identical args, verified by tests/test_r6.py T12).

PRE-DECLARED RULES (frozen 2026-11-12 before any r6 GPU run; reports/r6_design.md section 4.3):
  per arm, pooled over seeds 23/24 (per-feed mean of the two paired deltas, SE = sd/sqrt(n_feeds)), train-only preds at
  the holdout-selected checkpoint (2-epoch runs select the last step in 29/29 earlier runs):
  PASS (recipe change / stronger member):  val delta > 2 SE  AND  holdout delta >= -1 SE.
  FAMILY-CANDIDATE (same strength, different function), only if not PASS:
      val delta >= -0.0010 AND holdout delta >= -0.0005 (non-inferior, about -2 SE each)
      AND rho_x <= rho_bb - 0.005, with rho_x = mean(corr(arm_s23, base_s24), corr(arm_s24, base_s23)) and
          rho_bb = corr(base_s23, base_s24) (within-feed z, Pearson over all val candidates)
      AND the 2-member mixed ensemble {base_sA, arm_sB} >= the 2x-base ensemble on the holdout (pooled delta >= 0).
  otherwise FAIL. Slices (val M<50 / M>=50 / Tue / Thu / all-cold-positive / all-warm-positive; holdout late 20%) are
  reported; a slice < -2 SE is flagged WARN and must be re-checked in the 10-epoch pair (the holdout is blind to M<50).
  A/A (fork check): |r6_base_s23 - sx_base_s23| <= 0.0010 val and <= 0.0005 holdout, else STOP and audit the fork.
  e10 (2 seeds, vs F7 s101/s102, same seeds = same init and data order):
      recipe winner proceeds to a 6-seed family iff pooled val delta >= 0 AND holdout delta >= -1 SE;
      family candidate proceeds iff |val delta| <= 0.002 AND |holdout delta| <= 0.002 (gate (a) proxy).
Writes nn_x/r6/logs/r6_eval_<mode>[_<arms>].{json,txt}.
"""
import json
import os
import sys

assert os.environ.get("CUDA_VISIBLE_DEVICES", "x") == "", "CPU only"
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

sys.path.insert(0, "<repo>/src")
from metrics import ndcg_per_feed  # noqa: E402

ROOT = "<repo>"
R6 = f"{ROOT}/nn_x/r6"
DIRS = {"sx": (f"{ROOT}/nn_x/solvers/preds", f"{ROOT}/nn_x/solvers/logs"),
        "r6": (f"{R6}/preds", f"{R6}/logs"),
        "nn": (f"{ROOT}/preds", f"{ROOT}/nn/logs")}
ARMS = ["a4u", "wd01", "cset", "typed", "adrop0", "a4w"]
OUT = []


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    OUT.append(s)


def split_name(n):
    src, v = n.split(":", 1)
    return DIRS[src], v


def fz(s, fid, M):
    s = s.astype(np.float64)
    mu = np.bincount(fid, s, len(M)) / M
    sd = np.sqrt(np.maximum(np.bincount(fid, s * s, len(M)) / M - mu ** 2, 0)) + 1e-6
    return (s - mu[fid]) / sd[fid]


def setup(mode):
    if mode == "val":
        f = pl.read_parquet(f"{ROOT}/data/flat/val/feeds.parquet")
        M = f["M"].to_numpy().astype(np.int64)
        off = f["cand_off"].to_numpy().astype(np.int64)
        y = np.load(f"{ROOT}/data/flat/val/cand_target.npy").astype(np.float64)
        day = (f["timestamp"].to_numpy() + 3 * 3600) // 86400
        day = np.where(day <= day.min() + 1, 0, 1)
        fid = np.repeat(np.arange(len(M)), M)
        luo = np.load(f"{R6}/feat_cache/val/it_hist_all_luo.npy", mmap_mode="r")
        cold = np.nan_to_num(np.asarray(luo, dtype=np.float64), nan=0.0) == 0  # CPU2 definition: it_hist_all_luo == 0
        npos = np.bincount(fid, y, len(M))
        ncold = np.bincount(fid, y * cold, len(M))
        slices = {"all": np.ones(len(M), bool), "M<50": M < 50, "M>=50": M >= 50, "Tue": day == 0, "Thu": day == 1,
                  "coldpos": (npos > 0) & (ncold == npos), "warmpos": (npos > 0) & (ncold == 0)}
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
        assert (trM[hold] == M).all()
        ts = tr["timestamp"].to_numpy()[hold]
        fid = np.repeat(np.arange(len(M)), M)
        slices = {"all": np.ones(len(M), bool), "late20%": ts >= np.quantile(ts, 0.8)}
        path = lambda n: f"{split_name(n)[0][0]}/holdout/nn_{split_name(n)[1]}_holdout.npy"  # noqa: E731
    return dict(M=M, off=off, y=y, fid=fid, path=path, slices=slices, cache={})


def load_z(ctx, n):
    if n not in ctx["cache"]:
        ctx["cache"][n] = fz(np.load(ctx["path"](n)), ctx["fid"], ctx["M"])
    return ctx["cache"][n]


def per_feed(ctx, spec):
    """spec = 'a+b+...' (equal-weight per-feed z ensemble)."""
    sc = np.mean([load_z(ctx, n) for n in spec.split("+")], 0)
    return ndcg_per_feed(sc, ctx["y"], ctx["off"], ctx["M"])


def stats(d, slices):
    return {k: dict(delta=float(d[m].mean()), se=float(d[m].std() / np.sqrt(m.sum())), n=int(m.sum()),
                    z=float(d[m].mean() / (d[m].std() / np.sqrt(m.sum()) + 1e-12))) for k, m in slices.items()}


def fmt(st):
    return "  ".join(f"{k} {v['delta']:+.5f}+-{v['se']:.5f} (z {v['z']:+.2f})" for k, v in st.items())


def compare(ctx, pairs):
    ds, out = [], {"pairs": []}
    for a, b in pairs:
        pa, pb = per_feed(ctx, a), per_feed(ctx, b)
        ds.append(pb - pa)
        out["pairs"].append(dict(A=a, B=b, A_mean=float(pa.mean()), B_mean=float(pb.mean()),
                                 slices=stats(pb - pa, ctx["slices"])))
    out["pooled"] = stats(np.mean(ds, 0), ctx["slices"])
    return out


def corr(ctx, a, b):
    return float(np.corrcoef(load_z(ctx, a), load_z(ctx, b))[0, 1])


def log_readout(n):
    (pdir, ldir), v = split_name(n)
    j = json.load(open(f"{ldir}/{v}.json"))
    h = j["history"]
    best = max(h, key=lambda r: r["holdout_ndcg"])
    return dict(best_step=j["best_step"], total=h[-1]["step"], sel_val=best["val_ndcg"], sel_hold=best["holdout_ndcg"],
                final_val=h[-1]["val_ndcg"], final_hold=h[-1]["holdout_ndcg"], minutes=j["train_minutes"])


def arm_block(vctx, hctx, base, arm, seeds, rules=True):
    """base/arm: functions seed -> name. Returns the result dict and prints the readout."""
    pairs = [(base(s), arm(s)) for s in seeds]
    rv, rh = compare(vctx, pairs), compare(hctx, pairs)
    for p in rv["pairs"]:
        P(f"  VAL  {p['B']} {p['B_mean']:.5f} vs {p['A']} {p['A_mean']:.5f}: {fmt(p['slices'])}")
    P(f"  VAL  pooled: {fmt(rv['pooled'])}")
    for p in rh["pairs"]:
        P(f"  HOLD {p['B']} {p['B_mean']:.5f} vs {p['A']} {p['A_mean']:.5f}: {fmt(p['slices'])}")
    P(f"  HOLD pooled: {fmt(rh['pooled'])}")
    res = dict(val=rv, hold=rh)
    if len(seeds) == 2:
        s1, s2 = seeds
        rho_bb = corr(vctx, base(s1), base(s2))
        rho_x = 0.5 * (corr(vctx, arm(s1), base(s2)) + corr(vctx, arm(s2), base(s1)))
        rho_aa = corr(vctx, arm(s1), arm(s2))
        rho_same = 0.5 * (corr(vctx, arm(s1), base(s1)) + corr(vctx, arm(s2), base(s2)))
        rho_bb_h = corr(hctx, base(s1), base(s2))
        rho_x_h = 0.5 * (corr(hctx, arm(s1), base(s2)) + corr(hctx, arm(s2), base(s1)))
        P(f"  CORR val: base-base {rho_bb:.4f} | arm-base cross-seed {rho_x:.4f} | arm-arm {rho_aa:.4f} | "
          f"arm-base same-seed {rho_same:.4f};  holdout: base-base {rho_bb_h:.4f} | arm-base cross {rho_x_h:.4f}")
        A = f"{base(s1)}+{base(s2)}"
        mix = [(A, f"{base(s1)}+{arm(s2)}"), (A, f"{base(s2)}+{arm(s1)}")]
        ev, eh = compare(vctx, mix), compare(hctx, mix)
        P(f"  ENS VAL  2-member mixed vs 2x base: {fmt(ev['pooled'])}  (base pair {ev['pairs'][0]['A_mean']:.5f})")
        P(f"  ENS HOLD 2-member mixed vs 2x base: {fmt(eh['pooled'])}")
        e4v = compare(vctx, [(A, f"{A}+{arm(s1)}+{arm(s2)}")])
        e4h = compare(hctx, [(A, f"{A}+{arm(s1)}+{arm(s2)}")])
        P(f"  ENS 4-member (2 base + 2 arm) vs 2x base (info, unequal count): VAL {fmt({'all': e4v['pooled']['all']})}"
          f"  HOLD {fmt({'all': e4h['pooled']['all']})}")
        res.update(rho_bb=rho_bb, rho_x=rho_x, rho_aa=rho_aa, rho_same=rho_same, rho_bb_hold=rho_bb_h,
                   rho_x_hold=rho_x_h, ens_val=ev, ens_hold=eh, ens4_val=e4v, ens4_hold=e4h)
    if rules:
        dv, dh = rv["pooled"]["all"], rh["pooled"]["all"]
        warn = [f"val {k}" for k, s in rv["pooled"].items() if s["z"] < -2] + \
               [f"hold {k}" for k, s in rh["pooled"].items() if s["z"] < -2]
        if dv["z"] > 2 and dh["delta"] >= -dh["se"]:
            verdict = "PASS (recipe change / stronger member) -> 10-epoch pair"
        elif (len(seeds) == 2 and dv["delta"] >= -0.0010 and dh["delta"] >= -0.0005
              and res["rho_x"] <= res["rho_bb"] - 0.005 and res["ens_hold"]["pooled"]["all"]["delta"] >= 0):
            verdict = "FAMILY-CANDIDATE (non-inferior, lower correlation, useful in a mix) -> 10-epoch pair"
        else:
            verdict = "FAIL"
        res.update(verdict=verdict, warn=warn)
        P(f"  ==> {verdict}" + (f"   WARN slices < -2 SE: {warn}" if warn else ""))
    return res


def main():
    mode = sys.argv[1]
    vctx, hctx = setup("val"), setup("hold")
    res, tag = {}, mode
    if mode == "screen":
        arms = sys.argv[2:] or ARMS
        tag = "screen_" + "_".join(arms)
        seeds = (23, 24)
        base = lambda s: f"sx:sx_base_s{s}"  # noqa: E731
        P("== runs (log readouts: selected / final step)")
        for n in [base(s) for s in seeds] + [f"r6:r6_{a}_s{s}" for a in arms for s in seeds]:
            try:
                r = log_readout(n)
                P(f"  {n:24s} best {r['best_step']:>6}/{r['total']} val {r['sel_val']:.5f} hold {r['sel_hold']:.5f} | "
                  f"final val {r['final_val']:.5f} hold {r['final_hold']:.5f} | {r['minutes']:.0f} min")
            except FileNotFoundError:
                P(f"  {n:24s} (missing)")
        for a in arms:
            if not all(os.path.exists(vctx["path"](f"r6:r6_{a}_s{s}")) and os.path.exists(hctx["path"](f"r6:r6_{a}_s{s}"))
                       for s in seeds):
                P(f"\n== {a}: incomplete, skipped")
                continue
            P(f"\n== {a} vs sx_base (B = treatment)")
            res[a] = arm_block(vctx, hctx, base, lambda s, a=a: f"r6:r6_{a}_s{s}", seeds)
        P("\n== summary")
        for a, r in res.items():
            P(f"  {a:7s} val {r['val']['pooled']['all']['delta']:+.5f} (z {r['val']['pooled']['all']['z']:+.2f})  "
              f"hold {r['hold']['pooled']['all']['delta']:+.5f} (z {r['hold']['pooled']['all']['z']:+.2f})  "
              f"rho_x {r['rho_x']:.4f} vs rho_bb {r['rho_bb']:.4f}  -> {r['verdict']}")
    elif mode == "aa":
        P("== A/A fork check: r6_base_s23 (nn_x/r6 code, flags off) vs sx_base_s23 (nn_x/solvers code), same args/seed")
        r = arm_block(vctx, hctx, lambda s: f"sx:sx_base_s{s}", lambda s: f"r6:r6_base_s{s}", (23,), rules=False)
        dv = r["val"]["pairs"][0]["B_mean"] - r["val"]["pairs"][0]["A_mean"]
        dh = r["hold"]["pairs"][0]["B_mean"] - r["hold"]["pairs"][0]["A_mean"]
        rs = corr(vctx, "sx:sx_base_s23", "r6:r6_base_s23")
        ok = abs(dv) <= 0.0010 and abs(dh) <= 0.0005
        P(f"  |dval| {abs(dv):.5f} |dhold| {abs(dh):.5f} same-seed corr {rs:.4f} ==> "
          f"{'OK (GPU non-determinism level)' if ok else 'STOP: audit the fork before reading any arm'}")
        res = dict(aa=r, dval=dv, dhold=dh, corr=rs, ok=ok)
    elif mode == "e10":
        a = sys.argv[2]
        tag = f"e10_{a}"
        seeds = (101, 102)
        base = lambda s: f"nn:v17_pre2s_s{s}"  # noqa: E731
        arm = lambda s: f"r6:r6_{a}_e10_s{s}"  # noqa: E731
        for n in [base(s) for s in seeds] + [arm(s) for s in seeds]:
            r = log_readout(n)
            P(f"  {n:28s} best {r['best_step']:>6}/{r['total']} val {r['sel_val']:.5f} hold {r['sel_hold']:.5f} | "
              f"final val {r['final_val']:.5f} hold {r['final_hold']:.5f} | {r['minutes']:.0f} min")
        r = arm_block(vctx, hctx, base, arm, seeds, rules=False)
        dv, dh = r["val"]["pooled"]["all"], r["hold"]["pooled"]["all"]
        rec = dv["delta"] >= 0 and dh["delta"] >= -dh["se"]
        fam = abs(dv["delta"]) <= 0.002 and abs(dh["delta"]) <= 0.002
        P(f"  ==> recipe-winner rule {'PROCEED to 6-seed family' if rec else 'STOP'};  "
          f"family-candidate rule {'PROCEED' if fam else 'STOP'}")
        res = dict(e10=r, recipe_proceed=rec, family_proceed=fam)
    elif mode == "calib":
        P("== diversity calibration on the existing 2-epoch sx arms (information; sets the scale of rho_x)")
        base = lambda s: f"sx:sx_base_s{s}"  # noqa: E731
        for a in ("pl20", "tau30", "agedrop", "pl20_w1"):
            P(f"\n== sx_{a} vs sx_base")
            res[a] = arm_block(vctx, hctx, base, lambda s, a=a: f"sx:sx_{a}_s{s}", (23, 24))
        P("\n== summary")
        for a, r in res.items():
            P(f"  {a:8s} val {r['val']['pooled']['all']['delta']:+.5f} (z {r['val']['pooled']['all']['z']:+.2f}) hold "
              f"{r['hold']['pooled']['all']['delta']:+.5f}  rho_bb {r['rho_bb']:.4f} rho_x {r['rho_x']:.4f} rho_aa "
              f"{r['rho_aa']:.4f} rho_same {r['rho_same']:.4f}  mixed-ens hold "
              f"{r['ens_hold']['pooled']['all']['delta']:+.5f}  -> {r['verdict']}")
    json.dump(res, open(f"{R6}/logs/r6_eval_{tag}.json", "w"), indent=1, default=float)
    open(f"{R6}/logs/r6_eval_{tag}.txt", "w").write("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
