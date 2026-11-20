# Read-out of the stale-view fine-tune: pools of the members fine-tuned with stale-view probability 0, 0.5
# and 1.0, scored on fresh val and on the stale view of val, at member, pool and final-file level, against a
# control fine-tuned by the same script with the probability at 0.
# Run on CPU from the project root, as in the script's docstring:
#   CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 .venv/bin/python nn_x/w12/stale/staleft/pilot_read.py
# Produced out/pilot_read_sft*.txt and .json (the 8-member read is in ../results/stale_finetune_w8.txt).
# Kept as it ran: paths and names refer to the lab tree (see lab/ and docs/glossary.md).
# Original: lab/nn_x/w12/stale/staleft/pilot_read.py
#
"""staleft PILOT READ (DECLARATION.txt blocks 2 / 2b). CPU only; reads val labels only to score out-of-fold predictions.
  cd <repo> && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 ... .venv/bin/python nn_x/w12/stale/staleft/pilot_read.py [tags, default p05 p10 p0b]
Pools = member-equal mean of per-feed z; final = 0.5 z(FT pool) + 0.5 z(TO pool) (nn_x/w12/kd/build_r3.py form).
"""
import json
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"]:
    os.environ[k] = "12"
os.environ["NUMBA_CACHE_DIR"] = "<repo>/nn_x/w12/stale/staleft/out/numba_cache"  # never write a cache into src/
from pathlib import Path

import numba as nb
import numpy as np
import polars as pl

ROOT = Path("<repo>")
HERE = ROOT / "nn_x/w12/stale/staleft"
PR = HERE / "preds"
W5 = ROOT / "nn_x/w5/preds"
VETO = ROOT / "nn_x/w12/stale/veto/preds"
FLAT = ROOT / "data/flat"
BOUND = 1790497920
MEMBERS = os.environ.get("STALEFT_MEMBERS", "w5_e10_kd05w192_s951 w5_e10_kd05t25w192_s951 w5_e10_kd05w192_s952 w5_e10_kd05t25w192_s952").split()
SUF = os.environ.get("STALEFT_SUF", "sft")  # output name = <member>_<SUF>_<tag>
lines = []


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    lines.append(s)


@nb.njit(parallel=True)
def feed_z(off, M, x):  # src/stack_val.py feed_rank_z[:, 1]
    out = np.empty(len(x), np.float32)
    for f in nb.prange(len(M)):
        o, m = off[f], M[f]
        v = x[o:o + m]
        mu = v.mean(); sd = v.std() + 1e-9
        for k in range(m):
            out[o + k] = (v[k] - mu) / sd
    return out


@nb.njit(parallel=True)
def _ndcg(s, y, off, M):  # nn_x/w12/recheck/pool_composition/pc_common.py (pessimistic ties)
    n = len(M); out = np.ones(n, np.float64)
    for f in nb.prange(n):
        o, m = off[f], M[f]
        sv = s[o:o + m]; yv = y[o:o + m]
        npos = 0
        for k in range(m):
            if yv[k] > 0: npos += 1
        if npos == 0: continue
        order = np.argsort(-sv.astype(np.float64) + 0.0, kind='mergesort')
        dcg = 0.0
        i = 0
        while i < m:
            j = i
            while j + 1 < m and sv[order[j + 1]] == sv[order[i]]: j += 1
            if j == i:
                if yv[order[i]] > 0: dcg += 1.0 / np.log2(i + 2.0)
            else:
                cp = 0
                for k in range(i, j + 1):
                    if yv[order[k]] > 0: cp += 1
                for k in range(j - cp + 1, j + 1): dcg += 1.0 / np.log2(k + 2.0)
            i = j + 1
        idcg = 0.0
        for k in range(npos): idcg += 1.0 / np.log2(k + 2.0)
        out[f] = dcg / idcg
    return out


@nb.njit(parallel=True)
def feed_corr_top1(off, M, a, b):
    n = len(M); c = np.zeros(n); t = np.zeros(n)
    for f in nb.prange(n):
        o, m = off[f], M[f]
        x = a[o:o + m].astype(np.float64); y = b[o:o + m].astype(np.float64)
        xs = x.std(); ys = y.std()
        c[f] = ((x - x.mean()) * (y - y.mean())).mean() / (xs * ys) if xs > 0 and ys > 0 else 1.0
        t[f] = 1.0 if np.argmax(x) != np.argmax(y) else 0.0
    return c, t


fv = pl.read_parquet(FLAT / "val/feeds.parquet")
Mv = fv["M"].to_numpy().astype(np.int64); ov = fv["cand_off"].to_numpy().astype(np.int64)
yv = np.load(FLAT / "val/cand_target.npy").astype(np.float32)
ft = pl.read_parquet(FLAT / "test/feeds.parquet")
Mt = ft["M"].to_numpy().astype(np.int64); ot = ft["cand_off"].to_numpy().astype(np.int64)
pubf = ft["timestamp"].to_numpy().astype(np.int64) < BOUND
zv = lambda x: feed_z(ov, Mv, np.asarray(x, np.float32))  # noqa: E731
zt = lambda x: feed_z(ot, Mt, np.asarray(x, np.float32))  # noqa: E731
nd = lambda s: _ndcg(np.ascontiguousarray(s, np.float32), yv, ov, Mv)  # noqa: E731
sys.path.insert(0, str(ROOT / "src"))
import pmix  # noqa: E402

W = pmix.weights()
wpub, wpriv = W["public"], W["private"]


def ms(d, w=None):
    if w is None:
        return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))
    m, s = pmix.wmean_se(d, w)
    return float(m), float(s)


def f(d, w=None):
    m, s = ms(d, w)
    return f"{m:+.5f} +- {s:.5f}"


def row(d):
    big = Mv >= 50
    return f"{f(d)} | {f(d[big])} | {f(d[~big])} | {f(d, wpub)} | {f(d, wpriv)}"


def load(tag):
    """per-member arrays of a variant tag; None if incomplete."""
    o = {}
    for m in MEMBERS:
        b = PR / f"nn_{m}_{SUF}_{tag}"
        fs = {k: Path(str(b) + f"_{k}.npy") for k in ("val_oof", "val_oof_stale", "val_e0", "val_e0_stale", "test")}
        if not all(p.exists() for p in fs.values()):
            return None
        o[m] = {k: np.load(p) for k, p in fs.items()}
    return o


def pools(v):
    ftf = np.mean([zv(v[m]["val_oof"]) for m in MEMBERS], 0)
    fts = np.mean([zv(v[m]["val_oof_stale"]) for m in MEMBERS], 0)
    return ftf, fts


def main():
    tags = sys.argv[1:] or ["p05", "p10", "p0b"]
    res = {}
    c = load("p0")
    if c is None:
        P("control p0 incomplete"); return
    P(f"=== staleft PILOT READ; members {MEMBERS}; {len(Mv)} val feeds ({int((Mv >= 50).sum())} with M >= 50) ===")
    # train-only branch
    to_f = np.mean([zv(np.load(W5 / f"nn_{m}_trainonly_val.npy")) for m in MEMBERS], 0)
    to_s = np.mean([zv(c[m]["val_e0_stale"]) for m in MEMBERS], 0)
    P("\n-- checks of the train-only branch (epoch 0 of the control run) --")
    for m in MEMBERS:
        st = np.load(W5 / f"nn_{m}_trainonly_val.npy")
        e0 = c[m]["val_e0"]
        cc, _ = feed_corr_top1(ov, Mv, st, e0)
        msg = f"{m}: epoch-0 fresh vs stored train-only val: per-feed corr {cc.mean():.6f}, ndcg {nd(e0).mean():.5f} vs {nd(st).mean():.5f}"
        vp = VETO / f"cpu_{m}_staleval.npy"
        if vp.exists():
            cs = np.load(vp)
            c2, _ = feed_corr_top1(ov, Mv, cs, c[m]["val_e0_stale"])
            msg += f" | epoch-0 stale (GPU) vs veto CPU stale-val: per-feed corr {c2.mean():.6f}, ndcg {nd(c[m]['val_e0_stale']).mean():.5f} vs {nd(cs).mean():.5f}"
        P(msg)
    n_tof, n_tos = nd(to_f), nd(to_s)
    P(f"TO 4-pool: fresh {n_tof.mean():.5f}  stale {n_tos.mean():.5f}  drop {f(n_tos - n_tof)}")
    cf, cs_ = pools(c)
    fin = lambda a, b: 0.5 * zv(a) + 0.5 * zv(b)  # noqa: E731
    base = {"ft_fresh": nd(cf), "ft_stale": nd(cs_), "fin_fresh": nd(fin(cf, to_f)), "fin_stale": nd(fin(cs_, to_s))}
    # stored 3-seed FT of the same members (reference level)
    try:
        s3 = np.mean([zv(np.load(W5 / f"nn_{m}_n2cpv3s3_val_oof.npy")) for m in MEMBERS], 0)
        P(f"stored 3-seed FT 4-pool fresh {nd(s3).mean():.5f}; final {nd(fin(s3, to_f)).mean():.5f}")
    except FileNotFoundError:
        pass
    P("\n-- levels (4-pool) --")
    P("variant | FT fresh | FT stale | final fresh | final stale | FT member mean fresh | FT member mean stale")
    allv = {"p0": c}
    for t in tags:
        v = load(t)
        if v is not None:
            allv[t] = v
    lev = {}
    for t, v in allv.items():
        a, b = pools(v)
        lev[t] = {"ft_fresh": nd(a), "ft_stale": nd(b), "fin_fresh": nd(fin(a, to_f)), "fin_stale": nd(fin(b, to_s)),
                  "mem_fresh": np.mean([nd(v[m]["val_oof"]) for m in MEMBERS], 0), "mem_stale": np.mean([nd(v[m]["val_oof_stale"]) for m in MEMBERS], 0)}
        P(f"{t:4s} | {lev[t]['ft_fresh'].mean():.5f} | {lev[t]['ft_stale'].mean():.5f} | {lev[t]['fin_fresh'].mean():.5f} | {lev[t]['fin_stale'].mean():.5f} | "
          f"{lev[t]['mem_fresh'].mean():.5f} | {lev[t]['mem_stale'].mean():.5f}")
    P(f"control: final stale - final fresh {f(lev['p0']['fin_stale'] - lev['p0']['fin_fresh'])}; FT branch stale - fresh {f(lev['p0']['ft_stale'] - lev['p0']['ft_fresh'])}; "
      f"FT half worth (final - TO pool) fresh {f(lev['p0']['fin_fresh'] - n_tof)} stale {f(lev['p0']['fin_stale'] - n_tos)}")
    P("\n-- deltas against the P = 0 control, paired per feed: all | M>=50 | M<50 | pmix-public | pmix-private --")
    for t in allv:
        if t == "p0":
            continue
        res[t] = {}
        for key, name in [("fin_fresh", "FINAL fresh val"), ("fin_stale", "FINAL stale-val"), ("ft_fresh", "FT branch fresh val"), ("ft_stale", "FT branch stale-val"),
                          ("mem_fresh", "FT member mean fresh"), ("mem_stale", "FT member mean stale")]:
            d = lev[t][key] - lev["p0"][key]
            res[t][key] = {"all": ms(d), "M>=50": ms(d[Mv >= 50]), "M<50": ms(d[Mv < 50]), "pmix_pub": ms(d, wpub), "pmix_priv": ms(d, wpriv)}
            P(f"{t:4s} {name:22s} {row(d)}")
        a, b = res[t]["fin_fresh"]["all"], res[t]["fin_stale"]["all"]
        passed = a[0] >= 0 and b[0] >= 2 * b[1]
        near = a[0] >= -a[1] and b[0] >= b[1]
        res[t]["rule"] = {"pass": bool(passed), "within_1se": bool(near), "fresh_z": a[0] / a[1], "stale_z": b[0] / b[1]}
        P(f"{t:4s} RULE: final fresh {a[0]:+.5f} (z {a[0] / a[1]:+.2f}), final stale-val {b[0]:+.5f} (z {b[0] / b[1]:+.2f}) -> {'PASS' if passed else ('within 1 SE of passing' if near else 'FAIL')}")
    # DECLARATION 2d: ensemble mixes (reported, no decision). FT branch = member-equal z-mean over the members of several variants.
    if "p0b" in allv:
        P("\n-- ensemble mixes of 8 FT predictions (reported; reference of the same size = p0 + p0b): all | M>=50 | M<50 | pmix-public | pmix-private --")

        def ens(tags_):
            a = np.mean([zv(allv[t][m]["val_oof"]) for t in tags_ for m in MEMBERS], 0)
            b = np.mean([zv(allv[t][m]["val_oof_stale"]) for t in tags_ for m in MEMBERS], 0)
            return {"ft_fresh": nd(a), "ft_stale": nd(b), "fin_fresh": nd(fin(a, to_f)), "fin_stale": nd(fin(b, to_s))}
        ref = ens(["p0", "p0b"])
        P(f"reference p0+p0b: FT fresh {ref['ft_fresh'].mean():.5f} FT stale {ref['ft_stale'].mean():.5f} final fresh {ref['fin_fresh'].mean():.5f} final stale {ref['fin_stale'].mean():.5f}")
        for combo in (["p0", "p10"], ["p0", "p05"], ["p05", "p10"]):
            if all(t in allv for t in combo):
                e = ens(combo)
                nm = "+".join(combo)
                res[nm] = {}
                for key, name in [("fin_fresh", "FINAL fresh val"), ("fin_stale", "FINAL stale-val"), ("ft_fresh", "FT branch fresh val"), ("ft_stale", "FT branch stale-val")]:
                    d = e[key] - ref[key]
                    res[nm][key] = {"all": ms(d), "pmix_pub": ms(d, wpub), "pmix_priv": ms(d, wpriv)}
                    P(f"{nm:8s} {name:22s} {row(d)}")
                dm = 0.5 * (e["fin_fresh"] - ref["fin_fresh"]) + 0.5 * (e["fin_stale"] - ref["fin_stale"])
                P(f"{nm:8s} {'FINAL mix read':22s} {row(dm)}")
    # mix read (DECLARATION 2c) and the paired comparison p10 - p05 on it
    P("\n-- mix read = 0.5 fresh + 0.5 stale delta per feed (reported): all | M>=50 | M<50 | pmix-public | pmix-private --")
    for t in allv:
        if t == "p0":
            continue
        for key in ("fin", "ft"):
            dm = 0.5 * (lev[t][f"{key}_fresh"] - lev["p0"][f"{key}_fresh"]) + 0.5 * (lev[t][f"{key}_stale"] - lev["p0"][f"{key}_stale"])
            res[t][f"{key}_mix"] = {"all": ms(dm), "pmix_pub": ms(dm, wpub), "pmix_priv": ms(dm, wpriv)}
            P(f"{t:4s} {'FINAL' if key == 'fin' else 'FT branch'} mix read      {row(dm)}")
    if "p05" in allv and "p10" in allv:
        dm = 0.5 * (lev["p10"]["fin_fresh"] - lev["p05"]["fin_fresh"]) + 0.5 * (lev["p10"]["fin_stale"] - lev["p05"]["fin_stale"])
        P(f"p10 - p05, FINAL mix read  {row(dm)}")
        for key, name in (("fin_fresh", "FINAL fresh"), ("fin_stale", "FINAL stale")):
            P(f"p10 - p05, {name:15s} {row(lev['p10'][key] - lev['p05'][key])}")
    # per-member FT deltas (information)
    P("\n-- per member, FT prediction alone, delta against the member's control: fresh | stale --")
    for t in allv:
        if t == "p0":
            continue
        for m in MEMBERS:
            P(f"{t:4s} {m:28s} {f(nd(allv[t][m]['val_oof']) - nd(c[m]['val_oof']))} | {f(nd(allv[t][m]['val_oof_stale']) - nd(c[m]['val_oof_stale']))}")
    # label-free shift of the test predictions of the FT branch
    P("\n-- label-free shift of the TEST predictions (FT branch 4-pool) against the control: per-feed corr | top-1 changed; public / private --")
    tp0 = np.mean([zt(c[m]["test"]) for m in MEMBERS], 0)
    tto = np.mean([zt(np.load(W5 / f"nn_{m}_trainonly_test.npy")) for m in MEMBERS], 0)
    for t, v in allv.items():
        if t == "p0":
            continue
        tp = np.mean([zt(v[m]["test"]) for m in MEMBERS], 0)
        cc, t1 = feed_corr_top1(ot, Mt, tp, tp0)
        fc, ft1 = feed_corr_top1(ot, Mt, 0.5 * zt(tp) + 0.5 * zt(tto), 0.5 * zt(tp0) + 0.5 * zt(tto))
        res[t]["test_shift"] = {"ft_corr_pub": float(cc[pubf].mean()), "ft_corr_priv": float(cc[~pubf].mean()), "ft_top1_pub": float(t1[pubf].mean()), "ft_top1_priv": float(t1[~pubf].mean()),
                                "fin_corr_pub": float(fc[pubf].mean()), "fin_corr_priv": float(fc[~pubf].mean()), "fin_top1_pub": float(ft1[pubf].mean()), "fin_top1_priv": float(ft1[~pubf].mean())}
        P(f"{t:4s} FT branch: corr {cc[pubf].mean():.5f} / {cc[~pubf].mean():.5f} | top-1 changed {t1[pubf].mean():.4f} / {t1[~pubf].mean():.4f} || final file: corr {fc[pubf].mean():.5f} / {fc[~pubf].mean():.5f} | "
          f"top-1 changed {ft1[pubf].mean():.4f} / {ft1[~pubf].mean():.4f}")
    ot_ = os.environ.get("STALEFT_OUTTAG", "")
    json.dump(res, open(HERE / f"out/pilot_read_{SUF}{ot_}.json", "w"), indent=1)
    (HERE / f"out/pilot_read_{SUF}{ot_}.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
