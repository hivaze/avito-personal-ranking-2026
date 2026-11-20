"""HDROP-ALL diagnostic read-out (CPU, 12 threads). Rule: nn_x/tricks/HDROP-ALL/PREDECLARED.txt.
  cd <repo> && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 .venv/bin/python nn_x/tricks/HDROP-ALL/readout.py
-> nn_x/tricks/HDROP-ALL/RESULT.txt, result.json
"""
import json
import os
import sys

for _k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"]:
    os.environ[_k] = "12"
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

ROOT = "<repo>"
HERE = f"{ROOT}/nn_x/tricks/HDROP-ALL"
W5P = f"{ROOT}/nn_x/w5/preds"
sys.path.insert(0, f"{ROOT}/src")
from metrics import ndcg_per_feed  # noqa: E402

MEMBERS = ["w5_e10_base_s951", "w5_e10_base_s952"]
RATES = ["d10", "d20"]
K = 4
OUT = []


def P(s=""):
    print(s, flush=True)
    OUT.append(s)


def ctx_val():
    f = pl.read_parquet(f"{ROOT}/data/flat/val/feeds.parquet")
    M = f["M"].to_numpy().astype(np.int64)
    off = f["cand_off"].to_numpy().astype(np.int64)
    H = np.minimum(f["H"].to_numpy().astype(np.int64), 256)
    tue = (f["timestamp"].to_numpy() + 10800) // 86400 < 20720
    y = np.load(f"{ROOT}/data/flat/val/cand_target.npy").astype(np.float64)
    sl = {"M<50": M < 50, "M>=50": M >= 50, "Tue": tue, "Thu": ~tue,
          "H<64": H < 64, "H64-191": (H >= 64) & (H < 192), "H>=192": H >= 192}
    return dict(name="val", M=M, off=off, y=y, H=H, slices=sl, pdir=f"{HERE}/preds/val",
                stored=lambda v: f"{W5P}/nn_{v}_trainonly_val.npy")


def ctx_hold():
    meta = np.load(f"{ROOT}/preds/holdout/holdout_meta.npy")
    M, off = meta[0].astype(np.int64), meta[1].astype(np.int64)
    y = np.load(f"{ROOT}/preds/holdout/holdout_y.npy").astype(np.float64)
    H = np.load(f"{HERE}/preds/holdout/hold_H.npy").astype(np.int64)
    sl = {"H<64": H < 64, "H64-191": (H >= 64) & (H < 192), "H>=192": H >= 192}
    return dict(name="holdout", M=M, off=off, y=y, H=H, slices=sl, pdir=f"{HERE}/preds/holdout",
                stored=lambda v: f"{W5P}/holdout/nn_{v}_holdout.npy")


def fz(s, M):
    fid = np.repeat(np.arange(len(M)), M)
    s = s.astype(np.float64)
    mu = np.bincount(fid, s, len(M)) / M
    sd = np.sqrt(np.maximum(np.bincount(fid, s * s, len(M)) / M - mu ** 2, 0))
    return np.where(sd[fid] > 0, (s - mu[fid]) / np.where(sd[fid] > 0, sd[fid], 1), 0.0)


def zcorr(a, b, M):
    """pooled Pearson of per-feed z-scores (r6_eval.corr / rho_x)"""
    return float(np.corrcoef(fz(a, M), fz(b, M))[0, 1])


def mean_feed_pearson(a, b, M):
    fid = np.repeat(np.arange(len(M)), M)
    za, zb = fz(a, M), fz(b, M)
    va = np.bincount(fid, za * za, len(M))
    vb = np.bincount(fid, zb * zb, len(M))
    ok = (va > 0) & (vb > 0)
    r = np.bincount(fid, za * zb, len(M))[ok] / np.sqrt(va[ok] * vb[ok])
    return float(r.mean())


def st(d, slices=None):
    n = len(d)
    o = dict(delta=float(d.mean()), se=float(d.std(ddof=1) / np.sqrt(n)), n=int(n))
    o["z"] = o["delta"] / o["se"] if o["se"] > 0 else 0.0
    if slices:
        o["slices"] = {k: dict(delta=float(d[m].mean()), se=float(d[m].std(ddof=1) / np.sqrt(m.sum())), n=int(m.sum()))
                       for k, m in slices.items()}
    return o


def fmt(o):
    return f"{o['delta']:+.5f} +- {o['se']:.5f} (z {o['z']:+.2f})"


def run_split(c):
    M, off, y = c["M"], c["off"], c["y"]
    nd = lambda s: ndcg_per_feed(s, y, off, M)  # noqa: E731
    ld = lambda v, mode: np.load(f"{c['pdir']}/{v}~{mode}.npy").astype(np.float64)  # noqa: E731
    res = dict(n_feeds=int(len(M)), members={}, pooled={}, ensemble={})
    ref, ref_nd, tta, tta_nd, sp_nd = {}, {}, {}, {}, {}
    P(f"\n=== {c['name']} ({len(M)} feeds) ===")
    for v in MEMBERS:
        ref[v] = ld(v, "ref")
        ref_nd[v] = nd(ref[v])
        stored = np.load(c["stored"](v)).astype(np.float64)
        st_nd = nd(stored)
        rc = dict(max_abs_diff=float(np.abs(stored - ref[v]).max()), pearson=float(np.corrcoef(stored, ref[v])[0, 1]),
                  zcorr=zcorr(stored, ref[v], M), ndcg_replay=float(ref_nd[v].mean()), ndcg_stored=float(st_nd.mean()),
                  replay_minus_stored=st(ref_nd[v] - st_nd))
        res["members"][v] = dict(ref_ndcg=float(ref_nd[v].mean()), replay_check=rc, rates={})
        P(f"[{v}] full-history replay NDCG {ref_nd[v].mean():.5f} | stored {st_nd.mean():.5f} | replay check: max|d| "
          f"{rc['max_abs_diff']:.2e} Pearson {rc['pearson']:.8f} zcorr {rc['zcorr']:.8f}, replay-stored {fmt(rc['replay_minus_stored'])}")
        for r in RATES:
            ps = [ld(v, f"{r}_k{k}") for k in range(K)]
            t = np.mean(ps, 0)
            tta[(v, r)] = t
            tta_nd[(v, r)] = nd(t)
            sp = [nd(p) for p in ps]
            sp_nd[(v, r)] = np.mean(sp, 0)
            d_t = tta_nd[(v, r)] - ref_nd[v]
            d_s = sp_nd[(v, r)] - ref_nd[v]
            mr = dict(tta=st(d_t, c["slices"]), single_mean=st(d_s, c["slices"]),
                      single_each=[float((x - ref_nd[v]).mean()) for x in sp],
                      tta_zcorr=zcorr(t, ref[v], M), tta_feed_pearson=mean_feed_pearson(t, ref[v], M),
                      single_zcorr=[zcorr(p, ref[v], M) for p in ps],
                      single_feed_pearson=[mean_feed_pearson(p, ref[v], M) for p in ps],
                      tta_ndcg=float(tta_nd[(v, r)].mean()))
            res["members"][v]["rates"][r] = mr
            P(f"  {r}: TTA(4) delta {fmt(mr['tta'])} | single-pass mean delta {fmt(mr['single_mean'])} "
              f"(each {' '.join(f'{x:+.5f}' for x in mr['single_each'])})")
            P(f"       corr TTA vs full: zcorr {mr['tta_zcorr']:.6f}, mean per-feed Pearson {mr['tta_feed_pearson']:.6f}; "
              f"single-pass zcorr {np.mean(mr['single_zcorr']):.6f} (min {np.min(mr['single_zcorr']):.6f})")
            P("       TTA slices: " + "  ".join(f"{k} {o['delta']:+.5f}+-{o['se']:.5f}" for k, o in mr["tta"]["slices"].items()))
    # pooled over members (per-feed mean of the members' deltas) and the 2-member z-mean ensemble
    ens_ref = np.mean([fz(ref[v], M) for v in MEMBERS], 0)
    ens_ref_nd = nd(ens_ref)
    res["ensemble"]["ref_ndcg"] = float(ens_ref_nd.mean())
    P(f"\n[pooled over {len(MEMBERS)} members] (decisional on val at d10: D = TTA delta, C = mean member zcorr)")
    for r in RATES:
        dt = np.mean([tta_nd[(v, r)] - ref_nd[v] for v in MEMBERS], 0)
        ds = np.mean([sp_nd[(v, r)] - ref_nd[v] for v in MEMBERS], 0)
        C = float(np.mean([res["members"][v]["rates"][r]["tta_zcorr"] for v in MEMBERS]))
        Cf = float(np.mean([res["members"][v]["rates"][r]["tta_feed_pearson"] for v in MEMBERS]))
        Cs = float(np.mean([np.mean(res["members"][v]["rates"][r]["single_zcorr"]) for v in MEMBERS]))
        res["pooled"][r] = dict(tta=st(dt, c["slices"]), single_mean=st(ds, c["slices"]), C_zcorr=C, C_feed_pearson=Cf,
                                single_zcorr_mean=Cs)
        ens_t = np.mean([fz(tta[(v, r)], M) for v in MEMBERS], 0)
        de = nd(ens_t) - ens_ref_nd
        res["ensemble"][r] = dict(tta=st(de, c["slices"]), zcorr_vs_ref_ens=zcorr(ens_t, ens_ref, M))
        P(f"  {r}: TTA(4) pooled D {fmt(res['pooled'][r]['tta'])} | single-pass pooled {fmt(res['pooled'][r]['single_mean'])} "
          f"| C zcorr {C:.6f} (per-feed Pearson {Cf:.6f}; single-pass zcorr {Cs:.6f})")
        P("       pooled TTA slices: " + "  ".join(f"{k} {o['delta']:+.5f}+-{o['se']:.5f}"
                                                  for k, o in res["pooled"][r]["tta"]["slices"].items()))
        P(f"       2-member z-mean ensemble: ref {ens_ref_nd.mean():.5f}, TTA delta {fmt(res['ensemble'][r]['tta'])}, "
          f"zcorr vs ref ensemble {res['ensemble'][r]['zcorr_vs_ref_ens']:.6f}")
    return res


def decide(rv):
    D1, C1 = rv["pooled"]["d10"]["tta"]["delta"], rv["pooled"]["d10"]["C_zcorr"]
    D2 = rv["pooled"]["d20"]["tta"]["delta"]
    if abs(D1) <= 0.0003 and C1 >= 0.999:
        return "A", (f"A: |D(0.1)| = {abs(D1):.5f} <= 0.0003 AND C(0.1) = {C1:.6f} >= 0.999 -> the 10-ep model is already "
                     f"robust to history dropout; HDROP-ALL CLOSED")
    if D1 > 0.0005 or D2 > 0.0005:
        prior = max(D1, D2)
        return "B", f"B: TTA > +0.0005 (D(0.1) {D1:+.5f}, D(0.2) {D2:+.5f}) -> HDROP-ALL stays QUEUED with prior {prior:+.5f}"
    return "C", (f"C: neither A (|D(0.1)| {abs(D1):.5f} vs 0.0003, C(0.1) {C1:.6f} vs 0.999) nor B (D(0.1) {D1:+.5f}, "
                 f"D(0.2) {D2:+.5f} vs +0.0005) -> recorded and CLOSED")


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES", "x") == "", "CPU only"
    P("HDROP-ALL diagnostic read-out (rule: nn_x/tricks/HDROP-ALL/PREDECLARED.txt). Members w5_e10_base_s951/s952 "
      "(selected checkpoints). delta = mode - full history, paired per feed, SE = sd/sqrt(n_feeds).")
    rv = run_split(ctx_val())
    rh = run_split(ctx_hold())
    code, txt = decide(rv)
    P("\n=== DECISION (val, pre-declared) ===")
    P(txt)
    gap = {v: rv["members"][v]["rates"]["d10"]["tta"]["delta"] for v in MEMBERS}
    P("info (sweep's earlier wording, per-member drop-robustness gap < +0.0002 -> close): per-member TTA d10 deltas "
      + ", ".join(f"{v} {x:+.5f}" for v, x in gap.items()))
    json.dump(dict(val=rv, holdout=rh, decision=dict(code=code, text=txt)), open(f"{HERE}/result.json", "w"), indent=1)
    open(f"{HERE}/readout.txt", "w").write("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
