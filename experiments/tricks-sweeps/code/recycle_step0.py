# Probe for a second-pass re-ranker of the head of the list (file name changed from step0.py). It
# asks where the remaining error of the best hard-label pool sits: the NDCG ceiling of a perfect
# re-order of the top K, pair AUC inside the top 8, and the rank distribution of positives.
# How it was run: CPU, 12 threads, 2026-11-13 about 04:46 UTC, after its rule was written to
# lab/nn_x/tricks/recycle_step0/PREDECLARED.txt (04:44:58 UTC); the command line is not on record.
# What it produced: RESULT.txt (results/recycle_step0_RESULT.txt): a perfect re-order of the top 8
# recovers 29% of the headroom on val and 16% on the holdout; the bar was 40% on both.
# Kept as it ran: paths and names refer to the lab tree (see lab/ and docs/glossary.md).
# Original: lab/nn_x/tricks/recycle_step0/step0.py
"""RECYCLE Step 0 (CPU information readout; rule in PREDECLARED.txt, written before this ran).
(a) top-K perfect-reorder NDCG ceiling, (b) in-head (top-8) vs all-pair AUC, (c) positive rank distribution by M bucket,
(d) the same for the train-only pool. Populations: val G3 final, holdout G3 (20-member z-mean), val G3 TO."""
import os, sys, json
for v in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"]:
    os.environ[v] = "12"
import numpy as np
sys.path.insert(0, "<repo>/src")
from common import feed_meta, FLAT
from metrics import ndcg_per_feed

R = "<repo>"
OUT = f"{R}/nn_x/tricks/recycle_step0"
KS = [3, 5, 8, 16, 32]
HEAD = 8
BUCKETS = [(1, 1, "r1"), (2, 3, "r2-3"), (4, 8, "r4-8"), (9, 16, "r9-16"), (17, 10 ** 9, "r>16")]
LOG = []


def say(s=""):
    print(s, flush=True); LOG.append(s)


def zfeed(s, M):
    s = np.asarray(s, np.float64); n = len(M); fid = np.repeat(np.arange(n), M)
    mu = np.bincount(fid, s, n) / M
    sd = np.sqrt(np.maximum(np.bincount(fid, s * s, n) / M - mu ** 2, 0)) + 1e-9
    return (s - mu[fid]) / sd[fid]


def ratio_se(num, den):
    """ratio of means sum(num)/sum(den) with a feed-clustered delta-method SE."""
    n = len(num); r = num.sum() / den.sum()
    return r, (num - r * den).std(ddof=1) / np.sqrt(n) / den.mean()


def auc_pairs(s, t, fid, n):
    """Mann-Whitney U per feed (ties 0.5) on the given subset; returns per-feed U, P = npos*nneg, tied pairs."""
    o = np.lexsort((s, fid)); s_o, t_o, f_o = s[o], t[o], fid[o]
    N = len(o)
    newg = np.ones(N, bool); newg[1:] = (f_o[1:] != f_o[:-1]) | (s_o[1:] != s_o[:-1])
    gid = np.cumsum(newg) - 1
    pos = np.arange(N, dtype=np.float64)
    gmean = np.bincount(gid, pos) / np.bincount(gid)          # average position within each tie group
    newf = np.ones(N, bool); newf[1:] = f_o[1:] != f_o[:-1]
    fstart = np.maximum.accumulate(np.where(newf, np.arange(N), 0))
    rk = gmean[gid] - fstart + 1.0                              # 1-based average rank within feed (ascending score)
    npos = np.bincount(f_o, t_o, n); cnt = np.bincount(f_o, minlength=n).astype(np.float64)
    nneg = cnt - npos
    U = np.bincount(f_o, t_o * rk, n) - npos * (npos + 1) / 2
    gp = np.bincount(gid, t_o); gn = np.bincount(gid, 1 - t_o)
    gf = f_o[newg]
    ties = np.bincount(gf, gp * gn, n)
    return U, npos * nneg, ties


def analyse(name, s, y, M, off, extra_masks):
    s = np.asarray(s, np.float64); y = np.asarray(y, np.float64)
    n = len(M); fid = np.repeat(np.arange(n), M)
    nd = ndcg_per_feed(s, y, off, M)
    order = np.lexsort((y, -s, fid))
    t_o = y[order]; s_o = s[order]
    rank0 = np.arange(len(order)) - np.repeat(off, M)
    disc = 1.0 / np.log2(rank0 + 2.0)
    npos = np.bincount(fid, y, n)
    cumd = np.concatenate([[0.0], np.cumsum(1.0 / np.log2(np.arange(int(M.max())) + 2.0))])
    idcg = cumd[npos.astype(np.int64)]
    dcg = np.bincount(fid, t_o * disc, n)
    has = idcg > 0
    base = np.ones(n); base[has] = dcg[has] / idcg[has]
    assert np.allclose(base, nd), "own NDCG != ndcg_per_feed"
    H = 1.0 - base
    res = {"name": name, "n_feeds": int(n), "n_items": int(len(s)), "feeds_with_pos": int(has.sum()),
           "ndcg": float(base.mean()), "ndcg_se": float(base.std(ddof=1) / np.sqrt(n)), "headroom": float(H.mean())}
    say(f"\n=== {name}: feeds {n} (with >=1 positive {has.sum()}, {has.mean():.3f}), items {len(s)}, positives {int(npos.sum())}")
    say(f"NDCG {base.mean():.5f} +- {res['ndcg_se']:.5f}; headroom to 1.0 = {H.mean():.5f}")
    # (a) top-K reorder ceilings
    say("(a) top-K perfect-reorder ceiling (rest of the order unchanged)")
    say("    K   ceiling    gain +- SE           headshare +- SE   | tailshare  feeds(>=1 pos) w/o pos in top-K, their share of headroom")
    res["ceil"] = {}
    for K in KS:
        inh = rank0 < K
        h = np.bincount(fid, t_o * inh, n).astype(np.int64)
        tail = np.bincount(fid, t_o * disc * (~inh), n)
        c = np.ones(n); c[has] = (cumd[h[has]] + tail[has]) / idcg[has]
        g = c - base
        sh, sh_se = ratio_se(g, H)
        nohead = has & (h == 0)
        hr_nohead = H[nohead].sum() / H.sum()
        res["ceil"][K] = {"ceiling": float(c.mean()), "gain": float(g.mean()), "gain_se": float(g.std(ddof=1) / np.sqrt(n)),
                          "headshare": float(sh), "headshare_se": float(sh_se), "tailshare": float(1 - sh),
                          "feeds_nopos_in_head": float(nohead.sum() / has.sum()), "headroom_share_nopos_in_head": float(hr_nohead)}
        say(f"    {K:2d}  {c.mean():.5f}  {g.mean():+.5f} +- {g.std(ddof=1) / np.sqrt(n):.5f}   {sh:.4f} +- {sh_se:.4f}   | {1 - sh:.4f}     {nohead.sum() / has.sum():.4f}   {hr_nohead:.4f}")
        if K == HEAD:
            res["slices8"] = {}
            for lab, m in extra_masks.items():
                if m.sum() < 2:
                    say(f"       slice {lab:6s}: empty"); continue
                shm, shm_se = ratio_se(g[m], H[m])
                res["slices8"][lab] = {"n": int(m.sum()), "ndcg": float(base[m].mean()), "gain": float(g[m].mean()),
                                       "headshare": float(shm), "headshare_se": float(shm_se)}
                say(f"       slice {lab:6s} n {m.sum():6d}: NDCG {base[m].mean():.5f}  gain8 {g[m].mean():+.5f}  headshare8 {shm:.4f} +- {shm_se:.4f}")
    # (b) pair AUC: head (both in top-8) vs all pairs
    say("(b) within-feed pos-vs-neg pair AUC (ties 0.5); pooled = micro over pairs, macro = mean over feeds with >=1 pair")
    res["auc"] = {}
    sel_sets = {"head8": rank0 < HEAD, "head16": rank0 < 16, "all": np.ones(len(s), bool)}
    for lab, sel in sel_sets.items():
        U, P, T = auc_pairs(s_o[sel], t_o[sel], fid[sel], n)
        a, a_se = ratio_se(U, P)
        mf = P > 0
        mac = (U[mf] / P[mf]).mean(); mac_se = (U[mf] / P[mf]).std(ddof=1) / np.sqrt(mf.sum())
        res["auc"][lab] = {"pooled": float(a), "pooled_se": float(a_se), "macro": float(mac), "macro_se": float(mac_se),
                           "pairs": float(P.sum()), "feeds_with_pairs": int(mf.sum()), "tie_share": float(T.sum() / P.sum())}
        say(f"    {lab:6s}: pooled {a:.4f} +- {a_se:.4f} | macro {mac:.4f} +- {mac_se:.4f} | pairs {P.sum():.0f} in {mf.sum()} feeds | tied {T.sum() / P.sum():.5f}")
        for mlab, mm in extra_masks.items():
            m2 = mm & mf
            if m2.sum() < 2: continue
            am, am_se = ratio_se(U[m2], P[m2])
            res["auc"][lab][f"pooled_{mlab}"] = float(am)
            say(f"             {mlab:6s}: pooled {am:.4f} +- {am_se:.4f} ({m2.sum()} feeds)")
    # (c) positive rank distribution by M bucket
    say("(c) rank of positives in the base order: share at 1 / 2-3 / 4-8 / 9-16 / >16")
    pr = rank0[t_o > 0] + 1; pf = fid[t_o > 0]
    best = np.full(n, 10 ** 9); np.minimum.at(best, pf, pr)
    res["ranks"] = {}
    for lab, fm in [("all", np.ones(n, bool)), ("M<50", M < 50), ("M>=50", M >= 50)]:
        sel = fm[pf]
        if sel.sum() == 0:
            say(f"    {lab:6s}: no positives (bucket empty)"); res["ranks"][lab] = None; continue
        r_ = pr[sel]; b_ = best[fm & has]
        sh = [float(((r_ >= lo) & (r_ <= hi)).mean()) for lo, hi, _ in BUCKETS]
        shb = [float(((b_ >= lo) & (b_ <= hi)).mean()) for lo, hi, _ in BUCKETS]
        res["ranks"][lab] = {"n_pos": int(sel.sum()), "shares": dict(zip([b[2] for b in BUCKETS], sh)),
                             "best_pos_shares": dict(zip([b[2] for b in BUCKETS], shb)), "n_feeds_with_pos": int((fm & has).sum()),
                             "median_rank": float(np.median(r_)), "mean_M_feeds": float(M[fm].mean())}
        say(f"    {lab:6s} positives {sel.sum():6d}: " + "  ".join(f"{b[2]} {x:.4f}" for b, x in zip(BUCKETS, sh))
            + f"   (median rank {np.median(r_):.0f}, mean M {M[fm].mean():.1f})")
        say(f"    {lab:6s} best pos/feed {len(b_):5d}: " + "  ".join(f"{b[2]} {x:.4f}" for b, x in zip(BUCKETS, shb)))
    return res


def verdict(r):
    hs = r["ceil"][HEAD]["headshare"]; a = r["auc"]["head8"]["pooled"]
    return hs >= 0.40 and a < 0.75, hs, a


def main():
    yv = np.load(FLAT / "val/cand_target.npy")
    fv, Mv, offv = feed_meta("val"); offv = offv.astype(np.int64)
    day = (fv["timestamp"].to_numpy().astype(np.int64) + 10800) // 86400
    vmasks = {"M<50": Mv < 50, "M>=50": Mv >= 50, "Tue": day < 20720, "Thu": day >= 20720}
    Mh, offh = np.load(f"{R}/preds/holdout/holdout_meta.npy"); Mh = Mh.astype(np.int64); offh = offh.astype(np.int64)
    yh = np.load(f"{R}/preds/holdout/holdout_y.npy")
    mem = [f"v23_gru_pre2s_s{i}" for i in range(501, 507)] + [f"r6_cd_m02_r{i}" for i in range(101, 108)] \
        + [f"r6_cd_m02_p70k_r{i}" for i in range(101, 108)]
    assert len(mem) == 20
    acc = None
    for m in mem:
        z = zfeed(np.load(f"{R}/preds/holdout/nn_{m}_holdout.npy"), Mh); acc = z if acc is None else acc + z
    hold = (acc / len(mem)).astype(np.float32)
    nh = ndcg_per_feed(hold, yh, offh, Mh).mean()
    say(f"sanity: holdout G3 20-member z-mean NDCG {nh:.5f} (recorded 0.32379); holdout feeds {len(Mh)}, M min {Mh.min()}")
    assert abs(nh - 0.32379) < 1e-5, nh
    hmasks = {"M<50": Mh < 50, "M>=50": Mh >= 50}
    sv = np.load(f"{R}/preds/ens_G3_final_val.npy"); st = np.load(f"{R}/preds/ens_G3_to_val.npy")
    say(f"sanity: val G3 final NDCG {ndcg_per_feed(sv, yv, offv, Mv).mean():.5f} (recorded 0.40032); G3 TO {ndcg_per_feed(st, yv, offv, Mv).mean():.5f}")
    res = {}
    res["val_G3"] = analyse("P1 val G3 final (ens_G3_final_val)", sv, yv, Mv, offv, vmasks)
    res["hold_G3"] = analyse("P2 holdout G3 (20-member z-mean)", hold, yh, Mh, offh, hmasks)
    res["val_G3_TO"] = analyse("P3 val G3 train-only pool (ens_G3_to_val) [item (d), info]", st, yv, Mv, offv, vmasks)
    say("\n=== READING RULE (PREDECLARED.txt): QUEUED only if headshare_8 >= 0.40 AND pooled AUC_head8 < 0.75 on BOTH P1 and P2")
    ok_all = True
    for k in ["val_G3", "hold_G3", "val_G3_TO"]:
        ok, hs, a = verdict(res[k])
        if k != "val_G3_TO": ok_all &= ok
        g8 = res[k]["ceil"][HEAD]["gain"]
        say(f"  {k:10s}: headshare_8 {hs:.4f} ({'>=' if hs >= 0.40 else '<'} 0.40), AUC_head8 {a:.4f} ({'<' if a < 0.75 else '>='} 0.75)"
            f" -> {'pass' if ok else 'FAIL'}{'' if k != 'val_G3_TO' else ' (info only)'} | secondary §3 clause gain_8 {g8:+.5f} {'>=' if g8 >= 0.01 else '<'} +0.01")
    res["verdict"] = "QUEUED" if ok_all else "CLOSED"
    say(f"VERDICT: head-recycle family (RECYCLE v2, head-only cascade) {res['verdict']}")
    json.dump(res, open(f"{OUT}/step0_results.json", "w"), indent=1)
    open(f"{OUT}/step0_log.txt", "w").write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
