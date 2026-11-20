"""(2) Loss-metric alignment on REAL member logits (train-only members; no training).
Per feed and member (raw logits s, softmax at T=1 as in nn/model.py listwise_loss):
  ndcg (pessimistic), P_pos = softmax mass on positives, CE = -(1/K) sum_pos log p_i,
  gL1 = sum_j |p_j - y_j/K| (L1 size of the CE score-gradient; = 2(1-P_pos) when no positive has p_i > 1/K),
  lam = LambdaRank gradient size sum_{pos i, neg j} |dNDCG_ij| sigma(s_j - s_i)   (metric-aware gradient),
  Sm  = d NDCG / d delta for a uniform positive-margin shift delta (central difference, delta = 0.5 nat),
  Sl  = -d CE / d delta for the same shift (= 1 - P_pos),
  negative-gradient mass by negative rank bucket and by same/other category for CE (p_j) and LambdaRank (lam_j),
  CE split into category part -log P(cat_i) and within-category part -log p(i | cat_i),
  multi-positive "anti-gradient": positives with p_i > 1/K (CE pushes them DOWN).
Slices: M bin, K, positive coldness (label-defined, descriptive), H, user0, day.
Writes out/02_weights.txt, out/02_perfeed_val.npz, out/02_perfeed_hold.npz."""
import sys, time, os
import numpy as np, polars as pl
from numba import njit, prange
sys.path.insert(0, "<repo>/src")
from common import feed_meta, FLAT
from metrics import ndcg_per_feed

OUT = "<repo>/nn_x/insights/labels_loss/out"
PR = "<repo>/preds"
lines = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); lines.append(s)

MEMBERS = [f"r6_cd_m02_r10{i}" for i in range(1, 8)] + [f"v28_lr1_m02_s41{i}" for i in range(1, 8)] + \
          [f"v23_gru_pre2s_s50{i}" for i in range(1, 7)]
RB = np.array([0, 5, 20, 50, 100, 400])  # negative rank buckets (0-based rank)

@njit(parallel=True, cache=True)
def feed_stats(s, y, cat, off, M, out, negm_ce, negm_lam, negm_ce_rel, negm_lam_rel, rb):
    # out columns: 0 P_pos, 1 CE, 2 gL1, 3 lam, 4 CE_cat, 5 CE_within, 6 anti (sum max(p_i-1/K,0)), 7 n_anti_pos,
    # 8 ce_neg_samecat_mass, 9 lam_neg_samecat_mass, 10 best_pos_rank, 11 K, 12 lam_pos_vs_pos(0), 13 gL_neg_mass
    n = len(M)
    for f in prange(n):
        o = off[f]; m = M[f]
        ss = s[o:o + m].astype(np.float64); yy = y[o:o + m]; cc = cat[o:o + m]
        mx = ss.max()
        e = np.exp(ss - mx); Z = e.sum(); p = e / Z
        K = 0
        for j in range(m):
            K += yy[j]
        if K == 0:
            continue
        # ranks, pessimistic ties: order by score desc, then target asc
        order = np.argsort(-ss + 1e-12 * yy)  # tiny shift puts positives after tied negatives
        rank = np.empty(m, np.int64)
        for r in range(m):
            rank[order[r]] = r
        idcg = 0.0
        for r in range(K):
            idcg += 1.0 / np.log2(r + 2.0)
        Ppos = 0.0; ce = 0.0; cecat = 0.0; cewin = 0.0; anti = 0.0; nanti = 0; best = m
        for i in range(m):
            if yy[i] == 1:
                Ppos += p[i]
                ce -= np.log(p[i] + 1e-300)
                pc = 0.0
                for j in range(m):
                    if cc[j] == cc[i]:
                        pc += p[j]
                cecat -= np.log(pc + 1e-300)
                cewin -= np.log(p[i] / pc + 1e-300)
                if p[i] > 1.0 / K:
                    anti += p[i] - 1.0 / K; nanti += 1
                if rank[i] < best:
                    best = rank[i]
        out[f, 0] = Ppos; out[f, 1] = ce / K; out[f, 4] = cecat / K; out[f, 5] = cewin / K
        out[f, 6] = anti; out[f, 7] = nanti; out[f, 10] = best; out[f, 11] = K
        gl = 0.0; gneg = 0.0
        for j in range(m):
            g = p[j] - yy[j] / K
            gl += abs(g)
            if yy[j] == 0:
                gneg += p[j]
        out[f, 2] = gl; out[f, 13] = gneg
        lam = 0.0; ce_sc = 0.0; lam_sc = 0.0
        pidx = np.empty(K, np.int64); q = 0
        for i in range(m):
            if yy[i] == 1:
                pidx[q] = i; q += 1
        for j in range(m):
            if yy[j] == 1:
                continue
            dj = 1.0 / np.log2(rank[j] + 2.0)
            lj = 0.0; same = False
            for i in pidx:
                di = 1.0 / np.log2(rank[i] + 2.0)
                sig = 1.0 / (1.0 + np.exp(-(ss[j] - ss[i])))
                lj += abs(di - dj) / idcg * sig
                if cc[i] == cc[j]:
                    same = True
            lam += lj
            b = 0
            while b < len(rb) - 2 and rank[j] >= rb[b + 1]:
                b += 1
            negm_ce[f, b] += p[j]; negm_lam[f, b] += lj
            # relative to positives: 0 above best positive, 1 between positives, 2 below all positives
            rel = 0
            if rank[j] > best:
                rel = 1
                below_all = True
                for i in pidx:
                    if rank[i] > rank[j]:
                        below_all = False
                if below_all:
                    rel = 2
            negm_ce_rel[f, rel] += p[j]; negm_lam_rel[f, rel] += lj
            if same:
                ce_sc += p[j]; lam_sc += lj
        out[f, 3] = lam; out[f, 8] = ce_sc; out[f, 9] = lam_sc


def ce_per_feed(s, y, off, M):
    fid = np.repeat(np.arange(len(M)), M)
    s = s.astype(np.float64)
    mx = np.maximum.reduceat(s, off)
    e = np.exp(s - np.repeat(mx, M))
    Z = np.add.reduceat(e, off)
    lp = s - np.repeat(mx + np.log(Z), M)
    K = np.add.reduceat(y.astype(np.float64), off)
    return -np.bincount(fid, weights=lp * y, minlength=len(M)) / np.maximum(K, 1)


def run(split, members, M, off, y, cat, tag):
    n = len(M)
    acc = np.zeros((n, 14)); nm_ce = np.zeros((n, len(RB) - 1)); nm_lam = np.zeros((n, len(RB) - 1))
    nr_ce = np.zeros((n, 3)); nr_lam = np.zeros((n, 3))
    nd = np.zeros(n); Sm = np.zeros(n); Sl = np.zeros(n); Sm_z = np.zeros(n); Sl_z = np.zeros(n); sdl = np.zeros(n)
    used = 0
    for mname in members:
        fpath = f"{PR}/nn_{mname}_trainonly_val.npy" if split == "val" else f"{PR}/holdout/nn_{mname}_holdout.npy"
        if not os.path.exists(fpath):
            P("missing", fpath); continue
        s = np.load(fpath).astype(np.float32)
        t0 = time.time()
        o = np.zeros((n, 14)); a1 = np.zeros((n, len(RB) - 1)); a2 = np.zeros((n, len(RB) - 1))
        r1 = np.zeros((n, 3)); r2 = np.zeros((n, 3))
        feed_stats(s, y, cat, off, M, o, a1, a2, r1, r2, RB)
        acc += o; nm_ce += a1; nm_lam += a2; nr_ce += r1; nr_lam += r2
        nd += ndcg_per_feed(s, y, off, M)
        d = 0.5
        Sm += (ndcg_per_feed(s + d * y, y, off, M) - ndcg_per_feed(s - d * y, y, off, M)) / (2 * d)
        Sl += (ce_per_feed(s - d * y, y, off, M) - ce_per_feed(s + d * y, y, off, M)) / (2 * d)
        # the same shift in units of the feed's own logit sd (scale-free version)
        fid = np.repeat(np.arange(n), M)
        mu = np.bincount(fid, weights=s, minlength=n) / M
        sd = np.sqrt(np.maximum(np.bincount(fid, weights=(s - np.repeat(mu, M)) ** 2, minlength=n) / M, 1e-12))
        sdl += sd
        dz = 0.25 * np.repeat(sd, M)
        Sm_z += (ndcg_per_feed(s + dz * y, y, off, M) - ndcg_per_feed(s - dz * y, y, off, M)) / 0.5
        Sl_z += (ce_per_feed(s - dz * y, y, off, M) - ce_per_feed(s + dz * y, y, off, M)) / 0.5
        used += 1
        P(f"  {tag} {mname}: ndcg {ndcg_per_feed(s, y, off, M).mean():.5f}  ({time.time()-t0:.0f}s)")
    k = max(used, 1)
    return dict(acc=acc / k, nm_ce=nm_ce / k, nm_lam=nm_lam / k, nr_ce=nr_ce / k, nr_lam=nr_lam / k, nd=nd / k,
                Sm=Sm / k, Sl=Sl / k, Sm_z=Sm_z / k, Sl_z=Sl_z / k, sd=sdl / k, used=used)


def slice_table(name, labels, R, extra=None):
    acc = R["acc"]; nd = R["nd"]
    tot = dict(n=len(nd), head=(1 - nd).sum(), gl=acc[:, 2].sum(), lam=acc[:, 3].sum(), Sm=R["Sm"].sum(),
               Sl=R["Sl"].sum(), Smz=R["Sm_z"].sum(), Slz=R["Sl_z"].sum())
    P(f"\n-- slice: {name}")
    P("slice | feeds% | ndcg | headroom% | CEgrad% (gL1) | Lambda% | Sm% | Sl% | Sm/Sl (rel) | Smz/Slz (rel) | "
      "lam/gL (rel) | mean CE | P_pos | logit sd")
    for lab in labels:
        m = extra[lab]
        if m.sum() == 0: continue
        fs = m.mean()
        sm = R["Sm"][m].sum() / tot["Sm"]; sl = R["Sl"][m].sum() / tot["Sl"]
        smz = R["Sm_z"][m].sum() / tot["Smz"]; slz = R["Sl_z"][m].sum() / tot["Slz"]
        gl = acc[m, 2].sum() / tot["gl"]; lam = acc[m, 3].sum() / tot["lam"]
        P(f"{lab} | {fs*100:5.1f} | {nd[m].mean():.4f} | {(1-nd[m]).sum()/tot['head']*100:5.1f} | {gl*100:5.1f} | "
          f"{lam*100:5.1f} | {sm*100:5.1f} | {sl*100:5.1f} | {sm/sl:5.2f} | {smz/slz:5.2f} | {lam/gl:5.2f} | "
          f"{acc[m,1].mean():.3f} | {acc[m,0].mean():.3f} | {R['sd'][m].mean():.2f}")


def main():
    # ---------------- val ----------------
    f, M, off = feed_meta("val")
    y = np.asarray(np.load(FLAT / "val/cand_target.npy")).astype(np.int64)
    cat = np.asarray(np.load(FLAT / "val/cand_category_id.npy")).astype(np.int64)
    cold = np.asarray(np.load("<repo>/nn_x/r6/feat_cache/val/it_hist_all_luo.npy")) == 0
    P(f"val: {len(M)} feeds; members {len(MEMBERS)}")
    R = run("val", MEMBERS, M.astype(np.int64), off.astype(np.int64), y, cat, "val")
    np.savez_compressed(f"{OUT}/02_perfeed_val.npz", **{k: v for k, v in R.items() if k != "used"})
    acc = R["acc"]; K = acc[:, 11]
    fid = np.repeat(np.arange(len(M)), M)
    ncold_pos = np.bincount(fid, weights=(cold & (y == 1)).astype(float), minlength=len(M))
    H = f["H"].to_numpy(); u0 = f["user_id"].to_numpy() == 0
    day = ((f["timestamp"].to_numpy().astype(np.int64) + 3 * 3600) // 86400) % 7
    P(f"overall: ndcg {R['nd'].mean():.5f}; CE {acc[:,1].mean():.4f}; P_pos {acc[:,0].mean():.4f}; "
      f"CE category part {acc[:,4].mean():.4f} + within-category part {acc[:,5].mean():.4f}")
    MB = [5, 10, 20, 30, 50, 75, 100, 150, 200, 250, 301]
    sl = {f"M[{a},{b})": (M >= a) & (M < b) for a, b in zip(MB[:-1], MB[1:])}
    slice_table("M bin", list(sl), R, sl)
    sk = {"K=1": K == 1, "K=2": K == 2, "K=3": K == 3, "K>=4": K >= 4}
    slice_table("K (#positives)", list(sk), R, sk)
    sc = {"all-pos-cold": ncold_pos == K, "some-cold": (ncold_pos > 0) & (ncold_pos < K), "all-pos-warm": ncold_pos == 0}
    slice_table("positive coldness (label-defined, descriptive)", list(sc), R, sc)
    sh = {"H=0": H == 0, "H[1,32)": (H >= 1) & (H < 32), "H[32,128)": (H >= 32) & (H < 128),
          "H[128,256)": (H >= 128) & (H < 256), "H=256": H == 256}
    slice_table("history length H", list(sh), R, sh)
    su = {"user0": u0, "user>0": ~u0}
    slice_table("user 0", list(su), R, su)
    best = acc[:, 10] / M
    sb = {"bestpos top10%": best < 0.1, "10-30%": (best >= 0.1) & (best < 0.3), "30-60%": (best >= 0.3) & (best < 0.6),
          "bottom 40%": best >= 0.6}
    slice_table("best positive's rank percentile (label-defined, descriptive)", list(sb), R, sb)

    # negative-gradient mass by negative rank bucket / relation to positives / category
    P("\n-- where the gradient on NEGATIVES goes (val, member-averaged; share of total negative mass)")
    tot_ce = R["nm_ce"].sum(); tot_lam = R["nm_lam"].sum()
    P("neg rank bucket | CE mass% | Lambda mass% | ratio Lambda/CE")
    for b in range(len(RB) - 1):
        a = R["nm_ce"][:, b].sum() / tot_ce; c = R["nm_lam"][:, b].sum() / tot_lam
        P(f"[{RB[b]},{RB[b+1]}) | {a*100:5.1f} | {c*100:5.1f} | {c/a:5.2f}")
    for b, nm in enumerate(["above best positive", "between positives", "below all positives"]):
        a = R["nr_ce"][:, b].sum() / R["nr_ce"].sum(); c = R["nr_lam"][:, b].sum() / R["nr_lam"].sum()
        P(f"{nm} | {a*100:5.1f} | {c*100:5.1f} | {c/a:5.2f}")
    a = acc[:, 8].sum() / acc[:, 13].sum(); c = acc[:, 9].sum() / acc[:, 3].sum()
    P(f"negatives sharing a category with some positive | CE {a*100:5.1f} | Lambda {c*100:5.1f} | ratio {c/a:5.2f}")
    # same-category share of candidates (unweighted) for reference
    P(f"multi-positive anti-gradient: share of positives with p_i > 1/K: {acc[:,7].sum()/K.sum():.4f}; "
      f"anti mass / total CE grad mass {acc[:,6].sum()/acc[:,2].sum():.5f}; among K>=2 feeds: "
      f"{acc[K>=2,7].sum()/K[K>=2].sum():.4f}")
    P(f"CE decomposition by K: " + "; ".join(
        f"K={k}: cat {acc[K==k,4].mean():.3f} within {acc[K==k,5].mean():.3f}" for k in [1, 2, 3]))

    # ---------------- holdout (train random-like, M>=50) ----------------
    hm = np.load(f"{PR}/holdout/holdout_meta.npy"); hy = np.load(f"{PR}/holdout/holdout_y.npy").astype(np.int64)
    Mh, offh = hm[0].astype(np.int64), hm[1].astype(np.int64)
    # holdout category ids: rebuild from train fidx order used by holdout_eval (not stored) -> use 0 (no category split)
    cath = np.zeros(len(hy), np.int64)
    P(f"\nholdout: {len(Mh)} feeds (M>=50, random-like train)")
    Rh = run("hold", MEMBERS, Mh, offh, hy, cath, "hold")
    np.savez_compressed(f"{OUT}/02_perfeed_hold.npz", **{k: v for k, v in Rh.items() if k != "used"})
    Kh = Rh["acc"][:, 11]
    slh = {f"M[{a},{b})": (Mh >= a) & (Mh < b) for a, b in zip(MB[:-1], MB[1:])}
    slice_table("holdout M bin", list(slh), Rh, slh)
    skh = {"K=1": Kh == 1, "K=2": Kh == 2, "K=3": Kh == 3, "K>=4": Kh >= 4}
    slice_table("holdout K", list(skh), Rh, skh)
    open(f"{OUT}/02_weights.txt", "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
