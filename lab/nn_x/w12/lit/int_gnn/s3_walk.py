"""int_gnn s3: does a MULTI-HOP walk on the interaction graph carry label signal on val beyond what the pool already has?
CPU only, read-only on the project. usage: python s3_walk.py MODE   (MODE = train | all; graph as in s2_graph.py)

Scores per val candidate c of a feed with history H (weights w_h = 0.95^hpos of the newest occurrence; own user node skipped):
  s1 = 3-step walk item -> user -> item (P3 / whole-sequence co-visitation): sum_h w_h/deg(h) sum_{u in U(h) & U(c)} 1/deg(u)
  s2 = 5-step walk (one more item-user-item hop through any bridge item n != c):
       sum_{u in U(c)} 1/deg(u) sum_{n in I(u), n != c} b_n/deg(n),  b_n = the s1 mass on item n
  variants: divided by deg(c)^beta (beta 0, 0.5, 1; beta 1 = popularity-free mean over the candidate's users).
Reads (val, labels used ONLY for evaluation, user-grouped nested weights):
  - feed-mean within-feed AUC on: all rows / warm rows (deg>=1) / warm rows without a 1-hop link (the territory a
    transductive graph model would add over co-visitation), next to the AUC of log-degree and of the pool score;
  - nested residual on the W16 pool: z(pool) + w z(x), w in {0, .02, .05, .1, .2, .3} chosen on 4/5 of the feeds
    (GroupKFold by user, user-0 feeds as singletons), applied to the other 1/5; paired per-feed delta +- SE.
This is the residual lens of the project (known to recover only a fraction of a trained-input effect: COVERAGE_MAP row 323)."""
import os
import sys
sys.dont_write_bytecode = True
for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"]:
    os.environ[k] = "12"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from pathlib import Path
import numba as nb
import numpy as np
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from g_common import build_graph, log, FLAT, FEATS  # noqa: E402
sys.path.insert(0, "<repo>/src")
from metrics import ndcg_per_feed  # noqa: E402

ROOT = Path("<repo>")
NT = 8


@nb.njit(parallel=True, cache=True)
def walk(nf, coff, M, hoff, H, own, cand, hist, hpos, u_ptr, u_items, i_ptr, i_users, n_nodes, G, nthr, out):
    for t in nb.prange(nthr):
        ustamp = np.zeros(n_nodes, np.int32)
        ulist = np.empty(n_nodes, np.int32)
        a = np.zeros(n_nodes, np.float64)
        istamp = np.zeros(G, np.int32)
        b = np.zeros(G, np.float32)
        ep = 0
        for f in range(t, nf, nthr):
            ep += 1
            o = own[f]
            ho = hoff[f]
            nu = 0
            for j in range(H[f]):
                h = hist[ho + j]
                dup = False
                for k in range(j):
                    if hist[ho + k] == h:
                        dup = True
                        break
                if dup:
                    continue
                dh = i_ptr[h + 1] - i_ptr[h]
                if dh == 0:
                    continue
                w = 0.95 ** hpos[ho + j] / dh
                for e in range(i_ptr[h], i_ptr[h + 1]):
                    u = i_users[e]
                    if u == o:
                        continue
                    if ustamp[u] != ep:
                        ustamp[u] = ep
                        a[u] = 0.0
                        ulist[nu] = u
                        nu += 1
                    a[u] += w
            for k in range(nu):
                u = ulist[k]
                m = a[u] / (u_ptr[u + 1] - u_ptr[u])
                for e in range(u_ptr[u], u_ptr[u + 1]):
                    g = u_items[e]
                    if istamp[g] != ep:
                        istamp[g] = ep
                        b[g] = 0.0
                    b[g] += m
            co = coff[f]
            for i in range(M[f]):
                c = cand[co + i]
                dc = i_ptr[c + 1] - i_ptr[c]
                s1 = b[c] if istamp[c] == ep else 0.0
                s2 = 0.0
                for e in range(i_ptr[c], i_ptr[c + 1]):
                    u = i_users[e]
                    if u == o:
                        continue
                    acc = 0.0
                    for e2 in range(u_ptr[u], u_ptr[u + 1]):
                        n = u_items[e2]
                        if n != c and istamp[n] == ep:
                            acc += b[n] / (i_ptr[n + 1] - i_ptr[n])
                    s2 += acc / (u_ptr[u + 1] - u_ptr[u])
                out[co + i, 0] = s1
                out[co + i, 1] = s2
                out[co + i, 2] = dc


@nb.njit(cache=True)
def feed_auc(x, y, sub, off, M):
    """per-feed AUC of x for y within the row subset `sub` (ties 0.5); NaN if a class is missing."""
    n = len(M)
    out = np.full(n, np.nan)
    for f in range(n):
        o = off[f]
        npos = 0
        nneg = 0
        for i in range(M[f]):
            if sub[o + i]:
                if y[o + i] > 0:
                    npos += 1
                else:
                    nneg += 1
        if npos == 0 or nneg == 0:
            continue
        s = 0.0
        for i in range(M[f]):
            if sub[o + i] and y[o + i] > 0:
                xi = x[o + i]
                for j in range(M[f]):
                    if sub[o + j] and y[o + j] == 0:
                        xj = x[o + j]
                        if xi > xj:
                            s += 1.0
                        elif xi == xj:
                            s += 0.5
        out[f] = s / (npos * nneg)
    return out


def feed_z(x, fid, nF):
    cnt = np.bincount(fid, minlength=nF)
    mu = np.bincount(fid, weights=x, minlength=nF) / cnt
    d = x - mu[fid]
    sd = np.sqrt(np.bincount(fid, weights=d * d, minlength=nF) / cnt)
    return d / np.maximum(sd[fid], 1e-9)


def main():
    mode = sys.argv[1]
    nb.set_num_threads(NT)
    g = build_graph(mode)
    f = g["feeds"]["val"]
    M = f["M"].to_numpy().astype(np.int64)
    Hh = f["H"].to_numpy().astype(np.int64)
    coff = f["cand_off"].to_numpy().astype(np.int64)
    hoff = f["hist_off"].to_numpy().astype(np.int64)
    own = g["node_of_feed"]["val"].astype(np.int64)
    cand = g["cands"]["val"]
    hist = np.load(FLAT / "val/hist_item_id.npy").astype(np.int64)
    hpos = np.load(FLAT / "val/hist_hpos.npy").astype(np.int64)
    out = np.zeros((len(cand), 3), np.float64)
    walk(len(f), coff, M, hoff, Hh, own, cand, hist, hpos, g["u_ptr"], g["u_items"], g["i_ptr"], g["i_users"], g["n_nodes"], g["G"], NT, out)
    log("walk done")
    np.save(HERE / f"walk_{mode}_val.npy", out.astype(np.float32))
    s1, s2, deg = out[:, 0], out[:, 1], out[:, 2]
    y = np.load(FLAT / "val/cand_target.npy").astype(np.float64)
    nF = len(M)
    fid = np.repeat(np.arange(nF), M)
    user = f["user_id"].to_numpy().astype(np.int64)
    grp = np.where(user == 0, -1 - np.arange(nF), user)
    z = lambda v: feed_z(np.asarray(v, np.float64), fid, nF)  # noqa: E731
    pool = z(0.5 * z(np.load(ROOT / "preds/ens_W16kd_ft_val.npy")) + 0.5 * z(np.load(ROOT / "preds/ens_W16kd_to_val.npy")))
    nd = lambda s: ndcg_per_feed(s, y, coff, M)  # noqa: E731
    base = nd(pool)
    log(f"W16 pool val NDCG {base.mean():.5f} ({nF} feeds, {len(y)} rows); positives {int(y.sum())}")
    luo = np.asarray(np.load(FEATS / "val/it_hist_all_luo.npy"))
    cvi = np.asarray(np.load(FEATS / "val/seq_cv_invd20.npy"))
    warm = deg >= 1
    log(f"coverage: deg>=1 {warm.mean():.4f} | s1>0 {np.mean(s1 > 0):.4f} | s2>0 {np.mean(s2 > 0):.4f} | stored seq_cv_invd20>0 {np.mean(cvi > 0):.4f} | "
        f"s1>0 given stored co-vis>0 {np.mean(s1[cvi > 0] > 0):.4f} | positives: deg>=1 {np.mean(warm[y > 0]):.4f} s1>0 {np.mean(s1[y > 0] > 0):.4f} "
        f"s2>0 {np.mean(s2[y > 0] > 0):.4f} co-vis>0 {np.mean(cvi[y > 0] > 0):.4f}")
    dsafe = np.maximum(deg, 1.0)
    cols = {
        "logdeg": np.log1p(deg),
        "s1": s1, "s1/deg^.5 (RP3b)": s1 / dsafe ** 0.5, "s1/deg": s1 / dsafe,
        "s2": s2, "s2/deg^.5": s2 / dsafe ** 0.5, "s2/deg": s2 / dsafe,
        "stored seq_cv_invd20": cvi, "stored LUO count": luo, "W16 pool": pool,
    }
    subs = {"all rows": np.ones(len(y), bool), "warm (deg>=1)": warm, "warm, no 1-hop link (s1==0)": warm & (s1 == 0),
            "warm, no stored co-vis": warm & (cvi == 0), "1-hop linked (s1>0)": s1 > 0}
    for sn, sm in subs.items():
        line = f"[AUC] subset '{sn}': rows {sm.mean():.4f}, positives in subset {np.mean(sm[y > 0]):.4f} |"
        for cn, cv in cols.items():
            au = feed_auc(np.asarray(cv, np.float64), y, sm, coff, M)
            ok = ~np.isnan(au)
            line += f" {cn} {np.nanmean(au):.4f}±{np.nanstd(au) / np.sqrt(ok.sum()):.4f}"
        line += f" (feeds with both classes {int(ok.sum())})"
        log(line)
    # AUC of the walk scores AFTER removing the pool score: rank residual within feed (does it order pairs the pool gets wrong?)
    rng = np.random.default_rng(0)
    tie = rng.random(len(y)) * 1e-9
    for cn in ["s1", "s1/deg^.5 (RP3b)", "s2", "s2/deg^.5", "s2/deg", "logdeg"]:
        log(f"[single] {cn}: NDCG as the only score (random tie-break) {nd(cols[cn] + tie * (np.abs(cols[cn]).max() + 1)).mean():.5f}")
    # nested residual on the pool
    folds = list(GroupKFold(n_splits=5).split(np.arange(nF), groups=grp))
    grid = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3]
    npos_f = np.bincount(fid, weights=y, minlength=nF)
    coldpos = np.bincount(fid, weights=y * (luo == 0), minlength=nF)
    sl = {"M<50": M < 50, "M>=50": M >= 50, "all positives cold": (npos_f > 0) & (coldpos == npos_f), "all positives warm": (npos_f > 0) & (coldpos == 0)}

    def tf(v, kind):
        v = np.asarray(v, np.float64)
        if kind == "zlog":
            med = np.median(v[v > 0]) if np.any(v > 0) else 1.0
            return z(np.log1p(v / med))
        if kind == "warmz":  # z of log1p among the WARM rows of the feed only; cold rows get 0 (no cold/warm offset by construction)
            med = np.median(v[v > 0]) if np.any(v > 0) else 1.0
            lv = np.log1p(v / med)
            wm = warm.astype(np.float64)
            cnt = np.maximum(np.bincount(fid, weights=wm, minlength=nF), 1.0)
            mu = np.bincount(fid, weights=lv * wm, minlength=nF) / cnt
            d = (lv - mu[fid]) * wm
            sd = np.sqrt(np.bincount(fid, weights=d * d, minlength=nF) / cnt)
            return d / np.maximum(sd[fid], 1e-9)
        r = np.empty(len(v))  # within-feed percentile rank (ties averaged)
        from scipy.stats import rankdata
        for ff in range(nF):
            o = coff[ff]
            r[o:o + M[ff]] = (rankdata(v[o:o + M[ff]]) - 0.5) / M[ff]
        return z(r)

    for cn in ["s1", "s1/deg^.5 (RP3b)", "s1/deg", "s2", "s2/deg^.5", "s2/deg", "logdeg"]:
        for kind in ["zlog", "rank", "warmz"]:
            x = tf(cols[cn], kind)
            per = {w: nd(pool + w * x) for w in grid}
            delta = np.zeros(nF)
            ws = []
            for trf, vaf in folds:
                best = max(grid, key=lambda w: per[w][trf].mean())
                ws.append(best)
                delta[vaf] = per[best][vaf] - base[vaf]
            se = delta.std() / np.sqrt(nF)
            line = f"[nested W16] {cn} ({kind}): {delta.mean():+.5f} ± {se:.5f} (z {delta.mean() / max(se, 1e-12):+.2f}) w per fold {ws} |"
            for k2, m2 in sl.items():
                line += f" {k2} {delta[m2].mean():+.5f}"
            line += " | in-sample best fixed w: " + " ".join(f"{w}:{(per[w] - base).mean():+.5f}" for w in grid[1:])
            log(line)
    # the two walk columns together with opposite-sign freedom (popularity-free 2-hop on top of 1-hop)
    log("DONE")


if __name__ == "__main__":
    main()
