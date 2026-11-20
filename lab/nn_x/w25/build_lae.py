"""W25 G1/G5 scorers: group-level linear autoencoders (EASE-DAN, RDLAE, EASE) as in the 2025 winner, transferred to
item groups (k-means clusters cl4096 / cl1024 of the item embedding, and the 54 categories).

Fit (label-free): user x group BINARY matrix from ONE train snapshot per user (the user's latest train feed; every user-0 feed is its
own row); W fitted with the winner's hyper-parameters (study.md 1.3). Val/test histories never enter W (parity check: a second DAN W
fitted with the earliest val/test snapshot added is only compared on val scores).
Score: every feed from ITS OWN history only. s(c) = sum_j x_f[j] * W[j, g(c)] (= (x_f @ W)[g(c)]) with input vectors
  bin, cnt, int3 (= bin + 2*[type-1 event in group], the winner's binary + 2*contact), t1 (type-1 count), rec05 (sum 0.5^(hpos/5)),
  exp05 (sum exp(-0.05 hpos)); and the winner's "score_20" = x_f @ truncate(W^T, 300).
Outputs: data/<split>/w25_*.npy float32; train = rows of all p_random>0.5 feeds (data/train/rows.npy); val/test flat order.
"""
import json, sys, time
import numpy as np, numba as nb, scipy.sparse as sp
sys.path.insert(0, "<repo>/src")
from common import feed_meta, FLAT  # noqa: E402
from stack_val import feed_rank_z  # noqa: E402
from pathlib import Path

ROOT = Path("<repo>"); CLU = ROOT / "data/clu_work"; WD = ROOT / "nn_x/w25/data"
nb.set_num_threads(12)
T0 = time.time()


def log(*a):
    print(time.strftime("%H:%M:%S"), f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def hist_rows_of(feeds, hoff, H):
    """flat history row ids of the given feeds (concatenated, feed order) + per-feed local index."""
    n = H[feeds]
    rows = np.repeat(hoff[feeds] - np.r_[0, np.cumsum(n)[:-1]], n) + np.arange(n.sum())
    loc = np.repeat(np.arange(len(feeds)), n)
    return rows, loc


# ---------------- fits (verbatim logic of winner_autoencoders_model.py) ----------------
def fit_dan(X, reg=10.0, drop_p=0.04977334177467749, beta=0.9520854072258169, alpha=0.9520854072258169):
    X = X.astype(np.float64)
    item_counts = np.maximum(np.asarray(X.sum(0)).ravel(), 1.0)
    user_counts = np.maximum(np.asarray(X.sum(1)).ravel(), 1.0)
    XT = sp.diags(np.power(user_counts, -beta)) @ X
    G = (XT.T @ X).toarray()
    G[np.diag_indices_from(G)] += reg + drop_p / (1 - drop_p) * item_counts
    P = np.linalg.inv(G)
    B = np.eye(G.shape[0]) - P / np.diag(P)
    ipt = np.power(item_counts, -(1 - alpha))
    W = B * (1 / ipt).reshape(-1, 1) * ipt
    W[np.diag_indices_from(W)] = 0
    return W.astype(np.float32)


def fit_rdlae(X, reg=500.0, drop_p=0.7, xi=0.1):
    G = (X.T @ X).toarray().astype(np.float64)
    gamma = np.diag(G) * drop_p / (1 - drop_p) + reg
    G[np.diag_indices_from(G)] += gamma
    P = np.linalg.inv(G)
    dP = np.diag(P)
    cond = (1 - gamma * dP) > xi
    assert cond.sum() > 0
    lagr = ((1 - xi) / dP - gamma) * cond
    # paper form B = I - P diag(gamma + lambda): off-diagonals identical to the winner's code (which omits the I because its seen
    # nodes are masked at inference); the diagonal becomes xi (0.1) instead of -(1-xi), so a candidate whose group is in the
    # history is not penalised by a -0.9 self-term (our candidates are not masked).
    W = -(P * (gamma + lagr))
    W[np.diag_indices_from(W)] += 1.0
    return W.astype(np.float32)


def fit_ease(X, lam=500.0):
    G = (X.T @ X).toarray().astype(np.float64)
    G[np.diag_indices_from(G)] += lam
    P = np.linalg.inv(G)
    W = -P / np.diag(P)
    W[np.diag_indices_from(W)] = 0
    return W.astype(np.float32)


def truncate(G, k=300):
    """winner's utils.truncate: keep the top-k entries of every column (axis 0)."""
    col = np.argpartition(G, -k, axis=0)[-k:]
    mask = np.zeros_like(G, dtype=bool)
    np.put_along_axis(mask, col, True, axis=0)
    return np.where(mask, G, 0.0).astype(np.float32)


# ---------------- per-feed compressed histories ----------------
def compress(hg, hloc, het, hpos, K, nfeeds):
    """distinct (feed, group) pairs with weights [bin, cnt, int3, t1, rec05, exp05]. Returns ptr (nfeeds+1), g, w (n x 6)."""
    ok = hg >= 0
    key = hloc[ok].astype(np.int64) * K + hg[ok]
    t1 = (het[ok] == 1).astype(np.float32)
    r05 = np.power(0.5, hpos[ok].astype(np.float32) / 5.0)
    e05 = np.exp(-0.05 * hpos[ok].astype(np.float32))
    uk, inv = np.unique(key, return_inverse=True)
    n = len(uk)
    cnt = np.bincount(inv, minlength=n).astype(np.float32)
    t1c = np.bincount(inv, weights=t1, minlength=n).astype(np.float32)
    w = np.stack([np.ones(n, np.float32), cnt, 1 + 2 * (t1c > 0), t1c,
                  np.bincount(inv, weights=r05, minlength=n).astype(np.float32),
                  np.bincount(inv, weights=e05, minlength=n).astype(np.float32)], 1)
    floc = (uk // K).astype(np.int64); g = (uk % K).astype(np.int32)
    ptr = np.searchsorted(floc, np.arange(nfeeds + 1))
    return ptr, g, np.ascontiguousarray(w)


@nb.njit(parallel=True, cache=True)
def score_rows(cptr, cg, hptr, hg, hw, WT, vsel, out):
    """out[row, k] = sum_j hw[j, vsel[k]] * WT[cg[row], hg[j]] over the feed's history groups j; NaN if cg < 0."""
    nf = len(cptr) - 1
    for f in nb.prange(nf):
        for r in range(cptr[f], cptr[f + 1]):
            c = cg[r]
            if c < 0:
                for k in range(len(vsel)):
                    out[r, k] = np.nan
                continue
            for k in range(len(vsel)):
                v = vsel[k]
                acc = 0.0
                for j in range(hptr[f], hptr[f + 1]):
                    acc += hw[j, v] * WT[c, hg[j]]
                out[r, k] = acc


def main():
    ftr, Mtr, otr = feed_meta("train")
    Htr = ftr["H"].to_numpy().astype(np.int64); hofftr = ftr["hist_off"].to_numpy().astype(np.int64)
    uid = ftr["user_id"].to_numpy().astype(np.int64); ts = ftr["timestamp"].to_numpy().astype(np.int64)
    # ---- 1. one snapshot per user (latest train feed; user-0 feeds as own rows) ----
    order = np.lexsort((np.arange(len(uid)), ts, uid))  # by user, then time, then fidx
    us = uid[order]
    last = np.r_[us[1:] != us[:-1], True]
    snap = np.sort(np.concatenate([order[last & (us > 0)], np.where(uid == 0)[0]]))
    log(f"snapshot feeds {len(snap)} (users>0 {(uid[snap] > 0).sum()}, user0 {(uid[snap] == 0).sum()})")
    hrows, hloc = hist_rows_of(snap, hofftr, Htr)
    log(f"snapshot history rows {len(hrows)}")
    het = np.load(FLAT / "train/hist_event_type.npy", mmap_mode="r")[hrows]
    groups = {"cl4096": (4096, np.load(CLU / "hist_cl_4096_train.npy", mmap_mode="r")[hrows].astype(np.int32)),
              "cl1024": (1024, np.load(CLU / "hist_cl_1024_train.npy", mmap_mode="r")[hrows].astype(np.int32)),
              "cat": (54, np.load(FLAT / "train/hist_category_id.npy", mmap_mode="r")[hrows].astype(np.int32))}
    del hrows
    Ws = {}
    for gname, (K, hg) in groups.items():
        ok = hg >= 0
        key = np.unique(hloc[ok].astype(np.int64) * K + hg[ok])
        X = sp.csr_matrix((np.ones(len(key), np.float32), ((key // K).astype(np.int64), (key % K).astype(np.int32))),
                          shape=(len(snap), K))
        log(f"{gname}: X {X.shape} nnz {X.nnz}")
        Ws[(gname, "dan")] = fit_dan(X)
        Ws[(gname, "rdlae")] = fit_rdlae(X)
        if gname == "cl4096":
            Ws[(gname, "ease")] = fit_ease(X, 500.0)
            Ws[(gname, "dan_t300")] = truncate(Ws[(gname, "dan")].T, 300)      # the winner's Gt = truncate(W^T, 300)
            Ws[(gname, "rdlae_t300")] = truncate(Ws[(gname, "rdlae")].T, 300)
            key1 = np.unique(hloc[ok & (het == 1)].astype(np.int64) * K + hg[ok & (het == 1)])
            X1 = sp.csr_matrix((np.ones(len(key1), np.float32), ((key1 // K).astype(np.int64), (key1 % K).astype(np.int32))),
                               shape=(len(snap), K))
            Ws[(gname, "dant1W")] = fit_dan(X1)
            # label-free type-1 share per group over the snapshot histories (G2 analogue of the contact ratio)
            tot = np.bincount(hg[ok], minlength=K).astype(np.float64)
            t1s = np.bincount(hg[ok & (het == 1)], minlength=K).astype(np.float64)
            t1share = ((t1s + 1.0) / (tot + 1.0 / max(t1s.sum() / tot.sum(), 1e-6))).astype(np.float32)  # Bayesian, prior = global share
            np.save(WD / "grp_t1share_cl4096.npy", t1share)
            # parity variant: W with the earliest val/test snapshot of each user added (compared on val only)
            Xp = X
            for sp_ in ("val", "test"):
                f2, M2, o2 = feed_meta(sp_)
                u2 = f2["user_id"].to_numpy().astype(np.int64); t2 = f2["timestamp"].to_numpy().astype(np.int64)
                o = np.lexsort((np.arange(len(u2)), t2, u2)); us2 = u2[o]; first = np.r_[True, us2[1:] != us2[:-1]]
                sel = np.sort(np.concatenate([o[first & (us2 > 0)], np.where(u2 == 0)[0]]))
                hr, hl = hist_rows_of(sel, f2["hist_off"].to_numpy().astype(np.int64), f2["H"].to_numpy().astype(np.int64))
                g2 = np.load(CLU / f"hist_cl_4096_{sp_}.npy", mmap_mode="r")[hr].astype(np.int32); ok2 = g2 >= 0
                k2 = np.unique(hl[ok2].astype(np.int64) * K + g2[ok2])
                Xp = sp.vstack([Xp, sp.csr_matrix((np.ones(len(k2), np.float32), ((k2 // K).astype(np.int64), (k2 % K).astype(np.int32))),
                                                   shape=(len(sel), K))]).tocsr()
            Ws[(gname, "dan_all")] = fit_dan(Xp)
            log(f"parity W rows {Xp.shape[0]}")
        log(f"{gname}: W fitted ({[k[1] for k in Ws if k[0] == gname]})")
    del groups, het, hloc
    for (gname, w), Wm in Ws.items():
        np.save(WD / f"W_{gname}_{w}.npy", Wm)
    for (gname, w), Wm in Ws.items():
        Ws[(gname, w)] = np.ascontiguousarray(Wm.T)  # WT[c, j] for cache-friendly gathers
    # ---- 2. scoring ----
    VAR = {"bin": 0, "cnt": 1, "int3": 2, "t1": 3, "rec05": 4, "exp05": 5}
    SPEC = {  # column name -> (group, W, variant)
        "w25_dan_bin": ("cl4096", "dan", "bin"), "w25_dan_cnt": ("cl4096", "dan", "cnt"),
        "w25_rdlae_bin": ("cl4096", "rdlae", "bin"), "w25_rdlae_cnt": ("cl4096", "rdlae", "cnt"),
        "w25_ease_bin": ("cl4096", "ease", "bin"),
        "w25_dan_bin_t300": ("cl4096", "dan_t300", "bin"), "w25_rdlae_bin_t300": ("cl4096", "rdlae_t300", "bin"),
        "w25_dan1024_bin": ("cl1024", "dan", "bin"), "w25_rdlae1024_bin": ("cl1024", "rdlae", "bin"),
        "w25_dancat_bin": ("cat", "dan", "bin"), "w25_rdlaecat_bin": ("cat", "rdlae", "bin"),
        "w25_dan_int3": ("cl4096", "dan", "int3"), "w25_rdlae_int3": ("cl4096", "rdlae", "int3"),
        "w25_dan_t1": ("cl4096", "dan", "t1"), "w25_rdlae_t1": ("cl4096", "rdlae", "t1"),
        "w25_dan_rec05": ("cl4096", "dan", "rec05"), "w25_rdlae_rec05": ("cl4096", "rdlae", "rec05"),
        "w25_dan_exp05": ("cl4096", "dan", "exp05"), "w25_rdlae_exp05": ("cl4096", "rdlae", "exp05"),
        "w25_dant1W_bin": ("cl4096", "dant1W", "bin"), "w25_dant1W_t1": ("cl4096", "dant1W", "t1"),
        "w25_rdlae_int3_t300": ("cl4096", "rdlae_t300", "int3"),
        "w25_dan_all_bin": ("cl4096", "dan_all", "bin"),  # parity check column, val only
    }
    pr = np.load(ROOT / "eda/labels_position/out/train_prandom.npz")
    p = np.nan_to_num(pr["p_random"].astype(np.float32), nan=0.0)
    for split in ("val", "test", "train"):
        f, M, off = feed_meta(split)
        if split == "train":
            feeds = np.where(p > 0.5)[0]
            rows = np.repeat(off[feeds] - np.r_[0, np.cumsum(M[feeds])[:-1]], M[feeds]) + np.arange(M[feeds].sum())
            np.save(WD / "train/rows.npy", rows)
        else:
            feeds = np.arange(len(M)); rows = None
        cptr = np.r_[0, np.cumsum(M[feeds])]
        hr, hl = hist_rows_of(feeds, f["hist_off"].to_numpy().astype(np.int64), f["H"].to_numpy().astype(np.int64))
        het = np.load(FLAT / f"{split}/hist_event_type.npy", mmap_mode="r")[hr]
        hpos = np.load(FLAT / f"{split}/hist_hpos.npy", mmap_mode="r")[hr]
        cands = {"cl4096": np.load(CLU / f"cand_cl_4096_{split}.npy", mmap_mode="r"), "cl1024": np.load(CLU / f"cand_cl_1024_{split}.npy", mmap_mode="r"),
                 "cat": np.load(FLAT / f"{split}/cand_category_id.npy", mmap_mode="r")}
        hists = {"cl4096": np.load(CLU / f"hist_cl_4096_{split}.npy", mmap_mode="r"), "cl1024": np.load(CLU / f"hist_cl_1024_{split}.npy", mmap_mode="r"),
                 "cat": np.load(FLAT / f"{split}/hist_category_id.npy", mmap_mode="r")}
        log(f"{split}: {len(feeds)} feeds, {cptr[-1]} cand rows, {len(hr)} hist rows")
        cols = {}
        for gname, K in (("cl4096", 4096), ("cl1024", 1024), ("cat", 54)):
            hg = hists[gname][hr].astype(np.int32)
            hptr, g, w = compress(hg, hl, het, hpos, K, len(feeds))
            cg = cands[gname][rows] if rows is not None else cands[gname][:]
            cg = cg.astype(np.int32)
            specs = [(n, s) for n, s in SPEC.items() if s[0] == gname and (split == "val" or s[1] != "dan_all")]
            by_w = {}
            for n, s in specs:
                by_w.setdefault(s[1], []).append((n, VAR[s[2]]))
            for wname, lst in by_w.items():
                vsel = np.array([v for _, v in lst], np.int64)
                out = np.empty((len(cg), len(lst)), np.float32)
                score_rows(cptr, cg, hptr, g, w, Ws[(gname, wname)], vsel, out)
                for k, (n, _) in enumerate(lst):
                    cols[n] = out[:, k]
            log(f"{split} {gname}: {len(specs)} columns")
        # within-feed rank/z of the two main scores (NaN -> feed min before ranking)
        for n in ("w25_dan_bin", "w25_rdlae_bin"):
            x = cols[n].copy(); x = np.where(np.isnan(x), np.nanmin(x), x)
            rz = feed_rank_z(cptr[:-1].astype(np.int64), M[feeds].astype(np.int64), x.astype(np.float32))
            cols[n + "_rk"] = rz[:, 0]; cols[n + "_z"] = rz[:, 1]
        for n, x in cols.items():
            np.save(WD / f"{split}/{n}.npy", x.astype(np.float32))
        if split == "val":
            a, b = cols["w25_dan_bin"], cols["w25_dan_all_bin"]; ok = ~np.isnan(a)
            log(f"PARITY: corr(val dan_bin | W train-only, W with val/test snapshots) = {np.corrcoef(a[ok], b[ok])[0, 1]:.6f}")
        log(f"{split} written {len(cols)} columns; NaN share dan_bin {np.isnan(cols['w25_dan_bin']).mean():.5f}")
    names_g1 = ["w25_dan_bin", "w25_dan_cnt", "w25_rdlae_bin", "w25_rdlae_cnt", "w25_ease_bin", "w25_dan_bin_t300", "w25_rdlae_bin_t300",
                "w25_dan1024_bin", "w25_rdlae1024_bin", "w25_dancat_bin", "w25_rdlaecat_bin",
                "w25_dan_bin_rk", "w25_dan_bin_z", "w25_rdlae_bin_rk", "w25_rdlae_bin_z"]
    names_g5 = ["w25_dan_int3", "w25_rdlae_int3", "w25_dan_t1", "w25_rdlae_t1", "w25_dan_rec05", "w25_rdlae_rec05", "w25_dan_exp05",
                "w25_rdlae_exp05", "w25_dant1W_bin", "w25_dant1W_t1", "w25_rdlae_int3_t300"]
    gp = WD / "groups.json"
    G = json.loads(gp.read_text()) if gp.exists() else {}
    G["G1"] = names_g1; G["G5"] = names_g5
    gp.write_text(json.dumps(G, indent=1))
    log("done")


if __name__ == "__main__":
    main()
