"""Assemble per-row analysis tables for val and the 30K in-period holdout (read-only on project files).
Outputs (this dir): val.npz, hold.npz with
  feed-level: M, off, uid, ts, H, u0
  row-level : y, g (G3 pool z: val final 0.5z(ft)+0.5z(to); holdout train-only pool), gto, gft (val only),
              f1..f5, cat, item, cold (val: item unseen in train cands/hist; hold: -1),
              34 xt inputs (cvnd_pre2s, attach_xt transform), CPU replica of key GPU dense features
              (cos_u, smax, smax8, smean, lse, top3, ccnt, cshare, firstrec, fcs, + per-type same-cat counts e1..e4,
               max cos per event type e1/e2/e3/e4)
Run: CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 ... .venv/bin/python build.py
"""
import json, sys, time
import numpy as np, polars as pl
import numba as nb
sys.path.insert(0, "<repo>/src")
from stack_val import feed_rank_z
from metrics import ndcg_per_feed

ROOT = "<repo>"; FLAT = f"{ROOT}/data/flat"; P = f"{ROOT}/preds"; H_ = f"{P}/holdout"
FC = f"{ROOT}/nn/feat_cache"
OUT = "/tmp/scratchpad/dpa/nonlinear"
t0 = time.time()
log = lambda *a: print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)

XT = ["seq_cv_invd20", "seq_cv_pop", "seq_cv_norm", "it_hist_all_luo", "it_hist_et1_luo", "it_hist_et2_luo",
      "it_hist_et3_luo", "it_hist_et4_luo", "ctx_nd_hcent", "ctx_nd_knn_f1", "ctx_nd_knn_h", "ctx_nd_knn_h_dev",
      "ctx_nd_knn_u", "ctx_nd_knn_u_dev", "ctx_nd_max", "ctx_nd_top5h", "ctx_nd_top5u", "ctx_nd_ucent", "ctx_hn_top1w",
      "ctx_hn_rrw", "ctx_hn_bestrk", "ctx_hn_margin", "ctx_hn_soft05", "ctx_hn_e1_rrw", "ctx_hs_mutual", "ctx_hs_k128",
      "ctx_hs_e1", "ctx_hs_t10", "pre2_fe_z", "pre2_new_z", "pre2_fe_s", "pre2_new_s", "pre2_hmax32", "has:pre2_fe"]
assert len(XT) == 34
F10 = [f"v23_gru_pre2s_s{s}" for s in range(501, 507)]; F7x = [f"r6_cd_m02_r{s}" for s in range(101, 108)]
F7x70 = [f"r6_cd_m02_p70k_r{s}" for s in range(101, 108)]; G3 = F10 + F7x + F7x70


def xt_transform(n, x):
    x = x.astype(np.float32)
    if n.startswith("has:") or n.endswith("_rk"):
        pass
    elif n.endswith("_z"):
        x = np.clip(x, -5, 5)
    elif n.startswith(("seq_cv", "it_hist")):
        x = np.log1p(np.maximum(x, 0))
    else:
        x = np.clip(x, -10, 10)
    return np.nan_to_num(x, nan=0.0)


@nb.njit(parallel=True, fastmath=True)
def dense_cpu(M, coff, hoff, Hc, cemb, hemb, table, ue, ccat, hcat, het, out):
    """out [N, 25]: cos_u smax smax8 smax32 smean lse top3 ccnt_log cshare firstrec fcs logM logH
       maxcos_e1..e4 ccnt_e1..e4 (log1p) hn_cat_last8 hn_cat_last32 cn"""
    nf = len(M)
    for f in nb.prange(nf):
        m = M[f]; o = coff[f]; h = Hc[f]; ho = hoff[f]
        u = ue[f].astype(np.float64)
        un = np.sqrt((u * u).sum()); u = u / max(un, 1e-3)
        HE = np.zeros((h, 128), np.float64); hv = np.zeros(h, np.bool_)
        for k in range(h):
            e = hemb[ho + k]
            if e >= 0:
                v = table[e].astype(np.float64); nv = np.sqrt((v * v).sum())
                HE[k] = v / max(nv, 1e-3); hv[k] = True
        nh = hv.sum()
        # feed category counts
        for i in range(m):
            r = o + i
            e = cemb[r]
            c = np.zeros(128, np.float64); cn = 0.0
            if e >= 0:
                c = table[e].astype(np.float64); cn = np.sqrt((c * c).sum()); c = c / max(cn, 1e-3)
            out[r, 0] = (c * u).sum()
            smax = -1.5; smax8 = -1.5; smax32 = -1.5; ssum = 0.0; mx = np.full(4, -1.5, np.float64)
            t1 = -1.5; t2 = -1.5; t3 = -1.5; lse_acc = 0.0; lmax = -1e9
            S = np.empty(h, np.float64)
            for k in range(h):
                if hv[k]:
                    s = (c * HE[k]).sum(); S[k] = s
                    if s > lmax: lmax = s
            for k in range(h):
                if hv[k]:
                    s = S[k]
                    ssum += s
                    if s > smax: smax = s
                    if k < 8 and s > smax8: smax8 = s
                    if k < 32 and s > smax32: smax32 = s
                    lse_acc += np.exp(5 * (s - lmax))
                    t = het[ho + k]
                    if 1 <= t <= 4 and s > mx[t - 1]: mx[t - 1] = s
                    if s > t1:
                        t3 = t2; t2 = t1; t1 = s
                    elif s > t2:
                        t3 = t2; t2 = s
                    elif s > t3:
                        t3 = s
            out[r, 1] = smax; out[r, 2] = smax8; out[r, 3] = smax32
            out[r, 4] = ssum / max(nh, 1)
            out[r, 5] = (lmax + np.log(lse_acc) / 5 - np.log(max(nh, 1)) / 5) if nh > 0 else -1.5
            if nh >= 3:
                out[r, 6] = (t1 + t2 + t3) / 3
            elif nh == 2:
                out[r, 6] = (t1 + t2 + -1.5) / 3
            elif nh == 1:
                out[r, 6] = (t1 - 3.0) / 3
            else:
                out[r, 6] = -1.5
            # category affinity
            cc = ccat[r]; cnt = 0; first = -1; ce = np.zeros(4); c8 = 0; c32 = 0
            for k in range(h):
                if hcat[ho + k] == cc:
                    cnt += 1
                    if first < 0: first = k
                    t = het[ho + k]
                    if 1 <= t <= 4: ce[t - 1] += 1
                    if k < 8: c8 += 1
                    if k < 32: c32 += 1
            out[r, 7] = np.log1p(cnt); out[r, 8] = cnt / max(h, 1)
            out[r, 9] = 1.0 / (1.0 + first) if first >= 0 else 0.0
            fc = 0
            for j in range(m):
                if ccat[o + j] == cc: fc += 1
            out[r, 10] = fc / m
            out[r, 11] = np.log(m); out[r, 12] = np.log1p(h)
            for t in range(4):
                out[r, 13 + t] = mx[t]; out[r, 17 + t] = np.log1p(ce[t])
            out[r, 21] = c8 / 8.0; out[r, 22] = c32 / 32.0
            out[r, 23] = np.log1p(cn) if e >= 0 else 0.0
            out[r, 24] = 1.0 if e < 0 else 0.0


DNAMES = ["cos_u", "smax", "smax8", "smax32", "smean", "lse", "top3", "ccnt", "cshare", "firstrec", "fcs", "logM", "logH",
          "mx_e1", "mx_e2", "mx_e3", "mx_e4", "cc_e1", "cc_e2", "cc_e3", "cc_e4", "cat_last8", "cat_last32", "cnorm", "cmiss"]


def zpool(paths, off, M):
    acc = None
    for p in paths:
        z = feed_rank_z(off, M, np.load(p).astype(np.float32))[:, 1]
        acc = z if acc is None else acc + z
    return feed_rank_z(off, M, (acc / len(paths)).astype(np.float32))[:, 1]


def table_of(split):
    d = f"{ROOT}/data/competition/{split}/item_embeddings"
    meta = json.loads(open(f"{d}/data.meta").read())
    return np.memmap(f"{d}/data.memmap", dtype=np.int8, mode="r", shape=tuple(meta["shape"]))


def build(split):
    f = pl.read_parquet(f"{FLAT}/{split if split != 'hold' else 'train'}/feeds.parquet")
    src = "train" if split == "hold" else split
    Mall = f["M"].to_numpy().astype(np.int64); offall = f["cand_off"].to_numpy().astype(np.int64)
    Hall = f["H"].to_numpy().astype(np.int64); hoffall = f["hist_off"].to_numpy().astype(np.int64)
    if split == "hold":
        pr = np.nan_to_num(np.load(f"{ROOT}/eda/labels_position/out/train_prandom.npz")["p_random"].astype(np.float32), nan=0.0)
        pr[Mall < 50] = 0.0
        cand = np.where((pr > 0.9) & (Mall >= 50))[0]
        feeds = np.sort(np.random.default_rng(12345).choice(cand, 30000, replace=False))
    else:
        feeds = np.arange(len(f))
    M = Mall[feeds]; coff_src = offall[feeds]; Hn = Hall[feeds]; hoff_src = hoffall[feeds]
    rows = np.concatenate([np.arange(o, o + m) for o, m in zip(coff_src, M)])
    off = np.concatenate([[0], np.cumsum(M)[:-1]]).astype(np.int64)
    D = {"M": M, "off": off, "uid": f["user_id"].to_numpy()[feeds], "ts": f["timestamp"].to_numpy()[feeds],
         "H": Hn, "feeds": feeds}
    D["u0"] = (D["uid"] == 0)
    ld = lambda n: np.load(f"{FLAT}/{src}/{n}.npy", mmap_mode="r")
    D["y"] = np.asarray(ld("cand_target")[rows]).astype(np.float32)
    for k in range(1, 6):
        D[f"f{k}"] = np.asarray(ld(f"cand_f{k}")[rows]).astype(np.float32)
    D["cat"] = np.asarray(ld("cand_category_id")[rows]).astype(np.int16)
    D["item"] = np.asarray(ld("cand_item_id")[rows])
    log(split, "base rows", len(rows))
    if split == "hold":
        Mh, oh = np.load(f"{H_}/holdout_meta.npy"); yh = np.load(f"{H_}/holdout_y.npy")
        assert (Mh == M).all() and (oh == off).all() and np.array_equal(yh, D["y"]), "holdout reconstruction mismatch"
        D["g"] = zpool([f"{H_}/nn_{m}_holdout.npy" for m in G3], off, M)
        D["cold"] = np.full(len(rows), -1, np.int8)
    else:
        D["gft"] = zpool([f"{P}/nn_{m}_n2cpv3_val_oof.npy" for m in G3], off, M)
        D["gto"] = zpool([f"{P}/nn_{m}_trainonly_val.npy" for m in G3], off, M)
        D["g"] = (0.5 * D["gft"] + 0.5 * D["gto"]).astype(np.float32)
        ref = np.load(f"{P}/ens_G3_final_val.npy")
        log("corr with ens_G3_final_val", np.corrcoef(ref, D["g"])[0, 1])
        NITEM = 111_400_000
        seen = np.zeros(NITEM, bool)
        seen[np.load(f"{FLAT}/train/cand_item_id.npy", mmap_mode="r")[:]] = True
        seen[np.load(f"{FLAT}/train/hist_item_id.npy", mmap_mode="r")[:]] = True
        D["cold"] = (~seen[D["item"]]).astype(np.int8)
        del seen
    nd = ndcg_per_feed(D["g"], D["y"], off, M).mean()
    log(split, "G3 NDCG", round(nd, 5))
    for n in XT:
        base = n[4:] if n.startswith("has:") else n
        x = np.load(f"{FC}/{src}/{base}.npy", mmap_mode="r")
        x = np.asarray(x[rows])
        if n.startswith("has:"):
            x = (~np.isnan(x)).astype(np.float32)
            D["xt_" + n.replace(":", "_")] = x
        else:
            D["xt_" + n] = xt_transform(n, x)
    log(split, "xt done")
    # history (capped to newest 256 as in the trainer)
    Hc = np.minimum(Hn, 256)
    hrows = np.concatenate([np.arange(o, o + h) for o, h in zip(hoff_src, Hc)]) if Hc.sum() else np.zeros(0, np.int64)
    hoff = np.concatenate([[0], np.cumsum(Hc)[:-1]]).astype(np.int64)
    hl = lambda n: np.asarray(np.load(f"{FLAT}/{src}/{n}.npy", mmap_mode="r")[hrows])
    hemb = hl("hist_emb_idx").astype(np.int64); hcat = hl("hist_category_id").astype(np.int16)
    het = hl("hist_event_type").astype(np.int8); hes = hl("hist_event_source").astype(np.int8)
    hpos = hl("hist_hpos")
    cemb = np.asarray(ld("cand_emb_idx")[rows]).astype(np.int64)
    ue = np.asarray(np.load(f"{FLAT}/{src}/user_emb.npy", mmap_mode="r")[feeds])
    tab = table_of(src)
    # gather only needed rows of the table (unique)
    need = np.unique(np.concatenate([cemb[cemb >= 0], hemb[hemb >= 0]]))
    sub = np.empty((len(need), 128), np.int8)
    for ch in np.array_split(np.arange(len(need)), 200):
        sub[ch] = tab[need[ch]]
    remap = lambda a: np.where(a >= 0, np.searchsorted(need, np.maximum(a, 0)), -1)
    cemb_r, hemb_r = remap(cemb), remap(hemb)
    log(split, "table rows", len(need))
    out = np.zeros((len(rows), len(DNAMES)), np.float32)
    dense_cpu(M, off, hoff, Hc, cemb_r, hemb_r, sub, ue, D["cat"].astype(np.int64), hcat.astype(np.int64),
              het.astype(np.int64), out)
    for j, n in enumerate(DNAMES):
        D["d_" + n] = out[:, j]
    log(split, "dense done")
    D["hoff"] = hoff; D["Hc"] = Hc; D["hcat"] = hcat; D["het"] = het; D["hes"] = hes; D["hpos"] = hpos
    D["hemb_ok"] = (hemb >= 0)
    np.savez(f"{OUT}/{split}.npz", **D)
    log(split, "saved", len(rows))


if __name__ == "__main__":
    for s in sys.argv[1:]:
        build(s)
