"""Build one row-per-candidate analysis table per sample, all with the same columns.

Samples
  val      14,423 validation feeds, scored by FD16P20 (final score = 0.5 z(fine-tuned) + 0.5 z(train-only))
  holdout  30,000 random-order train feeds (same selection as lab/nn/holdout_eval.py), scored by the
           train-only branch of W8P20, whose 12 members never trained on them
  test     28,595 test feeds, scored by FD16P20 (no labels)

Outputs (folder data/ and private/ next to code/):
  data/disc.parquet        validation users of the discovery half, with labels (the analysts' sample)
  private/confirm.parquet  validation users of the confirmation half, with labels (sealed)
  private/holdout.parquet  holdout, with labels (sealed)
  private/test.parquet     test, no labels (built with --test)

Every column is computed from the feed's own row: its candidates, its user vector and its own history
snapshot. Nothing reads another feed of the same user. Rank ties are broken by row order (stable).
"""
from __future__ import annotations

import os

for _k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ.setdefault(_k, "1")      # 12 workers x 1 BLAS thread; set before numpy is imported

import argparse
import json
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path("<repo>")
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent.parent
FLAT = ROOT / "artifacts/data/flat"
FC = ROOT / "artifacts/feature_cache"
COMP = ROOT / "artifacts/data/competition"
CACHE_COLS = ["it_hist_all_luo", "it_hist_et1_luo", "it_hist_et2_luo", "it_hist_et3_luo", "it_hist_et4_luo",
              "xp_imp_c50", "seq_cv_invd20", "pre2_fe_z", "pre2_new_z"]
CONTACT, ORDER, FAV, LEAD = (2, 3, 9), (5, 6, 8), (4,), (7,)

G = {}  # per-process globals


def zscore_feeds(x: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    off = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    out = np.empty_like(x, dtype=np.float64)
    for o, m in zip(off, sizes):
        v = x[o:o + m].astype(np.float64)
        sd = v.std()
        out[o:o + m] = (v - v.mean()) / (sd if sd > 0 else 1.0)
    return out


def init_worker(split: str):
    d = FLAT / split
    G["split"] = split
    G["feeds"] = pl.read_parquet(d / "feeds.parquet")
    for k in ["cand_category_id", "cand_emb_idx", "cand_f1", "cand_f2", "cand_f3", "cand_f4", "cand_f5", "cand_item_id",
              "hist_category_id", "hist_emb_idx", "hist_event_type", "hist_event_source"]:
        G[k] = np.load(d / f"{k}.npy", mmap_mode="r")
    G["user_emb"] = np.load(d / "user_emb.npy", mmap_mode="r")
    meta = json.load(open(COMP / split / "item_embeddings/data.meta"))
    G["items"] = np.memmap(COMP / split / "item_embeddings/data.memmap", dtype=np.int8, mode="r", shape=tuple(meta["shape"]))
    G["cache"] = {c: np.load(FC / split / f"{c}.npy", mmap_mode="r") for c in CACHE_COLS}


def maxcos(sim: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if sim.shape[1] == 0 or not mask.any():
        return np.full(sim.shape[0], np.nan, np.float32)
    return sim[:, mask].max(1)


def feed_rows(args):
    """Columns for one feed: (fidx, score array of its rows in stored order) -> dict of arrays."""
    fidx, score = args
    t = G["feeds"].row(int(fidx), named=True)
    o, m = int(t["cand_off"]), int(t["M"])
    ho, hn = int(t["hist_off"]), int(t["H"])
    cat = np.asarray(G["cand_category_id"][o:o + m]).astype(np.int16)
    cemb = np.asarray(G["cand_emb_idx"][o:o + m])
    hc = np.asarray(G["hist_category_id"][ho:ho + hn]).astype(np.int16)
    ht = np.asarray(G["hist_event_type"][ho:ho + hn]).astype(np.int16)
    hs = np.asarray(G["hist_event_source"][ho:ho + hn]).astype(np.int16)
    he = np.asarray(G["hist_emb_idx"][ho:ho + hn])
    items = G["items"]
    cv_ok = cemb >= 0
    cvec = np.zeros((m, 128), np.float32)
    if cv_ok.any():
        cvec[cv_ok] = items[cemb[cv_ok]].astype(np.float32)
    hok = he >= 0
    hidx = np.nonzero(hok)[0]
    hvec = items[he[hok]].astype(np.float32) if len(hidx) else np.zeros((0, 128), np.float32)
    sim = (cvec @ hvec.T) / 65536.0 if len(hidx) else np.zeros((m, 0), np.float32)
    if sim.shape[1]:
        sim[~cv_ok] = np.nan
    u = np.asarray(G["user_emb"][int(fidx)]).astype(np.float32)
    uvec_missing = not u.any()
    out = {}
    out["ucos"] = np.where(cv_ok & (not uvec_missing), (cvec @ u) / 65536.0, np.nan).astype(np.float32)
    if sim.shape[1]:
        b = np.nanargmax(np.where(np.isnan(sim), -9, sim), 1)
        out["hmax"] = sim[np.arange(m), b].astype(np.float32)
        out["hmax_idx"] = hidx[b].astype(np.int16)
        out["hmax_type"] = ht[hidx[b]]
        out["hmax_src"] = hs[hidx[b]]
        ev_t = ht[hidx]; ev_s = hs[hidx]
        for k in (1, 5, 10, 32):
            out[f"h{k}"] = maxcos(sim, hidx < k)
        first5 = hidx < 5
        out["hmean5"] = (sim[:, first5].mean(1) if first5.any() else np.full(m, np.nan)).astype(np.float32)
        out["n_hdup95"] = (sim >= 0.95).sum(1).astype(np.int16)
        out["c_contact"] = maxcos(sim, np.isin(ev_t, CONTACT))
        out["c_order"] = maxcos(sim, np.isin(ev_t, ORDER))
        out["c_fav"] = maxcos(sim, np.isin(ev_t, FAV))
        out["c_lead"] = maxcos(sim, np.isin(ev_t, LEAD))
        out["c_vfeed"] = maxcos(sim, (ev_t == 1) & (ev_s == 1))
        out["c_vsearch"] = maxcos(sim, (ev_t == 1) & (ev_s == 2))
    else:
        nanf = np.full(m, np.nan, np.float32)
        out.update(hmax=nanf, hmax_idx=np.full(m, -1, np.int16), hmax_type=np.full(m, -1, np.int16),
                   hmax_src=np.full(m, -1, np.int16), h1=nanf, h5=nanf, h10=nanf, h32=nanf, hmean5=nanf,
                   n_hdup95=np.zeros(m, np.int16), c_contact=nanf, c_order=nanf, c_fav=nanf, c_lead=nanf,
                   c_vfeed=nanf, c_vsearch=nanf)
    # category history (0 / -1 when the category is absent from the history)
    if hn:
        views = np.isin(ht, (1, 10))
        cats_u, inv = np.unique(hc, return_inverse=True)
        cnt_v = np.bincount(inv, weights=views, minlength=len(cats_u))
        cnt_a = np.bincount(inv, weights=~views, minlength=len(cats_u))
        newest = np.full(len(cats_u), 10**4, int)
        np.minimum.at(newest, inv, np.arange(hn))
        crank = np.empty(len(cats_u), int)
        crank[np.argsort(-(cnt_v + cnt_a), kind="stable")] = np.arange(1, len(cats_u) + 1)
        pos = np.minimum(np.searchsorted(cats_u, cat), len(cats_u) - 1)
        found = cats_u[pos] == cat
        out["cat_views"] = np.where(found, cnt_v[pos], 0).astype(np.int16)
        out["cat_actions"] = np.where(found, cnt_a[pos], 0).astype(np.int16)
        out["cat_newest_idx"] = np.where(found, newest[pos], -1).astype(np.int16)
        out["cat_user_rank"] = np.where(found, crank[pos], 0).astype(np.int16)
        out["same_cat_newest"] = cat == hc[0]
    else:
        z16 = np.zeros(m, np.int16)
        out.update(cat_views=z16, cat_actions=z16, cat_newest_idx=np.full(m, -1, np.int16), cat_user_rank=z16,
                   same_cat_newest=np.zeros(m, bool))
    out["cat_hist_share"] = ((out["cat_views"].astype(np.float32) + out["cat_actions"]) / max(hn, 1)).astype(np.float32)
    # candidate set context
    cc = (cvec @ cvec.T) / 65536.0
    np.fill_diagonal(cc, -9)
    cc[~cv_ok] = -9; cc[:, ~cv_ok] = -9
    out["cand_maxcos"] = np.where(cv_ok, cc.max(1) if m > 1 else -9, np.nan).astype(np.float32)
    out["n_cand_dup95"] = (cc >= 0.95).sum(1).astype(np.int16)
    cu, cinv, ccount = np.unique(cat, return_inverse=True, return_counts=True)
    out["cat_n_in_feed"] = ccount[cinv].astype(np.int16)
    # model order
    rk = np.empty(m, int); rk[np.argsort(-score, kind="stable")] = np.arange(1, m + 1)
    out["rank"] = rk.astype(np.int16)
    cat_rank = np.empty(m, int)
    for c in range(len(cu)):
        idx = np.nonzero(cinv == c)[0]
        cat_rank[idx[np.argsort(rk[idx])]] = np.arange(1, len(idx) + 1)
    out["cat_rank_in_feed"] = cat_rank.astype(np.int16)
    srt = np.sort(score)[::-1]
    nxt = np.empty(m); nxt[np.argsort(-score, kind="stable")] = np.append(srt[1:], srt[-1])
    out["gap_next"] = (score - nxt).astype(np.float32)
    out["gap_top"] = (srt[0] - score).astype(np.float32)
    f2 = np.asarray(G["cand_f2"][o:o + m])
    nz = f2[f2 > 0]
    if len(nz):
        vals, cnts = np.unique(nz, return_counts=True)
        out["f2_home"] = f2 == vals[cnts.argmax()]
    else:
        out["f2_home"] = np.zeros(m, bool)
    for k in ["f1", "f2", "f3", "f4", "f5"]:
        out[k] = np.asarray(G[f"cand_{k}"][o:o + m]).astype(np.float32)
    out["cat"] = cat
    out["item_id"] = np.asarray(G["cand_item_id"][o:o + m])
    for c in CACHE_COLS:
        out[c] = np.asarray(G["cache"][c][o:o + m]).astype(np.float32)
    ts = int(t["timestamp"]) + 3 * 3600
    feedcols = dict(fidx=int(fidx), user_id=int(t["user_id"]), M=m, H=hn, guest=int(t["user_id"]) == 0,
                    uvec_missing=uvec_missing, hour=(ts % 86400) // 3600, weekday=(ts // 86400 + 3) % 7,
                    newest_type=int(ht[0]) if hn else -1, newest_src=int(hs[0]) if hn else -1,
                    newest_cat=int(hc[0]) if hn else -1, n_hist_cats32=len(np.unique(hc[:32])) if hn else 0,
                    n_feed_cats=len(cu), row0=o)
    for k, v in feedcols.items():
        out[k] = np.full(m, v)
    out["row"] = np.arange(o, o + m)
    return out


def build(split: str, fidx: np.ndarray, scores: dict[str, np.ndarray], offsets: np.ndarray, sizes: np.ndarray,
          labels: np.ndarray | None) -> pl.DataFrame:
    """scores: name -> flat array aligned with the concatenation of the feeds in `fidx` (in that order)."""
    loc = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    jobs = [(int(f), scores["score"][l:l + s].astype(np.float64)) for f, l, s in zip(fidx, loc, sizes)]
    with get_context("spawn").Pool(12, initializer=init_worker, initargs=(split,)) as pool:   # fork deadlocks polars
        parts = pool.map(feed_rows, jobs, chunksize=64)
    cols = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    for name, arr in scores.items():
        cols[name] = arr.astype(np.float32)
    if labels is not None:
        cols["label"] = labels.astype(np.int8)
    df = pl.DataFrame(cols)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    from avito_ranker.ensemble.build import final_score, load_pool
    from avito_ranker.ensemble.feeds import load_feed_index
    from avito_ranker.ensemble.zscore import feed_zscore
    (HERE / "data").mkdir(exist_ok=True); (HERE / "private").mkdir(exist_ok=True)
    if a.test:
        F = load_feed_index("test")
        b = load_pool("FD16P20kd", "test")
        zf, zt = feed_zscore(F, b.finetuned), feed_zscore(F, b.train_only)
        sc = final_score(b, F, 0.5)
        df = build("test", np.arange(F.n_feeds), {"score": sc, "s_ft": zf, "s_to": zt}, F.offsets, F.sizes, None)
        df.write_parquet(HERE / "private/test.parquet")
        print("test rows", df.height)
        return
    # validation
    F = load_feed_index("val")
    b = load_pool("FD16P20kd", "val")
    zf, zt = feed_zscore(F, b.finetuned), feed_zscore(F, b.train_only)
    sc = final_score(b, F, 0.5)
    y = np.load(FLAT / "val/cand_target.npy")
    df = build("val", np.arange(F.n_feeds), {"score": sc, "s_ft": zf, "s_to": zt}, F.offsets, F.sizes, y)
    # discovery / confirmation halves by user; every user-0 feed is its own group
    u = F.table["user_id"].to_numpy().astype(np.int64)
    g = np.where(u == 0, -(np.arange(F.n_feeds) + 1), u)
    groups = np.unique(g)
    rng = np.random.default_rng(1117)
    conf_groups = set(rng.choice(groups, len(groups) // 2, replace=False).tolist())
    is_conf = np.array([x in conf_groups for x in g])
    feed_conf = pl.DataFrame({"fidx": np.arange(F.n_feeds), "confirm": is_conf})
    df = df.join(feed_conf, on="fidx", how="left")
    df.filter(~pl.col("confirm")).drop("confirm").write_parquet(HERE / "data/disc.parquet")
    df.filter(pl.col("confirm")).drop("confirm").write_parquet(HERE / "private/confirm.parquet")
    print("val rows", df.height, "disc feeds", int((~is_conf).sum()), "confirm feeds", int(is_conf.sum()))
    # holdout
    tr = pl.read_parquet(FLAT / "train/feeds.parquet")
    M = tr["M"].to_numpy()
    pr = np.nan_to_num(np.load(ROOT / "artifacts/exploration/train_prandom.npz")["p_random"].astype(np.float32), nan=0.0)
    cand = np.where((pr > 0.9) & (M >= 50))[0]
    hold = np.sort(np.random.default_rng(12345).choice(cand, 30000, replace=False))
    meta = np.load(ROOT / "artifacts/predictions/holdout/holdout_meta.npy")
    assert (meta[0] == M[hold]).all(), "holdout selection does not match the stored holdout"
    hy = np.load(ROOT / "artifacts/predictions/holdout/holdout_y.npy")
    yt = np.load(FLAT / "train/cand_target.npy", mmap_mode="r")
    off = tr["cand_off"].to_numpy()
    rows = np.concatenate([np.arange(o, o + s) for o, s in zip(off[hold], M[hold])])
    assert (np.asarray(yt[rows]) == hy).all(), "holdout labels do not match"
    sizes = M[hold].astype(np.int64)
    W5 = ROOT / "lab/nn_x/w5/preds/holdout"; AB = ROOT / "lab/nn_x/abl/preds/holdout"
    full = [W5 / f"nn_w5_e10_kd05w192_s95{k}_holdout.npy" for k in range(1, 5)] + \
           [W5 / f"nn_w5_e10_kd05t25w192_s95{k}_holdout.npy" for k in range(1, 5)]
    light = [AB / f"nn_abl_e10_pop1kd05t25_s95{k}_holdout.npy" for k in range(1, 5)]
    pf = np.mean([zscore_feeds(np.load(p), sizes) for p in full], 0)
    pl_ = np.mean([zscore_feeds(np.load(p), sizes) for p in light], 0)
    sc = zscore_feeds(0.8 * zscore_feeds(pf, sizes) + 0.2 * zscore_feeds(pl_, sizes), sizes)
    dh = build("train", hold, {"score": sc, "s_ft": np.full(len(sc), np.nan), "s_to": sc}, off[hold], sizes, hy)
    dh.write_parquet(HERE / "private/holdout.parquet")
    print("holdout rows", dh.height)


if __name__ == "__main__":
    main()
