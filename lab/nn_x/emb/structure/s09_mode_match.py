"""(3e) Mode match: for feeds whose history is dominated by one side of the bimodal dims (6 > 12, 96 > 40), compare candidates on the
same side vs the other side: f1..f5 means (is any f a distance-like quantity?) and click rate. VAL, descriptive (no fitting)."""
import json, numpy as np, polars as pl

OUT = "<repo>/nn_x/emb/structure/"
V = "<repo>/data/flat/val/"
NV = 2_516_711
E = np.fromfile("<repo>/data/competition/val/item_embeddings/data.memmap", dtype=np.int8).reshape(NV, 128)
feeds = pl.read_parquet(V + "feeds.parquet"); F = len(feeds)
ce = np.load(V + "cand_emb_idx.npy"); cf = np.load(V + "cand_fidx.npy").astype(np.int64); tg = np.load(V + "cand_target.npy").astype(np.float64)
cc = np.load(V + "cand_category_id.npy").astype(np.int64)
he = np.load(V + "hist_emb_idx.npy"); hf = np.load(V + "hist_fidx.npy").astype(np.int64)
fs = {k: np.load(V + f"cand_f{k}.npy").astype(np.float64) for k in range(1, 6)}
ok = ce >= 0; hok = he >= 0
res = {}
for d, thr in [(6, 12), (96, 40)]:
    cin = np.zeros(len(ce), bool); cin[ok] = E[ce[ok], d] > thr
    hin = np.zeros(len(he), bool); hin[hok] = E[he[hok], d] > thr
    nh = np.bincount(hf[hok], minlength=F); sh = np.bincount(hf[hok], weights=hin[hok], minlength=F) / np.maximum(nh, 1)
    for side, fm in [("hist_mostly_IN(share>0.8)", (sh > 0.8) & (nh >= 20)), ("hist_mostly_OUT(share<0.05)", (sh < 0.05) & (nh >= 20))]:
        rows = ok & fm[cf]
        same = cin if "IN" in side else ~cin
        r = {"n_feeds": int(fm.sum()), "cand_rows": int(rows.sum()), "share_cands_same_side": float(same[rows].mean()),
             "click_rate_same_side": float(tg[rows & same].mean()), "click_rate_other_side": float(tg[rows & ~same].mean()) if (rows & ~same).sum() else None}
        # f-means same vs other side, within the same category (category-demeaned by feed rows)
        for k in range(1, 6):
            r[f"f{k}_same_minus_other"] = float(fs[k][rows & same].mean() - fs[k][rows & ~same].mean()) if (rows & ~same).sum() else None
        # within-feed: share of feeds where the other-side candidates have lower click rate
        res[f"dim{d}_{side}"] = r
        print(d, side, r, flush=True)
json.dump(res, open(OUT + "s09_mode_match.json", "w"), indent=1)
