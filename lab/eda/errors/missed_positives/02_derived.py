"""Derived raw features for the missed-positives lens (val and test, identical definitions; own-feed info only).

Per candidate:
  d_item_hcnt        events in own history with the exact same item_id
  d_item_hmin        min hpos of that item in own history (999 if none)
  d_item_het1        same, event_type 1 only (count)
  d_item_hcontact    same item with event_type in {2,3} (contact-like) (count)
  d_dup95_h          history events (with emb) with cos >= 0.95 to the candidate, exact item excluded
  d_dup90_h          same with cos >= 0.90
  d_dup95_hmin       min hpos among cos>=0.95 history events (999 if none)
  d_cat_last1        candidate category == category of the newest history event
  d_cat_last3cnt     same-category events among the 3 newest
  d_ncat_feed        number of candidates of the same category in the feed
  d_uemb_norm        user embedding norm / 256 (feed constant)
  d_h_ncat_ratio     unique history categories / min(H, 256) (feed constant; exploration propensity)
  d_h_et1_32         share of event_type 1 among the 32 newest events (feed constant)
"""
import os
os.environ.setdefault("NUMBA_NUM_THREADS", "16")
import sys
sys.path.insert(0, "<repo>/src")
from common import *
import numba as nb

OUT = Path("<repo>/eda/errors/missed_positives/derived")
NAMES = ["d_item_hcnt", "d_item_hmin", "d_item_het1", "d_item_hcontact", "d_dup95_h", "d_dup90_h", "d_dup95_hmin",
         "d_cat_last1", "d_cat_last3cnt", "d_ncat_feed", "d_uemb_norm", "d_h_ncat_ratio", "d_h_et1_32"]


@nb.njit(parallel=True, cache=True)
def kern(coff, M, hoff, H, citem, ccat, cemb, hitem, hcat, hemb, het, hpos, table, uemb, out):
    for f in nb.prange(len(M)):
        o, m, ho, h = coff[f], M[f], hoff[f], H[f]
        un = 0.0
        for k in range(uemb.shape[1]):
            un += float(uemb[f, k]) ** 2
        un = np.sqrt(un) / 256.0
        # feed-level history stats
        seen = np.zeros(128, np.int64)
        ncat = 0
        n32 = 0; e1 = 0
        for j in range(h):
            c = hcat[ho + j]
            if seen[c] == 0:
                ncat += 1
            seen[c] += 1
            if hpos[ho + j] < 32:
                n32 += 1
                if het[ho + j] == 1:
                    e1 += 1
        ncat_ratio = ncat / max(min(h, 256), 1)
        et1_32 = e1 / n32 if n32 > 0 else np.nan
        # newest events (hist rows are hpos-ascending, verify by searching min hpos)
        last_cat = -1
        best = 1 << 30
        for j in range(h):
            if hpos[ho + j] < best:
                best = hpos[ho + j]; last_cat = hcat[ho + j]
        cnt_feed = np.zeros(128, np.int64)
        for i in range(m):
            cnt_feed[ccat[o + i]] += 1
        for i in range(m):
            ci = o + i
            it = citem[ci]; cc = ccat[ci]
            n_it = 0; it_min = 999; it_e1 = 0; it_ct = 0
            d95 = 0; d90 = 0; d95min = 999
            l3 = 0
            ce = cemb[ci]
            for j in range(h):
                hj = ho + j
                if hpos[hj] < 3 and hcat[hj] == cc:
                    l3 += 1
                if hitem[hj] == it:
                    n_it += 1
                    if hpos[hj] < it_min:
                        it_min = hpos[hj]
                    if het[hj] == 1:
                        it_e1 += 1
                    if het[hj] == 2 or het[hj] == 3:
                        it_ct += 1
                    continue
                he = hemb[hj]
                if ce < 0 or he < 0:
                    continue
                dot = 0
                for k in range(table.shape[1]):
                    dot += np.int32(table[ce, k]) * np.int32(table[he, k])
                cs = dot / 65536.0
                if cs >= 0.90:
                    d90 += 1
                    if cs >= 0.95:
                        d95 += 1
                        if hpos[hj] < d95min:
                            d95min = hpos[hj]
            out[ci, 0] = n_it; out[ci, 1] = it_min; out[ci, 2] = it_e1; out[ci, 3] = it_ct
            out[ci, 4] = d95; out[ci, 5] = d90; out[ci, 6] = d95min
            out[ci, 7] = 1.0 if cc == last_cat else 0.0
            out[ci, 8] = l3
            out[ci, 9] = cnt_feed[cc]
            out[ci, 10] = un; out[ci, 11] = ncat_ratio; out[ci, 12] = et1_32


def run(split):
    f, M, off = feed_meta(split)
    H = f["H"].to_numpy().astype(np.int64); hoff = f["hist_off"].to_numpy()
    d = FLAT / split
    L = lambda n: np.load(d / f"{n}.npy")
    meta = json.loads((Path("<repo>/data/competition") / split / "item_embeddings/data.meta").read_text())
    table = np.memmap(Path("<repo>/data/competition") / split / "item_embeddings/data.memmap", dtype=np.int8,
                      mode="r", shape=tuple(meta["shape"]))
    table = np.ascontiguousarray(table)
    out = np.zeros((int(M.sum()), len(NAMES)), np.float32)
    kern(off.astype(np.int64), M, hoff.astype(np.int64), H, L("cand_item_id"), L("cand_category_id").astype(np.int64),
         L("cand_emb_idx").astype(np.int64), L("hist_item_id"), L("hist_category_id").astype(np.int64),
         L("hist_emb_idx").astype(np.int64), L("hist_event_type").astype(np.int64), L("hist_hpos").astype(np.int64),
         table, L("user_emb"), out)
    (OUT / split).mkdir(parents=True, exist_ok=True)
    for i, n in enumerate(NAMES):
        np.save(OUT / split / f"{n}.npy", out[:, i])
    print(split, {n: (float(np.nanmean(out[:, i])), float((out[:, i] > 0).mean())) for i, n in enumerate(NAMES)}, flush=True)


if __name__ == "__main__":
    import json
    for s in sys.argv[1:] or ["val", "test"]:
        run(s)
