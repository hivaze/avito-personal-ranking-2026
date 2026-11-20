"""P2 build: cross-user activity on the exact candidate item inside the scored period (val + test feeds themselves),
split by time relative to the query feed, plus the dynamics of f5 (the item's engagement rate) across feeds.

For a query row (feed q at time t, user u, item i) the sources are the other 43,017 val + test feeds g with
user(g) != u when u != 0 (user-0 feeds: only q itself is excluded). Equal timestamps are skipped.
Columns (per query row, float32), E = source feed earlier than t (AS-OF, legal), L = later than t (future):
  0 HE_all   history events on i in earlier sources        8 HL_all   ... later sources
  1 HE_fc    of which type 1 & source 1 (opened from feed) 9 HL_fc
  2 HE_rec   of which hpos < 16                            10 HL_rec
  3 HE_ct    of which types 2, 3, 4 (contact / favourite)  11 HL_ct
  4 CE       earlier source feeds showing i as candidate   12 CL
  5 f5_prev  f5 of i at the latest earlier appearance (val/test other users, else last train appearance), NaN if none
  6 dt_prev  hours since that appearance                   13 f5_next  f5 at the earliest later appearance (other users)
  7 n_tr     train candidate rows of i (exposure count)    14 dt_next  hours to it
  15 HL4_all later history events within 96 h (public-like horizon); 16 HL4_fc
Writes out/p2_val.npy, out/p2_test.npy and the column names.
"""
import sys, json
sys.path.insert(0, "<repo>/lab/nn_x/w12/leader/data_channels")
from dc_lib import *  # noqa
import numba as nb

NAMES = ["HE_all", "HE_fc", "HE_rec", "HE_ct", "CE", "f5_prev", "dt_prev", "n_tr",
         "HL_all", "HL_fc", "HL_rec", "HL_ct", "CL", "f5_next", "dt_next", "HL4_all", "HL4_fc"]

v = Split("val"); t = Split("test")
nv, nt = v.n, t.n
ts = np.concatenate([v.ts, t.ts]); uid = np.concatenate([v.user, t.user])
# history events
hg = np.concatenate([v.hfid, t.hfid + nv]).astype(np.int64)
hi = np.concatenate([v.hitem, t.hitem]).astype(np.int64)
hty = np.concatenate([v.htype, t.htype]).astype(np.int8)
hsr = np.concatenate([v.hsrc, t.hsrc]).astype(np.int8)
hps = np.concatenate([v.hpos, t.hpos]).astype(np.int16)
# candidate rows (exposures and queries)
cg = np.concatenate([v.fid, t.fid + nv]).astype(np.int64)
ci = np.concatenate([v.item, t.item]).astype(np.int64)
cf5 = np.concatenate([np.load(f"{FLAT}/val/cand_f5.npy"), np.load(f"{FLAT}/test/cand_f5.npy")]).astype(np.float32)
log("events", len(hg), "cand rows", len(cg))

# dense item ids over the union
items, inv = np.unique(np.concatenate([ci, hi]), return_inverse=True)
cid = inv[:len(ci)]; hid = inv[len(ci):]
nI = len(items)
log("items", nI)

# last train appearance of each item (train feeds are time-sorted, so the last row is the latest)
tr_item = np.load(f"{FLAT}/train/cand_item_id.npy")
tr_fidx = np.load(f"{FLAT}/train/cand_fidx.npy")
tr_f5 = np.load(f"{FLAT}/train/cand_f5.npy")
tr_ts = pl.read_parquet(f"{FLAT}/train/feeds.parquet")["timestamp"].to_numpy().astype(np.int64)
pos = np.searchsorted(items, tr_item)
pos[pos >= nI] = 0
hit = items[pos] == tr_item
log("train rows", len(tr_item), "rows whose item is in val/test", hit.sum())
rows = np.flatnonzero(hit)
dense = pos[rows]
n_tr = np.bincount(dense, minlength=nI).astype(np.float32)
last_row = np.full(nI, -1, np.int64)
last_row[dense] = rows                  # later rows overwrite earlier ones (numpy assignment keeps the last)
has_tr = last_row >= 0
tr_last_f5 = np.full(nI, np.nan, np.float32); tr_last_ts = np.zeros(nI, np.int64)
tr_last_f5[has_tr] = tr_f5[last_row[has_tr]]
tr_last_ts[has_tr] = tr_ts[tr_fidx[last_row[has_tr]]]
del tr_item, tr_fidx, tr_f5, pos, hit, rows, dense
log("train last-appearance table done; items with a train appearance", has_tr.sum())

# CSR of history events by item, sorted by source time
oh = np.lexsort((ts[hg], hid)); hp_ = np.searchsorted(hid[oh], np.arange(nI + 1))
H_g = hg[oh]; H_t = hty[oh]; H_s = hsr[oh]; H_p = hps[oh]
oc = np.lexsort((ts[cg], cid)); cp_ = np.searchsorted(cid[oc], np.arange(nI + 1))
C_g = cg[oc]; C_f5 = cf5[oc]
log("CSR built")


@nb.njit(parallel=True, cache=True)
def kernel(qg, qi, ts, uid, hp_, H_g, H_t, H_s, H_p, cp_, C_g, C_f5, tr_last_f5, tr_last_ts, n_tr, out):
    for r in nb.prange(len(qg)):
        g0 = qg[r]; it = qi[r]; t0 = ts[g0]; u0 = uid[g0]
        for z in range(17):
            out[r, z] = 0.0
        out[r, 5] = np.nan; out[r, 6] = np.nan; out[r, 13] = np.nan; out[r, 14] = np.nan
        out[r, 7] = n_tr[it]
        for e in range(hp_[it], hp_[it + 1]):
            g = H_g[e]
            if g == g0:
                continue
            if u0 != 0 and uid[g] == u0:
                continue
            dt = ts[g] - t0
            if dt == 0:
                continue
            fc = 1.0 if (H_t[e] == 1 and H_s[e] == 1) else 0.0
            rc = 1.0 if H_p[e] < 16 else 0.0
            ct = 1.0 if (H_t[e] >= 2 and H_t[e] <= 4) else 0.0
            if dt < 0:
                out[r, 0] += 1.0; out[r, 1] += fc; out[r, 2] += rc; out[r, 3] += ct
            else:
                out[r, 8] += 1.0; out[r, 9] += fc; out[r, 10] += rc; out[r, 11] += ct
                if dt <= 96 * 3600:
                    out[r, 15] += 1.0; out[r, 16] += fc
        best_prev = -1; best_next = -1
        for e in range(cp_[it], cp_[it + 1]):
            g = C_g[e]
            if g == g0:
                continue
            if u0 != 0 and uid[g] == u0:
                continue
            dt = ts[g] - t0
            if dt == 0:
                continue
            if dt < 0:
                out[r, 4] += 1.0
                best_prev = e                     # sorted by time: the last earlier one wins
            else:
                out[r, 12] += 1.0
                if best_next < 0:
                    best_next = e
        if best_prev >= 0:
            out[r, 5] = C_f5[best_prev]; out[r, 6] = (t0 - ts[C_g[best_prev]]) / 3600.0
        elif not np.isnan(tr_last_f5[it]):
            out[r, 5] = tr_last_f5[it]; out[r, 6] = (t0 - tr_last_ts[it]) / 3600.0
        if best_next >= 0:
            out[r, 13] = C_f5[best_next]; out[r, 14] = (ts[C_g[best_next]] - t0) / 3600.0


for name, lo, hi_ in (("val", 0, v.N), ("test", v.N, v.N + t.N)):
    out = np.zeros((hi_ - lo, 17), np.float32)
    kernel(cg[lo:hi_], cid[lo:hi_], ts, uid, hp_, H_g, H_t, H_s, H_p, cp_, C_g, C_f5, tr_last_f5, tr_last_ts, n_tr, out)
    np.save(f"{OUT}/out/p2_{name}.npy", out)
    log(name, out.shape, "nonzero shares:", {NAMES[j]: round(float((np.nan_to_num(out[:, j]) != 0).mean()), 4) for j in range(17)})
json.dump(NAMES, open(f"{OUT}/out/p2_names.json", "w"))
log("done")
