"""Clogit LR of "item seen again (other feeds/users, val+test stream) within H hours after the feed" vs H, on val."""
import sys
sys.path.insert(0, "<repo>/feats_v2/life_")
sys.path.insert(0, "<repo>/src")
from common import *
import ss_sparse_stream as S
import clogit_screen as C
import numba as nb

@nb.njit(parallel=True)
def next_dt(q_item, q_t, q_fuid, q_user, off, et, efu, eu, ek, kmask, out):
    for r in nb.prange(len(q_item)):
        it = q_item[r]; best = -1
        for e in range(off[it], off[it+1]):
            if efu[e] == q_fuid[r] or (q_user[r] != 0 and eu[e] == q_user[r]) or not kmask[ek[e]]:
                continue
            dt = et[e] - q_t[r]
            if dt > 0:
                best = dt; break
        out[r] = best

ev = S.merge_streams([(1, S.stream_events("val")), (2, S.stream_events("test"))])
fq = S.load_feeds("val"); cfi = np.load(FLAT/"val/cand_fidx.npy"); qi = np.load(FLAT/"val/cand_item_id.npy")
qt = fq["timestamp"].to_numpy().astype(np.int64)[cfi]; qf = cfi.astype(np.int64) + (1 << 28); qu = fq["user_id"].to_numpy().astype(np.int64)[cfi]
f, M, off = feed_meta("val"); y = np.load(FLAT/"val/cand_target.npy").astype(np.float64)
oof = np.load(C.BASE_OOF)
npos = np.bincount(cfi, weights=y, minlength=len(M)); rows = (npos > 0)[cfi]
y2 = y[rows]; fid2 = cfi[rows]; starts = np.r_[0, np.flatnonzero(np.diff(fid2)) + 1]; posf = npos[npos > 0]
s = (oof[rows] - oof[rows].mean()) / oof[rows].std()
ll0, _ = C.fit(s[:, None], y2, starts, fid2, posf)
for kname, km in [("any", [1,1,0,0]), ("cand", [1,0,0,0]), ("hist", [0,1,0,0]), ("et1", [0,0,0,1])]:
    o = np.empty(len(qi), np.int64)
    next_dt(qi, qt, qf, qu, ev["off"], ev["t"], ev["fuid"], ev["user"], ev["kind"], np.array(km, np.bool_), o)
    line = f"{kname:5s}"
    for H in [6, 12, 24, 48, 72, 96, 144, 1000]:
        ind = ((o > 0) & (o <= H * 3600)).astype(np.float64)
        ll1, w = C.fit(np.stack([s, ind[rows]], 1), y2, starts, fid2, posf)
        line += f" | H={H}: share={ind.mean():.3f} LR={2*(ll1-ll0):5.1f} b={w[1]:+.2f}"
    print(line, flush=True)
