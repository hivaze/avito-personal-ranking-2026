"""Review R1: independent re-check of the TRUNC premise (data_usage 14_rev_index_check).
Uses feeds.parquet cand_off directly (not data_usage's feed table) and p_random from eda train_prandom.npz."""
import sys; sys.path.insert(0, "<repo>/src")
from common import *
f, M, off = feed_meta("train")
n = len(M)
pos = np.load(FLAT / "train/cand_pos.npy").astype(np.int64)
y = np.load(FLAT / "train/cand_target.npy").astype(np.float32)
z = np.load("<repo>/eda/labels_position/out/train_prandom.npz")
pr = np.full(n, np.nan, np.float32); pr[z["fidx"]] = z["p_random"]
print("p_random NaN share", np.isnan(pr).mean(), " p>0.5 share", np.nanmean(pr > 0.5))
fid = np.repeat(np.arange(n), M)
assert off[0] == 0 and np.all(np.diff(off) == M[:-1])
# rank of each candidate within its feed by shown position
o = np.lexsort((pos, fid)); rank = np.empty(len(pos), np.int64); rank[o] = np.arange(len(pos)) - off[fid[o]]
rk_true = M[fid] - 1 - rank
rev_nn = np.clip(M[fid] - 1 - pos, 0, None)
maxpos = np.maximum.reduceat(pos, off)
gaps = maxpos - (M - 1)
print(f"feeds with max(pos) > M-1 (gaps): {np.mean(gaps > 0):.4f}; mean gaps {gaps.mean():.2f}; min pos==0 share {np.mean(np.minimum.reduceat(pos, off)==0):.4f}")
prf = np.nan_to_num(pr, nan=0.0)
for nm, fm in [("prod M>=50", (prf <= 0.5) & (M >= 50)), ("prod M<50", (prf <= 0.5) & (M < 50)), ("rand M>=50", (prf > 0.5) & (M >= 50)), ("rand M<50 (p>0.5)", (prf > 0.5) & (M < 50))]:
    mk = fm[fid]
    nf = fm.sum()
    a = mk & (rev_nn == 0); b = mk & (rk_true == 0)
    base = y[mk].mean()
    print(f"{nm:18s} feeds {nf:8d} base {base:.4f} | NN rev0: {a.sum()/nf:5.2f} cand/feed, rate {y[a].mean():.4f} | true last rate {y[b].mean():.4f}; rk1 {y[mk&(rk_true==1)].mean():.4f} rk2 {y[mk&(rk_true==2)].mean():.4f}; rk3-9 {y[mk&(rk_true>=3)&(rk_true<10)].mean():.4f}")
    # share of positives that are the true last item, and share of feeds whose last item is positive
    print(f"{'':18s} share of positives at true last: {y[b].sum()/y[mk].sum():.4f}; feeds with >=1 pos: {np.mean(np.add.reduceat(y, off)[fm] > 0):.4f}")
# what does last-item positive mean: is the last item the max-pos item and is it the LAST click?
ypos_last = np.zeros(n, bool)
# last positive's rank from end
lastpos_rk = np.full(n, -1)
idx = np.where(y > 0)[0]
# for each feed min rk_true among positives
mn = np.full(n, 10**9); np.minimum.at(mn, fid[idx], rk_true[idx])
has = mn < 10**9
for nm, fm in [("prod M>=50", (prf <= 0.5) & (M >= 50)), ("prod M<50", (prf <= 0.5) & (M < 50)), ("rand M>=50", (prf > 0.5) & (M >= 50))]:
    mm = fm & has
    print(f"{nm}: dist of rank-from-end of the LAST positive (0/1/2/3-9/10+): {[round(float(np.mean(mn[mm]==k)),4) for k in (0,1,2)]} {np.mean((mn[mm]>=3)&(mn[mm]<10)):.4f} {np.mean(mn[mm]>=10):.4f}")
