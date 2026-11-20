"""Build per-split caches: member z (ft/to), labels, folds, feed context. Verifies G3 reproduction and the numba NDCG."""
from lib import *
import time
import polars as pl
from common import feed_meta, FLAT
from metrics import ndcg_per_feed
from sklearn.model_selection import GroupKFold

T0 = time.time()
CACHE.mkdir(parents=True, exist_ok=True)
log = open(OUT / 'build_cache.log', 'w')
def pr(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); log.write(s + '\n'); log.flush()

def cold_share(off, M, cold):
    return np.add.reduceat(cold.astype(np.float64), off) / M

# ---------------- val ----------------
fv, M, off = feed_meta('val'); M = M.astype(np.int64); off = off.astype(np.int64)
y = np.load(FLAT / 'val/cand_target.npy').astype(np.float64)
u = fv['user_id'].to_numpy(); ts = fv['timestamp'].to_numpy().astype(np.int64); H = fv['H'].to_numpy().astype(np.int64)
g = np.where(u == 0, -1 - np.arange(len(M)), u)
fold = np.zeros(len(M), np.int8)
for k, (_, ho) in enumerate(GroupKFold(n_splits=5).split(np.arange(len(M)), groups=g)):
    fold[ho] = k
ref_fold = np.load(ROOT / 'nn_x/stack/nnstack/data/val_fold.npy')
pr('fold identical to nnstack val_fold.npy:', bool((ref_fold == fold).all()), np.bincount(fold))
cl = np.unique(g, return_inverse=True)[1]
luo = np.nan_to_num(np.asarray(np.load(ROOT / 'nn_x/r6/feat_cache/val/it_hist_all_luo.npy', mmap_mode='r'), np.float64))
cold = (luo == 0)
tue = (ts + 10800) // 86400 < 20720
f1 = np.load(FLAT / 'val/cand_f1.npy').astype(np.float32)
np.savez(CACHE / 'val_meta.npz', M=M, off=off, y=y, user=u, ts=ts, H=H, fold=fold, cl=cl, cold=cold.astype(np.float32),
         cshare=cold_share(off, M, cold), tue=tue, u0=(u == 0), f1=f1)
for half, pat in [('ft', 'nn_{}_n2cpv3_val_oof.npy'), ('to', 'nn_{}_trainonly_val.npy')]:
    Z = np.stack([feed_z(off, M, np.load(P / pat.format(m)).astype(np.float32)) for m in MEMBERS])
    np.save(CACHE / f'val_z_{half}.npy', Z)
    pr(f'[{time.time()-T0:.0f}s] val {half} z', Z.shape)
Zf = np.load(CACHE / 'val_z_ft.npy'); Zt = np.load(CACHE / 'val_z_to.npy')
pf = feed_z(off, M, Zf.mean(0)); pt = feed_z(off, M, Zt.mean(0)); g3 = feed_z(off, M, 0.5 * pf + 0.5 * pt)
ens = np.load(P / 'ens_G3_final_val.npy')
pr('val G3 rebuilt ndcg (metrics.py)', ndcg_per_feed(g3, y, off, M).mean(), 'ens_G3_final_val', ndcg_per_feed(ens, y, off, M).mean(),
   'max|diff| vs ens', float(np.abs(g3 - feed_z(off, M, ens)).max()))
a = ndcg_per_feed(g3.astype(np.float64), y, off, M); b = ndcg_nb(off, M, y, g3.astype(np.float64))
pr('numba ndcg vs metrics.py max|diff|', float(np.abs(a - b).max()))
# tie test
s_t = np.round(g3.astype(np.float64), 1)
pr('numba ndcg vs metrics.py with heavy ties max|diff|', float(np.abs(ndcg_per_feed(s_t, y, off, M) - ndcg_nb(off, M, y, s_t)).max()))
np.save(CACHE / 'val_g3.npy', g3); np.save(CACHE / 'val_pool_ft.npy', pf); np.save(CACHE / 'val_pool_to.npy', pt)

# ---------------- holdout ----------------
Mh, offh = np.load(HD / 'holdout_meta.npy'); Mh = Mh.astype(np.int64); offh = offh.astype(np.int64)
yh = np.load(HD / 'holdout_y.npy').astype(np.float64)
tr = pl.read_parquet(FLAT / 'train/feeds.parquet', columns=['timestamp', 'M', 'user_id', 'H', 'cand_off'])
prr = np.nan_to_num(np.load(ROOT / 'eda/labels_position/out/train_prandom.npz')['p_random'].astype(np.float32), nan=0.0)
trM = tr['M'].to_numpy(); prr[trM < 50] = 0.0
cand = np.where((prr > 0.9) & (trM >= 50))[0]
hold = np.sort(np.random.default_rng(12345).choice(cand, 30000, replace=False))
assert (trM[hold] == Mh).all()
uh = tr['user_id'].to_numpy()[hold]; tsh = tr['timestamp'].to_numpy()[hold].astype(np.int64); Hh = tr['H'].to_numpy()[hold].astype(np.int64)
assert (uh == np.load(ROOT / 'nn_x/stack/nnstack/data/hold_user.npy')).all()
gh = np.where(uh == 0, -1 - np.arange(len(Mh)), uh); clh = np.unique(gh, return_inverse=True)[1]
coldh = np.load(ROOT / 'nn_x/stack/nnstack/data/hold_cold.npy') > 0.5   # it_hist_all_luo==0 on the holdout rows (nnstack prep)
# verify on the first 200 feeds directly from the train feature cache
trluo = np.load(ROOT / 'nn_x/r6/feat_cache/train/it_hist_all_luo.npy', mmap_mode='r'); troff = tr['cand_off'].to_numpy()
chk = np.concatenate([np.asarray(trluo[troff[h]:troff[h] + Mh[i]]) for i, h in enumerate(hold[:200])])
pr('holdout cold flag check (200 feeds):', bool(((np.nan_to_num(chk) == 0) == coldh[:len(chk)]).all()))
dow = (((tsh + 10800) // 86400) + 3) % 7
# holdout f1 for candidate features
trf1 = np.load(FLAT / 'train/cand_f1.npy', mmap_mode='r')
f1h = np.concatenate([np.asarray(trf1[troff[h]:troff[h] + Mh[i]]) for i, h in enumerate(hold)]).astype(np.float32)
np.savez(CACHE / 'hold_meta.npz', M=Mh, off=offh, y=yh, user=uh, ts=tsh, H=Hh, cl=clh, cold=coldh.astype(np.float32),
         cshare=cold_share(offh, Mh, coldh), dow=dow, u0=(uh == 0), f1=f1h, late=(tsh >= np.quantile(tsh, 0.8)))
Zh = np.stack([feed_z(offh, Mh, np.load(HD / f'nn_{m}_holdout.npy').astype(np.float32)) for m in MEMBERS])
np.save(CACHE / 'hold_z_to.npy', Zh)
ph = feed_z(offh, Mh, Zh.mean(0)); np.save(CACHE / 'hold_pool_to.npy', ph)
pr(f'[{time.time()-T0:.0f}s] holdout TO pool ndcg', ndcg_nb(offh, Mh, yh, ph.astype(np.float64)).mean(),
   'nnstack hold_pool_G3_to', ndcg_nb(offh, Mh, yh, np.load(ROOT / 'nn_x/stack/nnstack/data/hold_pool_G3_to.npy').astype(np.float64)).mean())

# ---------------- test ----------------
ft_, Mt, offt = feed_meta('test'); Mt = Mt.astype(np.int64); offt = offt.astype(np.int64)
ut = ft_['user_id'].to_numpy(); tst = ft_['timestamp'].to_numpy().astype(np.int64)
lt = np.nan_to_num(np.asarray(np.load(ROOT / 'nn_x/r6/feat_cache/test/it_hist_all_luo.npy', mmap_mode='r'), np.float64)); coldt = lt == 0
np.savez(CACHE / 'test_meta.npz', M=Mt, off=offt, y=np.zeros(Mt.sum()), user=ut, ts=tst, H=ft_['H'].to_numpy().astype(np.int64),
         cl=np.arange(len(Mt)), cold=coldt.astype(np.float32), cshare=cold_share(offt, Mt, coldt), u0=(ut == 0),
         public=(tst < 1790497920), f1=np.load(FLAT / 'test/cand_f1.npy').astype(np.float32))
for half, pat in [('ft', 'nn_{}_n2cpv3_test.npy'), ('to', 'nn_{}_trainonly_test.npy')]:
    Z = np.stack([feed_z(offt, Mt, np.load(P / pat.format(m)).astype(np.float32)) for m in MEMBERS])
    np.save(CACHE / f'test_z_{half}.npy', Z)
Zf = np.load(CACHE / 'test_z_ft.npy'); Zt = np.load(CACHE / 'test_z_to.npy')
g3t = feed_z(offt, Mt, 0.5 * feed_z(offt, Mt, Zf.mean(0)) + 0.5 * feed_z(offt, Mt, Zt.mean(0)))
enst = np.load(P / 'ens_G3_final_test.npy')
pr('test G3 rebuilt vs ens_G3_final_test max|z diff|', float(np.abs(g3t - feed_z(offt, Mt, enst)).max()))
np.save(CACHE / 'test_g3.npy', g3t)
pr(f'[{time.time()-T0:.0f}s] done')
