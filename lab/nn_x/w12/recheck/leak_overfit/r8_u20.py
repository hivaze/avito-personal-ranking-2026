"""R8: U20 witness composition (kd_audit#2) and seen-user shares in val / public / private."""
from lo_common import *
ftr = pl.read_parquet(FLAT/'train/feeds.parquet', columns=['user_id', 'timestamp', 'M'])
uid = ftr['user_id'].to_numpy().astype(np.int64); n = len(uid)
xf = np.where(uid == 0, np.arange(n) % 5, (uid*2654435761 % 4294967296) % 5)
hidx = np.load(R/'nn_x/w12/recheck/kd_audit/hold_idx.npy')
print(' holdout feeds by user fold:', np.bincount(xf[hidx], minlength=5), '| user-0 feeds in holdout', int((uid[hidx] == 0).sum()))
tru = np.unique(uid[uid != 0])
uv = fv['user_id'].to_numpy().astype(np.int64); seen_v = np.isin(uv, tru) & (uv != 0)
xv = (uv*2654435761 % 4294967296) % 5
print(f' val feeds {len(uv)}: user in train {int(seen_v.sum())} ({seen_v.mean():.3f}), of them fold 0: {int((seen_v & (xv == 0)).sum())}; user-0 feeds {int((uv == 0).sum())}')
ut = ft['user_id'].to_numpy().astype(np.int64); ts = ft['timestamp'].to_numpy().astype(np.int64); pub = ts < 1790497920
seen_t = np.isin(ut, tru) & (ut != 0)
vu = np.unique(uv[uv != 0]); seen_tv = np.isin(ut, vu) & (ut != 0)
print(f' test public {int(pub.sum())}: user in train {seen_t[pub].mean():.3f}, user in val {seen_tv[pub].mean():.4f}, user-0 {np.mean(ut[pub]==0):.3f} | private {int((~pub).sum())}: user in train {seen_t[~pub].mean():.3f}, user in val {seen_tv[~pub].mean():.4f}, user-0 {np.mean(ut[~pub]==0):.3f}')
print(f' M<50 share: val {np.mean(Mv<50):.3f} public {np.mean(Mt[pub]<50):.3f} private {np.mean(Mt[~pub]<50):.3f}; mean M {Mv.mean():.1f} / {Mt[pub].mean():.1f} / {Mt[~pub].mean():.1f}')
day = (ts+10800)//86400
print(' test days (MSK):', {int(d): (int((day[pub] == d).sum()), int((day[~pub] == d).sum())) for d in np.unique(day)}, '(public, private)')
