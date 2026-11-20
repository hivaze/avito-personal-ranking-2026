"""Label-free: how test-like are the stale-val inputs (nn_x/r6/stale_cf/both_*.npy, val rows) in the gate's own statistic? Same G-form as gate.py:
val rows of feeds with M >= 50, stored columns and stale-val columns, against private (and public) test rows of feeds with M >= 50. Information only."""
from hz2_common import *
CF = ROOT / 'nn_x/r6/stale_cf'
kn = knots()
def rows_of(off, m):
    return np.repeat(off - np.concatenate([[0], np.cumsum(m)[:-1]]), m) + np.arange(int(m.sum()))
fv = feeds('val'); Mv = fv['M'].to_numpy().astype(np.int64); sel = np.flatnonzero(Mv >= 50)
vrow = rows_of(fv['cand_off'].to_numpy().astype(np.int64)[sel], Mv[sel])
vb = age_bucket(age_hours(np.load(FLAT / 'val/cand_f4.npy')[vrow], kn))
fte = feeds('test'); tts = fte['timestamp'].to_numpy().astype(np.int64); tM = fte['M'].to_numpy().astype(np.int64)
out = [f'val feeds with M >= 50: {len(sel)}, rows {len(vrow)}; row share by age bucket ' + ' '.join(f'{x:.3f}' for x in np.bincount(vb, minlength=7) / len(vb))]
for part in ('priv', 'pub'):
    ts_ = np.flatnonzero(((tts < PUB_TS) if part == 'pub' else (tts >= PUB_TS)) & (tM >= 50))
    trow = rows_of(fte['cand_off'].to_numpy().astype(np.int64)[ts_], tM[ts_])
    tb = age_bucket(age_hours(np.load(FLAT / 'test/cand_f4.npy')[trow], kn))
    for col in SEQ + LUO:
        rx = xt_transform(col, np.load(FC / 'test' / f'{col}.npy')[trow]); sd = float(rx.std())
        for vn, path in (('val stored', FC / 'val' / f'{col}.npy'), ('stale-val', CF / f'both_{col}.npy')):
            x = np.load(path); assert len(x) == int(Mv.sum()), (col, len(x))
            x = xt_transform(col, x[vrow])
            w = np.bincount(vb, minlength=7) / len(vb)
            gz = np.array([abs(float((x[vb == k] == 0).mean()) - float((rx[tb == k] == 0).mean())) for k in range(7)])
            gm = np.array([abs(float(x[vb == k].mean()) - float(rx[tb == k].mean())) / sd for k in range(7)])
            out.append(f'{part:4s} {col:16s} {vn:10s}: dZ {float((w * gz).sum()):.4f}  largest bucket {float(np.max(np.where(w >= 0.05, gz, 0))):.4f}  dM {float((w * gm).sum()):.4f}  '
                       f'zero share {float((x == 0).mean()):.4f} (test part {float((rx == 0).mean()):.4f}) | zero share by bucket ' + ' '.join(f'{float((x[vb == k] == 0).mean()):.3f}' for k in range(7)))
print('\n'.join(out)); open(LOGS / 'staleval_parity.txt', 'w').write('\n'.join(out) + '\n')
