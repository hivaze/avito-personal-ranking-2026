"""poplag: label-free diagnostics around the v3 gate (information only, no decision): (a) xp_imp_c50 and it_hist_all_luo zero share by age bucket for
holdout-day groups (which rows carry the residual), (b) share of left-censored partial-exposure rows by day group, (c) per-feed cold-share distribution."""
import sys
from pl_common import *
kn = knots(); mini = CACHE / 'mini'
crow = np.load(mini / 'crow.npy'); f4h = np.load(mini / 'c_f.npy')[:, 3]; Mh = np.load(mini / 'M.npy'); ts = np.load(mini / 'ts.npy')
fr = np.repeat(np.arange(len(Mh)), Mh); hday = day_of(ts); age_h = age_hours(f4h, kn); bkh = age_bucket(age_h)
fte = feeds('test'); Mt = fte['M'].to_numpy().astype(np.int64); tts = fte['timestamp'].to_numpy().astype(np.int64)
fi = np.load(FLAT / 'test/cand_fidx.npy'); bkt = age_bucket(age_hours(np.load(FLAT / 'test/cand_f4.npy'), kn))
part = {'pub': (tts < PUB_TS)[fi] & (Mt[fi] >= 50), 'priv': (tts >= PUB_TS)[fi] & (Mt[fi] >= 50)}
DAY0 = 20702
groups = {'Sep 6-8': (hday <= 20704), 'Sep 9-12': (hday >= 20705) & (hday <= 20708), 'Sep 13-14': (hday >= 20709) & (hday <= 20710), 'Sep 17-19': hday >= 20713, 'days >= Sep 13': hday >= 20709}
for view in ('priv', 'pub'):
    D = np.load(CACHE / f'view_{view}_D.npy'); L = np.load(CACHE / f'view_{view}_L.npy')
    cd = (hday - L)[fr]; tpost_day = (ts[fr] - 3600.0 * np.nan_to_num(age_h, nan=1e7) + MSK) / DAY
    lc = (tpost_day < DAY0) & (D > np.maximum(cd, DAY0)) & (D < 20716)      # created before Sep 6, partial exposure: shifted window
    print(f'\n##### view {view}_v3 (label-free; information only) #####')
    print(f' row classes: young / no dense exposure (D <= Sep 6 or D = c_f with listing after c_f) and exact windows {float((~lc & (D < 20716)).mean()):.3f}; '
          f'left-censored shifted windows {float(lc.mean()):.3f}; full window (E >= 14 d) {float((D == 20716).mean()):.3f}')
    for nm in ('xp_imp_c50', 'it_hist_all_luo', 'seq_cv_pop'):
        x = np.load(CACHE / f'lag_{view}_v3_{nm}.npy'); t = np.load(FC / 'test' / f'{nm}.npy')[part[view]]; bt = bkt[part[view]]
        zt = np.array([np.mean(t[bt == b] == 0) for b in range(7)])
        print(f' {nm}: zero share by age bucket, lagged holdout by day group minus test {view} (M >= 50); test: ' + ' '.join(f'{AGE_NAMES[b]} {zt[b]:.3f}' for b in range(7)))
        for g, m in groups.items():
            mr = m[fr]; w = np.bincount(bkh[mr], minlength=7) / mr.sum()
            zl = np.array([np.mean(x[mr & (bkh == b)] == 0) for b in range(7)])
            dZ = float((w * np.abs(zl - zt)).sum()); mb = float(np.max(np.where(w >= 0.05, np.abs(zl - zt), 0)))
            print(f'   {g:15s} feeds {int(m.sum()):6d} left-censored rows {float(lc[mr].mean()):.3f} | ' + ' '.join(f'{zl[b] - zt[b]:+.3f}' for b in range(7)) + f' | dZ {dZ:.4f} max bucket {mb:.3f}')
    # per-feed cold share (it_hist_all_luo == 0), feed level
    x0 = np.asarray(np.load(FC / 'train/it_hist_all_luo.npy', mmap_mode='r')[crow]); xl = np.load(CACHE / f'lag_{view}_v3_it_hist_all_luo.npy')
    cs0 = np.bincount(fr, weights=(x0 == 0)) / Mh; csl = np.bincount(fr, weights=(xl == 0)) / Mh
    tt = np.load(FC / 'test/it_hist_all_luo.npy'); ok = ((tts < PUB_TS) if view == 'pub' else (tts >= PUB_TS)) & (Mt >= 50)
    cst = (np.bincount(fi, weights=(tt == 0), minlength=len(Mt)) / Mt)[ok]
    q = lambda a: ' '.join(f'{v:.3f}' for v in np.quantile(a, [0.1, 0.25, 0.5, 0.75, 0.9]))
    print(f' per-feed cold share (it_hist_all_luo = 0), q10 q25 q50 q75 q90 | share of feeds >= 0.45:\n   holdout lag 0 {q(cs0)} | {float((cs0 >= 0.45).mean()):.3f}\n   holdout {view}_v3 {q(csl)} | {float((csl >= 0.45).mean()):.3f}\n   test {view} (M >= 50) {q(cst)} | {float((cst >= 0.45).mean()):.3f}')
