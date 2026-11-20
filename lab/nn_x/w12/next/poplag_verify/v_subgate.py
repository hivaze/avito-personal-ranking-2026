"""poplag_verify step G (LABEL-FREE): the five gate criteria G1-G5 of nn_x/w12/next/poplag/DECLARATION.txt, re-computed with my own code on
(a) the SUBSET = holdout feeds with day >= Sep 13 (the pre-declared subset read needs all five to hold for the private-like view), (b) the complement, (c) all 30,000 feeds.
Columns: it_hist_* / seq_cv_* = the study's lag_<view>_v3 files (confirmed by my own rebuild on 600 + 400 sampled feeds, v_ithist.py / v_sample.py);
xp_imp_c50 = MY full rebuild (v_gate.py). Reference = test rows of the matching part from feeds with M >= 50 (as declared); all-M reference as a sensitivity column.
  cd <repo>/nn_x/w12/next/poplag_verify && PV_THREADS=4 CUDA_VISIBLE_DEVICES="" ../../../../.venv/bin/python v_subgate.py"""
import os
os.environ.setdefault('PV_THREADS', '4')
from v_common import *
import time, json

T0 = time.time()


def lg(*a):
    print(f'[{time.time() - T0:5.0f}s]', *a, flush=True)


EDGES = np.array([0, 1, 3, 7, 10, 24, 76, np.inf]); BN = ['<1d', '1-3d', '3-7d', '7-10d', '10-24d', '24-76d', '>76d+cap']
KN = np.load(ROOT / 'data/feats/life_f4_age_knots.npy')


def bucket(f4):
    u = f4.astype(np.float64) * 64.0; k = np.clip(np.floor(u).astype(np.int64), 1, 62)
    a = np.where(f4.astype(np.float64) < 63 / 64, KN[k] + (u - k) * (KN[k + 1] - KN[k]), np.nan)
    return np.clip(np.searchsorted(EDGES, np.where(np.isnan(a), 1e9, a / 24.0), side='right') - 1, 0, 6)


ftr, hold, _ = my_holdout()
fM = ftr['M'].to_numpy().astype(np.int64); fco = ftr['cand_off'].to_numpy().astype(np.int64)
hd = (ftr['timestamp'].to_numpy().astype(np.int64)[hold] + TZ) // DAY
hM = fM[hold]; fr = np.repeat(np.arange(len(hold)), hM)
rows = np.concatenate([np.arange(o, o + m) for o, m in zip(fco[hold], hM)])
bk = bucket(np.asarray(np.load(FLAT / 'train/cand_f4.npy', mmap_mode='r')[rows]))
late_f = hd >= SEP6 + 7; late = late_f[fr]
fte = pl.read_parquet(FLAT / 'test/feeds.parquet'); tts = fte['timestamp'].to_numpy().astype(np.int64); tM = fte['M'].to_numpy().astype(np.int64)
t_fx = np.load(FLAT / 'test/cand_fidx.npy').astype(np.int64); t_bk = bucket(np.load(FLAT / 'test/cand_f4.npy'))
tpub = (tts < PUB_TS)[t_fx]; tbig = (tM >= 50)[t_fx]
lg(f'holdout feeds {len(hold)}: subset (day >= Sep 13) {int(late_f.sum())} feeds / {int(late.sum())} rows; complement {int((~late_f).sum())} feeds / {int((~late).sum())} rows')
PRIM = ['it_hist_all_luo', 'seq_cv_pop', 'xp_imp_c50']
SEC = ['it_hist_et1_luo', 'it_hist_et2_luo', 'it_hist_et3_luo', 'it_hist_et4_luo', 'seq_cv_invd20', 'seq_cv_norm']


def stats(raw, b):
    raw = np.asarray(raw, np.float32)
    return raw, b


def model_scale(name, raw):
    x = np.asarray(raw, np.float32)
    x = np.log1p(np.maximum(x, 0)) if name.startswith(('seq_cv', 'it_hist')) else np.clip(x, -10, 10)      # attach_xt transform of nn_x/w5/data.py
    return np.nan_to_num(x, nan=0.0).astype(np.float64)


def per_bucket(name, raw, b):
    x = model_scale(name, raw); z = (np.asarray(raw) == 0).astype(np.float64)
    n = np.bincount(b, minlength=7).astype(np.float64)
    return n, np.bincount(b, weights=z, minlength=7) / np.maximum(n, 1), np.bincount(b, weights=x, minlength=7) / np.maximum(n, 1), x.std()


def crit(name, hraw, hb, traw, tb):
    n, zh, mh, _ = per_bucket(name, hraw, hb); nt, zt, mt, sd = per_bucket(name, traw, tb)
    w = n / n.sum(); gz = zh - zt
    return dict(dZ=float((w * np.abs(gz)).sum()), maxb=float(np.max(np.where(w >= 0.05, np.abs(gz), 0.0))), argb=BN[int(np.argmax(np.where(w >= 0.05, np.abs(gz), 0.0)))],
                dM=float((w * np.abs(mh - mt)).sum() / sd), gap=[float(v) for v in gz], w=[float(v) for v in w], zh=[float(v) for v in zh], zt=[float(v) for v in zt])


OUT = {}
for nm in ('priv', 'pub'):
    tsel = (tpub if nm == 'pub' else ~tpub)
    ref = tsel & tbig
    for scope, hs in (('SUBSET days >= Sep 13', late), ('complement days < Sep 13', ~late), ('all 30,000 feeds', np.ones(len(rows), np.bool_))):
        print(f'\n################ view {nm}_v3 | {scope} | holdout rows {int(hs.sum())} | reference: {nm} test rows, M >= 50: {int(ref.sum())} ################', flush=True)
        print('column            | dZ lag0 | dZ lagged | largest bucket (>= 5% rows) | dM lagged | dZ / bucket / dM vs ALL-M test rows | zero share lag0 / lagged / test')
        fails = []; res = {}
        for cn in PRIM + SEC:
            l0 = np.asarray(np.load(FC / 'train' / f'{cn}.npy', mmap_mode='r')[rows])[hs]
            lagc = np.load(MYC / f'my_{nm}_v3_xp_imp_c50.npy') if cn == 'xp_imp_c50' else np.load(STUDY / f'lag_{nm}_v3_{cn}.npy')
            lagc = lagc[hs]; tr = np.load(FC / 'test' / f'{cn}.npy')
            g0 = crit(cn, l0, bk[hs], tr[ref], t_bk[ref]); g1 = crit(cn, lagc, bk[hs], tr[ref], t_bk[ref]); ga = crit(cn, lagc, bk[hs], tr[tsel], t_bk[tsel])
            f = []
            if cn in PRIM:
                if g1['dZ'] > 0.05: f.append('G1')
                if g1['maxb'] > 0.10: f.append('G2')
                if g1['dM'] > 0.20: f.append('G3')
                if cn == 'it_hist_all_luo' and g1['dZ'] > g0['dZ'] / 3: f.append('G4')
            else:
                if g1['dZ'] > 0.10 or g1['dM'] > 0.30: f.append('G5')
            fails += [f'{x}:{cn}' for x in f]; res[cn] = dict(dZ0=g0['dZ'], dZ=g1['dZ'], maxb=g1['maxb'], argb=g1['argb'], dM=g1['dM'], gap=g1['gap'], w=g1['w'], fail=f)
            print(f'{cn:17s} | {g0["dZ"]:.4f}  | {g1["dZ"]:.4f}    | {g1["maxb"]:.4f} ({g1["argb"]:8s})          | {g1["dM"]:.4f}    | {ga["dZ"]:.4f} / {ga["maxb"]:.4f} / {ga["dM"]:.4f}             | '
                  f'{np.mean(l0 == 0):.3f} / {np.mean(lagc == 0):.3f} / {np.mean(tr[ref] == 0):.3f}   {"FAIL " + ",".join(f) if f else "ok"}' + ('  [primary]' if cn in PRIM else ''))
        g4 = res['it_hist_all_luo']
        print(f'G4: it_hist_all_luo dZ lagged {g4["dZ"]:.4f} <= lag-0 dZ / 3 = {g4["dZ0"] / 3:.4f}: {g4["dZ"] <= g4["dZ0"] / 3}')
        for cn in PRIM:
            print(f'  {cn} gap(lagged - test) by age bucket [share of rows]: ' + ' | '.join(f'{BN[b]} {res[cn]["gap"][b]:+.4f} [{res[cn]["w"][b]:.3f}]' for b in range(7)))
        print(f'GATE {nm}_v3 on {scope}: {"PASS (G1-G5 all hold)" if not fails else "FAIL " + " ".join(fails)}', flush=True)
        OUT[f'{nm}|{scope}'] = dict(res=res, fails=fails)
json.dump(OUT, open(MYC / 'v_subgate.json', 'w'), indent=1)
lg('done')
