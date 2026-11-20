"""hz2n gate for the F family (ADDENDUM 2): G1n / G2n / G3n of the NEW head (pre2_new_s, pre2_new_z). Label-free.  python gate_new.py"""
import json
import sys

from enc_common import F, FC, FLAT, HERE, HZ2, N_TRAIN, PT, PUB_TS, ROOT, WS, VIEWS, bits_eq, feed_any, lg, mm, DAY, DAY0, MSK
import numpy as np
import polars as pl
from gate import ms, nz

fte = pl.read_parquet(f'{FLAT}/test/feeds.parquet')
tts, Mte = fte['timestamp'].to_numpy().astype(np.int64), fte['M'].to_numpy().astype(np.int64)
pub_rows = np.repeat(tts < PUB_TS, Mte)
tg = {}
for nm, (s, m) in {'val': ('val', None), 'public': ('test', pub_rows), 'private': ('test', ~pub_rows)}.items():
    luo, x, z = np.load(f'{FC}/{s}/it_hist_all_luo.npy'), np.load(f'{FC}/{s}/pre2_new_s.npy'), np.clip(np.load(f'{FC}/{s}/pre2_new_z.npy'), -5, 5)
    m = np.ones(len(luo), bool) if m is None else m
    cold = m & (luo == 0)
    tg[nm] = dict(ns_all=ms(x, m), ns_cold=ms(x, cold), nz_all=ms(z, m), nz_cold=ms(z, cold))
    lg('target', nm, json.dumps(tg[nm]))
f = pl.read_parquet(f'{FLAT}/train/feeds.parquet')
M = f['M'].to_numpy().astype(np.int64); coff = f['cand_off'].to_numpy().astype(np.int64)
slot = ((f['timestamp'].to_numpy().astype(np.int64) + MSK) // DAY - DAY0).astype(np.int8)
rslot = np.repeat(slot, M)
S = np.load(f'{FC}/train/it_hist_all_luo.npy')
x0, s0, z0 = np.load(f'{FC}/train/pre2_new.npy'), np.load(f'{FC}/train/pre2_new_s.npy'), np.load(f'{FC}/train/pre2_new_z.npy')
has = np.isfinite(x0)
idn = np.zeros(N_TRAIN, bool)
for k in (1, 2, 3):
    rows = np.load(f'{PT}/win{k}/cand_rows.npy')
    a = np.zeros(len(rows), bool)
    for c in ('v3', 'v4'):
        a |= np.asarray(mm(f'{WS}/pre_w{k}r_{c}_new_win{k}.npy')) != np.asarray(mm(f'{WS}/pre_w{k}r_{c}_new_noid_win{k}.npy'))
    idn[rows] = a
pr = np.load(f'{ROOT}/eda/labels_position/out/train_prandom.npz')
p = np.nan_to_num(pr['p_random'].astype(np.float32), nan=0.0); p[M < 50] = 0.0
cand = np.where((p > 0.9) & (M >= 50))[0]
hold = np.sort(np.random.default_rng(12345).choice(cand, 30000, replace=False))
pool = (p > 0.5) & (M >= 50); pool[hold] = False
PP = np.load(f'{HZ2}/feats/train/hz2_part_pub.npy')
groups = {'all days': np.ones(N_TRAIN, bool), 'days >= Sep 13': rslot >= 7, 'review pop (trPool private-like >= Sep 13)': np.repeat(pool & ~PP & (slot >= 7), M)}
res = {'targets': tg, 'views': {}}
for vn, (base, keepf) in VIEWS.items():
    d = f'{HERE}/{vn}F/train'
    V = np.load(f'{base}/train/it_hist_hz_all.npy'); keep = np.load(keepf) if keepf else np.zeros(N_TRAIN, bool)
    cold = np.where(keep, S, V) == 0
    nc = np.load(f'{d}/hz2n_newly_cold.npy')
    x1, s1, z1 = np.load(f'{d}/pre2_hz_new.npy'), np.load(f'{d}/pre2_hz_new_s.npy'), np.load(f'{d}/pre2_hz_new_z.npy')
    still = idn & bits_eq(x1, x0)
    ch_s, ch_z = ~bits_eq(s1, s0), ~bits_eq(z1, z0)
    zb, za = np.clip(z0, -5, 5), np.clip(z1, -5, 5)
    rv = {}
    for gn, g in groups.items():
        c = cold & g; A = c & has; B = nc & has & idn & g; C = nc & has & ~idn & g; D = (S == 0) & has & g
        r = dict(rows=int(g.sum()), cold_rows=int(c.sum()), G1n_before_count=int((c & idn).sum()), G1n_before=float((c & idn).sum() / max(c.sum(), 1)),
                 G1n_after_count=int((c & still).sum()), G1n_after=float((c & still).sum() / max(c.sum(), 1)), G1n_as_built_count=int(((S == 0) & g & idn).sum()),
                 G3n_rows_changed=dict(new_s=float((ch_s & g).sum() / g.sum()), new_z=float((ch_z & g).sum() / g.sum())),
                 sets={'A': (ms(s0, A), ms(s1, A)), 'B': (ms(s0, B), ms(s1, B)), 'C': (ms(s0, C), ms(s1, C)), 'D': (ms(s0, D), ms(s1, D))},
                 sets_z={'A': (ms(zb, A), ms(za, A)), 'B': (ms(zb, B), ms(za, B))})
        smd = {}
        for pn, (bf, af) in r['sets'].items():
            for tn in ('private', 'public'):
                smd[f'new_s {pn} vs {tn}'] = (round((bf[1] - tg[tn]['ns_cold'][1]) / tg[tn]['ns_all'][2], 4), round((af[1] - tg[tn]['ns_cold'][1]) / tg[tn]['ns_all'][2], 4))
        for pn, (bf, af) in r['sets_z'].items():
            smd[f'new_z {pn} vs private'] = (round((bf[1] - tg['private']['nz_cold'][1]) / tg['private']['nz_all'][2], 4), round((af[1] - tg['private']['nz_cold'][1]) / tg['private']['nz_all'][2], 4))
        r['SMD_before_after'] = smd
        rv[gn] = r
        lg(f'{vn}F | {gn} | G1n before {r["G1n_before"]:.4f} ({r["G1n_before_count"]}) after {r["G1n_after"]:.6f} ({r["G1n_after_count"]}); as built {r["G1n_as_built_count"]} | rows changed new_s '
           f'{r["G3n_rows_changed"]["new_s"]:.4f} new_z {r["G3n_rows_changed"]["new_z"]:.4f} | SMD (before, after): ' + ' | '.join(f'{k_}: {v_}' for k_, v_ in smd.items()))
    res['views'][vn + 'F'] = rv
json.dump(res, open(f'{HERE}/gate_new.json', 'w'), indent=1)
lg('GATE NEW DONE')
