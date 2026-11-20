"""hz2n ADDENDUM 3 (view c only): feats_hnFS = feats_hnF + seq_cv_hz_{invd20,pop,norm}.npy = where(view it_hist_all == 0, 0, stored).  python build_seq0.py"""
import json
import os

from enc_common import FC, HERE, N_TRAIN, ROOT, bits_eq, lg
import numpy as np

V = np.load(f'{ROOT}/data/feats_new/train/it_hist_hz_all.npy')
cold = V == 0
d = f'{HERE}/feats_hnFS/train'
os.makedirs(d, exist_ok=True)
res = {'view_cold_rows': int(cold.sum())}
for n, hn in (('seq_cv_invd20', 'seq_cv_hz_invd20'), ('seq_cv_pop', 'seq_cv_hz_pop'), ('seq_cv_norm', 'seq_cv_hz_norm')):
    st = np.load(f'{FC}/train/{n}.npy')
    assert st.dtype == np.float32 and st.shape == (N_TRAIN,)
    x = np.where(cold, np.float32(0), st).astype(np.float32)
    tmp = f'{d}/.{hn}.tmp.npy'
    np.save(tmp, x); os.replace(tmp, f'{d}/{hn}.npy')
    back = np.load(f'{d}/{hn}.npy')
    res[n] = {'rows_changed': int((~bits_eq(back, st)).sum()), 'S1_rows_changed_with_view_counter_pos': int((~bits_eq(back, st) & ~cold).sum()),
              'S2_rows_pos_given_view_cold': int(((back > 0) & cold).sum()), 'stored_rows_pos_given_view_cold': int(((st > 0) & cold).sum()),
              'stored_nan': int(np.isnan(st).sum()), 'stored_neg': int((st < 0).sum())}
    lg(n, json.dumps(res[n]))
for n in sorted(os.listdir(f'{HERE}/feats_hnF/train')):
    if not os.path.lexists(f'{d}/{n}'):
        os.symlink(os.path.realpath(f'{HERE}/feats_hnF/train/{n}'), f'{d}/{n}')
os.makedirs(f'{HERE}/feats_hnFS/aux', exist_ok=True)
if not os.path.lexists(f'{HERE}/feats_hnFS/aux/pre2_hz_fe_has.npy'):
    os.symlink(f'{HERE}/feats_hn/aux/pre2_hz_fe_has.npy', f'{HERE}/feats_hnFS/aux/pre2_hz_fe_has.npy')
res['pass'] = bool(all(res[n]['S1_rows_changed_with_view_counter_pos'] == 0 and res[n]['S2_rows_pos_given_view_cold'] == 0 for n in ('seq_cv_invd20', 'seq_cv_pop', 'seq_cv_norm')))
json.dump(res, open(f'{HERE}/build_seq0.json', 'w'), indent=1)
lg('S1 / S2', 'PASS' if res['pass'] else 'FAIL', sorted(os.listdir(d)))
