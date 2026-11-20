"""If a stacker is accepted: copy its val OOF / test scores to NEW preds files and write subs/cand_S2_nnstk_<tag>.parquet."""
import sys; sys.path.insert(0, '<repo>/src')
import numpy as np
from pathlib import Path
from make_sub import make_sub
O = Path('<repo>/nn_x/stack/nnstack/out'); P = Path('<repo>/preds'); S = Path('<repo>/subs')
tag = sys.argv[1]
for n in [P / f'nnstk_{tag}_val.npy', P / f'nnstk_{tag}_test.npy', S / f'cand_S2_nnstk_{tag}.parquet']:
    assert not n.exists(), f'refusing to overwrite {n}'
v = np.load(O / f'{tag}_val_oof.npy').astype(np.float32); t = np.load(O / f'{tag}_test.npy').astype(np.float32)
np.save(P / f'nnstk_{tag}_val.npy', v); np.save(P / f'nnstk_{tag}_test.npy', t)
make_sub(t, str(S / f'cand_S2_nnstk_{tag}.parquet')); print('written', S / f'cand_S2_nnstk_{tag}.parquet')
