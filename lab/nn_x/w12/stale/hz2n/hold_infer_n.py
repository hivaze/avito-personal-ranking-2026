"""hz2n: CPU fp32 scoring of the 30,000 holdout feeds THROUGH one of the hz2n view directories (label-free; writes predictions only). nn_x/w12/stale/hz2/hold_infer.py
imported UNCHANGED (the reader the hz2 review validated against the trainer's GPU file); this wrapper (pattern of hold_infer2.py) only replaces its attach() so that
the encoder-score columns are read from the view too, and redirects its directories to nn_x/w12/stale/hz2n/ (cache_hold/), so nothing is written into hz2/.
Usage (as hold_infer.py):
  cd <repo>/nn_x/w12/stale/hz2n && CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=12 ../../../../.venv/bin/python hold_infer_n.py \
      <feats_n | feats_nF | feats_hn | feats_hnFS | ...> <member,member,...> <late | early | full>
  -> cache_hold/hpred_<dir>_<sub>_<member>.npy (rows of feeds outside the subset are NaN).  `python hold_infer_n.py selftest` checks attach() without scoring.
A column is read from the view iff its view file exists in <dir>/train/ (popularity names of hz2_common.HZ2_NAME + the encoder names below); otherwise the stored
column. As in hold_infer.py no keep mask is applied (keep rows exist only in feeds of Sep 6-8; use the subset "late" = days >= Sep 13 for the lagged-holdout read).
Members: names in nn_x/w5/ckpt or nn_x/abl/ckpt (ckpt_path of hold_infer.py), e.g. w5_kcd_ref_s951@last.
"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, '<repo>/nn_x/w12/stale/hz2')
import hold_infer as HI  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

ME = Path('<repo>/nn_x/w12/stale/hz2n')
ENC = {'pre2_fe': 'pre2_hz_fe', 'pre2_fe_z': 'pre2_hz_fe_z', 'pre2_fe_s': 'pre2_hz_fe_s', 'pre2_new': 'pre2_hz_new', 'pre2_new_z': 'pre2_hz_new_z', 'pre2_new_s': 'pre2_hz_new_s'}
NAMES = {**HI.HZ2_NAME, **ENC}


def attach_n(s, names, vdir):
    cols, used = [], []
    for n in names:
        base = n[4:] if n.startswith('has:') else n
        p = vdir / 'train' / f'{NAMES[base]}.npy' if base in NAMES else None
        if p is not None and p.exists():
            x = np.asarray(np.load(p, mmap_mode='r')[s.crow]); used.append(n)
        else:
            x = np.asarray(np.load(HI.FC / 'train' / f'{base}.npy', mmap_mode='r')[s.crow])
        x = (~np.isnan(x)).astype(np.float32) if n.startswith('has:') else HI.xt_transform(n, x)
        cols.append(torch.from_numpy(np.nan_to_num(x, nan=0.0)).to(torch.float16))
    s.xt = torch.stack(cols, 1)
    return used


HI._orig_attach = HI.attach   # the validated reader's attach, kept for the selftest
HI.attach = attach_n
HI.HERE = ME
HI.CACHE = ME / 'cache_hold'


def selftest():
    """attach_n on the holdout rows: each view column == fp16(transform(view file)) at the rows, each other column == the stored column; against hold_infer.attach of the base view."""
    sys.path.insert(0, '<repo>/nn_x/w12/stale/hz2n')
    import test_cd_arms6c as T
    names = T.xt_presets()['cvnd_pre2s']
    s = HI.load_mini()
    base_attach = sys.modules['hold_infer'].__dict__['_orig_attach']
    for dn, basedir in (('feats_n', Path(T.HZ2) / 'feats'), ('feats_nF', Path(T.HZ2) / 'feats'), ('feats_hnFS', None)):
        used = attach_n(s, names, ME / dn)
        xn = s.xt.clone()
        exp_used = [n for n in names if (n[4:] if n.startswith('has:') else n) in NAMES and (ME / dn / 'train' / f'{NAMES[n[4:] if n.startswith("has:") else n]}.npy').exists()]
        assert used == exp_used, (used, exp_used)
        for j, n in enumerate(names):
            base = n[4:] if n.startswith('has:') else n
            src = (ME / dn / 'train' / f'{NAMES[base]}.npy') if n in used else (HI.FC / 'train' / f'{base}.npy')
            x = np.asarray(np.load(src, mmap_mode='r')[s.crow])
            x = (~np.isnan(x)).astype(np.float32) if n.startswith('has:') else HI.xt_transform(n, x)
            e = torch.from_numpy(np.nan_to_num(x, nan=0.0)).to(torch.float16)
            assert torch.equal(xn[:, j].view(torch.int16), e.view(torch.int16)), (dn, n)
        msg = ''
        if basedir is not None:  # popularity columns identical to the validated reader on the base view; only encoder columns differ
            base_attach(s, names, basedir)
            diff = [n for j, n in enumerate(names) if not torch.equal(xn[:, j].view(torch.int16), s.xt[:, j].view(torch.int16))]
            assert set(diff) <= set(ENC), diff
            msg = f'; vs hold_infer.attach on {basedir.name}: columns that differ {diff}'
        print(f'selftest PASS [{dn}]: {s.n} holdout feeds, {s.n_cand} rows; columns read from the view: {used}{msg}', flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'selftest':
        selftest()
    else:
        HI.CACHE.mkdir(exist_ok=True)
        HI.main()
