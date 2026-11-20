"""Proof that cd_arms6b is a drop-in for cd_arms6 (CPU, MiniSplit of real train feeds; read-only imports, no bytecode written).
  T1 text: every function / class of cd_arms6 that 6b does not mark "# 6b" is AST-identical; the changed set is exactly {luo_cols, setup_hzcd} + new apply_keep.
  T2 HZCD_DIR unset (the existing five files): on 200 random train feeds, same columns, same hz tensor, and HzTrainView.gather returns bit-identical batches
     (every key) for both modules; the five LUO columns differ from the plain gather on some cells, every other column is the plain gather.
  T3 HZCD_DIR = hz2 view: 8 columns swapped for the W8 preset (3 seq_cv + 5 it_hist); each equals fp16(attach_xt transform of the hz2 file) gathered per cell;
     every other xt column and every other batch key equals the plain gather. pop-light xt list (no it_hist, xp_imp_c50): 1 column swapped.
  T4 HZ2_KEEP: rows flagged True read the stored values in all swapped columns; other rows read hz2.
Usage: CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 python test_cd_arms6b.py   (run AFTER the hz2 files exist; T3 / T4 are skipped otherwise)
"""
import ast
import importlib.util
import os
import sys

sys.dont_write_bytecode = True
assert os.environ.get('CUDA_VISIBLE_DEVICES', 'x') == '', 'run with CUDA_VISIBLE_DEVICES=""'
import numpy as np
import polars as pl
import torch

torch.set_num_threads(12)
ROOT = '<repo>'
HERE = os.path.dirname(os.path.abspath(__file__))
W5 = f'{ROOT}/nn_x/w5'
sys.path.insert(0, f'{W5}/tests')
sys.path.insert(1, W5)
import warnings
warnings.filterwarnings('ignore')
from mini_split import MiniSplit  # noqa: E402

VAR_DIR = os.environ.get('HZ2_TEST_DIR', f'{HERE}/feats')
out = []


def P(*a):
    s = ' '.join(str(x) for x in a)
    out.append(s)
    print(s, flush=True)


def load_mod(path, name, env):
    old = {k: os.environ.get(k) for k in ('HZCD_DIR', 'HZ2_KEEP')}
    for k in old:
        os.environ.pop(k, None)
    os.environ.update(env)
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    for k, v in old.items():
        os.environ.pop(k, None)
        if v is not None:
            os.environ[k] = v
    return m


def xt_presets():
    """XT_PRESETS / XT_REL of nn_x/w5/train.py without importing it (the block is plain literals)."""
    src = open(f'{W5}/train.py').read()
    a = src.index('XT_PRESETS = {"cv"')
    b = src.index('\n\n\ndef parse():')
    ns = {}
    exec(src[a:b], ns)
    return ns['XT_PRESETS']


def bits_equal(a, b):
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    if a.is_floating_point():
        return bool(torch.equal(a.contiguous().view(-1).view(torch.int32 if a.dtype == torch.float32 else torch.int16),
                                b.contiguous().view(-1).view(torch.int32 if b.dtype == torch.float32 else torch.int16)))
    return bool(torch.equal(a, b))


def batches_equal(b1, b2, skip_xt_cols=None):
    assert set(b1) == set(b2), (set(b1) ^ set(b2))
    bad = []
    for k in b1:
        x, y = b1[k], b2[k]
        if k == 'xt' and skip_xt_cols is not None:
            keep = [c for c in range(x.shape[-1]) if c not in skip_xt_cols]
            x, y = x[..., keep], y[..., keep]
        if not bits_equal(x, y):
            bad.append(k)
    return bad


def main():
    # ---------------- T1
    s6, s6b = open(f'{W5}/cd_arms6.py').read(), open(f'{HERE}/cd_arms6b.py').read()
    t6, t6b = ast.parse(s6), ast.parse(s6b)
    d6 = {n.name: ast.dump(n) for n in t6.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    d6b = {n.name: ast.dump(n) for n in t6b.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    changed = sorted(k for k in d6 if d6[k] != d6b.get(k))
    new = sorted(k for k in d6b if k not in d6)
    P(f'T1 defs of cd_arms6: {sorted(d6)}; changed in 6b: {changed}; new in 6b: {new}')
    assert changed == ['luo_cols', 'setup_hzcd'] and new == ['apply_keep'] and set(d6) <= set(d6b)
    P('T1 PASS: HzTrainView, xt_transform, load_hz, hz_path, hz_stats, save_holdout_hz are AST-identical')
    # ---------------- feeds
    f = pl.read_parquet(f'{ROOT}/data/flat/train/feeds.parquet')
    nF = len(f)
    feeds = np.sort(np.random.default_rng(20261115).choice(nF, 200, replace=False))
    M = f['M'].to_numpy().astype(np.int64)[feeds]; coff = f['cand_off'].to_numpy().astype(np.int64)[feeds]
    crow = np.concatenate([np.arange(o, o + m) for o, m in zip(coff, M)])
    PRE = xt_presets()
    xt_names = PRE['cvnd_pre2s']
    ms = MiniSplit('train', feeds, xt_names=xt_names)
    P(f'MiniSplit: {len(feeds)} random train feeds, {len(crow)} candidate rows, {len(xt_names)} xt columns (preset cvnd_pre2s)')
    fb = [np.arange(s, min(200, s + 25)) for s in range(0, 200, 25)]
    C6 = load_mod(f'{W5}/cd_arms6.py', 'cd_arms6_ref', {})
    # ---------------- T2
    B = load_mod(f'{HERE}/cd_arms6b.py', 'cd_arms6b_default', {})
    assert B.HZ_DIR == C6.HZ_DIR and B.KEEP_PATH == ''
    c6, cb = C6.luo_cols(xt_names), B.luo_cols(xt_names)
    assert c6 == cb and [c[0] for c in c6] == [3, 4, 5, 6, 7], (c6, cb)
    h6 = C6.load_hz('train', c6, ms.device, rows=crow); hb = B.load_hz('train', cb, ms.device, rows=crow)
    assert bits_equal(h6, hb)
    v6, vb = C6.HzTrainView(ms, c6, h6), B.HzTrainView(ms, cb, hb)
    n_diff_cells = 0
    for ids in fb:
        b6, bb, b0 = v6.gather(ids, with_pos=True), vb.gather(ids, with_pos=True), ms.gather(ids, with_pos=True)
        bad = batches_equal(b6, bb)
        assert not bad, f'T2: keys differ between cd_arms6 and cd_arms6b: {bad}'
        bad0 = batches_equal(bb, b0, skip_xt_cols=[3, 4, 5, 6, 7])
        assert not bad0, f'T2: non-LUO content differs from the plain gather: {bad0}'
        n_diff_cells += int((bb['xt'][..., 3:8] != b0['xt'][..., 3:8]).sum())
    st6, stb = C6.hz_stats(ms, v6), B.hz_stats(ms, vb)
    assert st6 == stb
    P(f'T2 PASS: HZCD_DIR unset -> columns {[c[1] for c in cb]}; hz tensors bit-identical; {len(fb)} batches x {len(b6)} keys bit-identical between cd_arms6 and '
      f'cd_arms6b; LUO cells that differ from the plain gather: {n_diff_cells}; hz_stats identical')
    if not os.path.exists(f'{VAR_DIR}/train/seq_cv_hz_norm.npy'):
        P(f'T3 / T4 SKIPPED: {VAR_DIR}/train has no hz2 files yet')
        open(f'{HERE}/logs/test_cd_arms6b.log', 'w').write('\n'.join(out) + '\n')
        return
    # ---------------- T3
    B2 = load_mod(f'{HERE}/cd_arms6b.py', 'cd_arms6b_hz2', {'HZCD_DIR': VAR_DIR})
    c2 = B2.luo_cols(xt_names)
    assert [c[0] for c in c2] == [0, 1, 2, 3, 4, 5, 6, 7] and [c[1] for c in c2] == xt_names[:8], c2
    h2 = B2.load_hz('train', c2, ms.device, rows=crow)
    v2 = B2.HzTrainView(ms, c2, h2)
    brute = torch.stack([torch.from_numpy(B2.xt_transform(hn, np.asarray(np.load(f'{VAR_DIR}/train/{hn}.npy', mmap_mode='r')[crow]))).to(torch.float16)
                         for _, _, hn in c2], 1)
    assert bits_equal(brute, h2)
    changed_cells = 0
    for ids in fb:
        b2, b0 = v2.gather(ids, with_pos=True), ms.gather(ids, with_pos=True)
        bad0 = batches_equal(b2, b0, skip_xt_cols=list(range(8)))
        assert not bad0, f'T3: content outside the 8 swapped columns differs from the plain gather: {bad0}'
        want = brute[b2['ci']].float()
        assert bits_equal(b2['xt'][..., :8].contiguous(), want.contiguous())
        changed_cells += int((b2['xt'][..., :8] != b0['xt'][..., :8]).sum())
    st = B2.hz_stats(ms, v2)
    P(f'T3 PASS: HZCD_DIR = {VAR_DIR}: 8 columns swapped {[c[1] for c in c2]}; every swapped cell == fp16(attach_xt transform(hz2 file)); all other keys / columns == plain '
      f'gather; cells changed vs stored {changed_cells}')
    for s_ in st:
        P(f'     col {s_["col"]} {s_["luo"]:16s} -> {s_["hz"]:17s} zero rate {s_["zero_luo"]:.3f} -> {s_["zero_hz"]:.3f}, rows changed {s_["changed"]:.4f}')
    # pop-light xt list: the 8 popularity columns removed, xp_imp_c50 appended (nn_x/abl pop1: --xt_exclude <8 pop> --xt_add xp_imp_c50)
    pop_names = [n for n in xt_names if n not in xt_names[:8]] + ['xp_imp_c50']
    cp = B2.luo_cols(pop_names)
    assert len(cp) == 1 and cp[0][1] == 'xp_imp_c50' and cp[0][0] == len(pop_names) - 1, cp
    msp = MiniSplit('train', feeds[:40], xt_names=pop_names)
    crow40 = np.concatenate([np.arange(o, o + m) for o, m in zip(coff[:40], M[:40])])
    hp = B2.load_hz('train', cp, msp.device, rows=crow40)
    vp = B2.HzTrainView(msp, cp, hp)
    bp, b0 = vp.gather(np.arange(40), with_pos=True), msp.gather(np.arange(40), with_pos=True)
    assert not batches_equal(bp, b0, skip_xt_cols=[cp[0][0]])
    wantp = torch.from_numpy(B2.xt_transform('xp_imp_hz_c50', np.asarray(np.load(f'{VAR_DIR}/train/xp_imp_hz_c50.npy', mmap_mode='r')[crow40]))).to(torch.float16)
    assert bits_equal(bp['xt'][..., cp[0][0]].contiguous(), wantp[bp['ci']].float().contiguous())
    try:
        C6.luo_cols(pop_names)
        raise SystemExit('cd_arms6 accepted a pop-light xt list?')
    except AssertionError:
        pass
    P(f'T3b PASS: pop-light xt list ({len(pop_names)} columns, no it_hist): cd_arms6 refuses it; cd_arms6b with the hz2 dir swaps exactly xp_imp_c50 (column {cp[0][0]})')
    # ---------------- T4
    keep_full = np.load(f'{VAR_DIR}/train/hz2_keep_rows.npy', mmap_mode='r')
    keep = torch.from_numpy(np.asarray(keep_full[crow]))
    h4 = B2.apply_keep(h2.clone(), ms.xt, c2, keep)
    for k, (c, _, _) in enumerate(c2):
        assert bits_equal(h4[keep, k].contiguous(), ms.xt[keep, c].contiguous()) and bits_equal(h4[~keep, k].contiguous(), h2[~keep, k].contiguous())
    P(f'T4 PASS: HZ2_KEEP: {int(keep.sum())} of {len(keep)} sample rows flagged; flagged rows == stored xt in all 8 columns, the others == hz2')
    open(f'{HERE}/logs/test_cd_arms6b.log', 'w').write('\n'.join(out) + '\n')


if __name__ == '__main__':
    main()
