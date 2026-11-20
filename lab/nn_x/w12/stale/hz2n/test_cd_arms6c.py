"""hz2n proofs P1 and P4 (CPU, MiniSplit of real train feeds; read-only imports, no bytecode written; pattern of nn_x/w12/stale/hz2/test_cd_arms6b.py + _t5.py).
  P1  identity: cd_arms6c vs nn_x/w5/cd_arms6b.py when HZCD_DIR has no encoder-score file (unset = data/feats_new; hz2/feats and hz2/feats_L with and without
      HZ2_KEEP): AST (changed defs exactly {luo_cols, setup_hzcd}, no new def), same module constants, same column lists (W8 preset and pop-light list), bit-identical
      load_hz tensors, bit-identical HzTrainView.gather batches (every key) on 2,000 random train feeds, identical hz_stats, setup_hzcd log line identical apart from the tag.
  P4b end to end on the sample: setup_hzcd of cd_arms6c with each new directory (HZ2_KEEP where the view has one): the log line names the encoder columns; every
      gathered cell of a swapped column == keep ? stored : fp16(attach_xt transform(view file)); encoder columns show the no-ID values on newly cold rows and the
      stored bits elsewhere; every other key / column == plain gather; popularity columns bit-identical to the 6b view of the base directory.
      Optional hooks: has: column (a directory with the has file in train/), partial view refused, raw-ID preset refused, transform-branch equality of the name pairs.
  P4c tiny cooldown on CPU: the real Ranker of a W8 student checkpoint; 30 optimiser steps on batches gathered through the view (finite loss, weights move);
      label-free output sensitivity: within-feed z of the logits, plain vs base view vs new view (eval mode, before any step).
Holdout feeds (train.py --holdout 30000 rule) are excluded from every sample, so no holdout label is even loaded.
Usage: CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 python test_cd_arms6c.py [p1|p4b|p4c ...]   -> logs/test_cd_arms6c_<part>.log
"""
import ast
import importlib.util
import os
import re
import sys
import types

sys.dont_write_bytecode = True
assert os.environ.get('CUDA_VISIBLE_DEVICES', 'x') == '', 'run with CUDA_VISIBLE_DEVICES=""'
import warnings

warnings.filterwarnings('ignore')
import numpy as np
import polars as pl
import torch

torch.set_num_threads(12)
ROOT = '<repo>'
HERE = os.path.dirname(os.path.abspath(__file__))
W5 = f'{ROOT}/nn_x/w5'
HZ2 = f'{ROOT}/nn_x/w12/stale/hz2'
FC, F = f'{ROOT}/nn/feat_cache', f'{ROOT}/data/feats'
sys.path.insert(0, f'{W5}/tests')
sys.path.insert(1, W5)
from mini_split import MiniSplit  # noqa: E402

M6B, M6C = f'{W5}/cd_arms6b.py', f'{HERE}/cd_arms6c.py'
ENC_ALL = ('pre2_fe_z', 'pre2_fe_s', 'pre2_new_z', 'pre2_new_s')
is_F = lambda dn: dn.endswith(('F', 'FS'))  # noqa: E731
BASES = {'feats_n': (f'{HZ2}/feats', True), 'feats_Ln': (f'{HZ2}/feats_L', True), 'feats_hn': (f'{ROOT}/data/feats_new', False)}
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
        it = torch.int32 if a.dtype == torch.float32 else torch.int16
        return bool(torch.equal(a.contiguous().view(-1).view(it), b.contiguous().view(-1).view(it)))
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


def sample(n, seed):
    f = pl.read_parquet(f'{ROOT}/data/flat/train/feeds.parquet')
    Mall = f['M'].to_numpy().astype(np.int64)
    pr = np.load(f'{ROOT}/eda/labels_position/out/train_prandom.npz')
    p = np.nan_to_num(pr['p_random'].astype(np.float32), nan=0.0); p[Mall < 50] = 0.0
    cand = np.where((p > 0.9) & (Mall >= 50))[0]
    hold = np.sort(np.random.default_rng(12345).choice(cand, 30000, replace=False))  # train.py --holdout 30000
    ok = np.ones(len(f), bool); ok[hold] = False
    feeds = np.sort(np.random.default_rng(seed).choice(np.flatnonzero(ok), n, replace=False))
    M = Mall[feeds]; coff = f['cand_off'].to_numpy().astype(np.int64)[feeds]
    crow = np.concatenate([np.arange(o, o + m) for o, m in zip(coff, M)])
    slot = ((f['timestamp'].to_numpy().astype(np.int64)[feeds] + 10800) // 86400 - 20702)
    return feeds, M, coff, crow, slot


def keep_sample(base, crow, tag):
    os.makedirs(f'{HERE}/cache', exist_ok=True)
    kp = f'{HERE}/cache/keep_{tag}_{len(crow)}.npy'
    np.save(kp, np.asarray(np.load(f'{base}/train/hz2_keep_rows.npy', mmap_mode='r')[crow]))
    return kp


def wrap(mod, crow):
    orig = mod.load_hz
    mod.load_hz = lambda split, cols, device, rows=None: orig(split, cols, device, rows=crow)
    return mod


def norm_line(s):
    return re.sub(r'; \d+s$', '', s).replace('[6c: ', '[6b: ')


NS = types.SimpleNamespace(hcap='', variant='t')


def p1():
    s6b, s6c = open(M6B).read(), open(M6C).read()
    tb, tc = ast.parse(s6b), ast.parse(s6c)
    db = {n.name: ast.dump(n) for n in tb.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    dc = {n.name: ast.dump(n) for n in tc.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    changed = sorted(k for k in db if db[k] != dc.get(k)); new = sorted(k for k in dc if k not in db)
    ab = {n.targets[0].id: ast.dump(n.value) for n in tb.body if isinstance(n, ast.Assign)}
    ac = {n.targets[0].id: ast.dump(n.value) for n in tc.body if isinstance(n, ast.Assign)}
    P(f'P1.T1 defs of cd_arms6b: {sorted(db)}; changed in 6c: {changed}; new defs in 6c: {new}; module constants changed: {sorted(k for k in ab if ab[k] != ac.get(k))}; '
      f'new module constants: {sorted(k for k in ac if k not in ab)}')
    assert changed == ['luo_cols', 'setup_hzcd'] and new == [] and all(ab[k] == ac[k] for k in ab) and sorted(k for k in ac if k not in ab) == ['ENC_GROUPS', 'ENC_TO_HZ']
    imp = lambda t: [ast.dump(n) for n in t.body if isinstance(n, (ast.Import, ast.ImportFrom))]  # noqa: E731
    assert imp(tb) == imp(tc)
    P('P1.T1 PASS: hz_path, xt_transform, load_hz, apply_keep, HzTrainView, hz_stats, save_holdout_hz are AST-identical; imports identical; only ENC_TO_HZ / ENC_GROUPS added')
    feeds, M, coff, crow, slot = sample(2000, 20261116)
    PRE = xt_presets(); xt_names = PRE['cvnd_pre2s']
    pop_names = [n for n in xt_names if n not in xt_names[:8]] + ['xp_imp_c50']
    ms = MiniSplit('train', feeds, xt_names=xt_names)
    P(f'MiniSplit: {len(feeds)} random non-holdout train feeds (seed 20261116), {len(crow)} candidate rows, {len(xt_names)} xt columns (cvnd_pre2s); feeds per day slot '
      f'{dict(zip(*[x.tolist() for x in np.unique(slot, return_counts=True)]))}')
    fb = [np.arange(s, min(len(feeds), s + 50)) for s in range(0, len(feeds), 50)]
    envs = {'unset (data/feats_new)': {}, 'hz2/feats + HZ2_KEEP': {'HZCD_DIR': f'{HZ2}/feats', 'HZ2_KEEP': keep_sample(f'{HZ2}/feats', crow, 'a')},
            'hz2/feats_L + HZ2_KEEP': {'HZCD_DIR': f'{HZ2}/feats_L', 'HZ2_KEEP': keep_sample(f'{HZ2}/feats_L', crow, 'b')}, 'hz2/feats, no keep': {'HZCD_DIR': f'{HZ2}/feats'}}
    for en, env in envs.items():
        B, C = load_mod(M6B, 'b_' + str(len(en)), env), load_mod(M6C, 'c_' + str(len(en)), env)
        assert B.HZ_DIR == C.HZ_DIR and B.KEEP_PATH == C.KEEP_PATH and B.LUO_TO_HZ == C.LUO_TO_HZ and B.EXTRA_TO_HZ == C.EXTRA_TO_HZ
        cb, cc = B.luo_cols(xt_names), C.luo_cols(xt_names)
        assert cb == cc, (cb, cc)
        try:
            pb = B.luo_cols(pop_names)
        except AssertionError:
            pb = 'AssertionError'
        try:
            pc = C.luo_cols(pop_names)
        except AssertionError:
            pc = 'AssertionError'
        assert pb == pc, (pb, pc)
        hb, hc = B.load_hz('train', cb, ms.device, rows=crow), C.load_hz('train', cc, ms.device, rows=crow)
        assert bits_equal(hb, hc)
        wrap(B, crow); wrap(C, crow)
        lb, lc = [], []
        vb = B.setup_hzcd(NS, ms, xt_names, np.arange(5), 10, 100, lb.append); vc = C.setup_hzcd(NS, ms, xt_names, np.arange(5), 10, 100, lc.append)
        assert vb.cols == vc.cols and vb.col_idx == vc.col_idx and bits_equal(vb.hz, vc.hz)
        ncell = 0
        for ids in fb:
            b1, b2 = vb.gather(ids, with_pos=True), vc.gather(ids, with_pos=True)
            bad = batches_equal(b1, b2)
            assert not bad, f'{en}: keys differ between 6b and 6c: {bad}'
            ncell += int(b1['cmask'].sum())
        assert B.hz_stats(ms, vb) == C.hz_stats(ms, vc)
        assert norm_line(lb[-1]) == norm_line(lc[-1]), (lb[-1], lc[-1])
        assert 'ENCODER-SCORE' not in lc[-1]
        P(f'P1.T2 PASS [{en}]: columns {[c[1] for c in cc]} identical ({len(cc)}); pop-light list -> {pb if isinstance(pb, str) else [c[1] for c in pb]} identical; load_hz '
          f'bit-identical {tuple(hc.shape)}; {len(fb)} batches x {len(b1)} keys bit-identical ({ncell} real candidate cells); hz_stats identical; log line identical apart from the tag')
    P('   6c log line (unset env is the last but three; shown: hz2/feats, no keep): ' + lc[-1][:520] + ' ...')
    P('P1 PASS')


def expect_cols(dn, xt_names):
    base = f'{HERE}/{dn}'   # the directory itself (its symlinks carry the base view; feats_hnFS adds seq_cv files)
    encn = ENC_ALL if is_F(dn) else ENC_ALL[:2]
    pop = [n for n in xt_names if os.path.exists(f"{base}/train/{ {**{'it_hist_all_luo': 'it_hist_hz_all', 'it_hist_et1_luo': 'it_hist_hz_et1', 'it_hist_et2_luo': 'it_hist_hz_et2', 'it_hist_et3_luo': 'it_hist_hz_et3', 'it_hist_et4_luo': 'it_hist_hz_et4', 'seq_cv_invd20': 'seq_cv_hz_invd20', 'seq_cv_pop': 'seq_cv_hz_pop', 'seq_cv_norm': 'seq_cv_hz_norm', 'xp_imp_c50': 'xp_imp_hz_c50'}}.get(n, '__none__')}.npy")]
    return [n for n in xt_names if n in pop or n in encn]


def p4b():
    feeds, M, coff, crow, slot = sample(2000, 20261116)
    PRE = xt_presets(); xt_names = PRE['cvnd_pre2s']
    pop_names = [n for n in xt_names if n not in xt_names[:8]] + ['xp_imp_c50']
    ms = MiniSplit('train', feeds, xt_names=xt_names)
    msp = MiniSplit('train', feeds, xt_names=pop_names)
    fb = [np.arange(s, min(len(feeds), s + 50)) for s in range(0, len(feeds), 50)]
    foff = np.concatenate([[0], np.cumsum(M)[:-1]])
    S = np.asarray(np.load(f'{FC}/train/it_hist_all_luo.npy', mmap_mode='r')[crow])
    noid = np.asarray(np.load(f'{F}/train/pre2_fe_noid.npy', mmap_mode='r')[crow])
    C0 = load_mod(M6C, 'c_probe', {})
    probe = np.array([np.nan, -30, -7, -5.5, -1, 0, 0.3, 1, 4.9, 6, 11, 50], np.float32)
    for sn, hn in C0.ENC_TO_HZ.items():
        if sn.startswith('has:'):
            v = np.array([0, 1], np.float32)
            assert np.array_equal(C0.xt_transform(hn, v), v)
        else:
            assert np.array_equal(C0.xt_transform(sn, probe), C0.xt_transform(hn, probe)), (sn, hn)
    for sn, hn in {**C0.LUO_TO_HZ, **C0.EXTRA_TO_HZ}.items():
        assert np.array_equal(C0.xt_transform(sn, probe), C0.xt_transform(hn, probe)), (sn, hn)
    P('P4b.T0 PASS: xt_transform(hz name) == xt_transform(stored name) on a probe vector for every pair of LUO_TO_HZ / EXTRA_TO_HZ / ENC_TO_HZ (has: pair: identity on {0, 1})')
    for dn in ('feats_n', 'feats_Ln', 'feats_hn', 'feats_nP', 'feats_LnP', 'feats_hnP', 'feats_nF', 'feats_LnF', 'feats_hnF', 'feats_hnFS'):
        base, has_keep = BASES[dn.rstrip('PFS')]
        d = f'{HERE}/{dn}'
        env = {'HZCD_DIR': d}
        keep_np = np.zeros(len(crow), bool)
        if has_keep:
            env['HZ2_KEEP'] = keep_sample(base, crow, dn)
            keep_np = np.load(env['HZ2_KEEP'])
        C = wrap(load_mod(M6C, 'c_' + dn, env), crow)
        logs = []
        view = C.setup_hzcd(NS, ms, xt_names, np.arange(5), 10, 100, logs.append)
        names = [c[1] for c in view.cols]
        want_names = expect_cols(dn, xt_names)
        assert names == want_names, (names, want_names)
        line = logs[-1]
        encn = [n for n in names if n in ENC_ALL]
        assert encn == [n for n in xt_names if n in (ENC_ALL if is_F(dn) else ENC_ALL[:2])]
        assert f'[6c: {len(names)} swapped columns {names}; HZ2_KEEP rows on stored values: {int(keep_np.sum())}; ENCODER-SCORE VIEW ON: {len(encn)} of them are encoder-score columns ' \
               f"{encn} <- {[n.replace('pre2_', 'pre2_hz_') for n in encn]}" in line and d in line, line
        keep = torch.from_numpy(keep_np)
        brute = torch.stack([torch.from_numpy(C.xt_transform(hn, np.asarray(np.load(f'{d}/train/{hn}.npy', mmap_mode='r')[crow]))).to(torch.float16) for _, _, hn in view.cols], 1)
        cidx = [c[0] for c in view.cols]
        want_all = torch.where(keep[:, None], ms.xt[:, cidx], brute)
        assert bits_equal(view.hz, want_all)
        # the 6b view of the BASE directory: popularity columns must be the same bits
        Bb = wrap(load_mod(M6B, 'b_' + dn, {**env, 'HZCD_DIR': base}), crow)
        vbase = Bb.setup_hzcd(NS, ms, xt_names, np.arange(5), 10, 100, [].append)
        kpop = [k for k, n in enumerate(names) if n not in ENC_ALL]
        bn = [c[1] for c in vbase.cols]
        assert bits_equal(view.hz[:, [names.index(n) for n in bn]].contiguous(), vbase.hz)
        if dn.endswith('S'):  # ADDENDUM 3: seq_cv columns == fp16(log1p(where(view counter == 0, 0, stored))); stored bits where the view counter is > 0
            Vs = torch.from_numpy(np.asarray(np.load(f'{base}/train/it_hist_hz_all.npy', mmap_mode='r')[crow]) == 0)
            for n in ('seq_cv_invd20', 'seq_cv_pop', 'seq_cv_norm'):
                k_, c_ = names.index(n), xt_names.index(n)
                assert bits_equal(view.hz[~Vs, k_].contiguous(), ms.xt[~Vs, c_].contiguous()) and float(view.hz[Vs, k_].abs().max()) == 0.0
            assert [n for n in names if n not in bn and n not in ENC_ALL] == ['seq_cv_invd20', 'seq_cv_pop', 'seq_cv_norm']
        else:
            assert [names[k] for k in kpop] == bn
        # independent newly-cold mask and encoder values
        V = np.asarray(np.load(f'{base}/train/it_hist_hz_all.npy', mmap_mode='r')[crow])
        nc = (V == 0) & (S > 0) & ~keep_np
        assert np.array_equal(nc, np.asarray(np.load(f'{d}/train/hz2n_newly_cold.npy', mmap_mode='r')[crow]))
        kz, ks = names.index('pre2_fe_z'), names.index('pre2_fe_s')
        cz, cs = xt_names.index('pre2_fe_z'), xt_names.index('pre2_fe_s')
        hfe = np.asarray(np.load(f'{d}/train/pre2_hz_fe.npy', mmap_mode='r')[crow])
        src_score = noid if not dn.endswith('P') else hfe   # as asked: the stored no-ID file; P: the fe-map no-ID score (file pre2_hz_fe)
        exp_s = torch.from_numpy(np.nan_to_num(np.clip(((src_score + 20) / 6).astype(np.float32), -10, 10), nan=0.0)).to(torch.float16)
        nct = torch.from_numpy(nc)
        assert bits_equal(view.hz[nct, ks].contiguous(), exp_s[nct].contiguous()), 'pre2_fe_s on newly cold rows is not the no-ID value'
        assert bits_equal(view.hz[~nct, ks].contiguous(), ms.xt[~nct, cs].contiguous()), 'pre2_fe_s changed outside the newly cold rows'
        fnc = np.add.reduceat(nc.astype(np.int64), foff) > 0
        rin = torch.from_numpy(np.repeat(fnc, M))
        assert bits_equal(view.hz[~rin, kz].contiguous(), ms.xt[~rin, cz].contiguous()), 'pre2_fe_z changed in a feed without a newly cold row'
        ncells = dict(nc=0, fs_diff=0, fz_diff=0, keep=0, real=0)
        for ids in fb:
            b, b0 = view.gather(ids, with_pos=True), ms.gather(ids, with_pos=True)
            assert not batches_equal(b, b0, skip_xt_cols=cidx)
            assert bits_equal(b['xt'][..., cidx].contiguous(), want_all[b['ci']].float().contiguous())
            m = b['cmask']
            ncells['real'] += int(m.sum()); ncells['nc'] += int((nct[b['ci']] & m).sum()); ncells['keep'] += int((keep[b['ci']] & m).sum())
            ncells['fs_diff'] += int(((b['xt'][..., cs] != b0['xt'][..., cs]) & m).sum()); ncells['fz_diff'] += int(((b['xt'][..., cz] != b0['xt'][..., cz]) & m).sum())
            kk = keep[b['ci']]
            assert bits_equal(b['xt'][..., cidx][kk].contiguous(), b0['xt'][..., cidx][kk].contiguous())
            nn_ = nct[b['ci']] & m
            assert bits_equal(b['xt'][..., cs][nn_].contiguous(), exp_s[b['ci']][nn_].float().contiguous())
        st = {s_['luo']: s_ for s_ in C.hz_stats(ms, view)}
        extra = ''
        if is_F(dn):  # new head: no-ID value (file pre2_hz_new) on newly cold rows, stored bits elsewhere; z stored in feeds without a newly cold row
            kz2, ks2 = names.index('pre2_new_z'), names.index('pre2_new_s')
            cz2, cs2 = xt_names.index('pre2_new_z'), xt_names.index('pre2_new_s')
            hnew = np.asarray(np.load(f'{d}/train/pre2_hz_new.npy', mmap_mode='r')[crow])
            exp2 = torch.from_numpy(np.nan_to_num(np.clip(((hnew + 20) / 6).astype(np.float32), -10, 10), nan=0.0)).to(torch.float16)
            assert bits_equal(view.hz[nct, ks2].contiguous(), exp2[nct].contiguous()) and bits_equal(view.hz[~nct, ks2].contiguous(), ms.xt[~nct, cs2].contiguous())
            assert bits_equal(view.hz[~rin, kz2].contiguous(), ms.xt[~rin, cz2].contiguous())
            # fe-head columns of F are the as-asked files (same bits as the as-asked directory)
            assert os.path.samefile(f'{d}/train/pre2_hz_fe_z.npy', f'{HERE}/{dn.rstrip("FS")}/train/pre2_hz_fe_z.npy') and os.path.samefile(f'{d}/train/pre2_hz_fe_s.npy', f'{HERE}/{dn.rstrip("FS")}/train/pre2_hz_fe_s.npy')
            extra = (f'; NEW head: pre2_new_s == (pre2_hz_new + 20) / 6 on newly cold rows and stored bits elsewhere, pre2_new_z stored bits in feeds without a newly cold row, rows changed '
                     f'pre2_new_z {st["pre2_new_z"]["changed"]:.4f} pre2_new_s {st["pre2_new_s"]["changed"]:.4f}; fe-head files are the as-asked files (same inode)')
        # pop-light list
        Cp = wrap(load_mod(M6C, 'cp_' + dn, env), crow)
        cp = Cp.luo_cols(pop_names)
        vp = Cp.setup_hzcd(NS, msp, pop_names, np.arange(5), 10, 100, [].append)
        bp, bp0 = vp.gather(fb[0], with_pos=True), msp.gather(fb[0], with_pos=True)
        assert not batches_equal(bp, bp0, skip_xt_cols=[c[0] for c in cp])
        kzp = [c[1] for c in cp].index('pre2_fe_s')
        assert bits_equal(vp.hz[:, kzp].contiguous(), view.hz[:, ks].contiguous())
        P(f'P4b PASS [{dn}]: {len(names)} swapped columns {names}; log tag + encoder clause present; view.hz == keep ? stored : fp16(transform(file)) on all {len(crow)} sample rows; '
          f'popularity columns bit-identical to the 6b view of {base}; newly cold rows in sample {int(nc.sum())} (mask == independent recomputation); pre2_fe_s == no-ID value on them '
          f'and stored bits elsewhere; pre2_fe_z stored bits in the {int((~fnc).sum())} feeds without a newly cold row; {len(fb)} batches: real cells {ncells["real"]}, newly cold cells '
          f'{ncells["nc"]}, keep cells {ncells["keep"]}, cells where pre2_fe_s / pre2_fe_z differ from the plain gather {ncells["fs_diff"]} / {ncells["fz_diff"]}; other keys / columns '
          f'== plain gather; rows changed pre2_fe_z {st["pre2_fe_z"]["changed"]:.4f} pre2_fe_s {st["pre2_fe_s"]["changed"]:.4f}; pop-light list -> {[c[1] for c in cp]}{extra}')
        if dn == 'feats_n':
            P('   log line: ' + line)
    # optional hooks
    td = f'{HERE}/cache/t_has/train'; os.makedirs(td, exist_ok=True)
    for n in os.listdir(f'{HERE}/feats_n/train'):
        if not os.path.lexists(f'{td}/{n}'):
            os.symlink(os.path.realpath(f'{HERE}/feats_n/train/{n}'), f'{td}/{n}')
    if not os.path.lexists(f'{td}/pre2_hz_fe_has.npy'):
        os.symlink(f'{HERE}/feats_n/aux/pre2_hz_fe_has.npy', f'{td}/pre2_hz_fe_has.npy')
    Ch = wrap(load_mod(M6C, 'c_has', {'HZCD_DIR': f'{HERE}/cache/t_has', 'HZ2_KEEP': keep_sample(f'{HZ2}/feats', crow, 'feats_n')}), crow)
    lh = []
    vh = Ch.setup_hzcd(NS, ms, xt_names, np.arange(5), 10, 100, lh.append)
    nh = [c[1] for c in vh.cols]
    assert nh[-3:] == ['pre2_fe_z', 'pre2_fe_s', 'has:pre2_fe'] and len(nh) == 11 and 'ENCODER-SCORE VIEW ON: 3 of them' in lh[-1]
    assert bits_equal(vh.hz[:, 10].contiguous(), ms.xt[:, xt_names.index('has:pre2_fe')].contiguous())
    P(f'P4b.T-has PASS: with pre2_hz_fe_has.npy in train/ the has: column is swapped too (11 columns) and equals the stored has column on all {len(crow)} sample rows (it is unchanged by construction)')
    tp = f'{HERE}/cache/t_partial/train'; os.makedirs(tp, exist_ok=True)
    for n in os.listdir(f'{HERE}/feats_hn/train'):
        if n != 'pre2_hz_fe_z.npy' and not os.path.lexists(f'{tp}/{n}'):  # (feats_hn: no new-head file, so only the fe group is partial)
            os.symlink(os.path.realpath(f'{HERE}/feats_hn/train/{n}'), f'{tp}/{n}')
    Cq = load_mod(M6C, 'c_partial', {'HZCD_DIR': f'{HERE}/cache/t_partial'})
    try:
        Cq.luo_cols(xt_names)
        raise SystemExit('partial encoder view accepted?')
    except AssertionError as e:
        P(f'P4b.T-partial PASS: a directory with pre2_hz_fe_s.npy but no pre2_hz_fe_z.npy is refused: {e}')
    Cr = load_mod(M6C, 'c_raw', {'HZCD_DIR': f'{HERE}/feats_hn'})
    try:
        Cr.luo_cols(PRE['cvnd_pre2'])
        raise SystemExit('raw-ID preset accepted?')
    except AssertionError as e:
        P(f'P4b.T-rawid PASS: preset cvnd_pre2 (pre2_id_contrib as an input) is refused with an encoder view: {e}')
    P('P4b PASS')


def p4c():
    import model as Mo
    feeds, M, coff, crow, slot = sample(600, 20261117)
    PRE = xt_presets()
    ck = torch.load(f'{W5}/ckpt/w5_e10_kd05w192_s951@77500.pt', map_location='cpu', weights_only=False)
    xt_names = ck['cfg'].get('xt_names') or PRE['cvnd_pre2s']
    assert xt_names == PRE['cvnd_pre2s']
    ms = MiniSplit('train', feeds, xt_names=xt_names)
    P(f'P4c: checkpoint w5_e10_kd05w192_s951@77500 (the re-cool harness init), {len(feeds)} random non-holdout train feeds (seed 20261117), {len(crow)} rows')
    fb = ms.batches(np.arange(len(feeds)), cand_budget=4096, max_feeds=48, shuffle=True, rng=np.random.default_rng(0))

    def zfeed(lg, m):
        lg = lg.float().masked_fill(~m, 0.0)
        n = m.sum(1, keepdim=True).clamp_min(1)
        mu = lg.sum(1, keepdim=True) / n
        sd = (((lg - mu) ** 2) * m).sum(1, keepdim=True).div(n).sqrt().clamp_min(1e-6)
        return (lg - mu) / sd

    views = {}
    for dn in ('base hz2/feats (6b view)', 'feats_n', 'feats_nP', 'feats_nF', 'feats_Ln', 'feats_hn'):
        if dn.startswith('base'):
            base, has_keep, d, mod = f'{HZ2}/feats', True, f'{HZ2}/feats', M6B
        else:
            (base, has_keep), d, mod = BASES[dn.rstrip('PF')], f'{HERE}/{dn}', M6C
        env = {'HZCD_DIR': d}
        if has_keep:
            env['HZ2_KEEP'] = keep_sample(base, crow, 'c' + dn.split()[0])
        C = wrap(load_mod(mod, 'm_' + dn.split()[0], env), crow)
        lg_ = []
        views[dn] = C.setup_hzcd(NS, ms, xt_names, np.arange(5), 10, 100, lg_.append)
    model = Mo.Ranker(**ck['cfg'])
    model.load_state_dict(ck['state_dict'])
    model.eval()
    ncf = {dn: torch.from_numpy(np.asarray(np.load(f'{HERE}/{dn}/train/hz2n_newly_cold.npy', mmap_mode='r')[crow])) for dn in views if not dn.startswith('base')}
    acc = {dn: [] for dn in views}
    accb = {dn: [] for dn in views if not dn.startswith('base')}
    with torch.no_grad():
        for ids in fb[:12]:
            b0 = ms.gather(ids, with_pos=True)
            z0 = zfeed(model(b0, model.dense(b0), use_pos=True), b0['cmask'])
            zs = {}
            for dn, v in views.items():
                b = v.gather(ids, with_pos=True)
                zs[dn] = zfeed(model(b, model.dense(b), use_pos=True), b['cmask'])
                acc[dn].append((zs[dn] - z0)[b0['cmask']].abs())
            for dn in accb:
                ref = zs['base hz2/feats (6b view)'] if dn in ('feats_n', 'feats_nP', 'feats_nF') else None
                if ref is not None:
                    m = b0['cmask']; nn_ = ncf[dn][b0['ci']] & m
                    accb[dn].append(((zs[dn] - ref)[m].abs().mean().item(), (zs[dn] - ref)[nn_].abs().mean().item() if nn_.any() else float('nan'), (zs[dn] - ref)[m & ~nn_].abs().mean().item()))
    for dn in views:
        P(f'P4c sensitivity (eval mode, no step, label-free): mean |within-feed z of the logit, view - plain gather| over real cells: {dn}: {torch.cat(acc[dn]).mean().item():.4f}')
    for dn, v in accb.items():
        if v:
            a_ = np.nanmean(np.array(v), 0)
            P(f'P4c encoder swap alone ({dn} minus the 6b view of hz2/feats, same model, same feeds): mean |dz| all cells {a_[0]:.4f}; newly cold cells {a_[1]:.4f}; other cells {a_[2]:.4f}')
    # tiny cooldown: 30 optimiser steps through the feats_n view (CPU fp32; NOT comparable to any GPU number; a smoke of the code path, no result)
    for dn in ('feats_n', 'feats_nP', 'feats_nF'):
        model = Mo.Ranker(**ck['cfg']); model.load_state_dict(ck['state_dict']); model.train()
        w0 = torch.cat([p_.detach().flatten()[:2000] for p_ in model.parameters()]).clone()
        opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
        losses = []
        for step in range(30):
            ids = fb[step % len(fb)]
            b = views[dn].gather(ids, with_pos=True)
            lg = model(b, model.dense(b), use_pos=True)
            loss = Mo.listwise_loss(lg, b['y'], b['cmask'])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss))
        w1 = torch.cat([p_.detach().flatten()[:2000] for p_ in model.parameters()])
        assert all(np.isfinite(losses)) and float((w1 - w0).abs().sum()) > 0
        P(f'P4c PASS [{dn}]: 30 AdamW steps on batches gathered through HzTrainView(cd_arms6c) with the real Ranker: losses finite (first {losses[0]:.4f}, last {losses[-1]:.4f}, '
          f'min {min(losses):.4f}, max {max(losses):.4f}); weights moved (sum |dw| on a probe {float((w1 - w0).abs().sum()):.4f})')
    P('P4c PASS')


if __name__ == '__main__':
    parts = sys.argv[1:] or ['p1', 'p4b', 'p4c']
    os.makedirs(f'{HERE}/logs', exist_ok=True)
    for part in parts:
        out.clear()
        {'p1': p1, 'p4b': p4b, 'p4c': p4c}[part]()
        open(f'{HERE}/logs/test_cd_arms6c_{part}.log', 'w').write('\n'.join(out) + '\n')
