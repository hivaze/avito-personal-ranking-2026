"""u20wit read-out (2): the natural TRANSDUCTIVE witness for U20 (CPU fp32 replays vs CPU fp32 replays only).
  cd <repo> && CUDA_VISIBLE_DEVICES="" .venv/bin/python nn_x/w12/next/u20wit/u20_wit_read.py [--A u20a --B u20b --C u20c --seeds 951,952 --dom tr --nocheck]
Needs cache/sc/w5_e10_<arm>_s<seed>[@last]__<dom>.npy from wit_score.py (sets W0S, W0L, H0).
Witness = train feeds of FOLD-0 users, XPOOL v2 posterior p_random > 0.9, outside the 30K holdout: W0S (M < 50), W0L (M >= 50); H0 = the fold-0 holdout feeds.
  B - A on W0S / W0L = value of having trained on the feed's INPUTS with teacher-only targets (never its labels): the analogue of test feeds as a transfer set.
  C - A on W0S / W0L = IN-SAMPLE for C (C trained on these labels): an upper bound only, never evidence.
  B - A on H0        = same-user channel without the feed itself (B saw the user's other feeds unlabelled).
This is INFORMATION next to the declared rule (u20_read.py decides); no threshold is attached to it.
Statistic: member delta paired per feed, pooled over the seeds; 2-pool = mean per-feed z of the 2 seeds. Train labels of the witness rows are read for evaluation only."""
from wit_common import *
import argparse, torch
from stack_val import feed_rank_z
ap = argparse.ArgumentParser(); ap.add_argument('--A', default='u20a'); ap.add_argument('--B', default='u20b'); ap.add_argument('--C', default='u20c')
ap.add_argument('--seeds', default='951,952'); ap.add_argument('--dom', default='tr'); ap.add_argument('--nocheck', action='store_true', help='skip the U20 flag asserts (known-answer tests on other arms)')
a = ap.parse_args(); seeds = [int(s) for s in a.seeds.split(',')]; ARMS = (a.A, a.B, a.C)
sm = np.load(CACHE / 'sub_meta.npz'); sub = CACHE / 'sub'
M = sm['M'].astype(np.int64); off = np.concatenate([[0], np.cumsum(M)[:-1]]).astype(np.int64); y = np.load(sub / 'cand_target.npy').astype(np.float32)
fid = np.repeat(np.arange(len(M)), M); sid = sm['set_id']; uid = sm['uid']; pold = sm['pold']
name = lambda arm, s: f'w5_e10_{arm}_s{s}'
E = lambda x: ndcg_feeds(np.nan_to_num(x), y, off, M); Z = lambda x: feed_rank_z(off, M, np.nan_to_num(np.asarray(x, np.float32)))[:, 1]

# ---- membership of the witness feeds in each arm's training set, recomputed label-free from train.py's rules (l.284-318, 430-437)
tm = train_meta(); n = tm['n']; isH = np.zeros(n, bool); isH[tm['hold']] = True; xf = tm['xf']; S = witness_sets(tm)
trainA = ~isH & (xf != 0); trainB = ~isH; unlabB = trainB & (xf == 0); poolA = trainA & (tm['pold'] > 0.5); poolB = trainB & (tm['pold'] > 0.5) & ~unlabB; poolC = ~isH & (tm['pold'] > 0.5)
print('MEMBERSHIP (recomputed from train.py rules; label-free)')
print(f'  u20a: training feeds {int(trainA.sum())} (fold 0 dropped: {int((~isH & (xf == 0)).sum())}), cooldown pool {int(poolA.sum())} | u20b: training feeds {int(trainB.sum())}, unlabelled {int(unlabB.sum())}, cooldown pool {int(poolB.sum())} | u20c: training feeds {int(trainB.sum())}, cooldown pool {int(poolC.sum())}')
for k in ('W0S', 'W0L', 'H0'):
    i = S[k]
    print(f'  {k}: {len(i)} feeds | in u20a training {int(trainA[i].sum())} | in u20b training {int(trainB[i].sum())}, of them unlabelled (teacher-only target, aux weight 0) {int(unlabB[i].sum())}, in u20b cooldown pool {int(poolB[i].sum())} '
          f'| in u20c training (labelled) {int(trainB[i].sum())}, in u20c cooldown pool {int(poolC[i].sum())}')
for arm in ARMS:
    for s in seeds:
        lg = W5L / f'{name(arm, s)}.log'
        if lg.exists():
            ls = [l.strip()[:150] for l in open(lg) if l.startswith(('w12 xfold', 'U20 xfold', 'cooldown mix', 'train feeds'))]
            print(f'  log {name(arm, s)}: ' + ' || '.join(ls[:4]))
ys = np.bincount(fid, weights=y, minlength=len(M)); print(f'  witness feeds without a positive (the only label-dependent factor of the unlabelled target, train.py l.502): {int((ys[np.isin(sid, (0, 1))] == 0).sum())}')

def checks(arm, s, suf):
    ck = torch.load(CKPT / f'{name(arm, s)}{suf}.pt', map_location='cpu', weights_only=False); g = ck['args']
    want = {a.A: (0, False), a.B: (0, True), a.C: (-1, False)}[arm]
    assert (g['xfold'], bool(g['xfold_unlab'])) == want and g['kd_file'].endswith('teacher_xf0.npy') and g['kd_T'] == 3 and g['kd_w'] == 0.5 and g['d'] == 192 and g['seed'] == s, (arm, s, g)
    return ck.get('step', 'sel')

for suf, tag in (('', 'SELECTED checkpoints'), ('@last', '@last checkpoints')):
    def full(arm, s):  # a score file counts only when W0S, W0L and H0 are completely scored (wit_score.py saves after every set)
        p_ = CACHE / 'sc' / f'{name(arm, s)}{suf}__{a.dom}.npy'
        return p_.exists() and not np.isnan(np.load(p_)[np.isin(sid, (0, 1, 2))[fid]]).any()
    have = {arm: [s for s in seeds if full(arm, s)] for arm in ARMS}
    print(f'\n=== {tag}, domain token {a.dom} === scored: ' + ', '.join(f'{arm} {have[arm]}' for arm in ARMS))
    if not any(have.values()): continue
    e, zz = {}, {}
    for arm in ARMS:
        for s in have[arm]:
            if not a.nocheck: checks(arm, s, suf)
            x = np.load(CACHE / 'sc' / f'{name(arm, s)}{suf}__{a.dom}.npy')
            assert not np.isnan(x[np.isin(sid, (0, 1, 2))[fid]]).any(), f'{name(arm, s)}{suf}: W0S / W0L / H0 not fully scored'
            e[arm, s] = E(x); zz[arm, s] = Z(x)
    pool = {arm: E(np.mean([zz[arm, s] for s in have[arm]], 0)) for arm in ARMS if have[arm]}
    w0s, w0l, h0 = sid == 0, sid == 1, sid == 2
    ctrl = all(not np.isnan(np.load(CACHE / 'sc' / f'{name(arm, s)}{suf}__{a.dom}.npy')[np.isin(sid, (3, 4))[fid]]).any() for arm in ARMS for s in have[arm])
    ROWS = [('W0S short, all', w0s), ('  M 5-9', w0s & (M < 10)), ('  M 10-19', w0s & (M >= 10) & (M < 20)), ('  M 20-34', w0s & (M >= 20) & (M < 35)), ('  M 35-49', w0s & (M >= 35)),
            ('  excl. user-0 bucket', w0s & (uid != 0)), ('W0L long, all', w0l), ('  in C cooldown pool', w0l & (pold > 0.5)), ('  not in C pool', w0l & (pold <= 0.5)), ('  old p>0.9 (holdout-like)', w0l & (pold > 0.9)),
            ('  M 50-99', w0l & (M < 100)), ('  M 100-300', w0l & (M >= 100)), ('  excl. user-0 bucket', w0l & (uid != 0)), ('W0S + W0L', w0s | w0l), ('H0 fold-0 holdout', h0)]
    if ctrl: ROWS += [('W1S control (fold 1)', sid == 3), ('W1L control (fold 1)', sid == 4)]  # labelled and in-sample for A, B and C alike
    D = {}
    print(f'  {"feeds":27s} {"n":>6s} | ' + ' | '.join(f'{arm:>5s} member  pool' for arm in ARMS if have[arm]) + ' | B-A member (per seed) ; B-A pool | C-A member ; C-A pool [IN-SAMPLE for C on W0*] | B-C member')
    for rn, mk in ROWS:
        t = f'  {rn:27s} {int(mk.sum()):6d} | ' + ' | '.join(f'{np.mean([e[arm, s][mk].mean() for s in have[arm]]):.5f} {pool[arm][mk].mean():.5f}' for arm in ARMS if have[arm])
        for X, Y, lab in ((a.B, a.A, 'B-A'), (a.C, a.A, 'C-A'), (a.B, a.C, 'B-C')):
            ss = [s for s in have[X] if s in have[Y]]
            if not ss: continue
            d = np.mean([e[X, s][mk] - e[Y, s][mk] for s in ss], 0); D[rn, lab] = d
            t += f' | {fmt(d)}' + (' (' + ' '.join(f'{np.mean(e[X, s][mk] - e[Y, s][mk]):+.5f}' for s in ss) + ')' if lab == 'B-A' else '')
            if lab != 'B-C' and have[X] == have[Y]: t += f' ; {fmt(pool[X][mk] - pool[Y][mk])}'
        print(t)
    if ('W0L long, all', 'B-A') in D:
        dl, dh, ds = D['W0L long, all', 'B-A'], D['H0 fold-0 holdout', 'B-A'], D['W0S short, all', 'B-A']
        # short witness re-weighted to the M mix of val M < 50 (exact M, post-stratified; W0S has mean M 28 against 23 in val)
        Mv = pl.read_parquet(FLAT / 'val/feeds.parquet')['M'].to_numpy().astype(np.int64); Ms = M[w0s]
        cv = np.bincount(Mv[Mv < 50], minlength=50)[:50].astype(float); cw = np.bincount(Ms, minlength=50)[:50].astype(float)
        w = np.where(cw[Ms] > 0, cv[Ms] / np.maximum(cw[Ms], 1), 0.0); mu = (w * ds).sum() / w.sum(); sew = np.sqrt((w ** 2 * (ds - mu) ** 2).sum()) / w.sum()
        print(f'  SHORT witness re-weighted to the val M<50 size mix (exact M; effective n {w.sum() ** 2 / (w ** 2).sum():.0f}): B - A {mu:+.5f}+-{sew:.5f}')
        print(f'  TRANSDUCTIVE READ ({tag}): B - A on natural SHORT witness feeds {fmt(ds)} (n {len(ds)}); on LONG witness feeds {fmt(dl)} (n {len(dl)}); both {fmt(D["W0S + W0L", "B-A"])}')
        print(f'    contrast "this feed\'s inputs seen" minus "only the user\'s other feeds seen" (long: W0L minus H0, independent feed sets): {dl.mean() - dh.mean():+.5f}+-{np.sqrt(se(dl) ** 2 + se(dh) ** 2):.5f}')
        dm = D['  old p>0.9 (holdout-like)', 'B-A']
        print(f'    same contrast on the population-matched subset (W0L feeds with old p_random > 0.9, the holdout\'s own selection rule; n {len(dm)}): {dm.mean() - dh.mean():+.5f}+-{np.sqrt(se(dm) ** 2 + se(dh) ** 2):.5f}')
        if ('W0L long, all', 'C-A') in D:
            cs, cl = D['W0S short, all', 'C-A'], D['W0L long, all', 'C-A']
            print(f'    in-sample bound (C trained on these labels; NOT evidence): C - A short {fmt(cs)}, long {fmt(cl)}; B - A as a share of that bound: short {ds.mean() / cs.mean():+.3f}, long {dl.mean() / cl.mean():+.3f}')
    # mechanism (label-free): does a model follow the teacher more closely on feeds it saw as unlabelled inputs? within-feed Pearson corr to teacher_xf0 (mean logit)
    T = np.asarray(np.load(ROOT / 'nn_x/w12/kd/out/teacher_xf0.npy', mmap_mode='r')[np.load(sub / 'rows.npy')], np.float64); cT = T - (np.bincount(fid, weights=T) / M)[fid]; vT = np.bincount(fid, weights=cT * cT)
    def tcorr(x):
        x = np.nan_to_num(x).astype(np.float64); cx = x - (np.bincount(fid, weights=x) / M)[fid]; return np.bincount(fid, weights=cx * cT) / np.sqrt(np.bincount(fid, weights=cx * cx) * vT + 1e-30)
    tc = {arm: np.mean([tcorr(np.load(CACHE / 'sc' / f'{name(arm, s)}{suf}__{a.dom}.npy')) for s in have[arm]], 0) for arm in ARMS if have[arm]}
    eT = E(T.astype(np.float32))
    print('  teacher_xf0 (GPU bf16 file, 3 hard-label members that never saw fold 0) NDCG: ' + ', '.join(f'{k} {eT[mk].mean():.5f}' for k, mk in (('W0S', w0s), ('W0L', w0l), ('H0', h0))))
    print('  mean within-feed corr to the teacher (member mean over seeds): ' + ' | '.join(f'{k}: ' + ' '.join(f'{arm} {tc[arm][mk].mean():.4f}' for arm in tc) + (f' ; B-A {fmt(tc[a.B][mk] - tc[a.A][mk])}' if a.A in tc and a.B in tc else '')
                                                                                         for k, mk in (('W0S', w0s), ('W0L', w0l), ('H0', h0))))
    # anchor: CPU replay vs the stored GPU holdout file on H0, per member
    hold = tm['hold']; Mh = tm['M'][hold]; oh = np.concatenate([[0], np.cumsum(Mh)[:-1]]); h0i = np.where(xf[hold] == 0)[0]; hrows = np.concatenate([np.arange(oh[i], oh[i] + Mh[i]) for i in h0i]); rH = h0[fid]
    t = '  anchor on H0, CPU fp32 minus stored GPU bf16 file, per member: '
    for arm in ARMS:
        for s in have[arm]:
            p = W5P / f'holdout/nn_{name(arm, s)}{suf}_holdout.npy'
            if p.exists():
                g = np.zeros(len(y), np.float32); g[rH] = np.load(p)[hrows]; t += f'{arm}_s{s} {fmt(e[arm, s][h0] - E(g)[h0])}; '
    print(t)
print('DONE')
