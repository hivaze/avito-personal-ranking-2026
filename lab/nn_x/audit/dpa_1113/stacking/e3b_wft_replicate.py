"""E3b: is the feed-conditional w_ft preference (M<50 -> more train-only weight) a property of G3's members, or does it
replicate on member sets that never entered G3 (fresh families)? For each family pool: per-M-bucket in-sample w_ft
curve, and the nested (FT folds) M-conditional w_ft fit vs w_ft = 0.5 and vs the unconditional fit."""
from lib import *
import re, glob, time
from common import feed_meta

T0 = time.time()
V = S('val')
log = open(OUT / 'e3b_wft_replicate.txt', 'w')
def pr(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); log.write(s + '\n'); log.flush()
have = lambda m: (P / f'nn_{m}_n2cpv3_val_oof.npy').exists() and (P / f'nn_{m}_trainonly_val.npy').exists()
FAMS = {
    'G3(ref)': MEMBERS,
    'F5cd': ['v15_cd_all_e10', 'v15_cd_all_s81', 'v15_cd_all_s82', 'v15_cd_allprew_s83', 'v15_cd_allprew_s84', 'v15_cd_allprew_s85'],
    'F6': ['v13_lr1_mix1', 'v14_s92', 'v14_s93', 'v14_s94', 'v14_s95', 'v14_s96'],
    'F9': [f'v22_pre2s_lr1_s{s}' for s in range(401, 407)],
    'F12': [f'v26_d192_pre2s_s{s}' for s in range(701, 708)],
    'F7(v17)': [f'v17_pre2s_s{s}' for s in (101, 102, 103, 105, 106, 107)],
}
for pre in ['v27_gru_m02', 'v28_lr1_m02', 'v29_d192_m02', 'v30_f7_m02']:
    ms = sorted({re.sub(r'^nn_|_trainonly_val\.npy$', '', Path(f).name) for f in glob.glob(str(P / f'nn_{pre}_s*_trainonly_val.npy'))})
    FAMS[pre] = [m for m in ms if '@' not in m and have(m)]
WFT = np.array([0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0])
bM = np.digitize(V.M, [50, 100])
rows = []
for nm, ms in FAMS.items():
    ms = [m for m in ms if have(m)]
    if len(ms) < 3: pr(f'{nm}: only {len(ms)} members, skipped'); continue
    zf = V.z(np.mean([V.z(np.load(P / f'nn_{m}_n2cpv3_val_oof.npy').astype(np.float32)) for m in ms], 0).astype(np.float32))
    zt = V.z(np.mean([V.z(np.load(P / f'nn_{m}_trainonly_val.npy').astype(np.float32)) for m in ms], 0).astype(np.float32))
    NV = np.stack([V.ev(w * zf.astype(np.float64) + (1 - w) * zt.astype(np.float64)) for w in WFT], 1)
    i5 = 4
    curve = ' | '.join(f'{lab}: ' + ' '.join(f'{w:.1f}:{NV[bM == b, j].mean() - NV[bM == b, i5].mean():+.4f}' for j, w in enumerate(WFT))
                       for b, lab in enumerate(['M<50', '50-99', '>=100']))
    # nested: unconditional and M-conditional argmax of w_ft
    oofu = np.zeros(V.n); oofc = np.zeros(V.n); ch = []
    for k in range(5):
        tr = V.fold != k; te = V.fold == k
        ju = int(np.argmax(NV[tr].mean(0))); oofu[te] = NV[te, ju]
        cc = []
        for b in range(3):
            jb = int(np.argmax(NV[tr & (bM == b)].mean(0))); oofc[te & (bM == b)] = NV[te & (bM == b), jb]; cc.append(WFT[jb])
        ch.append(cc)
    base = NV[:, i5]
    a, sa = se_cl(oofc - base, V.cl); b_, sb = se_cl(oofc - oofu, V.cl); c, sc = se_cl(oofu - base, V.cl)
    m50 = V.M < 50
    d0, s0 = se_cl(NV[:, 0] - base, V.cl, m50); d0b, s0b = se_cl(NV[:, 0] - base, V.cl, ~m50)
    pr(f'{nm:14s} n={len(ms):2d} w.5 val {base.mean():.5f} | Mcond-vs-w.5 {a:+.5f} +- {sa:.5f} (z {a / sa:+.2f}) | Mcond-vs-uncond {b_:+.5f} +- {sb:.5f} '
       f'(z {b_ / sb:+.2f}) | uncond-vs-w.5 {c:+.5f} (z {c / sc:+.2f}) | TO-only (w0) vs w.5: M<50 {d0:+.5f} (z {d0 / s0:+.2f}) M>=50 {d0b:+.5f} (z {d0b / s0b:+.2f})'
       f'\n      fold choices (M<50, 50-99, >=100): {ch}\n      in-sample curve vs w.5: {curve}')
    rows.append((nm, d0, s0, d0b, s0b))
pr(f'[{time.time()-T0:.0f}s] done')
