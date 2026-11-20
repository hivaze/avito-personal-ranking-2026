"""Pre-declared final-ensemble decision (CURRENT_STATUS 3n): families, strength gate, candidates Ca-Cd, holdout choice, val confirm."""
import sys; sys.path.insert(0, '<repo>/src')
from common import *
from metrics import ndcg_per_feed
from stack_val import feed_rank_z
P = Path('<repo>/preds'); H = P / 'holdout'
FAM = {
 'F5': ['v12_all_e10','v12_all_s81','v12_all_s82','v12_allprew_s83','v12_allprew_s84','v12_allprew_s85'],
 'F5cd': ['v15_cd_all_e10','v15_cd_all_s81','v15_cd_all_s82','v15_cd_allprew_s83','v15_cd_allprew_s84','v15_cd_allprew_s85'],
 'F6': ['v13_lr1_mix1','v14_s92','v14_s93','v14_s94','v14_s95','v14_s96'],
 'F7': [f'v17_pre2s_s{s}' for s in range(101,108)],
 'F7p': [f'v17_pre2s_s{s}@77500' for s in range(101,107)],
}
fv, Mv, ov = feed_meta('val'); yv = np.load(FLAT/'val/cand_target.npy')
Mh, oh = np.load(H/'holdout_meta.npy'); Mh = Mh.astype(np.int64); oh = oh.astype(np.int64); yh = np.load(H/'holdout_y.npy')
zv = lambda x: feed_rank_z(ov, Mv, np.asarray(x, np.float32))[:,1]
zh = lambda x: feed_rank_z(oh, Mh, np.asarray(x, np.float32))[:,1]
ev = lambda s: ndcg_per_feed(s, yv, ov, Mv); eh = lambda s: ndcg_per_feed(s, yh, oh, Mh)
se = lambda d: d.std()/np.sqrt(len(d))
V = {}; HO = {}
for f, ms in FAM.items():
    for m in ms:
        V[(m,'ft')] = zv(np.load(P/f'nn_{m}_n2cpv3_val_oof.npy')); V[(m,'to')] = zv(np.load(P/f'nn_{m}_trainonly_val.npy'))
        HO[m] = zh(np.load(H/f'nn_{m}_holdout.npy'))
# strength gate: mean single-member train-only score
print('family single-member train-only means (val / holdout) and family mixes:')
st = {}
for f, ms in FAM.items():
    sv = np.mean([ev(V[(m,'to')]).mean() for m in ms]); sh = np.mean([eh(HO[m]).mean() for m in ms]); st[f] = (sv, sh)
bv = max(s[0] for s in st.values()); bh = max(s[1] for s in st.values())
gated = [f for f in FAM if st[f][0] >= bv - 0.002 and st[f][1] >= bh - 0.002]
for f in FAM: print(f'  {f:5s} n={len(FAM[f])} single to val {st[f][0]:.5f} hold {st[f][1]:.5f}  gate={"PASS" if f in gated else "fail"}')
def fam_mix_v(ms): return 0.5*zv(np.mean([V[(m,'ft')] for m in ms],0)) + 0.5*zv(np.mean([V[(m,'to')] for m in ms],0))
def fam_h(ms): return zh(np.mean([HO[m] for m in ms],0))
def fam_equal(fs): return sum(zv(fam_mix_v(FAM[f])) for f in fs), sum(fam_h(FAM[f]) for f in fs)
def mem_equal(fs):
    ms = [m for f in fs for m in FAM[f]]; return fam_mix_v(ms), fam_h(ms)
C = {'Ca {F5}': fam_equal(['F5']), 'Cb {F5,F6}': fam_equal(['F5','F6']), 'Cc {F5,F6,F7}': fam_equal(['F5','F6','F7']),
     f'Cd member-equal {"+".join(gated)}': mem_equal(gated)}
ref_v = ev(C['Ca {F5}'][0]); ref_h = eh(C['Ca {F5}'][1])
tue = (fv['timestamp'].to_numpy()+10800)//86400 < 20720
print('\ncandidates: holdout (selector) | val vs Ca with SE and slices')
res = {}
for k, (sv, sh) in C.items():
    pv, ph = ev(sv), eh(sh); d = pv-ref_v; dh = ph-ref_h; res[k] = ph.mean()
    sl = ' '.join(f'{n} {d[m].mean():+.4f}({d[m].mean()/max(se(d[m]),1e-9):+.1f})' for n, m in [('M<50',Mv<50),('M>=50',Mv>=50),('Tue',tue),('Thu',~tue)])
    print(f'  {k:40s} hold {ph.mean():.5f} ({dh.mean():+.5f}+-{se(dh):.5f}) | val {pv.mean():.5f} d {d.mean():+.5f}+-{se(d):.5f} z {d.mean()/se(d):+.2f} | {sl}')
best = max(res, key=res.get); print('\nholdout-selected candidate:', best)
