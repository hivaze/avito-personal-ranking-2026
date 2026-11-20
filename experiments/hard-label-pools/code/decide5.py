# Decision rule written at 2026-11-12 15:20 UTC, after the public board had scored the val-best pool
# M2 (0.392434) below G3 (0.392820), for pools with 12 new from-scratch members. Score S = min(V,
# V - D): V is the val delta against the 38-member pool PF7x and D is the part of the train-only
# val delta that the holdout does not confirm. A pool must beat S of the 20-member pool G3.
# How it was run: CPU, 2026-11-12, output saved as decide5_out.txt; `decide5.py refonly` tests the
# code on the reference pools.
# What it produced: results/decide5_out.txt: no pool with the new members beat G3 on S (best
# +0.00007 against +0.00027), so nothing was submitted and G3 stayed pick 1.
# Kept as it ran: paths and names refer to the lab tree (see lab/ and docs/glossary.md).
# Original: lab/nn_x/picks/decide5.py
"""PRE-DECLARED 2026-11-12 15:20 UTC (CURRENT_STATUS Log), before any number of a pool containing NEW members existed:
tonight's 2 submissions. See the Log entry for the rule; this file implements it exactly.
NEW = complete from-scratch m02 members (v27_gru_m02_*, v28_lr1_m02_*, v29_d192_m02_*, v30_f7_m02_*), each passing
member gate (a): single train-only val >= 0.39467 AND holdout >= 0.31799.
Candidates N1 = G3+NEW, N2 = F10+F7x+NEW, N3 = PF7x+NEW, N4 = F10+NEW (only if |NEW| >= 12); references PF7x, G3, M2.
Proxies vs PF7x: V (final 0.5z(ft)+0.5z(to) val), Vto (train-only val), H (holdout, train-only), D = Vto - H,
S = min(V, V - D). Eligible: H >= -1 SE, no val slice < -2 SE (M<50, M>=50, Tue, Thu, coldpos, warmpos), presub PASS
(presub is run after this script on the chosen files). A = max S among eligible N*, B = next; each must beat S(G3)."""
import sys, glob, re; sys.path.insert(0, '<repo>/src')
from common import *
from metrics import ndcg_per_feed
from stack_val import feed_rank_z
P = Path('<repo>/preds'); H = P / 'holdout'
FAM = {
 'F5cd': ['v15_cd_all_e10','v15_cd_all_s81','v15_cd_all_s82','v15_cd_allprew_s83','v15_cd_allprew_s84','v15_cd_allprew_s85'],
 'F6': ['v13_lr1_mix1','v14_s92','v14_s93','v14_s94','v14_s95','v14_s96'], 'F9': [f'v22_pre2s_lr1_s{s}' for s in range(401,407)],
 'F10': [f'v23_gru_pre2s_s{s}' for s in range(501,507)], 'F12': [f'v26_d192_pre2s_s{s}' for s in range(701,708)],
 'F7x': [f'r6_cd_m02_r{s}' for s in range(101,108)], 'F7x70': [f'r6_cd_m02_p70k_r{s}' for s in range(101,108)],
}
have = lambda m: all(p.exists() for p in [P/f'nn_{m}_n2cpv3_val_oof.npy', P/f'nn_{m}_n2cpv3_test.npy', P/f'nn_{m}_trainonly_val.npy', P/f'nn_{m}_trainonly_test.npy', H/f'nn_{m}_holdout.npy'])
cand_new = sorted({re.sub(r'^nn_|_trainonly_val\.npy$', '', Path(f).name) for pat in ['v27_gru_m02_s*', 'v28_lr1_m02_s*', 'v29_d192_m02_s*', 'v30_f7_m02_s*'] for f in glob.glob(str(P/f'nn_{pat}_trainonly_val.npy'))})
cand_new = [m for m in cand_new if '@' not in m and have(m)]
if 'refonly' in sys.argv: cand_new = []  # code test on the reference pools only (numbers already known)
fv, Mv, ov = feed_meta('val'); yv = np.load(FLAT/'val/cand_target.npy')
Mh, oh = np.load(H/'holdout_meta.npy'); Mh = Mh.astype(np.int64); oh = oh.astype(np.int64); yh = np.load(H/'holdout_y.npy')
zv = lambda x: feed_rank_z(ov, Mv, np.asarray(x, np.float32))[:,1]; zh = lambda x: feed_rank_z(oh, Mh, np.asarray(x, np.float32))[:,1]
ev = lambda s: ndcg_per_feed(s, yv, ov, Mv); eh = lambda s: ndcg_per_feed(s, yh, oh, Mh); se = lambda d: d.std()/np.sqrt(len(d))
V, HO = {}, {}
def load(m):
    if m not in HO:
        V[(m,'ft')] = zv(np.load(P/f'nn_{m}_n2cpv3_val_oof.npy')); V[(m,'to')] = zv(np.load(P/f'nn_{m}_trainonly_val.npy')); HO[m] = zh(np.load(H/f'nn_{m}_holdout.npy'))
NEW = []
print('NEW member gate (a): single to val >= 0.39467 and holdout >= 0.31799')
for m in cand_new:
    load(m); sv, sh = ev(V[(m,'to')]).mean(), eh(HO[m]).mean(); ok = sv >= 0.39467 and sh >= 0.31799
    print(f'  {m:22s} to val {sv:.5f} hold {sh:.5f} {"PASS" if ok else "fail"}'); NEW += [m] if ok else []
print(f'NEW eligible: {len(NEW)} {NEW}')
for f in FAM.values():
    for m in f: load(m)
def pool(ms):
    ft = zv(np.mean([V[(m,'ft')] for m in ms],0)); to = zv(np.mean([V[(m,'to')] for m in ms],0))
    return ev(0.5*ft + 0.5*to), ev(to), eh(zh(np.mean([HO[m] for m in ms],0)))
fam = lambda *fs: [m for f in fs for m in FAM[f]]
C = {'PF7x': fam('F5cd','F6','F7x','F9','F10','F12'), 'G3': fam('F10','F7x','F7x70'), 'M2': fam('F7x','F7x70')}
if NEW:
    C['N1=G3+NEW'] = fam('F10','F7x','F7x70') + NEW; C['N2=F10+F7x+NEW'] = fam('F10','F7x') + NEW; C['N3=PF7x+NEW'] = fam('F5cd','F6','F7x','F9','F10','F12') + NEW
    if len(NEW) >= 12: C['N4=F10+NEW'] = fam('F10') + NEW
R = {k: pool(ms) for k, ms in C.items()}
v0, t0, h0 = R['PF7x']
tue = (fv['timestamp'].to_numpy()+10800)//86400 < 20720
cold = np.nan_to_num(np.load('<repo>/nn_x/r6/feat_cache/val/it_hist_all_luo.npy').astype(np.float64), nan=0.0) == 0
fid = np.repeat(np.arange(len(Mv)), Mv); npos = np.bincount(fid, yv, len(Mv)); ncold = np.bincount(fid, yv*cold, len(Mv))
SL = [('M<50',Mv<50),('M>=50',Mv>=50),('Tue',tue),('Thu',~tue),('coldpos',(npos>0)&(ncold==npos)),('warmpos',(npos>0)&(ncold==0))]
out = {}
for k, (v, t, h) in R.items():
    dv, dt, dh = v - v0, t - t0, h - h0
    Vd, Vtd, Hd = dv.mean(), dt.mean(), dh.mean(); D = Vtd - Hd; S = min(Vd, Vd - D)
    zs = {n: dv[m].mean()/max(se(dv[m]),1e-9) for n, m in SL}
    elig = Hd >= -se(dh) and min(zs.values()) >= -2
    out[k] = dict(S=S, elig=elig, n=len(C[k]))
    print(f'{k:16s} n={len(C[k]):3d} val {v.mean():.5f} hold {h.mean():.5f} | V {Vd:+.5f}+-{se(dv):.5f} Vto {Vtd:+.5f} H {Hd:+.5f}+-{se(dh):.5f} D {D:+.5f} S {S:+.5f} | '
          + ' '.join(f'{n} {z:+.1f}' for n, z in zs.items()) + f' | {"eligible" if elig else "NOT eligible"}')
sg = out['G3']['S']
rank = sorted([k for k in out if k.startswith('N') and out[k]['elig']], key=lambda k: -out[k]['S'])
print(f'S(G3) = {sg:+.5f}; eligible N ranked by S: ' + ', '.join(f'{k} {out[k]["S"]:+.5f}' for k in rank))
pick = [k for k in rank if out[k]['S'] > sg][:2]
print('SUBMISSIONS (pending presub_gate PASS):', pick if pick else 'none beats G3 on S')
# output-only addition (17:00, no logic change): save member lists of every candidate + the picks for the build step
import json as _j
_j.dump({'picks': pick, 'members': C, 'S': {k: out[k]['S'] for k in out}, 'eligible': {k: bool(out[k]['elig']) for k in out}, 'NEW': NEW},
        open('/tmp/scratchpad/decide5_picks.json', 'w'), indent=1)
