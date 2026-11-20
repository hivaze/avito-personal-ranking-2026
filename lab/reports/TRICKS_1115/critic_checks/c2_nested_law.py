"""CRITIC check 2 (CPU, read-only): nested and law checks behind the plan's slot arguments. Written before any of its numbers was read; every look is printed.
A  w_ft: curve of FD16P20 and of the mean of four disjoint 8-member wide pools; NESTED pick (5 user-grouped folds x 40 assignments) against w_ft 0.5.
B  blend bonus of a z-blend of two files built from DISJOINT members, against the mean of the two (6 pairs of 12-member files + the synthesis's BL).
C  ENSA against the expected FD16P20-type file (mean over all 495 draws of 4 of the 12 all-data wide pop1 students).
D  spread of the mean member pair correlation (mechanism clause of the weight-decay-0 rule).
E  anchored public forecast, leave-one-out over the nine scored files: with the val residual term, without it, nearest scored file only.
F  the probe's trigger: probability under three truths."""
import sys, itertools, json
sys.path.insert(0, '<repo>/reports/TRICKS_1115/critic_checks')
from critic_lib import *
from math import erf, sqrt
L = []
def w_(s=''):
    print(s, flush=True); L.append(s)
Phi = lambda x: 0.5 * (1 + erf(x / sqrt(2)))
rng = np.random.default_rng(20261115)
# ---------------- A
w_('## A. w_ft (delta against w_ft 0.5, paired per feed)')
WS = [0.0, 0.25, 0.35, 0.5, 0.6, 0.65, 0.75, 0.85, 1.0]
FDP = [f'abl_fd10_pop1kd05w192_s{s}' for s in S4]
POOLS = {'FD16P20': branches(wide('fd10', S8), FDP), 'W8P20': branches(wide('e10', S4), [f'abl_e10_pop1kd05t25_s{s}' for s in S4]),
         'e10 s951-954': branches(wide('e10', S4)), 'e10 s955-958': branches(wide('e10', S4b)), 'fd10 s951-954': branches(wide('fd10', S4)), 'fd10 s955-958': branches(wide('fd10', S4b))}
CUR = {k: np.stack([ndcg(fin(b, wft=w)) for w in WS], 1) for k, b in POOLS.items()}
DEL = {k: c - c[:, [WS.index(0.5)]] for k, c in CUR.items()}
DEL['mean of 4 disjoint 8-pools'] = np.mean([DEL[k] for k in ('e10 s951-954', 'e10 s955-958', 'fd10 s951-954', 'fd10 s955-958')], 0)
for k in ('FD16P20', 'W8P20', 'mean of 4 disjoint 8-pools'):
    w_(f'{k}:'); [w_(f'   w_ft {w:.2f}: {three(DEL[k][:, i])}') for i, w in enumerate(WS) if w != 0.5]
def nested(D, crit_w, reps=40):
    res = []
    for r in range(reps):
        fo = user_folds(rng); oos = np.zeros(NV); picks = []
        for k in range(5):
            tr = fo != k; cw = np.ones(NV) if crit_w is None else crit_w
            j = int(np.argmax((D[tr] * cw[tr, None]).sum(0) / cw[tr].sum())); picks.append(WS[j]); oos[~tr] = D[~tr, j]
        res.append((oos.mean(), np.average(oos, weights=WPUB), np.average(oos, weights=WPRIV), np.mean(picks)))
    a = np.array(res); return a.mean(0), a.std(0)
w_('NESTED pick of w_ft among the 9 weights (out-of-sample delta against 0.5; mean and sd over 40 fold assignments):')
for k in ('FD16P20', 'mean of 4 disjoint 8-pools'):
    for cn, cw in (('plain val', None), ('pmix-private', WPRIV)):
        m, s = nested(DEL[k], cw); w_(f'   {k} | picked by {cn}: mean picked w_ft {m[3]:.2f} | out-of-sample val {m[0]:+.5f} ({s[0]:.5f}) | pmix-public {m[1]:+.5f} ({s[1]:.5f}) | pmix-private {m[2]:+.5f} ({s[2]:.5f})')
i6 = WS.index(0.6); pos = [int(wm(DEL[k][:, i6], WPRIV)[0] > 0) for k in ('e10 s951-954', 'e10 s955-958', 'fd10 s951-954', 'fd10 s955-958')]
w_(f'   the four pools at 0.60, pmix-private one by one: ' + ', '.join(f'{k} {f3(wm(DEL[k][:, i6], WPRIV))}' for k in ('e10 s951-954', 'e10 s955-958', 'fd10 s951-954', 'fd10 s955-958')))
# ---------------- B
w_(); w_('## B. z-blend of two files built from disjoint members, minus the mean of the two files (per feed)')
F12 = {'heldout A (e10 s951-954 + e10 pop d128)': branches(wide('e10', S4), [f'abl_e10_pop1kd05t25_s{s}' for s in S4]),
       'heldout B (e10 s955-958 + e10 pop wide)': branches(wide('e10', S4b), [f'abl_e10_pop1kd05w192_s{s}' for s in S4]),
       'alldata A (fd10 s951-954 + fd pop s951-954)': branches(wide('fd10', S4), FDP),
       'alldata B (fd10 s955-958 + fd pop s955-958)': branches(wide('fd10', S4b), [f'abl_fd10_pop1kd05w192_s{s}' for s in S4b])}
ZF = {k: fin(b) for k, b in F12.items()}; EF = {k: ndcg(v) for k, v in ZF.items()}
for k in F12: w_(f'   file {k}: final val {EF[k].mean():.5f}')
bon = []
for a, b in itertools.combinations(F12, 2):
    e = ndcg(0.5 * z(ZF[a]) + 0.5 * z(ZF[b])); d = e - 0.5 * (EF[a] + EF[b]); bon.append(d)
    w_(f'   blend({a.split(" (")[0]}, {b.split(" (")[0]}) - mean: {three(d)} | sd per feed {d.std():.5f}')
mb = np.mean(bon, 0); w_(f'   MEAN over the 6 disjoint pairs: {three(mb)}')
zW = stored('W8P20kd'); zF = stored('FD16P20kd'); eW, eF = ndcg(zW), ndcg(zF); eBL = ndcg(0.5 * z(zW) + 0.5 * z(zF)); dBL = eBL - 0.5 * (eW + eF)
w_(f'   BL = blend(W8P20, FD16P20) - mean: {three(dBL)} | sd per feed {dBL.std():.5f} -> public sampling SE {dBL.std() / np.sqrt(NPUB):.5f} | BL final val {eBL.mean():.5f}')
w_(f'   BL minus the 6-pair mean bonus (is BL an outlier of the same quantity?): {three(dBL - mb)}')
base = (0.3950094 + 0.394892) / 2
for nm, bonus in (('synthesis: BL val residual at the public mix', wm(dBL, WPUB)[0]), ('zero bonus', 0.0), ('mean bonus of the 6 disjoint pairs at the public mix', wm(mb, WPUB)[0])):
    sd = dBL.std() / np.sqrt(NPUB); f = base + bonus; w_(f'   BL public forecast = mean of the two public scores {base:.6f} + bonus [{nm}] {bonus:+.6f} = {f:.6f} +- {sd:.6f} -> P(> 0.3950094) {1 - Phi((0.3950094 - f) / sd):.2f}')
# ---------------- C
w_(); w_('## C. ENSA against the expected FD16P20-type file (all 495 draws of 4 of the 12 all-data wide pop1 students on the FD16 base)')
P12 = [f'abl_fd10_pop1kd05w192_s{s}' for s in range(951, 963)]; bFT, bTO = pool(wide('fd10', S8)); zb = (z(bFT), z(bTO))
acc = np.zeros(NV); lv = []
for c in itertools.combinations(range(12), 4):
    p = pool([P12[i] for i in c]); e = ndcg(fin((0.8 * zb[0] + 0.2 * z(p[0]), 0.8 * zb[1] + 0.2 * z(p[1])))); acc += e; lv.append(e.mean())
exp = acc / len(lv); lv = np.array(lv); eA = ndcg(stored('ENSAkd', d=R / 'nn_x/w12/ens_1115/stage')); eB = ndcg(stored('ENSBkd', d=R / 'nn_x/w12/ens_1115/stage'))
w_(f'   495 draws: mean final val {lv.mean():.5f}, sd {lv.std():.5f}, min {lv.min():.5f}, max {lv.max():.5f}; realised FD16P20 {eF.mean():.5f} is above {(lv < eF.mean() - 1e-9).sum()} of 495')
w_(f'   FD16P20 minus the expected draw: {three(eF - exp)}')
w_(f'   ENSA minus the expected draw:    {three(eA - exp)}')
w_(f'   ENSB minus the expected draw:    {three(eB - exp)}')
w_(f'   W8P20 minus the expected draw:   {three(eW - exp)}')
# ---------------- D
w_(); w_('## D. mean pairwise correlation of member z (train-only, val), reference recipe kd05w192: spread over seed subsets')
def cm(names): 
    Z = np.stack([mz(m)[1] for m in names]); return np.corrcoef(Z)
ref8 = [f'w5_e10_kd05w192_s{s}' for s in S8]; C8 = cm(ref8); iu = np.triu_indices(8, 1)
w_(f'   kd05w192 s951-958: mean pair correlation {C8[iu].mean():.4f}, sd over the 28 pairs {C8[iu].std():.4f}, min {C8[iu].min():.4f}, max {C8[iu].max():.4f}')
sub = [np.mean([C8[i, j] for i, j in itertools.combinations(c, 2)]) for c in itertools.combinations(range(8), 4)]
w_(f'   over the 70 subsets of 4 seeds: mean {np.mean(sub):.4f}, sd {np.std(sub):.4f}, range {np.min(sub):.4f} .. {np.max(sub):.4f}; s951-954 {np.mean([C8[i, j] for i, j in itertools.combinations(range(4), 2)]):.4f}, s955-958 {np.mean([C8[i, j] for i, j in itertools.combinations(range(4, 8), 2)]):.4f}')
t8 = [f'w5_e10_kd05t25w192_s{s}' for s in S8]; f8 = [f'w5_fd10_kd05w192_s{s}' for s in S8]
for nm, nn in (('kd05t25w192 s951-958 (held-out)', t8), ('fd10 kd05w192 s951-958 (all data)', f8)):
    c = cm(nn); w_(f'   {nm}: mean pair correlation {c[iu].mean():.4f}')
# jackknife SE of the 8-seed mean pair correlation (leave one seed out)
jk = np.array([np.mean([C8[i, j] for i, j in itertools.combinations([k for k in range(8) if k != d], 2)]) for d in range(8)])
w_(f'   leave-one-seed-out jackknife SE of the 8-seed mean: {np.sqrt(7 / 8 * ((jk - jk.mean()) ** 2).sum()):.4f}')
# ---------------- E
w_(); w_('## E. anchored public forecast, leave-one-out over the nine scored files (own code)')
SC = {'W8': ('W8kd', 0.5, 0.394486), 'W16': ('W16kd', 0.5, 0.39448337), 'W8P20': ('W8P20kd', 0.5, 0.3950094), 'W8TO': ('W8kd', 0.0, 0.394009), 'S16': ('S16kd', 0.5, 0.3940782),
      'FD16P20': ('FD16P20kd', 0.5, 0.394892), 'FD16P20m': ('FD16P20_m02_04kd', 0.5, 0.3947465), 'C3': ('Q2a8s3', 0.5, 0.393454), 'G3': ('G3', 0.5, 0.392820)}
E = {k: ndcg(stored(t, wft=w)) for k, (t, w, _) in SC.items()}
w_('   final val rebuilt: ' + ', '.join(f'{k} {E[k].mean():.5f}' for k in SC))
def fc(c, an, mode):
    A = np.stack([E[a] for a in an], 1); p = np.array([SC[a][2] for a in an])
    if mode == 'nearest':   # the scored file with the smallest per-feed sd of the difference, plus the pmix-public delta
        j = int(np.argmin([(c - A[:, i]).std() for i in range(A.shape[1])])); return p[j] + wm(c - A[:, j], WPUB)[0], p[j]
    a0 = A[:, 0]; X = A[:, 1:] - a0[:, None]; yy = c - a0; b = np.linalg.lstsq(X - X.mean(0), yy - yy.mean(), rcond=None)[0]; w = np.concatenate([[1 - b.sum()], b])
    r = c - A @ w; return p @ w + wm(r, WPUB)[0], p @ w
er = {'anchored with the val residual (the synthesis form)': [], 'anchored weights only (residual set to 0)': [], 'nearest scored file + pmix-public delta': [], 'nearest scored file only': [], 'level line 0.0074 + 0.963 val': []}
for k in SC:
    an = [a for a in SC if a != k]; f1, f0 = fc(E[k], an, 'ls'); n1, n0 = fc(E[k], an, 'nearest'); pub = SC[k][2]
    er['anchored with the val residual (the synthesis form)'].append(pub - f1); er['anchored weights only (residual set to 0)'].append(pub - f0)
    er['nearest scored file + pmix-public delta'].append(pub - n1); er['nearest scored file only'].append(pub - n0); er['level line 0.0074 + 0.963 val'].append(pub - (0.0074 + 0.963 * E[k].mean()))
for nm, e in er.items():
    e = np.array(e); w_(f'   {nm}: RMS error {np.sqrt((e ** 2).mean()):.6f} | mean {e.mean():+.6f} | per file ' + ' '.join(f'{x:+.5f}' for x in e))
KD = [i for i, k in enumerate(SC) if k in ('W8P20', 'FD16P20', 'FD16P20m')]
w_('   same, only the three files with the popularity-light admixture (W8P20, FD16P20, FD16P20m): ' + ' | '.join(f'{nm.split(" (")[0]}: RMS {np.sqrt((np.array(e)[KD] ** 2).mean()):.6f}' for nm, e in er.items()))
# ---------------- F
w_(); w_('## F. the probe trigger (reading >= W8 - 0.00005 = 0.39444; paired public SE of the probe against W8 from check 1: 0.00052)')
se = 0.00052
for nm, d in (('val reading -0.00078', -0.00078), ('pmix-public reading -0.00072', -0.00072), ('scan forecast with the fold term -0.00051', -0.00051), ('anchored forecast -0.00096', 0.393528 - 0.394486), ('true delta -0.00033 (the smallest loss at which 0.60 is at least as good as 0.50 on a quadratic through the three W8 points)', -0.00033), ('true delta 0', 0.0)):
    w_(f'   truth = {nm}: P(trigger) {1 - Phi((-0.00005 - d) / se):.2f}')
d0 = 0.394009 - 0.394486
for d1 in (-0.00005, 0.0, +0.0003):
    post = (d1 * 0.00058 ** 2 + (-0.00051) * se ** 2) / (0.00058 ** 2 + se ** 2); b = post - d0; a = 2 * (d0 + post); wopt = 0.5 - b / (2 * a); g6 = a * 0.01 + b * 0.1
    w_(f'   probe reads W8 {d1:+.5f}: posterior delta of the fine-tuned-only file {post:+.5f} (prior -0.00051 sd 0.00058 [R int_ft 3.4]); quadratic through the W8 points: optimum {wopt:.2f}, gain of 0.60 over 0.50 {g6:+.6f}')
open(OUT / 'c2_nested_law.txt', 'w').write('\n'.join(L) + '\n')
