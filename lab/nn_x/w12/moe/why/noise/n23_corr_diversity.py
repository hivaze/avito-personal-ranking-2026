"""why/noise N2 + N3.
N2: per-feed correlation of the three screen deltas dX = pool(X 951-954) - pool(Ra) (X = A, B, C), val and holdout, and the same with the
    shared reference broken (A vs Ra, B vs Rb, C vs Rbar70 etc.); share of per-feed delta variance that is the reference's own term.
N3: ensemble diversity per 4-group: mean pairwise Pearson correlation of member z (row-weighted; equals the M-weighted mean within-feed
    correlation because z is standardised per feed), ambiguity (K-1)/K (1 - rho), pool gain = pool NDCG - mean member NDCG (paired per
    feed), compared with the 70 reference 4-subsets (seed-level spread) and with historical w192 arms; same-seed vs cross-seed correlation
    (does sharing the seed / data order make an arm member a near copy of the reference member?)."""
import itertools, json
from nz_common import *

m = meta()
out = []
P = lambda *a: (print(*a, flush=True), out.append(' '.join(str(x) for x in a)))
REF = [nm('kd05w192', s) for s in S8]
HIST8 = ['kd05t25w192', 'kd08w192', 'kwwd0', 'kwwd01', 'kwwd05', 'kwaux0', 'kd05w192ps']
HIST4 = ['kd05w224', 'kd05w192gru', 'kdmixw192', 'kwaux1', 'kwd2', 'kwh30', 'kwh6', 'kwr32', 'kwsb2', 'kwwd10', 'kw256d2', 'kd05g2w192', 'kd03g2w192']
MEM = REF + [nm(a, s) for a in ARMS.values() for s in S4A] + [nm(a, s) for a in HIST8 for s in S8] + [nm(a, s) for a in HIST4 for s in S4A]
ix = {v: i for i, v in enumerate(MEM)}
# ---- correlation matrices (row-weighted Pearson of per-feed z; z has per-feed mean 0 / sd 1, so corr = sum z_i z_j / sum 1)
C = {}
for sp, k in (('val', 0), ('hold', 1)):
    X = np.stack([Z(v)[k] for v in MEM]).astype(np.float32)
    G = (X @ X.T).astype(np.float64) / X.shape[1]
    dg = np.sqrt(np.diag(G)); C[sp] = G / dg[:, None] / dg[None, :]
    del X
np.savez(OUT / 'n3_corr.npz', val=C['val'], hold=C['hold'], names=np.array(MEM))
rho = lambda sp, vs: float(np.mean([C[sp][ix[a], ix[b]] for a, b in itertools.combinations(vs, 2)]))

# ================= N2
P('== N2: per-feed correlation of the screen deltas (pool 951-954 minus a reference pool), paired per feed')
Ra, Rb = [REF[i] for i in range(4)], [REF[i] for i in range(4, 8)]
ra, rb = pool_ndcg(Ra), pool_ndcg(Rb)
ap = {k: pool_ndcg([nm(a, s) for s in S4A]) for k, a in ARMS.items()}
subs = list(itertools.combinations(range(8), 4))
rbar = [np.mean([pool_ndcg([REF[i] for i in c])[k] for c in subs], 0) for k in (0, 1)]
for k, sp in ((0, 'val'), (1, 'holdout')):
    dA, dB, dC = ap['A'][k] - ra[k], ap['B'][k] - ra[k], ap['C'][k] - ra[k]
    c = np.corrcoef([dA, dB, dC])
    P(f'  {sp:7s} shared Ra    : corr(dA,dB) {c[0, 1]:+.3f} corr(dA,dC) {c[0, 2]:+.3f} corr(dB,dC) {c[1, 2]:+.3f} | sd dA {dA.std():.4f} dB {dB.std():.4f} dC {dC.std():.4f}')
    eA, eB = ap['A'][k] - ra[k], ap['B'][k] - rb[k]
    fA, fC = ap['A'][k] - rb[k], ap['C'][k] - ra[k]
    gB, gC = ap['B'][k] - rb[k], ap['C'][k] - rb[k]
    P(f'  {sp:7s} split refs   : corr(A-Ra, B-Rb) {np.corrcoef(eA, eB)[0, 1]:+.3f} corr(A-Rb, C-Ra) {np.corrcoef(fA, fC)[0, 1]:+.3f} corr(B-Rb... A-Ra) n/a'
      f' | corr(A-Rb, B-Rb) {np.corrcoef(fA, gB)[0, 1]:+.3f} corr(B-Rb, C-Rb) {np.corrcoef(gB, gC)[0, 1]:+.3f}')
    # arm-arm direct correlation (no reference at all): corr(A - mean of the other two?) -> report corr of arm pool NDCG residuals vs Rbar70
    hA, hB, hC = ap['A'][k] - rbar[k], ap['B'][k] - rbar[k], ap['C'][k] - rbar[k]
    c2 = np.corrcoef([hA, hB, hC])
    P(f'  {sp:7s} vs Rbar70    : corr(dA,dB) {c2[0, 1]:+.3f} corr(dA,dC) {c2[0, 2]:+.3f} corr(dB,dC) {c2[1, 2]:+.3f}')
    rr = ra[k] - rbar[k]
    P(f'  {sp:7s} reference term (Ra - Rbar70): per-feed var share in dA {rr.var() / dA.var():.2f} dB {rr.var() / dB.var():.2f} dC {rr.var() / dC.var():.2f};'
      f' mean {rr.mean():+.5f}+-{se(rr):.5f} = share of the mean delta: A {-rr.mean() / dA.mean():+.2f} B {-rr.mean() / dB.mean():+.2f} C {-rr.mean() / dC.mean():+.2f}')
# member-level deltas (paired by seed), per seed
P('  member-level (mean member NDCG) deltas vs the same-seed reference, per seed (val / holdout):')
for k_, a in ARMS.items():
    rows = []
    for s in S4A:
        mv_, mh_ = member_ndcg(nm(a, s)); rv_, rh_ = member_ndcg(nm('kd05w192', s))
        rows.append(f's{s} {mv_.mean() - rv_.mean():+.5f}/{mh_.mean() - rh_.mean():+.5f}')
    P(f'    {k_}: ' + ' | '.join(rows))
P('  reference member NDCG per seed (val / holdout): ' + ' | '.join(f's{s} {member_ndcg(v)[0].mean():.5f}/{member_ndcg(v)[1].mean():.5f}' for s, v in zip(S8, REF)))

# ================= N3
P('')
P('== N3: diversity and pool gain per 4-group (selected checkpoints, member-equal z pools)')


def group_stats(vs):
    r = {}
    for k, sp, spn in ((0, 'val', 'val'), (1, 'hold', 'holdout')):
        mem = np.mean([member_ndcg(v)[k] for v in vs], 0); pl = pool_ndcg(vs)[k]
        r[spn] = dict(rho=rho(sp, vs), mem=mem, pool=pl, gain=pl - mem)
    return r


groups = {'Ra(ref 951-954)': Ra, 'Rb(ref 955-958)': Rb}
groups.update({f'{k}:{a}': [nm(a, s) for s in S4A] for k, a in ARMS.items()})
for a in HIST8:
    groups[f'{a} a-half'] = [nm(a, s) for s in S4A]; groups[f'{a} b-half'] = [nm(a, s) for s in S4B]
for a in HIST4:
    groups[a] = [nm(a, s) for s in S4A]
GS = {g: group_stats(vs) for g, vs in groups.items()}
RS = [group_stats([REF[i] for i in c]) for c in subs]
res = {}
for spn in ('val', 'holdout'):
    rg = np.array([x[spn]['gain'].mean() for x in RS]); rr = np.array([x[spn]['rho'] for x in RS])
    P(f'  [{spn}] 70 reference 4-subsets: gain mean {rg.mean():+.5f} sd {rg.std(ddof=1):.5f} [min {rg.min():+.5f} max {rg.max():+.5f}]; rho mean {rr.mean():.4f} sd {rr.std(ddof=1):.4f}'
      f' [min {rr.min():.4f} max {rr.max():.4f}]; corr(gain, 1-rho) over subsets {np.corrcoef(rg, 1 - rr)[0, 1]:+.2f}')
    gA = GS['Ra(ref 951-954)'][spn]['gain']
    P(f'  [{spn}] {"group":28s} {"rho":>7s} {"ambig":>7s} {"member":>8s} {"pool":>8s} {"gain":>9s}  gain-gain(Ra) paired  | gain rank among 70 ref subsets')
    rows = {}
    for g in groups:
        x = GS[g][spn]; dg = x['gain'] - gA
        rk = int((rg < x['gain'].mean()).sum())
        P(f'  [{spn}] {g:28s} {x["rho"]:.4f} {0.75 * (1 - x["rho"]):.4f} {x["mem"].mean():.5f} {x["pool"].mean():.5f} {x["gain"].mean():+.5f}  {fmt(dg)}'
          f'  | {rk}/70 below')
        rows[g] = dict(rho=x['rho'], mem=float(x['mem'].mean()), pool=float(x['pool'].mean()), gain=float(x['gain'].mean()), gain_se=se(x['gain']),
                       dgain_vs_Ra=float(dg.mean()), dgain_vs_Ra_se=se(dg))
    # cross-group regression gain ~ rho (reference subsets + every group): is a low gain explained by high member similarity?
    allr = np.array([GS[g][spn]['rho'] for g in groups]); allg = np.array([GS[g][spn]['gain'].mean() for g in groups])
    sl, ic = np.polyfit(rr, rg, 1)
    P(f'  [{spn}] linear fit over the 70 ref subsets: gain = {ic:+.4f} + {sl:+.4f} * rho; predicted gain at the arms\' rho: '
      + ', '.join(f'{k}: pred {ic + sl * GS[f"{k}:{a}"][spn]["rho"]:+.5f} obs {GS[f"{k}:{a}"][spn]["gain"].mean():+.5f}' for k, a in ARMS.items()))
    sl2, ic2 = np.polyfit(allr, allg, 1)
    P(f'  [{spn}] linear fit over all {len(groups)} groups (arms + halves): gain = {ic2:+.4f} + {sl2:+.4f} * rho; corr {np.corrcoef(allr, allg)[0, 1]:+.2f}')
    res[spn] = dict(ref70_gain=rg.tolist(), ref70_rho=rr.tolist(), groups=rows)
# ---- seed-level spread of a 4-pool gain: 35 disjoint split differences
for spn in ('val', 'holdout'):
    rg = {c: x[spn]['gain'].mean() for c, x in zip(subs, RS)}
    dd = np.array([rg[c] - rg[tuple(sorted(set(range(8)) - set(c)))] for c in subs if 0 in c])
    P(f'  [{spn}] seed-level sd of a 4-pool gain (35 disjoint splits, rms(diff)/sqrt2): {np.sqrt((dd ** 2).mean() / 2):.5f}')
# ---- same-seed vs cross-seed correlation with the reference member
P('')
P('== same-seed vs cross-seed member correlation (Pearson of per-feed z; val / holdout)')
for spn in ('val', 'hold'):
    cc = C[spn]
    rr_ = [cc[ix[REF[i]], ix[REF[j]]] for i, j in itertools.combinations(range(8), 2)]
    P(f'  [{spn}] ref-ref (different seeds, 28 pairs): mean {np.mean(rr_):.4f} sd {np.std(rr_, ddof=1):.4f} min {np.min(rr_):.4f} max {np.max(rr_):.4f}')
    for lab, arms_, seeds in [(k, [a], S4A) for k, a in ARMS.items()] + [(a, [a], S8) for a in HIST8]:
        a = arms_[0]
        same = [cc[ix[nm(a, s)], ix[nm('kd05w192', s)]] for s in seeds]
        cross = [cc[ix[nm(a, s)], ix[nm('kd05w192', t)]] for s in seeds for t in seeds if t != s]
        within = [cc[ix[nm(a, s)], ix[nm(a, t)]] for s, t in itertools.combinations(seeds, 2)]
        P(f'  [{spn}] {lab:12s}: arm-ref same seed {np.mean(same):.4f} | arm-ref cross seed {np.mean(cross):.4f} (same-cross {np.mean(same) - np.mean(cross):+.4f})'
          f' | arm-arm within {np.mean(within):.4f} [{np.min(within):.4f}, {np.max(within):.4f}]')
    # within-reference halves
    wa = [cc[ix[REF[i]], ix[REF[j]]] for i, j in itertools.combinations(range(4), 2)]
    wb = [cc[ix[REF[i]], ix[REF[j]]] for i, j in itertools.combinations(range(4, 8), 2)]
    xab = [cc[ix[REF[i]], ix[REF[j]]] for i in range(4) for j in range(4, 8)]
    P(f'  [{spn}] ref within Ra {np.mean(wa):.4f} within Rb {np.mean(wb):.4f} across Ra-Rb {np.mean(xab):.4f}')
json.dump(res, open(OUT / 'n23_corr_diversity.json', 'w'))
open(OUT / 'n23_corr_diversity.txt', 'w').write('\n'.join(out) + '\n')
