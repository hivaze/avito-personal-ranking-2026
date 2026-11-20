"""Compact tables for the report (all numbers from res_C*.pkl, pl_*.npz, res_xmodel.pkl)."""
import pickle, sys; sys.path.insert(0, '<repo>/src')
from lib import *
import pmix
fams = families(); names = [x[0] for x in fams]; elig = np.array([x[5] for x in fams]); iM = names.index('C(M50)'); iA = names.index('A'); W = pmix.weights()
R = {c: pickle.load(open(HERE / f'res_{c}.pkl', 'rb')) for c in COMPS}; PA = np.load(HERE / 'pl_PLA_FD16.npz'); PW = np.load(HERE / 'pl_PLA_W16.npz'); PG = np.load(HERE / 'pl_PLG_C1.npz'); P2 = np.load(HERE / 'pl_PLA2.npz'); XM = pickle.load(open(HERE / 'res_xmodel.pkl', 'rb'))
pv = lambda obs, null: (1 + (null >= obs).sum()) / (1 + len(null)); vec = lambda c, fi: R[c]['vec'][fi] - R[c]['plain_vec']; t2n = P2['nested'].mean(0)
def nested_table(keys):
    out = ['| family | par | rules | C1 in-sample | C1 nested - plain (z) | p_A | C1 nested - fixed M-rule | C1 nested - nested C(M50) | p_A,M | p_G | C2 nested - plain | C3 nested - plain (p_A W16) | C4 nested - plain | twins->C1 other models + other users | T2 = mean(C1, C3) | p2 |',
           '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for k in keys:
        fi = names.index(k); d = vec('C1', fi); m, s = se(d); an = PA['nested']; dM = (R['C1']['vec'][fi] - R['C1']['vec'][iM]); dg = PG['nested'][:, fi] - PG['nested'][:, iA]
        d2 = (vec('C1', fi) + vec('C3', fi)) / 2; m2, s2 = se(d2); m3 = vec('C3', fi).mean()
        out.append(f'| {k}{"" if elig[fi] else " [diag]"} | {fams[fi][4]} | {len(fams[fi][3])} | {R["C1"]["ins"][fi] - R["C1"]["plain"]:+.5f} | {m:+.5f} +- {s:.5f} ({m / s:+.1f}) | {pv(m, an[:, fi]):.3f} | {fmt(R["C1"]["vec"][fi] - R["C1"]["mrule_vec"])} | '
                   + (f'{fmt(dM)} | {pv(dM.mean(), an[:, fi] - an[:, iM]):.3f}' if fi != iM else '0 | -') + f' | ' + ('-' if fams[fi][1] == ('one',) else f'{pv(m - vec("C1", iA).mean(), dg):.3f}') + f' | {fmt(vec("C2", fi))} | {fmt(vec("C3", fi))} ({pv(m3, PW["nested"][:, fi]):.3f}) | {fmt(vec("C4", fi))} | '
                   f'{fmt(XM[("C3", "C1")]["vec"][fi] - R["C1"]["plain_vec"])} | {m2:+.5f} +- {s2:.5f} | {pv(m2, t2n[:, fi]):.3f} |')
    return '\n'.join(out)
def variant_table():
    out = ['| variant | C1 nested - plain | p_A | C1 - fixed M-rule | C2 | C3 | C4 | families chosen most often on C1 (of 25 fits) |', '|---|---|---|---|---|---|---|---|']
    from collections import Counter
    for v in ('B-all', 'B-3', 'G'):
        x = R['C1']['var'][v]; d = x['vec'] - R['C1']['plain_vec']; ch = Counter(z for l in x['chosen'] for z in l).most_common(4)
        out.append(f'| {v} | {fmt(d)} | {pv(d.mean(), PA["var_" + v]):.3f} | {fmt(x["vec"] - R["C1"]["mrule_vec"])} | ' + ' | '.join(fmt(R[c]['var'][v]['vec'] - R[c]['plain_vec']) for c in ('C2', 'C3', 'C4')) + ' | ' + ', '.join(f'{a} x{b}' for a, b in ch) + ' |')
    return '\n'.join(out)
def fixed_table():
    out = ['| fixed M-rule (0.2 / 0.4 by M >= 50, w_ft 0.5) minus plain | C1 | C2 | C3 | C4 |', '|---|---|---|---|---|']
    d = {c: R[c]['mrule_vec'] - R[c]['plain_vec'] for c in COMPS}
    out.append('| val | ' + ' | '.join(fmt(d[c]) for c in COMPS) + ' |'); out.append('| pmix public | ' + ' | '.join(fmt(d[c], W['public']) for c in COMPS) + ' |'); out.append('| pmix private | ' + ' | '.join(fmt(d[c], W['private']) for c in COMPS) + ' |')
    out.append('| fixed M-rule minus nested C(M50) | ' + ' | '.join(fmt(R[c]['mrule_vec'] - R[c]['vec'][iM]) for c in COMPS) + ' |')
    out.append('| nested C(M50) minus plain, pmix public | ' + ' | '.join(fmt(vec(c, iM), W['public']) for c in COMPS) + ' |'); out.append('| nested C(M50) minus plain, pmix private | ' + ' | '.join(fmt(vec(c, iM), W['private']) for c in COMPS) + ' |')
    return '\n'.join(out)
def chosen_rules(k):
    """rules chosen for family k in the in-sample fit of each composition."""
    fi = names.index(k); return ' ; '.join(f'{c}: ' + ' | '.join('a %.1f' % decode(x)[0] if decode(x)[0] == decode(x)[1] and decode(x)[2] == 0.5 else str(decode(x)) for x in R[c]['ins_rule'][fi]) for c in COMPS)
if __name__ == '__main__':
    top = [names[i] for i in np.argsort(-np.where(elig, R['C1']['nested'], -9))[:8]]
    keys = top + [k for k in ('C(M50)', 'C(Mq)', 'G(M3q)', 'G(M3r)', 'D(M50)', 'I(M50)', 'A', 'B', 'E(CS45)', 'E(CLq)', 'E(Hq)', 'L(CL3q)', 'C(DAY)') if k not in top]
    print(nested_table(keys)); print(); print(variant_table()); print(); print(fixed_table()); print(); print('C(M50) in-sample rules:', chosen_rules('C(M50)')); print('G(M3r):', chosen_rules('G(M3r)')); print('A:', chosen_rules('A'))
