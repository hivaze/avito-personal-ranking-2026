"""Placebo read-out (DECLARATION block 5 / 6): p-values of the observed nested quantities against PL-A (pseudo popularity-light pool + permuted roles) and PL-G (permuted gates)."""
import pickle, sys
from lib import *
fams = families(); names = [x[0] for x in fams]; elig = np.array([x[5] for x in fams]); iM = names.index('C(M50)'); iA = names.index('A')
pv = lambda obs, null: (1 + (null >= obs).sum()) / (1 + len(null))
def read(comp, pla_key, plg_key, top=30):
    R = pickle.load(open(HERE / f'res_{comp}.pkl', 'rb')); obs = R['nested'] - R['plain']; oins = R['ins'] - R['plain']
    A = np.load(HERE / f'pl_PLA_{pla_key}.npz'); nA = len(A['plain']); G = np.load(HERE / f'pl_PLG_{plg_key}.npz') if plg_key and (HERE / f'pl_PLG_{plg_key}.npz').exists() else None
    print(f'\n################ observed = {comp} {COMPS[comp]} | PL-A on {pla_key}: {nA} draws' + (f' | PL-G on {plg_key}: {len(G["plain"])} permutations' if G is not None else ''))
    an, ai = A['nested'], A['ins']
    print(f'PL-A whole-search calibration (gains over the draw\'s own PLAIN):')
    print(f'  nested, all {elig.sum()} eligible families pooled: mean {an[:, elig].mean():+.5f} (SE of the mean over draws {an[:, elig].mean(1).std() / np.sqrt(nA):.5f}); per-family sd over draws: median {np.median(an[:, elig].std(0)):.5f}')
    print(f'  in-sample, per family: mean over families of the mean optimism (in-sample minus nested) {(ai - an)[:, elig].mean():+.5f}; by number of parameters: ' + ', '.join(f'{k} par {(ai - an)[:, [i for i, x in enumerate(fams) if x[4] == k and x[5]]].mean():+.5f}' for k in (2, 3, 4)))
    gmax = ai[:, elig].max(1); print(f'  greedy in-sample best of all eligible rules (the "scan"): mean {gmax.mean():+.5f} sd {gmax.std():.5f} q05 {np.quantile(gmax, .05):+.5f} q50 {np.median(gmax):+.5f} q95 {np.quantile(gmax, .95):+.5f} | observed {oins[elig].max():+.5f} (p {pv(oins[elig].max(), gmax):.3f})')
    best_nested = an[:, elig].max(1); print(f'  best NESTED family per draw (max over {elig.sum()} families; the bias of picking the top row of a nested table): mean {best_nested.mean():+.5f} q95 {np.quantile(best_nested, .95):+.5f} | observed top row {obs[elig].max():+.5f} ({names[int(np.argmax(np.where(elig, obs, -9)))]}) p {pv(obs[elig].max(), best_nested):.3f}')
    for v in ('B-all', 'B-3', 'G'):
        x = A['var_' + v]; o = R['var'][v]['nested'] - R['plain']; print(f'  variant {v:6s}: placebo nested mean {x.mean():+.5f} sd {x.std():.5f} q95 {np.quantile(x, .95):+.5f} | observed {o:+.5f} p_A {pv(o, x):.3f}')
    x = A['mrule']; o = (R['mrule_vec'] - R['plain_vec']).mean(); print(f'  fixed rule 0.2 / 0.4 by M (not searched) on placebo pools: mean {x.mean():+.5f} sd {x.std():.5f} q95 {np.quantile(x, .95):+.5f} | observed {o:+.5f} p {pv(o, x):.3f}')
    print(f'{"family":16s} par | observed: in-sample  nested   | PL-A nested: mean     sd      q95    p_A   | PL-A in-sample mean | obs nested - C(M50)  PL-A diff: sd     q95    p_A,M' + ('  | obs nested - A   PL-G diff: mean    sd     q95    p_G' if G is not None else ''))
    order = np.argsort(-obs)
    rows = []
    for fi in order[:top].tolist() + [i for i in (iM, iA) if i not in order[:top]]:
        dn = an[:, fi] - an[:, iM]; od = obs[fi] - obs[iM]
        s = f'{names[fi]:16s} {fams[fi][4]}   | {oins[fi]:+.5f}  {obs[fi]:+.5f} | {an[:, fi].mean():+.5f} {an[:, fi].std():.5f} {np.quantile(an[:, fi], .95):+.5f} {pv(obs[fi], an[:, fi]):.3f} | {ai[:, fi].mean():+.5f}            | {od:+.5f}   {dn.std():.5f} {np.quantile(dn, .95):+.5f} {pv(od, dn):.3f}'
        if G is not None:
            dg = G['nested'][:, fi] - G['nested'][:, iA]; og = obs[fi] - obs[iA]; s += f'  | {og:+.5f}   {dg.mean():+.5f} {dg.std():.5f} {np.quantile(dg, .95):+.5f} {pv(og, dg):.3f}'
        print(s + ('' if elig[fi] else '  [DIAG]'))
    return R, A, G
if __name__ == '__main__':
    read('C1', 'FD16', 'C1', int(sys.argv[1]) if len(sys.argv) > 1 else 30)
    if (HERE / 'pl_PLA_W16.npz').exists(): read('C3', 'W16', 'C3' if (HERE / 'pl_PLG_C3.npz').exists() else None, int(sys.argv[1]) if len(sys.argv) > 1 else 30)
