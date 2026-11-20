"""ADDENDUM 3 read: T2 = mean of the nested gain on C1 and C3, against the matched placebo PL-A2."""
import pickle, sys; sys.path.insert(0, '<repo>/src')
from lib import *
import pmix
fams = families(); names = [x[0] for x in fams]; elig = np.array([x[5] for x in fams]); iM = names.index('C(M50)'); W = pmix.weights()
R1 = pickle.load(open(HERE / 'res_C1.pkl', 'rb')); R3 = pickle.load(open(HERE / 'res_C3.pkl', 'rb')); Pz = np.load(HERE / 'pl_PLA2.npz'); nd = Pz['nested'].shape[1]
pv = lambda obs, null: (1 + (null >= obs).sum()) / (1 + len(null))
t2n = Pz['nested'].mean(0)                     # [draw, fam]
vec = lambda R, fi: R['vec'][fi] - R['plain_vec']
T2 = np.array([(vec(R1, fi).mean() + vec(R3, fi).mean()) / 2 for fi in range(len(fams))])
print(f'PL-A2: {nd} matched draws. placebo T2 pooled over eligible families: mean {t2n[:, elig].mean():+.5f}; per-family sd over draws: median {np.median(t2n[:, elig].std(0)):.5f} (single member set: {np.median(Pz["nested"][0][:, elig].std(0)):.5f} FD16, {np.median(Pz["nested"][1][:, elig].std(0)):.5f} W16)')
print(f'correlation over draws of the FD16 and W16 placebo nested gains (per family, median): {np.median([np.corrcoef(Pz["nested"][0][:, i], Pz["nested"][1][:, i])[0, 1] for i in np.flatnonzero(elig)]):.2f}')
d2 = (vec(R1, iM) + vec(R3, iM)) / 2; m, s = se(d2)
print(f'C(M50): T2 {m:+.5f}+-{s:.5f} (paired SE of the two-set mean; z {m / s:+.2f}) | placebo mean {t2n[:, iM].mean():+.5f} sd {t2n[:, iM].std():.5f} q95 {np.quantile(t2n[:, iM], .95):+.5f} | p2 {pv(m, t2n[:, iM]):.3f}'
      f' | pmix-public {fmt(d2, W["public"])} private {fmt(d2, W["private"])}')
fm = ((R1['mrule_vec'] - R1['plain_vec']) + (R3['mrule_vec'] - R3['plain_vec'])) / 2; m, s = se(fm); pm = Pz['mrule'].mean(0)
print(f'fixed MRULE: T2 {m:+.5f}+-{s:.5f} (z {m / s:+.2f}) | the same fixed rule on the placebo pools: mean {pm.mean():+.5f} sd {pm.std():.5f} max {pm.max():+.5f} | p2 {pv(m, pm):.3f} (the rule was chosen on these feeds: optimistic)'
      f' | pmix-public {fmt(fm, W["public"])} private {fmt(fm, W["private"])}')
gm = t2n[:, elig].max(1); print(f'family-wise: max over 95 families of placebo T2 per draw: mean {gm.mean():+.5f} q95 {np.quantile(gm, .95):+.5f} | observed max {T2[elig].max():+.5f} ({names[int(np.argmax(np.where(elig, T2, -9)))]}) p {pv(T2[elig].max(), gm):.3f}')
dd = t2n - t2n[:, [iM]]; gd = dd[:, elig].max(1); od = T2 - T2[iM]
print(f'family-wise for "beats C(M50)": max over families of placebo [T2(fam) - T2(C(M50))] per draw: mean {gd.mean():+.5f} q95 {np.quantile(gd, .95):+.5f} | observed max {od[elig].max():+.5f} ({names[int(np.argmax(np.where(elig, od, -9)))]}) p {pv(od[elig].max(), gd):.3f}')
for v in ('B-all', 'B-3', 'G'):
    o = ((R1['var'][v]['vec'] - R1['plain_vec']) + (R3['var'][v]['vec'] - R3['plain_vec'])) / 2; x = Pz['var_' + v].mean(0); print(f'variant {v}: T2 {fmt(o)} | placebo mean {x.mean():+.5f} sd {x.std():.5f} | p2 {pv(o.mean(), x):.3f}')
print(f'{"family":16s} par | T2 (paired SE)        z   | C1       C3      | placebo T2: mean    sd      q95     p2   | T2 - T2 C(M50)  placebo diff q95  p2,M | T2 - fixed MRULE')
order = np.argsort(-np.where(elig, T2, -9))[:int(sys.argv[1]) if len(sys.argv) > 1 else 25]
for fi in list(order) + [i for i in (iM, names.index('A')) if i not in order]:
    d2 = (vec(R1, fi) + vec(R3, fi)) / 2; m, s = se(d2)
    print(f'{names[fi]:16s} {fams[fi][4]}   | {m:+.5f}+-{s:.5f} {m / s:+5.2f} | {vec(R1, fi).mean():+.5f} {vec(R3, fi).mean():+.5f} | {t2n[:, fi].mean():+.5f} {t2n[:, fi].std():.5f} {np.quantile(t2n[:, fi], .95):+.5f} {pv(m, t2n[:, fi]):.3f} | {od[fi]:+.5f}        {np.quantile(dd[:, fi], .95):+.5f}        {pv(od[fi], dd[:, fi]):.3f} | {fmt(d2 - fm)}')
