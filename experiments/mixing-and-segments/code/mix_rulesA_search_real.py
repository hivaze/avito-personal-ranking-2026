# Runs the nested rule search of lib.py on the four real member compositions and prints, for every
# rule family, the in-sample gain, the nested gain over the plain weights and over the fixed
# feed-size rule.
# How it was run: on CPU from lab/nn_x/w12/mix/rulesA after repro.py had written the NDCG tables;
# the command line is not on record. Output: logs_search_real.txt and res_C*.pkl.
# What it produced: 100 families, 95 eligible, 131,547 rules. The first composition is copied in
# results/mix_rule_search_all_data_16.txt.
# Kept as it ran: paths and names refer to the lab tree (see lab/ and docs/glossary.md).
# Original: lab/nn_x/w12/mix/rulesA/search_real.py
"""Step 3: the declared nested search on the four real compositions (tables N_C*.npy from repro.py). Output res_C*.pkl + a text table."""
import time, pickle; t0 = time.time()
from lib import *
f, M, off = meta('val'); pr = props('val'); gt = gates(pr); parts = partitions(gt); fams = families(); FL = [folds(pr['uid'], s) for s in SEEDS]
print('families', len(fams), 'eligible', sum(x[5] for x in fams), 'rules', sum(len(x[3]) for x in fams), '| fold sizes seed 11:', np.bincount(FL[0]), flush=True)
# user-group check: no user (non-0) straddles folds
for fo in FL:
    u = pr['uid']; nz = u != 0; o = np.argsort(u[nz], kind='stable'); uu = u[nz][o]; ff = fo[nz][o]; same = uu[1:] == uu[:-1]; assert (ff[1:][same] == ff[:-1][same]).all()
for c in COMPS:
    N = np.load(HERE / f'N_{c}.npy'); r = search(N, parts, fams, FL, want_vec=True)
    plain = N[:, PLAIN]; mr = np.where(M >= 50, N[:, cidx(A04, A04, C05)], plain); r['plain_vec'] = plain; r['mrule_vec'] = mr
    pickle.dump(r, open(HERE / f'res_{c}.pkl', 'wb'))
    iM = r['names'].index('C(M50)'); vM = r['vec'][iM]
    print(f'\n=== {c} {COMPS[c]} plain {plain.mean():.5f} | fixed M-rule - plain {fmt(mr - plain)} | {time.time() - t0:.0f}s')
    print(f'{"family":18s} par rules | in-sample   | nested - plain        z    seeds(min..max)      | nested - MRULE        | nested - nested C(M50) | best rule on all val (a_ft, a_to, c per cell)')
    order = np.argsort(-r['nested'])
    for fi in order:
        name, key, ncell, R, npar, el, form = fams[fi]; v = r['vec'][fi]; d = v - plain; m_, s_ = se(d); sd_ = r['nested_seed'][:, fi] - plain.mean()
        print(f'{name:18s} {npar}  {len(R):5d} | {r["ins"][fi] - plain.mean():+.5f}    | {fmt(d)}  {m_ / max(s_, 1e-12):+5.2f}  {sd_.min():+.5f}..{sd_.max():+.5f} | {fmt(v - mr)} | {fmt(v - vM)}  | '
              + ' ; '.join(str(decode(x)) for x in r['ins_rule'][fi]) + ('' if el else '   [DIAGNOSTIC]'))
    for vname, vv in r['var'].items():
        d = vv['vec'] - plain; m_, s_ = se(d); sd_ = vv['nested_seed'] - plain.mean()
        from collections import Counter
        ch = Counter(x for l in vv['chosen'] for x in l).most_common(6)
        print(f'VARIANT {vname:6s}: nested - plain {fmt(d)} z {m_ / s_:+.2f} seeds {sd_.min():+.5f}..{sd_.max():+.5f} | - MRULE {fmt(vv["vec"] - mr)} | - nested C(M50) {fmt(vv["vec"] - vM)} | chosen (of 25): {ch}')
print('done', time.time() - t0)
