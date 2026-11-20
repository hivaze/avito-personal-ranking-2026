"""ADDENDUM 1 (post hoc design): X-MODEL read. Rule fitted on member set SRC, read on member set TGT: xm-same (all val feeds), xm-nest (training users of SRC -> held-out users of TGT); POOLED: fitted on SRC+TGT training users."""
import pickle
from lib import *
f, M, off = meta('val'); pr = props('val'); gt = gates(pr); parts = partitions(gt); fams = families(); FL = [folds(pr['uid'], s) for s in SEEDS]; n = len(M); K = 5
names = [x[0] for x in fams]; elig = np.array([x[5] for x in fams]); NT = {c: np.load(HERE / f'N_{c}.npy') for c in COMPS}
def xread(Nsel, Ntgt):
    """returns (xm_same[fam], vec[fam, n] nested per-feed on TGT with rules fitted on Nsel training users)."""
    nf = len(fams); same = np.zeros(nf); vec = np.zeros((nf, n)); ar = np.arange(n)
    for si, fold in enumerate(FL):
        Ss = {k: block_sums(Nsel, fold, c, nc, K) for k, (c, nc) in parts.items()}
        for fi, (name, key, ncell, R, npar, el, form) in enumerate(fams):
            Sk = Ss[key]; Tall = Sk.sum(0); cell = parts[key][0]
            if si == 0:
                i, _ = best(Tall, R); same[fi] = Ntgt[ar, R[i][cell]].mean()
            comb = np.empty(n, np.int64)
            for k in range(K):
                i, _ = best(Tall - Sk[k], R); mk = fold == k; comb[mk] = R[i][cell[mk]]
            vec[fi] += Ntgt[ar, comb] / len(FL)
    return same, vec
out = {}
for src, tgt in (('C3', 'C1'), ('C1', 'C3'), ('C4', 'C2'), ('C2', 'C4')):
    R = pickle.load(open(HERE / f'res_{tgt}.pkl', 'rb')); plain = R['plain_vec']; mr = R['mrule_vec']
    same, vec = xread(NT[src], NT[tgt]); psame, pvec = xread(NT[src] + NT[tgt], NT[tgt])
    out[(src, tgt)] = dict(same=same - plain.mean(), nest=vec.mean(1) - plain.mean(), pool=pvec.mean(1) - plain.mean(), vec=vec, pvec=pvec)
    sel = np.array([x[5] and x[4] <= 3 for x in fams])
    print(f'\n=== SRC {src} -> TGT {tgt}: fixed MRULE - PLAIN on TGT {fmt(mr - plain)} | eligible 2-3 par families: mean own in-sample {(R["ins"] - R["plain"])[sel].mean():+.5f} xm-same {out[(src, tgt)]["same"][sel].mean():+.5f} xm-nest {out[(src, tgt)]["nest"][sel].mean():+.5f}'
          f' own nested {(R["nested"] - R["plain"])[sel].mean():+.5f} pool-nest {out[(src, tgt)]["pool"][sel].mean():+.5f} | shared-feed share = 1 - xm-nest / xm-same = {1 - out[(src, tgt)]["nest"][sel].mean() / out[(src, tgt)]["same"][sel].mean():.2f}')
    print(f'{"family":16s} par | own in-sample | xm-same  | xm-nest (paired SE)   | own nested | pool-nest (paired SE)   | pool-nest - MRULE')
    order = np.argsort(-(R['nested'] - R['plain']))
    show = [i for i in order[:14]] + [names.index(x) for x in ('C(M50)', 'C(Mq)', 'G(M3q)', 'G(M3r)', 'A', 'B', 'D(M50)', 'I(M50)', 'E(CS45)', 'E(CLq)', 'F(Hq)', 'K(Mq,Hq)', 'K(M50,Hq)')]
    seen = set()
    for fi in show:
        if fi in seen: continue
        seen.add(fi)
        print(f'{names[fi]:16s} {fams[fi][4]}   | {R["ins"][fi] - R["plain"]:+.5f}      | {out[(src, tgt)]["same"][fi]:+.5f} | {fmt(vec[fi] - plain)}    | {R["nested"][fi] - R["plain"]:+.5f}   | {fmt(pvec[fi] - plain)}      | {fmt(pvec[fi] - mr)}')
pickle.dump({k: {kk: vv for kk, vv in v.items()} for k, v in out.items()}, open(HERE / 'res_xmodel.pkl', 'wb'))
