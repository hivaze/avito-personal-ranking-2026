"""Unit test: independent brute-force implementation of the nested read for a few families (explicit rule definitions from the feed properties), against lib.search results (res_C1.pkl)."""
import pickle, itertools
from lib import *
f, M, off = meta('val'); pr = props('val'); n = len(M); N = np.load(HERE / 'N_C1.npy'); R = pickle.load(open(HERE / 'res_C1.pkl', 'rb')); names = R['names']; ar = np.arange(n)
FL = [folds(pr['uid'], s) for s in SEEDS]; A = range(NA); C = range(NC)
g = {'M50': M >= 50, 'Mq': M >= 70, 'CS45': pr['CS'] >= .45, 'Hq': pr['H'] >= 190}
def brute(rules):
    """rules: list of per-feed combo-index arrays. returns in-sample best mean, nested mean (5 x 5 folds), first-max tie rule."""
    V = np.stack([N[ar, r] for r in rules])            # [rule, feed]
    ins = V.mean(1).max(); tot = 0.0
    for fold in FL:
        for k in range(5):
            tr = fold != k; i = int(np.argmax(V[:, tr].sum(1))); tot += V[i, ~tr].sum()
    return ins, tot / (n * len(FL))
tests = {
 'C(M50)': [np.where(g['M50'], cidx(a1, a1, C05), cidx(a0, a0, C05)) for a0 in A for a1 in A],
 'F(Hq)': [np.where(g['Hq'], cidx(a, a, c1), cidx(a, a, c0)) for c0 in C for c1 in C for a in A],
 'I(M50)': [np.where(g['M50'], cidx(f1, t1, C05), cidx(f0, t0, C05)) for f0 in A for f1 in A for t0 in A for t1 in A],
 'J(Mq,CS45)': [cidx(np.where(g['Mq'], a1, a0), np.where(g['Mq'], a1, a0), np.where(g['CS45'], c1, c0)) for a0 in A for a1 in A for c0 in C for c1 in C],
 'J(CS45,Mq)': [cidx(np.where(g['CS45'], a1, a0), np.where(g['CS45'], a1, a0), np.where(g['Mq'], c1, c0)) for a0 in A for a1 in A for c0 in C for c1 in C],
 'K(Mq,Hq)': [(lambda a: cidx(a, a, C05))(np.array(q)[2 * g['Mq'].astype(int) + g['Hq'].astype(int)]) for q in itertools.product(A, A, A, A)],
 'G(M3r)': [(lambda a: cidx(a, a, C05))(np.array(q)[(M >= 30).astype(int) + (M >= 100)]) for q in itertools.product(A, A, A)],
}
ok = True
for k, rules in tests.items():
    ins, nes = brute(rules); fi = names.index(k); d1 = abs(ins - R['ins'][fi]); d2 = abs(nes - R['nested'][fi]); d3 = abs(R['vec'][fi].mean() - R['nested'][fi]); good = max(d1, d2, d3) < 1e-12; ok &= good
    print(f'{k:12s} rules {len(rules):5d} | brute in-sample {ins - R["plain"]:+.6f} nested {nes - R["plain"]:+.6f} | lib {R["ins"][fi] - R["plain"]:+.6f} {R["nested"][fi] - R["plain"]:+.6f} | max abs diff {max(d1, d2, d3):.1e} {"OK" if good else "MISMATCH"}')
print('UNIT TEST', 'PASS' if ok else 'FAIL')
