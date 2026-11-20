"""Step 6: the declared single hypotheses h1-h4 (DECLARATION block 7). Fixed rules, full val, paired per-feed SE; no search."""
import sys; sys.path.insert(0, '<repo>/src')
from lib import *
import pmix
f, M, off = meta('val'); y = labels(); pr = props('val'); W = pmix.weights(); n = len(M); ar = np.arange(n)
big = M >= 50; AI = lambda a: int(round(a * 10)); CI = lambda c: int(round((c - 0.3) * 10))
def vec(N, a_ft, a_to, c):
    a_ft = np.broadcast_to(np.asarray(a_ft), n); a_to = np.broadcast_to(np.asarray(a_to), n); c = np.broadcast_to(np.asarray(c), n)
    return N[ar, cidx(a_ft, a_to, c)]
def line(tag, d):
    m, s = se(d); return f'{tag}: {fmt(d)} (z {m / s:+.2f}) | pmix-public {fmt(d, W["public"])} private {fmt(d, W["private"])} | M<50 {fmt(d[~big])} M>=50 {fmt(d[big])}'
NT = {c: np.load(HERE / f'N_{c}.npy') for c in COMPS}
aM = np.where(big, AI(0.4), AI(0.2)); a2 = AI(0.2); c5 = CI(0.5)
print('cells: CS>=.45', (pr['CS'] >= .45).mean().round(4), 'CL>=.45', (pr['CL'] >= .45).mean().round(4), '| M<30', (M < 30).mean().round(4), '30-99', ((M >= 30) & (M < 100)).mean().round(4), '>=100', (M >= 100).mean().round(4))
for c in COMPS:
    N = NT[c]; plain = vec(N, a2, a2, c5); mr = vec(N, aM, aM, c5)
    print(f'\n=== {c} {COMPS[c]}: plain {plain.mean():.5f} | MRULE - plain {fmt(mr - plain)}')
    for tag, cold in (('h1a CS>=.45', pr['CS'] >= .45), ('h1b CL>=.45', pr['CL'] >= .45)):
        cc = np.where(cold, CI(0.7), c5)
        print(line(f' {tag} c .7/.5 on a=.2  minus PLAIN', vec(N, a2, a2, cc) - plain) + f' | cold cell only {fmt((vec(N, a2, a2, cc) - plain)[cold])}')
        print(line(f' {tag} c .7/.5 on MRULE minus MRULE', vec(N, aM, aM, cc) - mr) + f' | cold cell only {fmt((vec(N, aM, aM, cc) - mr)[cold])}')
    print(line(' (ref, not a hypothesis) c = .7 everywhere on a=.2 minus PLAIN', vec(N, a2, a2, CI(0.7)) - plain))
    to = vec(N, a2, aM, c5); ftv = vec(N, aM, a2, c5)
    print(line(' h3-TO (M-rule in the train-only branch only) minus PLAIN', to - plain)); print(line(' h3-TO minus MRULE', to - mr))
    print(line(' h3-FT (M-rule in the fine-tuned branch only) minus PLAIN', ftv - plain)); print(line(' h3-FT minus MRULE', ftv - mr))
    print(f'   share of the MRULE gain carried: TO-only {(to - plain).mean() / (mr - plain).mean():.2f}  FT-only {(ftv - plain).mean() / (mr - plain).mean():.2f}')
    c3 = (M >= 30).astype(int) + (M >= 100)
    for tag, w3 in (('h4a (.2,.3,.4)', (0.2, 0.3, 0.4)), ('h4b (.1,.3,.5)', (0.1, 0.3, 0.5))):
        a3 = np.array([AI(x) for x in w3])[c3]; v = vec(N, a3, a3, c5)
        print(line(f' {tag} minus PLAIN', v - plain)); print(line(f' {tag} minus MRULE', v - mr) + ' | by cell M<30 / 30-99 / >=100: ' + ' / '.join(fmt((v - mr)[c3 == k]) for k in range(3)))
# h2: 8-member popularity-light pools, what exists now
print('\n=== h2: M-rule 0.2 / 0.5 vs 0.2 / 0.4 with 8 popularity-light members')
cache = {}
H2 = [('W16 + 4 wide + 4 d128 (W16P20x8)', 'W16', MEMBERS['P1w'] + MEMBERS['P1d']), ('W8 + 4 wide + 4 d128 (W8P20x8)', 'W8', MEMBERS['P1w'] + MEMBERS['P1d']),
      ('FD16 + 8 wide full-data (FD16P20w8)', 'FD16', MEMBERS['FDPw8']), ('FD16 + 4 wide + 4 d128 full-data (FD16P20x8)', 'FD16', MEMBERS['FDP'] + MEMBERS['FDPd']),
      ('FD8 + 8 wide full-data', 'FD8', MEMBERS['FDPw8']), ('FD8 + 4 wide + 4 d128 full-data', 'FD8', MEMBERS['FDP'] + MEMBERS['FDPd'])]
a5 = np.where(big, AI(0.5), AI(0.2))
for tag, b, pop in H2:
    miss = [m for m in pop for s in SUF['val'] if not (P / f'nn_{m}_{s}.npy').exists()]
    if miss: print(f' {tag}: NOT READ, files missing for {sorted(set(miss))}'); continue
    for m in MEMBERS[b] + pop:
        if (m, 'val') not in cache: cache[(m, 'val')] = member_z(m, 'val')
    N = table(pool(MEMBERS[b], 'val', cache) + pool(pop, 'val', cache), y); plain = vec(N, a2, a2, c5); m4 = vec(N, aM, aM, c5); m5 = vec(N, a5, a5, c5)
    print(f' {tag}: plain {plain.mean():.5f}'); print(line('    rule .2/.4 minus PLAIN', m4 - plain)); print(line('    rule .2/.5 minus PLAIN', m5 - plain)); print(line('    rule .2/.5 minus rule .2/.4', m5 - m4))
# the same contrast on the 4-member pools for reference (prior exposure [D]: lead's dose line)
for c in COMPS:
    N = NT[c]; print(line(f' (ref, 4-member pool) {c} rule .2/.5 minus rule .2/.4', vec(N, a5, a5, c5) - vec(N, aM, aM, c5)))
