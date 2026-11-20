"""Read of the sign-flip placebo (placebo_a.npz + placebo_b.npz). Row 0 of each array = the real data (fold seed 7001), rows 1.. = null draws."""
import sys
sys.path.insert(0, '<repo>/nn_x/w12/mix/review')
import numpy as np
from pathlib import Path
HERE = Path('<repo>/nn_x/w12/mix/review')
a = np.load(HERE / 'placebo_a.npz', allow_pickle=True); b = np.load(HERE / 'placebo_b.npz', allow_pickle=True); names = list(a['names'])
X = {k: np.concatenate([a[k][:1], a[k][1:], b[k][1:]]) for k in a.files if k != 'names'}
assert all(np.allclose(a[k][0], b[k][0]) for k in X); N = len(X['nest|C1']) - 1; print(f'null draws N = {N}; real row identical in both processes')
p = lambda obs, null: (1 + (null >= obs - 1e-15).sum()) / (1 + len(null))
ix = {nm: i for i, nm in enumerate(names)}
KEYS = ('C1', 'C2', 'C3', 'C4')
print('\n=== SANITY of my nested code under the sign-flip null: mean over draws of the nested gain (must be 0 within 3 SE of the mean) ===')
worst = 0
for k in KEYS:
    nl = X[f'nest|{k}'][1:]; m = nl.mean(0); se = nl.std(0) / np.sqrt(N); z = m / se; worst = max(worst, np.abs(z).max())
    print(f'  {k}: ' + ' '.join(f'{nm} {m[i]:+.6f} (z {z[i]:+.1f})' for i, nm in enumerate(names)))
for k in KEYS:
    for h in ('H3', 'H3a'):
        nl = X[f'{h}|{k}'][1:, 0]; z = nl.mean() / (nl.std() / np.sqrt(N)); worst = max(worst, abs(z)); print(f'  {h} {k}: null mean {nl.mean():+.6f} (z {z:+.1f})', end=' |')
print(f'\n  worst |z| = {worst:.2f}  -> {"PASS" if worst <= 3 else "FAIL"}')
print('\n=== in-sample optimism under the null: mean of (in-sample best) and q95, per family (C1) ===')
print('  ' + ' | '.join(f'{nm} {X["ins|C1"][1:, i].mean():+.5f} (q95 {np.quantile(X["ins|C1"][1:, i], .95):+.5f})' for i, nm in enumerate(names)))
print('\n=== P1: nested gain over PLAIN (fold seed 7001) against the null: obs [null sd] p ===')
for k in KEYS:
    print(f'  {k}: ' + ' | '.join(f'{nm} {X[f"nest|{k}"][0, i]:+.5f} [{X[f"nest|{k}"][1:, i].std():.5f}] p {p(X[f"nest|{k}"][0, i], X[f"nest|{k}"][1:, i]):.3f}' for i, nm in enumerate(names)))
T2 = 0.5 * (X['nest|C1'] + X['nest|C3']); T4 = sum(X[f'nest|{k}'] for k in KEYS) / 4; TF = 0.5 * (X['nest|C1'] + X['nest|C2'])
print('  T2 (C1, C3): ' + ' | '.join(f'{nm} {T2[0, i]:+.5f} p {p(T2[0, i], T2[1:, i]):.3f}' for i, nm in enumerate(names)))
print('  mean C1..C4: ' + ' | '.join(f'{nm} {T4[0, i]:+.5f} p {p(T4[0, i], T4[1:, i]):.3f}' for i, nm in enumerate(names)))
print('  full-data mean (C1, C2): ' + ' | '.join(f'{nm} {TF[0, i]:+.5f} p {p(TF[0, i], TF[1:, i]):.3f}' for i, nm in enumerate(names)))
print('\n=== P2 family-wise: max over my families of the nested gain ===')
for lab, sel in (('all 12 of mine', names), ('A-grid families only (A, C(M50), C(M70), F(Hq), G(M3q), K(M70,Hq))', names[:6])):
    idx = [ix[s] for s in sel]
    for k in ('C1', 'C3'):
        mx = X[f'nest|{k}'][:, idx].max(1); print(f'  {lab} on {k}: observed max {mx[0]:+.5f} ({sel[int(X[f"nest|{k}"][0, idx].argmax())]}) | null max mean {mx[1:].mean():+.5f} q95 {np.quantile(mx[1:], .95):+.5f} | family-wise p {p(mx[0], mx[1:]):.3f}')
    mx = T2[:, idx].max(1); print(f'  {lab} on T2: observed max {mx[0]:+.5f} ({sel[int(T2[0, idx].argmax())]}) | null q95 {np.quantile(mx[1:], .95):+.5f} | family-wise p {p(mx[0], mx[1:]):.3f}')
print('\n=== P3: nested (family - nested C(M50)) against the null ===')
for nm in ('F(Hq)', 'HSTUMP', 'K(M70,Hq)', 'G(M3q)', 'A', 'CONSTp'):
    i, j = ix[nm], ix['C(M50)']; line = f'  {nm}: '
    for k in KEYS:
        d = X[f'nest|{k}'][:, i] - X[f'nest|{k}'][:, j]; line += f'{k} {d[0]:+.5f} p {p(d[0], d[1:]):.3f} | '
    d = T2[:, i] - T2[:, j]; line += f'T2 {d[0]:+.5f} p {p(d[0], d[1:]):.3f} | '; d = T4[:, i] - T4[:, j]; line += f'mean C1..C4 {d[0]:+.5f} p {p(d[0], d[1:]):.3f}'
    print(line)
print('\n=== CHECK 5: M-rule under the null. H1 = nested MSTEP50, H3 = cross-set nested, HEAD = (H1 + H3) / 2, fixed = fixed rule (no selection in the null) ===')
HD = {}
for k in KEYS:
    h1 = X[f'nest|{k}'][:, ix['MSTEP50']]; h3 = X[f'H3|{k}'][:, 0]; hd = 0.5 * (h1 + h3); HD[k] = hd; fx = X[f'fix|{k}'][:, 0]; h1a = X[f'nest|{k}'][:, ix['C(M50)']]; h3a = X[f'H3a|{k}'][:, 0]; hda = 0.5 * (h1a + h3a); HD[k + 'a'] = hda
    print(f'  {k}: H1 {h1[0]:+.5f} p {p(h1[0], h1[1:]):.3f} | H3 {h3[0]:+.5f} p {p(h3[0], h3[1:]):.3f} | HEAD {hd[0]:+.5f} [null sd {hd[1:].std():.5f}] p {p(hd[0], hd[1:]):.3f} | A-grid H1 {h1a[0]:+.5f} p {p(h1a[0], h1a[1:]):.3f} H3 {h3a[0]:+.5f} p {p(h3a[0], h3a[1:]):.3f} HEAD {hda[0]:+.5f} p {p(hda[0], hda[1:]):.3f}'
          f' | fixed {fx[0]:+.5f} [null sd {fx[1:].std():.5f}] p {p(fx[0], fx[1:]):.3f}')
for lab, ks in (('full-data (C1, C2)', ('C1', 'C2')), ('twins (C3, C4)', ('C3', 'C4')), ('16-base (C1, C3)', ('C1', 'C3')), ('all four', KEYS)):
    hd = sum(HD[k] for k in ks) / len(ks); hda = sum(HD[k + 'a'] for k in ks) / len(ks); fx = sum(X[f'fix|{k}'][:, 0] for k in ks) / len(ks)
    print(f'  mean {lab}: HEAD {hd[0]:+.5f} [null sd {hd[1:].std():.5f}] p {p(hd[0], hd[1:]):.3f} | A-grid HEAD {hda[0]:+.5f} p {p(hda[0], hda[1:]):.3f} | fixed {fx[0]:+.5f} [null sd {fx[1:].std():.5f}] p {p(fx[0], fx[1:]):.3f}')
fx = sum(X[f'fix|{k}'][:, 0] for k in ('HA', 'HB', 'WA', 'WB')) / 4; print(f'  fixed rule, mean of the 4 member-disjoint halves: {fx[0]:+.5f} [null sd {fx[1:].std():.5f}] p {p(fx[0], fx[1:]):.3f} (no selection in this null)')
fx = sum(X[f'fix|{k}'][:, 0] for k in ('HA', 'HB')) / 2; print(f'  fixed rule, mean of the 2 full-data halves: {fx[0]:+.5f} [null sd {fx[1:].std():.5f}] p {p(fx[0], fx[1:]):.3f}')
print('\n=== P4 ORIGIN EMULATION: 75-rule winner selected in-sample on one member set, read on the other (same feeds, same signs) ===')
W5 = (0, .1, .2, .3, .4); GN = ('M>=50', 'cold>=.30', 'cold>=.45'); dec = lambda c: round(0.05 * (int(c) // 13), 2)
q = X['P4|C3sel']; print(f'  real: winner on C3 = gate {GN[int(q[0, 4])]} weights {dec(q[0, 5])} / {dec(q[0, 6])}, in-sample {q[0, 0]:+.5f}; its read on C1 {q[0, 1]:+.5f}, C2 {q[0, 2]:+.5f}, C4 {q[0, 3]:+.5f}')
print(f'  null: winner in-sample mean {q[1:, 0].mean():+.5f} (q95 {np.quantile(q[1:, 0], .95):+.5f}); transferred read on C1 mean {q[1:, 1].mean():+.5f} sd {q[1:, 1].std():.5f} q95 {np.quantile(q[1:, 1], .95):+.5f}; on C2 mean {q[1:, 2].mean():+.5f} sd {q[1:, 2].std():.5f}'
      f' q95 {np.quantile(q[1:, 2], .95):+.5f}; on C4 (shares the 4 popularity-light members) mean {q[1:, 3].mean():+.5f} q95 {np.quantile(q[1:, 3], .95):+.5f}')
f1, f2 = X['fix|C1'][0, 0], X['fix|C2'][0, 0]
print(f'  observed fixed M-rule on C1 {f1:+.5f}: selection-adjusted p {p(f1, q[1:, 1]):.3f}, minus null mean = {f1 - q[1:, 1].mean():+.5f} | on C2 {f2:+.5f}: p {p(f2, q[1:, 2]):.3f}, minus null mean = {f2 - q[1:, 2].mean():+.5f}')
print(f'  share of null draws whose winner uses the M gate: {(q[1:, 4] == 0).mean():.2f}; winner read on its own set (optimism) vs transferred: ratio of means {q[1:, 1].mean() / q[1:, 0].mean():.2f} (C1), {q[1:, 2].mean() / q[1:, 0].mean():.2f} (C2)')
q = X['P4|C1sel']; print(f'  mirror: winner on C1 = gate {GN[int(q[0, 4])]} weights {dec(q[0, 5])} / {dec(q[0, 6])}, in-sample {q[0, 0]:+.5f}; read on C3 {q[0, 1]:+.5f} (null mean {q[1:, 1].mean():+.5f}, p {p(q[0, 1], q[1:, 1]):.3f}), C4 {q[0, 2]:+.5f} (p {p(q[0, 2], q[1:, 2]):.3f}), C2 {q[0, 3]:+.5f}')
q = X['P5|C34sel']; fo = 0.5 * (f1 + f2)
print(f'  P5: winner on mean(C3, C4) = gate {GN[int(q[0, 2])]} weights {dec(q[0, 3])} / {dec(q[0, 4])}, in-sample {q[0, 0]:+.5f}; read on mean(C1, C2) {q[0, 1]:+.5f}; null transferred mean {q[1:, 1].mean():+.5f} sd {q[1:, 1].std():.5f} q95 {np.quantile(q[1:, 1], .95):+.5f}'
      f' | observed fixed M-rule on mean(C1, C2) {fo:+.5f}: selection-adjusted p {p(fo, q[1:, 1]):.3f}; minus null mean {fo - q[1:, 1].mean():+.5f}')
q = X['P5|C12sel']; print(f'  P5 mirror: winner on mean(C1, C2) = gate {GN[int(q[0, 2])]} weights {dec(q[0, 3])} / {dec(q[0, 4])}, in-sample {q[0, 0]:+.5f}; read on mean(C3, C4) {q[0, 1]:+.5f} (null mean {q[1:, 1].mean():+.5f}, p {p(q[0, 1], q[1:, 1]):.3f})')
