"""Systematic (seed-averaged) function difference arm vs reference, net of seed noise (stored preds, CPU).
Members unit-scaled within feed (z). With z_Xi = f_X + e_Xi:  E|z_Xi - z_Rj|^2 (i != j) = |f_X - f_R|^2 + s2_X + s2_R,  s2_X = E|z_Xi - z_Xj|^2 / 2.
delta^2 = mean_{i!=j} |z_Xi - z_Rj|^2 - s2_X - s2_R (mean squared distances per candidate). Null check: ref seeds {951,952} vs {953,954}."""
import itertools, sys, numpy as np, polars as pl
sys.path.insert(0, '<repo>/lab/nn_x/w12/moe/why/mech')
from mech_common import FLAT, W5, pred_paths
Mv = pl.read_parquet(FLAT / 'val/feeds.parquet')['M'].to_numpy().astype(np.int64)
Mh = np.load(W5 / 'preds/holdout/holdout_meta.npy')[0].astype(np.int64)
ARMS = dict(ref='kd05w192', A='kd05w192moeffn', B='kd05w192moevo', C='kd05w192b3ff1')
for sp, M in (('val', Mv), ('hold', Mh)):
    fid = np.repeat(np.arange(len(M)), M)
    def z(a, s):
        pv, ph = pred_paths(f'w5_e10_{ARMS[a]}_s{s}'); x = np.load(pv if sp == 'val' else ph).astype(np.float64)
        c = x - (np.bincount(fid, x, len(M)) / M)[fid]; return c / np.sqrt((c ** 2).mean())
    Z = {(a, s): z(a, s) for a in ARMS for s in (951, 952, 953, 954)}
    d2 = lambda p, q: float(((Z[p] - Z[q]) ** 2).mean())
    def s2(members): return np.mean([d2(p, q) for p, q in itertools.combinations(members, 2)]) / 2
    ref = [('ref', s) for s in (951, 952, 953, 954)]
    for a in ('A', 'B', 'C'):
        X = [(a, s) for s in (951, 952, 953, 954)]
        cross = np.mean([d2(p, q) for p in X for q in ref if p[1] != q[1]])
        same = np.mean([d2(p, q) for p in X for q in ref if p[1] == q[1]])
        dd = cross - s2(X) - s2(ref)
        print(f'{sp} {a}: seed-noise sd per member {np.sqrt(s2(X)):.3f} (ref {np.sqrt(s2(ref)):.3f}); cross-seed arm-ref RMS {np.sqrt(cross):.3f}; '
              f'same-seed arm-ref RMS {np.sqrt(same):.3f}; systematic delta^2 {dd:+.5f} -> delta {np.sqrt(max(dd, 0)):.3f} (units: within-feed logit sd)')
    g1, g2 = ref[:2], ref[2:]
    cross = np.mean([d2(p, q) for p in g1 for q in g2]); dd = cross - d2(g1[0], g1[1]) / 2 - d2(g2[0], g2[1]) / 2
    print(f'{sp} NULL ref{{951,952}} vs ref{{953,954}}: systematic delta^2 {dd:+.5f}')
