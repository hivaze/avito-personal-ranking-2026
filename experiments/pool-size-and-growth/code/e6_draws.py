# Tests whether the val edge of one member draw says anything about new feeds. The reference file
# used the first 4 of 12 popularity-light students and its val score is above 472 of the 495
# possible 4-of-12 files. For random halves of the val users the script correlates, across the 495
# draws, the file level on one half with the level on the other half.
# How it was run: on CPU from lab/nn_x/w12/ens_1115; the command line is not on record. Output:
# out/e6_draws.log and out/draws.json.
# What it produced: split-half correlation -0.156 on val, -0.027 at the public feed mix and
# -0.005 at the private mix, so the expected transferable part of the edge is 0
# (results/ens_e6_draws.log).
# Kept as it ran: paths and names refer to the lab tree (see lab/ and docs/glossary.md).
# Original: lab/nn_x/w12/ens_1115/e6_draws.py
"""e6: is the val edge of a favourable member draw a property of the members or of the val sample?  (FD16P20 = FD16 + the first 4 of the 12 full-data wide pop1 students; its final val is
above 472 of the 495 possible 4-of-12 draws.)  Split-half transfer: for random halves of the val USERS, the correlation across the 495 draws of (file level on half 1) with (file level on half 2).
reliability of the full-val draw excess as a predictor of a NEW feed sample = 2 r / (1 + r) (Spearman-Brown); expected transferable part of a draw's excess = reliability x excess.
Same for the base: 300 random 8-of-16 draws of the full-data wide students (pop fixed at all 12), and for 16-of-32 draws of all wide students.  Output out/draws.json."""
from ens_common import *
import itertools, pmix
m = Meta(); st = Store(m); WTS = pmix.weights(); Z = lambda x: m.z('val', x); rng = np.random.default_rng(1118)
uid = m.uid.copy(); u0 = uid == 0; uid[u0] = -1 - np.arange(len(uid))[u0]; uu, uinv = np.unique(uid, return_inverse=True)
def transfer(E, tag, realised=None):
    """E: draws x feeds per-feed NDCG."""
    out = {}
    for wn, w in (('val', np.ones(E.shape[1])), ('public', WTS['public']), ('private', WTS['private'])):
        rs = []
        for _ in range(300):
            h = rng.random(len(uu)) < 0.5; a = h[uinv]; la = (E[:, a] * w[a]).sum(1) / w[a].sum(); lb = (E[:, ~a] * w[~a]).sum(1) / w[~a].sum(); rs.append(np.corrcoef(la, lb)[0, 1])
        r = float(np.mean(rs)); rel = 2 * r / (1 + r) if r > -1 else float('nan'); lv = (E * w).sum(1) / w.sum()
        out[wn] = dict(split_half_r=r, r_sd=float(np.std(rs)), reliability=rel, draw_sd=float(lv.std()), true_sd=float(lv.std() * np.sqrt(max(rel, 0))))
        s = f'{tag} [{wn}]: draw sd {lv.std():.5f} | split-half r {r:+.3f} (sd over splits {np.std(rs):.3f}) -> reliability {rel:+.3f} -> sd of the transferable part {lv.std() * np.sqrt(max(rel, 0)):.5f}'
        if realised is not None:
            ex = float(lv[realised] - lv.mean()); out[wn].update(realised_excess=ex, expected_transfer=ex * max(rel, 0)); s += f' | realised draw excess {ex:+.5f} -> expected on new feeds {ex * max(rel, 0):+.5f}'
        print(s, flush=True)
    return out
res = {}
# --- pop draws: 4 of 12 with base FD16
bft = Z(st.pool('ft_val', FD16)); bto = Z(st.pool('to_val', FD16)); XF = st.mat('ft_val', FDP12).astype(np.float64); XT = st.mat('to_val', FDP12).astype(np.float64)
combs = list(itertools.combinations(range(12), 4)); E = np.empty((len(combs), len(m.Mv)), np.float32)
for i, c in enumerate(combs):
    c = list(c); E[i] = m.ev(0.5 * Z(0.8 * bft + 0.2 * Z(XF[c].mean(0))) + 0.5 * Z(0.8 * bto + 0.2 * Z(XT[c].mean(0))))
res['pop 4 of 12 (base FD16)'] = transfer(E, 'pop 4 of 12 (base FD16)', realised=combs.index((0, 1, 2, 3)))
e12 = m.ev(0.5 * Z(0.8 * bft + 0.2 * Z(XF.mean(0))) + 0.5 * Z(0.8 * bto + 0.2 * Z(XT.mean(0)))); d = e12 - E.mean(0); print(f'FD16 + all 12 pop minus the mean 4-of-12 file: {fmt(d)}'); res['all12_minus_mean4'] = (float(d.mean()), se(d))
# --- base draws: 8 of 16 full-data wide (pop = all 12)
pf = Z(XF.mean(0)); pt = Z(XT.mean(0)); BF = st.mat('ft_val', FD16).astype(np.float64); BT = st.mat('to_val', FD16).astype(np.float64); E = np.empty((300, len(m.Mv)), np.float32)
for i in range(300):
    c = rng.choice(16, 8, replace=False); E[i] = m.ev(0.5 * Z(0.8 * Z(BF[c].mean(0)) + 0.2 * pf) + 0.5 * Z(0.8 * Z(BT[c].mean(0)) + 0.2 * pt))
res['base 8 of 16 (pop FDP12)'] = transfer(E, 'base 8 of 16 full-data wide (pop FDP12)')
# --- base draws: 16 of 32 wide (held-out + full-data), pop = all 12
B2F = st.mat('ft_val', W16 + FD16).astype(np.float64); B2T = st.mat('to_val', W16 + FD16).astype(np.float64)
for i in range(300):
    c = rng.choice(32, 16, replace=False); E[i] = m.ev(0.5 * Z(0.8 * Z(B2F[c].mean(0)) + 0.2 * pf) + 0.5 * Z(0.8 * Z(B2T[c].mean(0)) + 0.2 * pt))
res['base 16 of 32 (pop FDP12)'] = transfer(E, 'base 16 of 32 wide (pop FDP12)')
json.dump(res, open(OUT / 'draws.json', 'w'), indent=0)
