# Forecast of the public score of an unscored file from the nine scored files of the last two
# generations: weights by least squares on per-feed val NDCG (summing to one), plus the mean
# residual at the public feed mix. Checked by leaving each scored file out in turn.
# How it was run: CPU, 2026-11-15 about 22:10 UTC; the method was fixed in the docstring before
# any number was read; the command line is not on record.
# What it produced: anchored_forecast.{txt,json} (results/anchored_forecast.txt): leave-one-out RMS
# error 0.00022 over the nine scored files; every staged file was forecast below the team best.
# The critic (lab/reports/TRICKS_1115/CRITIC.md section 6 item 4) found that the level line errs
# by 0.00034 on the same nine files, not the 0.0006 the synthesis compared it with.
# Kept as it ran: paths and names refer to the lab tree (see lab/ and docs/glossary.md).
# Original: lab/reports/TRICKS_1115/synth_checks/anchored_forecast.py
"""SYNTHESIS check (CPU, read-only on preds/, subs/, nn_x/*/stage/): anchored forecast of the PUBLIC score of an unscored file.
Method (fixed before any number was read, 2026-11-15T22:1xZ; one method, no scan):
  public(C) = sum_i w_i public(A_i) + mean over public feeds of r_f,   r_f = ndcg_f(C) - sum_i w_i ndcg_f(A_i),   sum_i w_i = 1
  A_i = scored files of the distillation era with stored branch val files; w = least squares on val per-feed NDCG under sum-to-one
  (variance-minimising control variate); the mean of r_f on public feeds is estimated by the pmix-public weighted mean on val.
  SE^2 = var(r_f) / 8562 (public sampling of the residual) + (pmix SE of the val mean of r_f)^2.
Validation: leave-one-out over the scored files (forecast each from the others) -> empirical error against the stated SE.
Targets: the four staged files, FD16P20vf, and ONE declared structure BL = 0.5 z(W8P20 file) + 0.5 z(FD16P20 file).
"""
import sys, json; sys.path.insert(0, '<repo>/src')
from common import *
from metrics import ndcg_per_feed
from stack_val import feed_rank_z
import pmix
from pathlib import Path
from scipy.stats import norm
R = Path('<repo>'); P = R / 'preds'; OUT = R / 'reports/TRICKS_1115/synth_checks'
fv, Mv, ov = feed_meta('val'); yv = np.load(FLAT / 'val/cand_target.npy')
zv = lambda x: feed_rank_z(ov, Mv, np.ascontiguousarray(np.asarray(x, np.float32)))[:, 1]
ev = lambda s: ndcg_per_feed(np.asarray(s, np.float32), yv, ov, Mv).astype(np.float64)
WT = pmix.weights(); WPUB, WPRIV = WT['public'], WT['private']
NPUB = 8562; BEST = 0.3950094
def fin(tag, d=P, w=0.5):
    return w * zv(np.load(d / f'ens_{tag}_ft_val.npy')) + (1 - w) * zv(np.load(d / f'ens_{tag}_to_val.npy'))
# scored files (public scores: docs/04_SUBMISSIONS.md section 2 and section 5, CURRENT_STATUS.md section 1)
SC = {  # name: (tag, w_ft, public, final val on record)
 'W8': ('W8kd', 0.5, 0.394486, 0.40165), 'W16': ('W16kd', 0.5, 0.39448337, 0.40158), 'W8P20': ('W8P20kd', 0.5, 0.3950094, 0.40214),
 'W8TO': ('W8kd', 0.0, 0.394009, 0.40157), 'S16': ('S16kd', 0.5, 0.3940782, 0.40173), 'FD16P20': ('FD16P20kd', 0.5, 0.394892, 0.40271),
 'FD16P20m': ('FD16P20_m02_04kd', 0.5, 0.3947465, 0.40298), 'C3': ('Q2a8s3', 0.5, 0.393454, 0.40101), 'G3': ('G3', 0.5, 0.392820, 0.40032)}
Z = {k: fin(t, w=w) for k, (t, w, _, _) in SC.items()}
E = {k: ev(z) for k, z in Z.items()}
lines = []
def w_(s=''):
    print(s, flush=True); lines.append(s)
w_('## path check: final val rebuilt from stored branch files against the value on record')
for k in SC: w_(f'{k}: rebuilt {E[k].mean():.5f} | record {SC[k][3]:.5f} | public {SC[k][2]:.6f}')
def forecast(c, anchors):
    A = np.stack([E[a] for a in anchors], 1); p = np.array([SC[a][2] for a in anchors])
    # sum-to-one least squares: c - a0 = (A[:,1:] - a0) b
    a0 = A[:, 0]; X = A[:, 1:] - a0[:, None]; yy = c - a0
    Xc = X - X.mean(0); yc = yy - yy.mean()
    b = np.linalg.lstsq(Xc, yc, rcond=None)[0] if X.shape[1] else np.zeros(0)
    w = np.concatenate([[1 - b.sum()], b]); r = c - A @ w
    m, s = pmix.wmean_se(r, WPUB); samp = r.std() / np.sqrt(NPUB)
    return float(p @ w + m), float(np.sqrt(s * s + samp * samp)), w, float(m), float(r.mean()), float(samp)
w_(); w_('## leave-one-out validation (each scored file forecast from the other eight)')
w_('file | public | forecast | error (public - forecast) | stated SE | error / SE | largest weights')
errs = []; zs = []
for k in SC:
    an = [a for a in SC if a != k]
    f, s, w, m, mv, samp = forecast(E[k], an); e = SC[k][2] - f; errs.append(e); zs.append(e / s)
    top = sorted(zip(an, w), key=lambda t: -abs(t[1]))[:3]
    w_(f'{k} | {SC[k][2]:.6f} | {f:.6f} | {e:+.6f} | {s:.6f} | {e / s:+.2f} | ' + ', '.join(f'{a} {x:+.2f}' for a, x in top))
errs = np.array(errs); zs = np.array(zs)
w_(f'LOO: mean error {errs.mean():+.6f} | RMS error {np.sqrt((errs ** 2).mean()):.6f} | RMS of error / SE {np.sqrt((zs ** 2).mean()):.2f} | max abs {np.abs(errs).max():.6f}')
kd = [k for k in SC if k not in ('C3', 'G3')]
ek = np.array([errs[list(SC).index(k)] for k in kd]); w_(f'LOO, distillation-era files only ({len(kd)}): RMS error {np.sqrt((ek ** 2).mean()):.6f}')
INFL = float(max(1.0, np.sqrt((zs ** 2).mean())))
w_(f'SE inflation used below = max(1, RMS of error / SE) = {INFL:.2f}')
ST1 = R / 'nn_x/w12/ens_1115/stage'
T = {
 'probe_W8_FTonly_wft100 (staged)': fin('W8kd', w=1.0),
 'cand_FD16P20_wft060 (staged)': fin('FD16P20kd', w=0.6),
 'cand_ENSAkd_stage (staged)': fin('ENSAkd', d=ST1),
 'cand_ENSBkd_stage (staged)': fin('ENSBkd', d=ST1),
 'FD16P20vf (subs/, unscored pick-2 default)': fin('FD16P20vfkd'),
 'BL = 0.5 z(W8P20) + 0.5 z(FD16P20) (declared structure, not built)': 0.5 * zv(Z['W8P20']) + 0.5 * zv(Z['FD16P20']),
}
w_(); w_('## targets (anchors = all nine scored files)')
w_('target | final val | val d vs FD16P20 | val d vs W8P20 | forecast public | SE (stated x inflation) | P(public > 0.3950094) | P(public > FD16P20 0.394892) | residual: val mean, pmix-public mean, public sampling SE | weights')
res = {}
for k, z in T.items():
    c = ev(z); f, s, w, m, mv, samp = forecast(c, list(SC)); s2 = s * INFL
    d1 = c - E['FD16P20']; d2 = c - E['W8P20']
    pb = 1 - norm.cdf((BEST - f) / s2); pf = 1 - norm.cdf((0.394892 - f) / s2)
    top = sorted(zip(SC, w), key=lambda t: -abs(t[1]))[:4]
    w_(f'{k} | {c.mean():.5f} | {d1.mean():+.5f}+-{d1.std() / np.sqrt(len(d1)):.5f} | {d2.mean():+.5f}+-{d2.std() / np.sqrt(len(d2)):.5f} | {f:.6f} | {s2:.6f} | {pb:.2f} | {pf:.2f} | {mv:+.6f}, {m:+.6f}, {samp:.6f} | ' + ', '.join(f'{a} {x:+.2f}' for a, x in top))
    res[k] = dict(val=float(c.mean()), forecast=f, se=s2, p_best=float(pb), p_fd16p20=float(pf))
# the blend bonus, direct
bl = ev(T['BL = 0.5 z(W8P20) + 0.5 z(FD16P20) (declared structure, not built)']); r = bl - 0.5 * (E['W8P20'] + E['FD16P20'])
m, s = pmix.wmean_se(r, WPUB); mp, sp = pmix.wmean_se(r, WPRIV)
w_(); w_(f'BL minus the mean of its two scored components, per feed: val {r.mean():+.6f}+-{r.std() / np.sqrt(len(r)):.6f} | pmix-public {m:+.6f}+-{s:.6f} | pmix-private {mp:+.6f}+-{sp:.6f} | sd per feed {r.std():.5f} -> public sampling SE {r.std() / np.sqrt(NPUB):.6f}')
w_(f'BL two-anchor forecast: (0.3950094 + 0.394892) / 2 + {m:+.6f} = {(0.3950094 + 0.394892) / 2 + m:.6f}')
json.dump(res, open(OUT / 'anchored_forecast.json', 'w'), indent=1)
open(OUT / 'anchored_forecast.txt', 'w').write('\n'.join(lines) + '\n')
