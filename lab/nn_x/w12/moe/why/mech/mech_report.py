"""w12/moe/why/mech: aggregate the mech_run.py outputs over seeds 951-954 (CPU). Writes out/report.txt (+ prints).
Ablation deltas: per feed, the mean over the 4 seeds of (ablated - intact NDCG of the same checkpoint); mean +- SE over feeds (n val 14,423,
holdout 30,000). 'resid' = scale-free residual perturbation sqrt(1 - rho^2)/rho of the within-feed-centred logits (= r for i.i.d. noise of
relative size r); 'noise-eq' = NDCG change that i.i.d. Gaussian noise of that residual size costs a reference member (out/calib.json,
interpolated in r^2); ratio = delta / noise-eq (> 1: the removed part carried more label signal than its size predicts if random)."""
import json
import sys
import numpy as np

sys.path.insert(0, '<repo>/lab/nn_x/w12/moe/why/mech')
from mech_common import OUT, se  # noqa: E402

ARMS = dict(ref='kd05w192', A='kd05w192moeffn', B='kd05w192moevo', C='kd05w192b3ff1')
SEEDS = (951, 952, 953, 954)
L = []
P = lambda *a: L.append(' '.join(str(x) for x in a))
cal = json.load(open(OUT / 'calib.json'))['noise']
RG = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]
NC = {sp: [0.0] + [float(np.mean([a for a, _ in cal[f'{sp}.r{r}']])) for r in RG[1:]] for sp in ('val', 'hold')}
noise_eq = lambda sp, r: float(np.interp(r ** 2, np.square(RG), NC[sp]))

R, N = {}, {}
for g, arm in ARMS.items():
    R[g] = [json.load(open(OUT / g / f'res_w5_e10_{arm}_s{s}.json')) for s in SEEDS]
    N[g] = [np.load(OUT / g / f'ndcg_w5_e10_{arm}_s{s}.npz') for s in SEEDS]

P('=' * 140)
P('ABLATIONS at inference (selected checkpoints; per-feed delta averaged over seeds 951-954; mean +- SE over feeds; paired within checkpoint)')
P(f'{"group":5s} {"ablation":26s} | {"val delta":>18s} {"resid":>6s} {"noise-eq":>9s} {"ratio":>6s} | {"holdout delta":>18s} {"resid":>6s} {"noise-eq":>9s} {"ratio":>6s} | per-seed val deltas')
ABL = {}
for g in ARMS:
    tags = [k[len('val.'):] for k in N[g][0].files if k.startswith('val.') and k != 'val.intact']
    for t in tags:
        row = f'{g:5s} {t:26s} |'
        for sp in ('val', 'hold'):
            d = np.mean([n[f'{sp}.{t}'].astype(np.float64) - n[f'{sp}.intact'] for n in N[g]], 0)
            rr = float(np.mean([r['ablations'][f'{sp}.{t}']['resid'] for r in R[g]]))
            ne = noise_eq(sp, rr)
            ABL[(g, t, sp)] = (float(d.mean()), se(d), rr, ne)
            row += f' {d.mean():+.5f}+-{se(d):.5f} {rr:6.3f} {ne:+.5f} {d.mean() / ne if ne < 0 else float("nan"):6.2f} |'
        row += ' ' + ' '.join(f'{r["ablations"][f"val.{t}"]["d"]:+.4f}' for r in R[g])
        P(row)
    P('-' * 140)

P('')
P('NOISE CALIBRATION (reference members, i.i.d. Gaussian noise of within-feed relative size r): ' +
  ' | '.join(f'r={r}: val {NC["val"][i]:+.5f} hold {NC["hold"][i]:+.5f}' for i, r in enumerate(RG) if r > 0))
cd = json.load(open(OUT / 'calib.json'))['dist']
P('FUNCTION DISTANCES (stored preds; unit within-feed sd; rel diff = RMS(a - b), corr = mean within-feed Pearson):')
for k, v in cd.items():
    P(f'   {k:50s} rel diff {v["reldiff"]:.3f}  corr {v["corr"]:.4f}  (n {v["n"]})')

P('')
P('=' * 140)
P('RESIDUAL-STREAM MAGNITUDES per candidate block (val, real candidates, within-feed centred RMS; mean over 4 seeds [min-max])')
for g in ARMS:
    blocks = R[g][0]['blocks']
    for b in blocks:
        def q(f):
            xs = [f(r['val.blocks'][b]) for r in R[g]]
            return f'{np.mean(xs):.3f} [{np.min(xs):.3f}-{np.max(xs):.3f}]'
        s = (f'{g:4s} {b:8s} upd/in {q(lambda x: x["rmsc_upd"] / x["rmsc_in"])}  attn/in {q(lambda x: x["rmsc_att"] / x["rmsc_in"])}  '
             f'ffn/in {q(lambda x: x["rmsc_ff"] / x["rmsc_in"])}')
        if 'rmsc_moe' in R[g][0]['val.blocks'][b]:
            s += (f'  routed/in {q(lambda x: x["rmsc_moe"] / x["rmsc_in"])}  routed/dense-ffn {q(lambda x: x["rmsc_moe"] / x["rmsc_ff"])}'
                  f'  cos(routed, dense) {q(lambda x: x["cos_moe_ff"])}')
        if 'vo_O_rms_ratio' in R[g][0]['val.blocks'][b]:
            s += (f'  routedV/denseV {q(lambda x: x["vo_V_rms_ratio"])} cos {q(lambda x: x["vo_V_cos"])}  routedO/denseO {q(lambda x: x["vo_O_rms_ratio"])}'
                  f' (centred {q(lambda x: x["vo_O_rmsc_ratio"])}) cos {q(lambda x: x["vo_O_cos"])}')
        P(s)
    P('')

P('=' * 140)
P('ATTENTION per candidate block (val; normalised entropy H/log(#valid keys) per head, sorted; mean over seeds of the per-head means; '
  'eff keys share = exp(H)/#keys; user-token mass (cross) / self mass (set))')
for g in ARMS:
    for b in R[g][0]['blocks']:
        ent = np.mean([sorted(r['val.blocks'][b]['att_ent']) for r in R[g]], 0)
        effk = np.mean([np.mean(r['val.blocks'][b]['att_effk']) for r in R[g]])
        mx = np.mean([np.mean(r['val.blocks'][b]['att_maxp']) for r in R[g]])
        ex = 'att_user' if b.startswith('cross') else 'att_self'
        um = np.mean([np.mean(r['val.blocks'][b][ex]) for r in R[g]])
        P(f'{g:4s} {b:8s} ent/head {np.round(ent, 3).tolist()}  mean {ent.mean():.3f}  eff-keys share {effk:.3f}  max p {mx:.3f}  {ex[4:]} mass {um:.3f}')
    P('')

P('=' * 140)
P('REPRESENTATION (val, within-feed-centred real candidate tokens): effective rank (exp entropy of the spectrum), participation ratio, '
  '#dims for 90% / 99% variance, top-1 share; logit R^2 of a linear readout of [layer] and [layer, X]; R^2 carried by the top-k principal directions')
for sp in ('val', 'hold'):
    P(f'-- {sp}')
    for g in ARMS:
        for ln in R[g][0][f'{sp}.layers']:
            def m(k):
                return np.mean([r[f'{sp}.layers'][ln][k] for r in R[g]])
            x = R[g][0][f'{sp}.layers'][ln]
            tk = ''
            if 'r2_topk' in x:
                ks = list(x['r2_topk'].keys())
                tk = ' top-k R2 ' + ' '.join(f'{k}:{np.mean([r[f"{sp}.layers"][ln]["r2_topk"][k] for r in R[g]]):.3f}' for k in ks)
            P(f'{g:4s} {ln:8s} dim {x["dim"]:3d} erank {m("erank"):6.1f} PR {m("pr"):5.1f} k90 {m("k90"):5.1f} k99 {m("k99"):5.1f} top1 {m("top1"):.3f} '
              f'R2 {m("r2_self"):.3f}' + (f' R2+X {m("r2_withX"):.3f}' if 'r2_withX' in x else '') + tk)
        P('')

P('=' * 140)
P('ROUTER STATE')
for b in R['A'][0]['blocks']:
    gb = np.array([r['val.blocks'][b]['moe_gbar'] for r in R['A']])
    P(f'A {b:8s} mean effective gate per expert (sum = scale 2) per seed: ' + ' | '.join(np.round(np.sort(x)[::-1], 3).astype(str).tolist().__str__() for x in gb))
for b in [b_ for b_ in R['B'][0]['blocks'] if 'vo_V_smean' in R['B'][0]['val.blocks'][b_]]:
    for Rr in 'VO':
        sm = np.array([r['val.blocks'][b][f'vo_{Rr}_smean'] for r in R['B']])
        ss = np.array([r['val.blocks'][b][f'vo_{Rr}_sstd'] for r in R['B']])
        mg = np.array([r['val.blocks'][b][f'vo_{Rr}_margin'] for r in R['B']])
        sg = np.array([r['val.blocks'][b][f'vo_{Rr}_selg'] for r in R['B']])
        P(f'B {b:8s} {Rr}-router: sigmoid gate mean {sm.mean():.3f} (per head/expert range {sm.min():.3f}-{sm.max():.3f}); sd across tokens mean {ss.mean():.3f} '
          f'(max {ss.max():.3f}); CV {np.mean(ss / sm):.3f}; selected-gate mean {sg.mean():.3f}; top-2 vs 3rd biased-score margin mean {mg.mean():.4f}')
txt = '\n'.join(L)
(OUT / 'report.txt').write_text(txt + '\n')
print(txt)
