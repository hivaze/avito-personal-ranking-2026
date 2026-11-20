"""w12/moe/why/mech: per-token gate magnitudes from the saved val routing records (CPU).
A: share of real candidates whose routed gate total (sum of the 2 renormalised gates; nominal = scale 2.0) is < 0.02 / < 0.2 / >= 1.98,
   and the share whose 8 raw affinities s = sqrt(softplus(router x)) are all < 1e-3 (router switched off).
B: per router, share of (token, head) whose selected sigmoid gates are both < 0.01 / < 0.05, and whose largest sigmoid gate is < 0.05."""
import json, numpy as np, sys
sys.path.insert(0, '<repo>/lab/nn_x/w12/moe/why/mech')
from mech_common import OUT
res = {}
for s in (951, 952, 953, 954):
    for b in ('cross.0', 'sets.0', 'cross.1', 'sets.1'):
        z = np.load(OUT / 'A' / f'route_w5_e10_kd05w192moeffn_s{s}_val_{b}.npz')
        g = z['gate'].astype(np.float32).sum(1); sa = z['s'].astype(np.float32)
        r = dict(gate_lt002=float((g < 0.02).mean()), gate_lt02=float((g < 0.2).mean()), gate_full=float((g >= 1.98).mean()),
                 router_off=float((sa.max(1) < 1e-3).mean()), s_max_median=float(np.median(sa.max(1))))
        res[f'A.{s}.{b}'] = r
        print('A', s, b, {k: round(v, 4) for k, v in r.items()})
for s in (951, 952, 953, 954):
    for b in ('cross.0', 'cross.1'):
        z = np.load(OUT / 'B' / f'route_w5_e10_kd05w192moevo_s{s}_val_{b}.npz')
        for R in 'VO':
            sel = z[f'sel{R}'].astype(np.int64); ok = sel[:, 0, 0] >= 0
            sv = z[f's{R}'].astype(np.float32)[ok]; sl = sel[ok]
            gsel = np.take_along_axis(sv, sl, -1)  # [T, h, 2]
            r = dict(sel_both_lt001=float((gsel.max(-1) < 0.01).mean()), sel_both_lt005=float((gsel.max(-1) < 0.05).mean()),
                     max_lt005=float((sv.max(-1) < 0.05).mean()), sel_gate_median=float(np.median(gsel)), sel_gate_p90=float(np.quantile(gsel, 0.9)))
            res[f'B.{s}.{b}.{R}'] = r
            print('B', s, b, R, {k: round(v, 4) for k, v in r.items()})
(OUT / 'gates.json').write_text(json.dumps(res, indent=1))
