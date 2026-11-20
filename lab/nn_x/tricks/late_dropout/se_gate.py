"""LATE-DROPOUT kill-gate constant (PREDECLARED.txt): SE_gate from the six w5_e10_base runs only (no arm data).
SE_gate = sqrt(mean over 15 seed pairs i<j and plateau evals t = 40000..77500 step 2500 of (h_i(t) - h_j(t))^2).
Also checks the six base runs share one recipe (args equal up to variant / seed) and prints h_base(60000) for s951/952.
Writes nn_x/tricks/late_dropout/se_gate.json."""
import itertools, json
import numpy as np
W = '<repo>/nn_x/w5/logs'
S = [951, 952, 953, 954, 955, 956]
J = {s: json.load(open(f'{W}/w5_e10_base_s{s}.json')) for s in S}
a0 = {k: v for k, v in J[951]['args'].items() if k not in ('variant', 'seed')}
for s in S:
    OFF = {'trunc_lifts': '', 'hzcd': False, 'val_mix': 0.0, 'val_fold': -1}  # later-version flags at their OFF default
    d = {k: (a0.get(k), v) for k, v in J[s]['args'].items() if k not in ('variant', 'seed') and a0.get(k) != v
         and not (k not in a0 and k in OFF and v == OFF[k])}
    assert not d, (s, d)
T = list(range(40000, 77501, 2500))
H = {s: {h['step']: h['holdout_ndcg'] for h in J[s]['history']} for s in S}
for s in S:
    assert all(t in H[s] for t in T + [60000]), s
D = np.array([[H[i][t] - H[j][t] for t in T] for i, j in itertools.combinations(S, 2)])
se = float(np.sqrt((D ** 2).mean()))
d60 = np.array([H[i][60000] - H[j][60000] for i, j in itertools.combinations(S, 2)])
out = dict(se_gate=se, n_pairs=len(D), steps=T, threshold=-2 * se, h_base_60000={s: H[s][60000] for s in S},
           rms_at_60000_only=float(np.sqrt((d60 ** 2).mean())), mean_pair_diff=float(D.mean()), sd_pair_diff=float(D.std()),
           per_step_rms={t: float(np.sqrt((D[:, k] ** 2).mean())) for k, t in enumerate(T)})
print(json.dumps(out, indent=1))
json.dump(out, open('<repo>/nn_x/tricks/late_dropout/se_gate.json', 'w'), indent=1)
