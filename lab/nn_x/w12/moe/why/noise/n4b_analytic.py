"""why/noise N4b: analytic (quadrature) version of the shared-reference null, conditional on the observed Ra - Rb, to replace the
kernel-weighted Monte Carlo of n4_power.py (ESS ~240). Model: 4-pool means e ~ N(0, sp^2) iid (sp from n4_power.json, pooled over 8 recipes);
D_k = e_k - e_Ra; given g = Ra - Rb, e_Ra ~ N(g/2, sp^2/2); arms independent of the references and of each other."""
import json
import numpy as np
from scipy.stats import norm
from pathlib import Path

OUT = Path('<repo>/lab/nn_x/w12/moe/why/noise')
p = json.load(open(OUT / 'n4_power.json'))
out = []
P = lambda *a: (print(*a), out.append(' '.join(str(x) for x in a)))
for sp_name, sp, g, obs in (('val', p['sp_v'], 0.00138, [-0.00086, -0.00144, -0.00069]), ('holdout', p['sp_h'], 0.00001, [0.00020, 0.00029, 0.00040])):
    e = np.linspace(g / 2 - 8 * sp / np.sqrt(2), g / 2 + 8 * sp / np.sqrt(2), 20001)
    w = norm.pdf(e, g / 2, sp / np.sqrt(2)); w /= w.sum()
    sgn = -1 if sp_name == 'val' else 1
    # P(all three D_k on the observed side): D_k < 0 <=> e_k < e_Ra (val), D_k > 0 <=> e_k > e_Ra (holdout)
    pk = norm.cdf(e / sp) if sgn < 0 else norm.sf(e / sp)
    pall = float(np.sum(w * pk ** 3))
    pall_u = 0.25
    m = np.mean(obs)
    sd_mean = sp * np.sqrt(1 / 3 + 1 / 2)
    pm = norm.cdf((m + g / 2) / sd_mean) if sgn < 0 else norm.sf((m + g / 2) / sd_mean)
    pm_u = norm.cdf(m / (sp * np.sqrt(1 / 3 + 1))) if sgn < 0 else norm.sf(m / (sp * np.sqrt(1 / 3 + 1)))
    sd_one = sp * np.sqrt(1 + 1 / 2)
    worst = min(obs) if sgn < 0 else max(obs)
    p1 = norm.cdf((worst + g / 2) / sd_one) if sgn < 0 else norm.sf((worst + g / 2) / sd_one)
    P(f'[{sp_name}] sigma_p {sp:.5f}, observed Ra-Rb {g:+.5f} -> E[e_Ra | Ra-Rb] {g / 2:+.5f} (sd {sp / np.sqrt(2):.5f})')
    P(f'  P(all three on the observed side): unconditional {pall_u:.3f}, conditional {pall:.3f}')
    P(f'  P(mean of three as extreme as {m:+.5f}): unconditional {pm_u:.4f}, conditional {pm:.3f}')
    P(f'  P(a single arm as extreme as {worst:+.5f}) conditional {p1:.3f}; P(at least one of three, independence approx) {1 - (1 - p1) ** 3:.3f}')
    P(f'  expected arm delta under the null given Ra-Rb: {-g / 2:+.5f}; observed arm deltas minus that: ' + ' '.join(f'{x + g / 2:+.5f}' for x in obs)
      + f' (null sd of each {sd_one:.5f})')
open(OUT / 'n4b_analytic.txt', 'w').write('\n'.join(out) + '\n')
