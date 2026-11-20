"""Stage 4: multiplicity-adjusted significance (cluster-bootstrap max-t, recentered) over ALL configs each study tried on val.
Study B: every val-nested stacker OOF in nnstack/out vs its own equal pool. Study A: 20 main + 20 TOonly hfit weightings vs that pool's ME.
Study C: PoP variants vs their union equal pools."""
import os, json
for _k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS', 'NUMBA_NUM_THREADS']: os.environ.setdefault(_k, '12')
src = open('<repo>/nn_x/stack/verify/verify.py').read().split('RES = {}')[0]
exec(compile(src, 'verify.py', 'exec'))
_, inv = np.unique(gv, return_inverse=True); G = inv.max() + 1; ng = np.bincount(inv).astype(float)
rng = np.random.default_rng(7); B = 3000
W = rng.multinomial(G, np.ones(G) / G, size=B).astype(np.float32)
def maxt(D, names, target):
    keep = [i for i, d in enumerate(D) if np.abs(d).max() > 0]
    if len(keep) < len(D): log('  dropped identical-to-base configs:', [names[i] for i in range(len(D)) if i not in keep])
    D = [D[i] for i in keep]; names = [names[i] for i in keep]
    D = np.stack(D, 1); mu = D.mean(0)
    Sg = np.stack([np.bincount(inv, D[:, c] - mu[c], G) for c in range(D.shape[1])], 1)
    se = np.sqrt(G / (G - 1) * (Sg ** 2).sum(0)) / len(D); t = mu / se
    Tb = (W @ Sg.astype(np.float32)) / (W @ ng.astype(np.float32))[:, None] / se
    mx = Tb.max(1); i = names.index(target)
    log(f'  {len(names)} configs; observed t: ' + ', '.join(f'{n} {x:+.2f}' for n, x in sorted(zip(names, t), key=lambda z: -z[1])[:8]) + ' ...')
    log(f'  null max-t quantiles 50/90/95/99: {np.quantile(mx, [.5, .9, .95, .99]).round(2).tolist()} | target {target} t {t[i]:+.2f} -> family-wise adjusted p {(mx >= t[i]).mean():.3f} (unadjusted 1-sided p {(Tb[:, i] >= t[i]).mean():.3f})')
    return t
# ---- Study B
ON = f'{R}/nn_x/stack/nnstack/out'; PO = {'G3': G3, 'PF7x': PF7x, 'M2': M2}
base = {k: nd(spec(v, 'val')) for k, v in PO.items()}
names, D = [], []
for fn in sorted(os.listdir(ON)):
    if not fn.endswith('_val_oof.npy'): continue
    tag = fn[:-12]
    if tag in ('g3_gate_b3_s1', 'g3_gate_b3_s2', 'v_gate_b3_s1', 'v_gate_b3_s2', 'g3_gate_b3', 'v_gate_b3'): continue  # replicates folded into g3gate3/pfgate3
    pool = 'G3' if tag.startswith('g3') else 'M2' if tag.startswith('m2') else 'PF7x'
    if os.path.exists(f'{ON}/{tag}.log'):
        a = json.loads(open(f'{ON}/{tag}.log').readline()[5:]); assert a['pool'] == pool, (tag, a['pool'])
    names.append(tag); D.append(nd(np.load(f'{ON}/{fn}')) - base[pool])
log('STUDY B (val-nested stackers):'); maxt(D, names, 'g3gate3')
# unselected replicates only (seed 0 was the run that triggered the replication; s1/s2 were declared afterwards)
zz = lambda r: Z(ov, Mv, np.load(f'{ON}/{r}_val_oof.npy'))
s12 = np.mean([zz('g3_gate_b3_s1'), zz('g3_gate_b3_s2')], 0)
report('S2 unselected replicates s1+s2 mean vs G3', s12, spec(G3, 'val'))
pf12 = np.mean([zz('v_gate_b3_s1'), zz('v_gate_b3_s2')], 0)
report('PF7x gate unselected replicates s1+s2 vs PF7x', pf12, spec(PF7x, 'val'))
# pooled gate evidence across the 3 pools it was run on (equal-weight mean of the 3 per-feed deltas)
dg = [nd(np.load(f'{ON}/g3gate3_val_oof.npy')) - base['G3'], nd(np.load(f'{ON}/pfgate3_val_oof.npy')) - base['PF7x'], nd(np.load(f'{ON}/m2_gate_b3_val_oof.npy')) - base['M2']]
dm = np.mean(dg, 0); m_, i_, c_ = se_pair(dm, gv); log(f'gate config averaged over G3/PF7x/M2 pools: {m_:+.6f} +- {max(i_, c_):.6f} (t {m_ / max(i_, c_):+.2f})')
# ---- Study A
H = f'{R}/nn_x/stack/hfit/logs'
POOLS = {'P6': PF7x, 'P3': G3, 'P11': P11, 'P9nr': ['F5cd', 'F6', 'F7', 'F9', 'F10', 'F12', 'V27', 'V28', 'V29']}
names, D = [], []
for pn, fams in POOLS.items():
    Rj = json.load(open(f'{H}/{pn}_W1-W2-W3-W4-W5.json')); assert Rj['families'] == fams
    me = spec(fams, 'val'); nme = nd(me)
    ftme = half_pool(fams, 'val', 'ft')
    for var, r in Rj['variants'].items():
        w = dict(zip(fams, r['w']))
        s_main = 0.5 * Z(ov, Mv, half_pool(fams, 'val', 'ft', w)) + 0.5 * Z(ov, Mv, half_pool(fams, 'val', 'to', w))
        s_to = 0.5 * Z(ov, Mv, ftme) + 0.5 * Z(ov, Mv, half_pool(fams, 'val', 'to', w))
        names += [f'{pn}_{var}', f'{pn}_{var}_TOonly']; D += [nd(s_main) - nme, nd(s_to) - nme]
log('STUDY A (holdout-fitted weightings, 20 main + 20 TOonly):'); maxt(D, names, 'P11_W1_TOonly')
# ---- Study C (PoP only; the ~100 aggregation comparisons are not all reconstructable here)
pops = {'PoP(PF7x,G3)': [PF7x, G3], 'PoP(PF7x,M2)': [PF7x, M2], 'PoP3': [PF7x, G3, M2]}
names, D = [], []
for n, pl_ in pops.items():
    s = np.mean([Z(ov, Mv, spec(p, 'val')) for p in pl_], 0); names.append(n); D.append(nd(s) - nd(spec(U45, 'val')))
log('STUDY C (PoP variants vs U45 union equal pool; subset of the 9 PoP + ~100 total comparisons):'); maxt(D, names, 'PoP3')
