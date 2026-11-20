"""e5: what would fine-tunes of EXISTING held-out members add today?  Members with train-only predictions but no fine-tune can only be read on the train-only branch.
Baseline held-out file H0 = W16 + (P1d + P1w) at 0.2 (every member has a fine-tune).  For every arm X without fine-tunes (>= 4 members, pool level within 0.001 of the wide students):
  TO-preview : final val of [FT branch of H0 unchanged | TO branch with base = W16 + X] minus H0     (half of the file moves)
  TO-branch  : train-only branch delta alone (val | in-period holdout)
  forecast   : if the fine-tuned branch moved like the train-only branch the file would move by about the TO-branch delta (both branches enter at 0.5); law value for seed-like members
               c (1/16 - 1/(16 + k)) with c from out/law.json (W16 fin).
Measured control where a fine-tune exists: W16 + HZL8 (8 held-out hz re-cools @last with fine-tunes) on both branches.  Output out/ft_preview.json."""
from ens_common import *
import pmix
m = Meta(); st = Store(m); WTS = pmix.weights(); Z = lambda x: m.z('val', x); law = json.load(open(OUT / 'law.json'))
pop = P1d + P1w; pf = Z(st.pool('ft_val', pop)); pt = Z(st.pool('to_val', pop)); ph = m.z('hold', st.pool('hold', pop))
ftb = 0.8 * Z(st.pool('ft_val', W16)) + 0.2 * pf; tob = lambda base: 0.8 * Z(st.pool('to_val', base)) + 0.2 * pt; hob = lambda base: 0.8 * m.z('hold', st.pool('hold', base)) + 0.2 * ph
e0 = m.ev(0.5 * Z(ftb) + 0.5 * Z(tob(W16))); t0_ = m.ev(tob(W16)); h0 = m.eh(hob(W16)); c = law['W16']['fin']['all']['c']
print(f'H0 = W16 + P1d + P1w: final val {e0.mean():.5f} TO branch {t0_.mean():.5f} holdout TO branch {h0.mean():.5f}; law c (W16 final) {c:.5f}')
ARMS = {'kd08w192 x8': e10('kd08w192', S8), 'kd05w192gru x4': e10('kd05w192gru'), 'kd05w224 x4': e10('kd05w224'), 'kdmixw192 x4': e10('kdmixw192'), 'kwh6 x4': e10('kwh6'), 'kwr32 x4': e10('kwr32'), 'kwd2 x4': e10('kwd2'),
        'gru + w224 + kdmix x12': e10('kd05w192gru') + e10('kd05w224') + e10('kdmixw192'),
        'RC ref@last x16': rc('ref'), 'RC ref2@last x16': rc('ref2'), 'RC th@last x16': rc('th'), 'RC hzt@last x16': rc('hzt'), 'RC hznFS@last x16': rc('hznFS'), 'RC xd30@last x16': rc('xd30'),
        'RC hz@last s955-958 x8': rc('hz', '@last', (955, 956, 957, 958)), 'RC ref + ref2 x32': rc('ref') + rc('ref2'), 'all no-FT wide-level arms x76': e10('kd05w192gru') + e10('kd05w224') + e10('kdmixw192') + rc('ref') + rc('ref2') + rc('th') + rc('hzt')}
res = {}
for nm, X in ARMS.items():
    assert not any(st.has('ft_val', x) for x in X), nm
    base = W16 + X; e = m.ev(0.5 * Z(ftb) + 0.5 * Z(tob(base))); d = e - e0; dt = m.ev(tob(base)) - t0_; dh = m.eh(hob(base)) - h0; k = len(X); pp, sp = pmix.wmean_se(dt, WTS['public']); pr, sr = pmix.wmean_se(dt, WTS['private'])
    res[nm] = dict(k=k, preview=(float(d.mean()), se(d)), to_branch=(float(dt.mean()), se(dt)), hold=(float(dh.mean()), se(dh)), to_pub=(float(pp), float(sp)), to_priv=(float(pr), float(sr)), law_seedlike=c * (1 / 16 - 1 / (16 + k)), ft_gpu_h=k * 1.0)
    print(f'{nm:32s} TO-preview (half the file) {fmt(d)} | TO branch val {fmt(dt)} pmix {pp:+.5f}/{pr:+.5f} | holdout {fmt(dh)} | law if seed-like {c * (1 / 16 - 1 / (16 + k)):+.5f} | fine-tune cost {k} GPU-h', flush=True)
# measured control: held-out hz re-cools that HAVE fine-tunes
base = W16 + HZL8; f = 0.5 * Z(0.8 * Z(st.pool('ft_val', base)) + 0.2 * pf) + 0.5 * Z(tob(base)); d = m.ev(f) - e0; dft = m.ev(0.8 * Z(st.pool('ft_val', base)) + 0.2 * pf) - m.ev(ftb); dto = m.ev(tob(base)) - t0_
prev = m.ev(0.5 * Z(ftb) + 0.5 * Z(tob(base))) - e0
res['_control_HZL8'] = dict(full=(float(d.mean()), se(d)), ft_branch=(float(dft.mean()), se(dft)), to_branch=(float(dto.mean()), se(dto)), preview=(float(prev.mean()), se(prev)))
print(f'CONTROL W16 + HZL8 (fine-tunes exist): both branches {fmt(d)} | FT branch {fmt(dft)} | TO branch {fmt(dto)} | its TO-preview would have read {fmt(prev)}')
# the same control on the all-data side: FD16 + FDXHZ16 against FD16 (pop FDP4)
pf2 = Z(st.pool('ft_val', FDP4)); pt2 = Z(st.pool('to_val', FDP4)); fb = lambda base: 0.8 * Z(st.pool('ft_val', base)) + 0.2 * pf2; tb = lambda base: 0.8 * Z(st.pool('to_val', base)) + 0.2 * pt2
e1 = m.ev(0.5 * Z(fb(FD16)) + 0.5 * Z(tb(FD16))); b2 = FD16 + FDXHZ; d = m.ev(0.5 * Z(fb(b2)) + 0.5 * Z(tb(b2))) - e1; dft = m.ev(fb(b2)) - m.ev(fb(FD16)); dto = m.ev(tb(b2)) - m.ev(tb(FD16)); prev = m.ev(0.5 * Z(fb(FD16)) + 0.5 * Z(tb(b2))) - e1
res['_control_FDXHZ'] = dict(full=(float(d.mean()), se(d)), ft_branch=(float(dft.mean()), se(dft)), to_branch=(float(dto.mean()), se(dto)), preview=(float(prev.mean()), se(prev)))
print(f'CONTROL FD16 + FDXHZ16 (fine-tunes exist): both branches {fmt(d)} | FT branch {fmt(dft)} | TO branch {fmt(dto)} | its TO-preview would have read {fmt(prev)}')
json.dump(res, open(OUT / 'ft_preview.json', 'w'), indent=0)
