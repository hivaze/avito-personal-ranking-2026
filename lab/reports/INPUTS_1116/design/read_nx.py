"""READ TOOL of the trained-input bundle arm (rule: reports/INPUTS_1116/design/RULE_LEDGER.txt, stages A and B). CPU only, read-only.

  cd <repo> && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 NUMBA_NUM_THREADS=12 \
     POLARS_MAX_THREADS=12 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python reports/INPUTS_1116/design/read_nx.py <ARM> <CTL> [out file]
  ARM, CTL = member lists "prefix:s1-s2[+prefix:s1-s2...]", e.g.
     stage A:  abl_e10_nx:951-958      w5_e10_base:951-956+abl_e10_base:957-958
     stage B:  abl_e10_nxkd05w192:951-958   w5_e10_kd05w192:951-958+w5_e10_kd05t25w192:951-958
  Members are resolved by nn_x/w12/recheck/pool_composition/pc_common.paths (train-only val, holdout; selected and @last).

Statistics (all paired per feed over the same feeds):
  member level = mean over the arm's members of per-feed NDCG minus the same for the control; SE = max(per-feed paired SE,
                 seed-level SE sqrt(sd_arm^2 / k_arm + sd_ctl^2 / k_ctl) of the member means); z = delta / SE
  pool level   = NDCG of the member-equal mean of per-feed z-scores (the project's pool) of the arm minus that of the control
  val          = 14,423 feeds, selected checkpoints; also with the private / public feed-mix weights of src/pmix.py, and M < 50 / M >= 50
  holdout      = the 30,000 in-period feeds; the DECISION read is on the lag-view subset = holdout feeds of Sep 12-19 (slot >= 6),
                 the only holdout feeds on which the lag-built inputs exist (has = 1); subsets val-like (Sep 12-14) and test-like (Sep 17-19)
  pooled z     = (z_val + z_holdout_subset) / sqrt(2), member level
Decision line = clauses of RULE_LEDGER.txt (PASS / REPLICATE / CLOSED). Nothing is selected by this tool.
"""
import os, sys
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'POLARS_MAX_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ.setdefault(_k, '12')
sys.dont_write_bytecode = True
sys.path.insert(0, '<repo>/nn_x/w12/recheck/pool_composition'); sys.path.insert(0, '<repo>/src')
sys.path.insert(0, '<repo>/nn_x/w12/stale/hz2')
import numpy as np
import pc_common as pc
import pmix
import hz2_common as hc


def members(spec):
    out = []
    for part in spec.split('+'):
        pre, rng_ = part.split(':'); a, b = (rng_.split('-') + [rng_])[:2]
        out += [f'{pre}_s{s}' for s in range(int(a), int(b) + 1)]
    return out


def main():
    arm, ctl = members(sys.argv[1]), members(sys.argv[2])
    fo = open(sys.argv[3], 'w') if len(sys.argv) > 3 else None

    def P(*a):
        s = ' '.join(str(x) for x in a); print(s, flush=True)
        if fo: fo.write(s + '\n')
    m = pc.Meta(); W = pmix.weights(); wpriv = np.asarray(W['private'], np.float64); wpub = np.asarray(W['public'], np.float64)
    ftr = hc.feeds('train'); hold, _ = hc.holdout_feeds(ftr)
    slot = (hc.day_of(ftr['timestamp'].to_numpy()[hold]) - hc.DAY0).astype(np.int64)
    assert len(slot) == len(m.Mh)
    sub = dict(all=np.ones(len(slot), bool), lagview=slot >= 6, vallike=(slot >= 6) & (slot <= 9), testlike=slot >= 11)

    def load(v, suf):
        pv, ph, _ = pc.paths(v, suf)
        zv = m.z('val', np.load(pv)); zh = m.z('hold', np.load(ph))
        return zv, zh, pc.ndcg_fast(zv, m.y['val'], m.off['val'], m.M['val']), pc.ndcg_fast(zh, m.y['hold'], m.off['hold'], m.M['hold'])

    def contrast(suf):
        A = [load(v, suf) for v in arm]; C = [load(v, suf) for v in ctl]
        res = {}
        for tag, i_z, i_n, sp in (('val', 0, 2, 'val'), ('hold', 1, 3, 'hold')):
            na = np.stack([a[i_n] for a in A]); nc = np.stack([c[i_n] for c in C])
            pa = pc.ndcg_fast(np.mean([a[i_z] for a in A], 0).astype(np.float32), m.y[sp], m.off[sp], m.M[sp])
            pc_ = pc.ndcg_fast(np.mean([c[i_z] for c in C], 0).astype(np.float32), m.y[sp], m.off[sp], m.M[sp])
            res[tag] = (na, nc, pa - pc_)
        return res

    def stat(na, nc, dpool, mask=None, w=None):
        if mask is None: mask = np.ones(na.shape[1], bool)
        d = na[:, mask].mean(0) - nc[:, mask].mean(0); dp = dpool[mask]
        if w is None:
            mu, se_pf = d.mean(), d.std() / np.sqrt(len(d)); mup, sep = dp.mean(), dp.std() / np.sqrt(len(dp))
            ma, mc = na[:, mask].mean(1), nc[:, mask].mean(1)
        else:
            ww = w[mask]; mu, se_pf = pmix.wmean_se(d, ww); mup, sep = pmix.wmean_se(dp, ww)
            ma = np.array([np.average(x[mask], weights=ww) for x in na]); mc = np.array([np.average(x[mask], weights=ww) for x in nc])
        se_seed = np.sqrt(ma.var(ddof=1) / len(ma) + mc.var(ddof=1) / len(mc)) if len(ma) > 1 and len(mc) > 1 else 0.0
        se = max(se_pf, se_seed)
        return dict(d=mu, se=se, se_pf=se_pf, se_seed=se_seed, z=mu / se, dp=mup, sep=sep, zp=mup / sep, n=int(mask.sum()),
                    pos=int((ma > mc.mean()).sum()), k=len(ma))

    def line(lab, r):
        P(f'  {lab:34s} n {r["n"]:6d} | member {r["d"]:+.5f} +- {r["se"]:.5f} (z {r["z"]:+.2f}; per-feed SE {r["se_pf"]:.5f}, seed SE {r["se_seed"]:.5f}; '
          f'{r["pos"]}/{r["k"]} arm members above the control mean) | pool {r["dp"]:+.5f} +- {r["sep"]:.5f} (z {r["zp"]:+.2f})')
    P(f'ARM  ({len(arm)}): {arm[0]} .. {arm[-1]}'); P(f'CTL  ({len(ctl)}): {ctl[0]} .. {ctl[-1]}')
    out = {}
    for suf, name in (('', 'selected checkpoints'), ('@last', 'last step')):
        R = contrast(suf); P(f'== {name}')
        na, nc, dp = R['val']; big = m.Mv >= 50
        out[suf, 'val'] = stat(na, nc, dp); line('val', out[suf, 'val'])
        out[suf, 'priv'] = stat(na, nc, dp, w=wpriv); line('val, private mix', out[suf, 'priv'])
        line('val, public mix', stat(na, nc, dp, w=wpub)); line('val M>=50', stat(na, nc, dp, big)); line('val M<50', stat(na, nc, dp, ~big))
        na, nc, dp = R['hold']
        for k_, lab in (('all', 'holdout, all feeds'), ('lagview', 'holdout, lag-view feeds (Sep 12-19)'), ('vallike', 'holdout, Sep 12-14 (val-like lag)'),
                        ('testlike', 'holdout, Sep 17-19 (test-like lag)')):
            out[suf, k_] = stat(na, nc, dp, sub[k_]); line(lab, out[suf, k_])
    v, p, h, hl = out['', 'val'], out['', 'priv'], out['', 'lagview'], out['@last', 'lagview']
    pooled = (v['z'] + h['z']) / np.sqrt(2)
    P(f'== DECISION (selected checkpoints; member level): z_val {v["z"]:+.2f}, z_holdout(lag-view feeds) {h["z"]:+.2f}, pooled z {pooled:+.2f}; '
      f'private-mix z {p["z"]:+.2f}; pool deltas val {v["dp"]:+.5f}, holdout(lag-view) {h["dp"]:+.5f}; holdout(lag-view) at the last step {hl["d"]:+.5f}')
    side = v['z'] >= 1 and h['z'] >= 1 and p['z'] >= -1 and v['dp'] >= 0 and h['dp'] >= 0 and hl['d'] >= 0
    if pooled >= 2 and side: verdict = 'PASS (subject to the stale-val clause, read separately)'
    elif 1.6 <= pooled < 2 and v['z'] >= 0 and h['z'] >= 0: verdict = 'REPLICATE (4 more seeds per arm, one re-read)'
    else: verdict = 'NOT PASSED -> CLOSED by the rule'
    P(f'   clauses: z_val >= 1, z_holdout >= 1, pooled >= 2, private-mix z >= -1, both pool deltas >= 0, last-step holdout >= 0  ->  {verdict}')
    if fo: fo.close()


if __name__ == '__main__':
    main()
