"""CRITIC c04: can the stage A / B rule of RULE_LEDGER.txt pass by noise, and is the power statement right?
(1) member-level NDCG of every stored member of the six families on val and on the lag-view holdout feeds (slots >= 6): seed sd, nominal per-feed
    SE of a k-vs-k member-mean contrast, correlation of the seed deviations between val and holdout.
(2) empirical A/A: the COMPLETE rule (the stat() of design/read_nx.py, copied) applied to random splits of stored members of ONE recipe:
    16 wide students 8 v 8 (stratified by teacher), hard-label pop1 4 v 4, hard-label base 3 v 3. Share PASS / REPLICATE; sd and correlation of z_val, z_holdout.
(3) Monte Carlo of the rule with the SE estimated from the members (as the tool does), seed noise correlated across witnesses, and the extra clauses
    the designer's p04 left out (last step, stale-val).  CPU only, read-only; writes critic/out/c04_rule_null.txt."""
import os, sys, itertools
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS', 'POLARS_MAX_THREADS'): os.environ[k] = '12'
CR = '<repo>/reports/INPUTS_1116/critic'
os.environ['NUMBA_CACHE_DIR'] = CR + '/out/numba_cache'
sys.dont_write_bytecode = True
sys.path.insert(0, '<repo>/nn_x/w12/recheck/pool_composition'); sys.path.insert(0, '<repo>/src'); sys.path.insert(0, '<repo>/nn_x/w12/stale/hz2')
import numpy as np
import pc_common as pc
import pmix
import hz2_common as hc
out = open(CR + '/out/c04_rule_null.txt', 'w')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); out.write(s + '\n'); out.flush()
m = pc.Meta(); Wt = pmix.weights(); wpriv = np.asarray(Wt['private'], np.float64)
ftr = hc.feeds('train'); hold, _ = hc.holdout_feeds(ftr); slot = (hc.day_of(ftr['timestamp'].to_numpy()[hold]) - hc.DAY0).astype(np.int64); lagv = slot >= 6
FAM = {'HL base (w5_e10_base)': [f'w5_e10_base_s{s}' for s in range(951, 957)], 'HL pop1 (abl_e10_pop1)': [f'abl_e10_pop1_s{s}' for s in range(951, 959)],
       'HL xdrop_pop': [f'abl_e10_xdrop_pop_s{s}' for s in range(951, 955)], 'HL pop2': [f'abl_e10_pop2_s{s}' for s in range(951, 955)],
       'KD w192': [f'w5_e10_kd05w192_s{s}' for s in range(951, 959)], 'KD t25w192': [f'w5_e10_kd05t25w192_s{s}' for s in range(951, 959)]}
Z = {}; ND = {}
def load(v, suf=''):
    pv, ph, _ = pc.paths(v, suf); zv = m.z('val', np.load(pv)); zh = m.z('hold', np.load(ph))
    return zv, zh, pc.ndcg_fast(zv, m.y['val'], m.off['val'], m.M['val']), pc.ndcg_fast(zh, m.y['hold'], m.off['hold'], m.M['hold'])
for fam, mem in FAM.items():
    for v in mem:
        for suf in ('', '@last'):
            Z[v, suf] = load(v, suf)
P('=== (1) member levels: val (14,423 feeds) and lag-view holdout (15,225 feeds of slots >= 6), selected checkpoints ===')
P('family | n | seed sd val | seed sd holdout lag-view | nominal per-feed SE of ONE member-pair difference val / lag-view | corr of seed deviations val vs lag-view')
dv_all, dh_all = [], []
for fam, mem in FAM.items():
    lv = np.array([Z[v, ''][2].mean() for v in mem]); lh = np.array([Z[v, ''][3][lagv].mean() for v in mem])
    d = Z[mem[0], ''][2] - Z[mem[1], ''][2]; d2 = (Z[mem[0], ''][3] - Z[mem[1], ''][3])[lagv]
    dv_all += list(lv - lv.mean()); dh_all += list(lh - lh.mean())
    P(f'{fam} | {len(mem)} | {lv.std(ddof=1):.5f} | {lh.std(ddof=1):.5f} | {d.std() / np.sqrt(len(d)):.5f} / {d2.std() / np.sqrt(len(d2)):.5f} | {np.corrcoef(lv, lh)[0, 1]:+.2f}')
dv_all, dh_all = np.array(dv_all), np.array(dh_all); hl_n = 6 + 8 + 4 + 4
P(f'pooled within-family deviations: hard-label members (n {hl_n}, df {hl_n - 4}): sd val {np.sqrt((dv_all[:hl_n] ** 2).sum() / (hl_n - 4)):.5f}, sd lag-view holdout {np.sqrt((dh_all[:hl_n] ** 2).sum() / (hl_n - 4)):.5f}, '
  f'corr {np.corrcoef(dv_all[:hl_n], dh_all[:hl_n])[0, 1]:+.2f}; students (n 16): sd val {np.sqrt((dv_all[hl_n:] ** 2).sum() / 14):.5f}, sd holdout {np.sqrt((dh_all[hl_n:] ** 2).sum() / 14):.5f}, corr {np.corrcoef(dv_all[hl_n:], dh_all[hl_n:])[0, 1]:+.2f}; '
  f'all 38: corr {np.corrcoef(dv_all, dh_all)[0, 1]:+.2f}')
P(f'hard-label members without abl_e10_pop1: sd val {np.sqrt((np.r_[dv_all[:6], dv_all[14:22]] ** 2).sum() / 11):.5f}')

def stat(na, nc, dpool, mask=None, w=None):   # copy of design/read_nx.py stat()
    if mask is None: mask = np.ones(na.shape[1], bool)
    d = na[:, mask].mean(0) - nc[:, mask].mean(0); dp = dpool[mask]
    if w is None:
        mu, se_pf = d.mean(), d.std() / np.sqrt(len(d)); mup = dp.mean(); ma, mc = na[:, mask].mean(1), nc[:, mask].mean(1)
    else:
        ww = w[mask]; mu, se_pf = pmix.wmean_se(d, ww); mup, _ = pmix.wmean_se(dp, ww)
        ma = np.array([np.average(x[mask], weights=ww) for x in na]); mc = np.array([np.average(x[mask], weights=ww) for x in nc])
    se_seed = np.sqrt(ma.var(ddof=1) / len(ma) + mc.var(ddof=1) / len(mc)); se = max(se_pf, se_seed)
    return dict(d=mu, se=se, z=mu / se, dp=mup, se_pf=se_pf, se_seed=se_seed)
def decide(arm, ctl):
    r = {}
    for suf in ('', '@last'):
        A = [Z[v, suf] for v in arm]; C = [Z[v, suf] for v in ctl]
        for tag, iz, i_n, sp in (('val', 0, 2, 'val'), ('hold', 1, 3, 'hold')):
            na = np.stack([a[i_n] for a in A]); nc = np.stack([c[i_n] for c in C])
            if suf == '':
                dp = pc.ndcg_fast(np.mean([a[iz] for a in A], 0).astype(np.float32), m.y[sp], m.off[sp], m.M[sp]) - pc.ndcg_fast(np.mean([c[iz] for c in C], 0).astype(np.float32), m.y[sp], m.off[sp], m.M[sp])
            else: dp = np.zeros(na.shape[1])
            if tag == 'val':
                r[suf, 'val'] = stat(na, nc, dp)
                if suf == '': r[suf, 'priv'] = stat(na, nc, dp, w=wpriv)
            else: r[suf, 'lag'] = stat(na, nc, dp, lagv)
    v, p, h, hl = r['', 'val'], r['', 'priv'], r['', 'lag'], r['@last', 'lag']
    pooled = (v['z'] + h['z']) / np.sqrt(2)
    side = v['z'] >= 1 and h['z'] >= 1 and p['z'] >= -1 and v['dp'] >= 0 and h['dp'] >= 0 and hl['d'] >= 0
    verdict = 'PASS' if (pooled >= 2 and side) else ('REPL' if (1.6 <= pooled < 2 and v['z'] >= 0 and h['z'] >= 0) else 'NO')
    return verdict, v['z'], h['z'], pooled, v['se_pf'], v['se_seed'], h['se_pf'], h['se_seed'], v['d'], h['d']
rng = np.random.default_rng(1116)
P('\n=== (2) empirical A/A of the complete rule (random splits of members of ONE recipe; both signs of every split are counted) ===')
P('family, split | signed contrasts | PASS | REPLICATE | sd z_val | sd z_holdout(lag-view) | corr(z_val, z_holdout) | share z_val >= 1 | share z_hold >= 1 | share pooled >= 2 | mean SE val (per-feed / seed) | mean SE holdout (per-feed / seed) | sd delta val / holdout')
def aa(name, splits):
    R = []
    for arm, ctl in splits:
        a = decide(arm, ctl); b = decide(ctl, arm); R += [a, b]
    vd = np.array([r[0] for r in R]); zv = np.array([r[1] for r in R]); zh = np.array([r[2] for r in R]); po = np.array([r[3] for r in R])
    P(f'{name} | {len(R)} | {(vd == "PASS").sum()} ({(vd == "PASS").mean():.3f}) | {(vd == "REPL").sum()} ({(vd == "REPL").mean():.3f}) | {zv.std():.2f} | {zh.std():.2f} | {np.corrcoef(zv, zh)[0, 1]:+.2f} | '
      f'{(zv >= 1).mean():.3f} | {(zh >= 1).mean():.3f} | {(po >= 2).mean():.3f} | {np.mean([r[4] for r in R]):.5f} / {np.mean([r[5] for r in R]):.5f} | {np.mean([r[6] for r in R]):.5f} / {np.mean([r[7] for r in R]):.5f} | '
      f'{np.std([r[8] for r in R]):.5f} / {np.std([r[9] for r in R]):.5f}')
    return zv, zh
a8, b8 = FAM['KD w192'], FAM['KD t25w192']; c4 = list(itertools.combinations(range(8), 4)); sp = []
for _ in range(150):
    i = c4[rng.integers(len(c4))]; j = c4[rng.integers(len(c4))]
    arm = [a8[k] for k in i] + [b8[k] for k in j]; ctl = [a8[k] for k in range(8) if k not in i] + [b8[k] for k in range(8) if k not in j]; sp.append((arm, ctl))
aa('16 wide students, 8 v 8 (4 + 4 of each teacher per side), 150 random splits', sp)
p8 = FAM['HL pop1 (abl_e10_pop1)']; sp = [([p8[k] for k in i], [p8[k] for k in range(8) if k not in i]) for i in c4 if 0 in i]
aa('hard-label pop1, 4 v 4, all 35 splits', sp)
b6 = FAM['HL base (w5_e10_base)']; sp = [([b6[k] for k in i], [b6[k] for k in range(6) if k not in i]) for i in itertools.combinations(range(6), 3) if 0 in i]
aa('hard-label base (the stored controls), 3 v 3, all 10 splits', sp)

P('\n=== (3) Monte Carlo of stage A (8 + 8 hard labels) with the SE the tool computes: max(per-feed SE, seed SE estimated from the 16 members) ===')
P('model: member level on a witness = effect + seed deviation; seed deviations N(0, s) with correlation rho between val and lag-view holdout; per-feed SE fixed at its nominal value;')
P('private mix: z correlated 0.7 with val, SE 0.00052, effect x0.5; pool-level sign clauses as in design/p04 (corr 0.8); last-step clause: delta_h + N(0, 0.5 SE_h) >= 0; stale-val clause: effect x0.5, own noise, >= -1 SE')
def mc(eff_v, eff_h, s_v, s_h, rho, pf_v=0.00036, pf_h=0.00029, n=200000, tail=0.0, tail_size=0.0029, full=True):
    e1 = rng.standard_normal((n, 16)); e2 = rng.standard_normal((n, 16))
    dv = s_v * e1; dh = s_h * (rho * e1 + np.sqrt(1 - rho ** 2) * e2)
    if tail > 0: dv = dv - tail_size * (rng.random((n, 16)) < tail)       # occasional member far below its family on val only (abl_e10_pop1_s951: -0.0029)
    av, cv = dv[:, :8].mean(1) + eff_v, dv[:, 8:].mean(1); ah, ch = dh[:, :8].mean(1) + eff_h, dh[:, 8:].mean(1)
    # feed-sampling noise of the fixed witness sets is NOT a property of the arm: under the null only seed noise moves the delta.
    sev = np.maximum(pf_v, np.sqrt(dv[:, :8].var(1, ddof=1) / 8 + dv[:, 8:].var(1, ddof=1) / 8)); seh = np.maximum(pf_h, np.sqrt(dh[:, :8].var(1, ddof=1) / 8 + dh[:, 8:].var(1, ddof=1) / 8))
    zv = (av - cv) / sev; zh = (ah - ch) / seh; pooled = (zv + zh) / np.sqrt(2)
    ok = (zv >= 1) & (zh >= 1) & (pooled >= 2)
    if full:
        e = rng.standard_normal((n, 5))
        zp = 0.5 * eff_v / 0.00052 + 0.7 * (zv - eff_v / sev) + np.sqrt(0.51) * e[:, 0]
        pv = zv * (0.00036 / 0.00052) * 0.8 + 0.6 * e[:, 1]; ph = zh * (0.00029 / 0.00042) * 0.8 + 0.6 * e[:, 2]
        last = (ah - ch) + 0.5 * seh * e[:, 3] >= 0; sv = 0.5 * eff_v / sev + e[:, 4] >= -1
        ok = ok & (zp >= -1) & (pv >= 0) & (ph >= 0) & last & sv
    return ok.mean(), zv.std(), zh.std(), np.corrcoef(zv, zh)[0, 1]
P('case | seed sd val / holdout | rho | P(PASS first read): effect 0 | +0.0003 | +0.0005 | +0.0010 | +0.0020   (holdout effect = 0.7 x val effect) | sd of z_val under the null')
for lab, s_v, s_h, rho, tail in (('seed noise as implied by the per-feed SE (0.00072 / 0.00058), independent witnesses: the designer\'s case', 0.00072, 0.00058, 0.0, 0.0),
                                 ('the same with the measured correlation', 0.00072, 0.00058, None, 0.0),
                                 ('seed sd of the stored controls and of pop1 pooled (measured), measured correlation', None, None, None, 0.0),
                                 ('as the first case + 1 member in 22 lands 0.0029 low on val only (as abl_e10_pop1_s951 did)', 0.00072, 0.00058, None, 1 / 22)):
    hlv = np.sqrt((dv_all[:hl_n] ** 2).sum() / (hl_n - 4)); hlh = np.sqrt((dh_all[:hl_n] ** 2).sum() / (hl_n - 4)); r_ = float(np.corrcoef(dv_all[:hl_n], dh_all[:hl_n])[0, 1])
    sv_ = hlv if s_v is None else s_v; sh_ = hlh if s_h is None else s_h; rr = max(r_, 0.0) if rho is None else rho
    cells = [mc(d, 0.7 * d, sv_, sh_, rr, tail=tail) for d in (0.0, 0.0003, 0.0005, 0.001, 0.002)]
    P(f'{lab} | {sv_:.5f} / {sh_:.5f} | {rr:.2f} | ' + ' | '.join(f'{c[0]:.3f}' for c in cells) + f' | {cells[0][1]:.2f}')
    cells = [mc(d, 0.7 * d, sv_, sh_, rr, tail=tail, full=False) for d in (0.0, 0.0003, 0.0005, 0.001, 0.002)]
    P(f'    the three z clauses only (no private / pool / last-step / stale-val clause): ' + ' | '.join(f'{c[0]:.3f}' for c in cells))
out.close()
