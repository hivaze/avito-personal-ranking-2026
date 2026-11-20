"""poplag_verify step H (LABELLED): the SUBSET READ pre-declared by the lead at 2026-11-15T01:08:21Z (last block of nn_x/w5/logs/E_kd_predeclared.txt).
Run only after v_subgate.py showed G1-G5 on the subset (private-like) and v_sample.py confirmed the staged predictions.
  Subset = holdout feeds with day >= Sep 13. S(w) = NDCG((1 - w) z(W8 pool) + w z(pop1kd pool)) - NDCG(z(W8 pool)); pools = member mean of per-feed z.
  S_lag: private-like lagged inputs (study's staged CPU fp32 predictions, confirmed on a sample by my own rebuild + rescoring);
  S_0: lag-0 inputs, CPU fp32 (study's replay, confirmed on samples and against the stored bf16 files); D = S_lag - S_0, paired per feed.
  RULE: S_lag >= -0.0002 AND D >= +1 SE(D) -> "bias explanation holds"; S_lag <= -0.0005 -> "view closed for student pools"; anything else -> "pick 2 at most".
NDCG / z / pooling = my own code (v_common.py). Everything else printed is a no-decision extra.
  cd <repo>/nn_x/w12/next/poplag_verify && PV_THREADS=4 CUDA_VISIBLE_DEVICES="" ../../../../.venv/bin/python v_read.py [pub]"""
import os
os.environ.setdefault('PV_THREADS', '4')
from v_common import *
import time, json

WITH_PUB = len(sys.argv) > 1 and sys.argv[1] == 'pub'
PUB_EARLY = all((MYC / f'pred_pub_v3_early_{v}.npy').exists() for v in W8 + P4)       # the complement part of the public-like view is printed only when all 12 members exist
Mh, oh = np.load(ROOT / 'nn_x/w5/preds/holdout/holdout_meta.npy'); Mh = Mh.astype(np.int64); oh = oh.astype(np.int64)
yh = np.load(ROOT / 'nn_x/w5/preds/holdout/holdout_y.npy').astype(np.float32)
ftr, hold, _ = my_holdout()
assert (ftr['M'].to_numpy()[hold] == Mh).all()
hd = (ftr['timestamp'].to_numpy().astype(np.int64)[hold] + TZ) // DAY
late = hd >= SEP6 + 7
fid = np.repeat(np.arange(len(Mh)), Mh)
ev = lambda s: ndcg_feed(s, yh, oh, Mh)
z = lambda x: zfeed(x, oh, Mh)
OUT = []


def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); OUT.append(s)


def load(view, v):
    if view == 'stored':
        return np.load(stored_path(v))
    if view in ('lag0', 'priv_v3'):
        return np.load(STUDY / f'pred_{view}_{v}.npy')
    x = np.empty(len(yh), np.float32)                                    # pub_v3: my own scoring, two parts
    lr = late[fid]
    x[lr] = np.load(MYC / f'pred_pub_v3_late_{v}.npy')
    x[~lr] = np.load(MYC / f'pred_pub_v3_early_{v}.npy') if PUB_EARLY else np.load(STUDY / f'pred_lag0_{v}.npy')[~lr]   # placeholder (never printed) while the early part is not scored
    return x


def pools(view):
    mem = {v: load(view, v) for v in W8 + P4}
    return np.mean([z(mem[v]) for v in W8], 0), np.mean([z(mem[v]) for v in P4], 0), {v: ev(mem[v]) for v in mem}


def stat(pw, pp, w):
    return ev((1 - w) * z(pw).astype(np.float64) + w * z(pp).astype(np.float64)) - ev(pw)


def ms(d, m):
    d = np.asarray(d, np.float64)[m]
    return d.mean(), d.std(ddof=1) / np.sqrt(len(d))


def f(d, m):
    a, b = ms(d, m)
    return f'{a:+.5f} +- {b:.5f}'


P(f'=== poplag_verify SUBSET READ {time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}; holdout feeds {len(Mh)}; subset (day >= Sep 13) {int(late.sum())} feeds, complement {int((~late).sum())}; +- = 1 SE paired per feed ===')
w0, p0, m0 = pools('lag0'); w1, p1, m1 = pools('priv_v3'); ws, ps, mst = pools('stored')
e_w0, e_p0, e_w1, e_p1 = ev(w0), ev(p0), ev(w1), ev(p1)
S0 = {w: stat(w0, p0, w) for w in (0.1, 0.2, 0.3)}; S1 = {w: stat(w1, p1, w) for w in (0.1, 0.2, 0.3)}; SS = {w: stat(ws, ps, w) for w in (0.1, 0.2, 0.3)}
ALL = np.ones(len(Mh), np.bool_)
RES = {}


def block(name, m, S1_, e_w1_, e_p1_, label):
    n = int(m.sum())
    P(f'\n##### {label}, {name}: {n} feeds #####')
    a, sa = ms(S1_[0.2], m); b, sb = ms(S0[0.2], m); dd, sd = ms(S1_[0.2] - S0[0.2], m)
    P(f' S_lag (w 0.2) = {a:+.5f} +- {sa:.5f} (z {a / sa:+.2f})')
    P(f' S_0   (w 0.2) = {b:+.5f} +- {sb:.5f} (z {b / sb:+.2f})   [same statistic from the STORED bf16 files on these feeds: {f(SS[0.2], m)}]')
    P(f' D = S_lag - S_0 = {dd:+.5f} +- {sd:.5f} (D / SE(D) = {dd / sd:+.2f}); per-feed corr(S_lag, S_0) = {np.corrcoef(S1_[0.2][m], S0[0.2][m])[0, 1]:.3f}')
    for w in (0.1, 0.3):
        P(f' w {w}: S_lag {f(S1_[w], m)} | S_0 {f(S0[w], m)} | D {f(S1_[w] - S0[w], m)}')
    P(f' W8 pool level: lag 0 {e_w0[m].mean():.5f} -> lagged {e_w1_[m].mean():.5f}: {f(e_w1_ - e_w0, m)}')
    P(f' pop1kd pool level: lag 0 {e_p0[m].mean():.5f} -> lagged {e_p1_[m].mean():.5f}: {f(e_p1_ - e_p0, m)}')
    P(f' pop1kd pool - W8 pool: lag 0 {f(e_p0 - e_w0, m)} | lagged {f(e_p1_ - e_w1_, m)} | change {f((e_p1_ - e_w1_) - (e_p0 - e_w0), m)}')
    return dict(n=n, S_lag=a, se_S_lag=sa, S_0=b, se_S_0=sb, D=dd, se_D=sd)


r = block('SUBSET (holdout days >= Sep 13)', late, S1, e_w1, e_p1, 'PRIVATE-LIKE view = DECISION READ')
RES['priv_subset'] = r
c1 = (r['S_lag'] >= -0.0002) and (r['D'] >= r['se_D'])
branch = 'bias explanation holds' if c1 else ('view closed for student pools' if r['S_lag'] <= -0.0005 else 'pick 2 at most')
P(f'\n RULE (01:08:21Z): S_lag >= -0.0002 [{r["S_lag"] >= -0.0002}] AND D >= +1 SE(D) [{r["D"] >= r["se_D"]}] -> holds; S_lag <= -0.0005 [{r["S_lag"] <= -0.0005}] -> closed; else pick 2 at most')
P(f' BRANCH: "{branch}"')
RES['branch'] = branch
P('\n================ NO-DECISION EXTRAS ================')
RES['priv_complement'] = block('complement (holdout days < Sep 13; the gate FAILS there)', ~late, S1, e_w1, e_p1, 'private-like view, no decision')
RES['priv_all'] = block('all 30,000 feeds (the gate FAILS; derived for completeness)', ALL, S1, e_w1, e_p1, 'private-like view, no decision')
P('\n member level change, private-like lagged - lag 0 (subset | complement), and CPU lag-0 level on the subset:')
for v in W8 + P4:
    P(f'  {v:28s} {f(m1[v] - m0[v], late)} | {f(m1[v] - m0[v], ~late)} | lag-0 CPU level on subset {m0[v][late].mean():.5f} (stored bf16 {mst[v][late].mean():.5f})')
# decomposition: which pool's inputs move the statistic (subset)
sa = stat(w1, p0, 0.2); sb = stat(w0, p1, 0.2)
P(f'\n decomposition on the subset (w 0.2): W8 lagged + pop1kd lag-0: {f(sa, late)} (change vs S_0 {f(sa - S0[0.2], late)}); W8 lag-0 + pop1kd lagged: {f(sb, late)} (change vs S_0 {f(sb - S0[0.2], late)})')
# by imposed lag and by holdout day
L = np.load(STUDY / 'view_priv_L.npy')
P(' by imposed lag (subset): ' + ' | '.join(f'L {l} (n {int((late & (L == l)).sum())}): S_lag {f(S1[0.2], late & (L == l))}, D {f(S1[0.2] - S0[0.2], late & (L == l))}' for l in np.unique(L)))
P(' by holdout day (all days), S_0 | S_lag | D:')
for d in np.unique(hd):
    m = hd == d
    P(f'   Sep {int(d - SEP6 + 6):2d} (n {int(m.sum())}{", subset" if d >= SEP6 + 7 else ""}): {f(S0[0.2], m)} | {f(S1[0.2], m)} | {f(S1[0.2] - S0[0.2], m)}')
# cold-share bins (per-feed share of rows whose lagged it_hist_all_luo is 0), subset
x = np.load(STUDY / 'lag_priv_v3_it_hist_all_luo.npy'); cs = np.bincount(fid, weights=(x == 0)) / Mh
P(f' cold share (rows with lagged it_hist_all_luo = 0) on the subset: mean {cs[late].mean():.3f}')
for lo, hi in ((0, 0.35), (0.35, 0.45), (0.45, 0.55), (0.55, 1.01)):
    m = late & (cs >= lo) & (cs < hi)
    if m.sum() > 200:
        P(f'   cold share [{lo}, {min(hi, 1.0)}) n {int(m.sum())}: S_0 {f(S0[0.2], m)} | S_lag {f(S1[0.2], m)} | D {f(S1[0.2] - S0[0.2], m)} | W8 pool change {f(e_w1 - e_w0, m)} | pop1kd pool change {f(e_p1 - e_p0, m)}')
# split-half stability of D on the subset (random halves of feeds)
rng = np.random.default_rng(1); idx = np.flatnonzero(late); dsub = (S1[0.2] - S0[0.2])[idx]; s1sub = S1[0.2][idx]
bs = np.array([[dsub[k].mean(), s1sub[k].mean()] for k in (rng.integers(0, len(idx), len(idx)) for _ in range(2000))])
P(f' bootstrap over subset feeds (2000 draws): P(S_lag >= -0.0002) = {np.mean(bs[:, 1] >= -0.0002):.3f}; P(S_lag <= -0.0005) = {np.mean(bs[:, 1] <= -0.0005):.3f}; P(D > 0) = {np.mean(bs[:, 0] > 0):.3f}')
if WITH_PUB:
    w2, p2, m2 = pools('pub_v3'); e_w2, e_p2 = ev(w2), ev(p2); S2 = {w: stat(w2, p2, w) for w in (0.1, 0.2, 0.3)}
    RES['pub_subset'] = block('SUBSET (holdout days >= Sep 13)', late, S2, e_w2, e_p2, 'PUBLIC-like view (my own scoring), no decision')
    if PUB_EARLY:
        RES['pub_complement'] = block('complement (holdout days < Sep 13; the gate FAILS there)', ~late, S2, e_w2, e_p2, 'PUBLIC-like view, no decision')
        RES['pub_all'] = block('all 30,000 feeds', ALL, S2, e_w2, e_p2, 'PUBLIC-like view, no decision')
    P(' member level change, public-like lagged - lag 0, subset: ' + ' | '.join(f'{v.split("_e10_")[1]} {f(m2[v] - m0[v], late)}' for v in W8 + P4))
    L2 = np.load(STUDY / 'view_pub_L.npy')
    P(' by imposed lag (subset, public-like): ' + ' | '.join(f'L {l} (n {int((late & (L2 == l)).sum())}): S_lag {f(S2[0.2], late & (L2 == l))}, D {f(S2[0.2] - S0[0.2], late & (L2 == l))}' for l in np.unique(L2)))
    P(f' private-like minus public-like on the subset: S {f(S1[0.2] - S2[0.2], late)}; W8 pool {f(e_w1 - e_w2, late)}; pop1kd pool {f(e_p1 - e_p2, late)}')
tag = ('pub' if PUB_EARLY else 'pub_subset_only') if WITH_PUB else 'priv'
(HERE / 'logs' / f'v_read_{tag}.txt').write_text('\n'.join(OUT) + '\n')
json.dump(RES, open(MYC / f'v_read_{tag}.json', 'w'), indent=1)
