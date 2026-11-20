"""poplag_verify step I (LABELLED, POST-HOC, NO DECISION): extra splits made AFTER the decision read was written down (logs/v_read_priv.txt).
None of this was pre-declared; it only describes where the subset result comes from.
  1. label-free: how much of the v3 dense window of a row lies AFTER the row's day, by holdout day (the "lagged" view is exposure-matched, not causal);
  2. S_0 / S_lag / D on Sep 13-14 (window contains later days) vs Sep 17-19 (window = past days only);
  3. decomposition (which pool's inputs move the statistic) on the complement and on all feeds.
  cd <repo>/nn_x/w12/next/poplag_verify && PV_THREADS=4 CUDA_VISIBLE_DEVICES="" ../../../../.venv/bin/python v_extra.py"""
import os
os.environ.setdefault('PV_THREADS', '4')
from v_common import *
import time

Mh, oh = np.load(ROOT / 'nn_x/w5/preds/holdout/holdout_meta.npy'); Mh = Mh.astype(np.int64); oh = oh.astype(np.int64)
yh = np.load(ROOT / 'nn_x/w5/preds/holdout/holdout_y.npy').astype(np.float32)
ftr, hold, _ = my_holdout()
hd = (ftr['timestamp'].to_numpy().astype(np.int64)[hold] + TZ) // DAY
late = hd >= SEP6 + 7
fid = np.repeat(np.arange(len(Mh)), Mh)
OUT = []


def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); OUT.append(s)


P(f'=== poplag_verify POST-HOC extras {time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())} (no decision; not pre-declared) ===')
# ---- 1. label-free: position of the v3 dense window relative to the row's day (private-like view)
D = np.load(STUDY / 'view_priv_D.npy').astype(np.int64)          # equal to my own derivation on all rows (logs/v_gate.log)
dt = hd[fid]
DATA_DAYS = np.array([d for d in range(SEP6, SEP20) if d - SEP6 not in (9, 10)])      # 12 dense days: Sep 6-14, Sep 17-19
nb = np.zeros(len(D), np.int64); na = np.zeros(len(D), np.int64)
for d in DATA_DAYS:
    inw = (d < D) & (np.abs(d - dt) >= 3)
    nb += inw & (d < dt); na += inw & (d > dt)
P(' label-free, private-like v3: dense days counted for a row (it_hist / seq_cv window = data days < D outside the +-2-day band), by holdout day:')
P('   day     | rows with >= 1 counted dense day | mean days BEFORE the row | mean days AFTER the row | rows whose window is ONLY later days | closest counted past day (old listings, D = Sep 20)')
for d in np.unique(hd):
    m = dt == d; anyw = (nb[m] + na[m]) > 0
    past = [x for x in DATA_DAYS if x < d and d - x >= 3]
    P(f'   Sep {int(d - SEP6 + 6):2d}{"*" if d >= SEP6 + 7 else " "} | {anyw.mean():.3f}                            | {nb[m].mean():.2f}                     | {na[m].mean():.2f}                    | {np.mean((nb[m] == 0) & (na[m] > 0)):.3f}                                | {("Sep " + str(int(max(past) - SEP6 + 6)) + f" ({int(d - max(past))} d before)") if past else "none"}')
P('   (* = subset days)')
for nm, m in (('subset', late[fid]), ('  Sep 13-14', (dt >= SEP6 + 7) & (dt <= SEP6 + 8)), ('  Sep 17-19', dt >= SEP6 + 11), ('complement', ~late[fid]), ('all', np.ones(len(D), np.bool_))):
    tot = nb[m].sum() + na[m].sum()
    P(f'   {nm:11s}: share of counted (row, dense day) pairs that are AFTER the row: {na[m].sum() / tot:.3f}; rows with any later day counted {np.mean(na[m] > 0):.3f}')

# ---- 2 / 3. labelled splits
ev = lambda s: ndcg_feed(s, yh, oh, Mh)
z = lambda x: zfeed(x, oh, Mh)
pool = lambda view, mem: np.mean([z(np.load(STUDY / f'pred_{view}_{v}.npy')) for v in mem], 0)
stat = lambda pw, pp, w=0.2: ev((1 - w) * z(pw).astype(np.float64) + w * z(pp).astype(np.float64)) - ev(pw)


def f(d, m):
    d = np.asarray(d, np.float64)[m]
    return f'{d.mean():+.5f} +- {d.std(ddof=1) / np.sqrt(len(d)):.5f}'


w0, p0, w1, p1 = pool('lag0', W8), pool('lag0', P4), pool('priv_v3', W8), pool('priv_v3', P4)
S0, S1, Sa, Sb = stat(w0, p0), stat(w1, p1), stat(w1, p0), stat(w0, p1)
e_w0, e_w1, e_p0, e_p1 = ev(w0), ev(w1), ev(p0), ev(p1)
G = {'subset (Sep 13-19)': late, '  Sep 13-14 (window has later days)': (hd >= SEP6 + 7) & (hd <= SEP6 + 8), '  Sep 17-19 (window = past days only)': hd >= SEP6 + 11,
     'complement (Sep 6-12)': ~late, '  Sep 6-8': hd <= SEP6 + 2, '  Sep 9-12': (hd >= SEP6 + 3) & (hd <= SEP6 + 6), 'all 30,000': np.ones(len(hd), np.bool_)}
P('\n labelled, w 0.2, private-like v3 (CPU fp32 both sides):')
P('   group                                 |     n | S_0                 | S_lag               | D = S_lag - S_0     | W8 pool change      | pop1kd pool change  | W8 inputs lagged only (vs S_0) | pop1kd input lagged only (vs S_0)')
for nm, m in G.items():
    P(f'   {nm:37s} | {int(m.sum()):5d} | {f(S0, m)} | {f(S1, m)} | {f(S1 - S0, m)} | {f(e_w1 - e_w0, m)} | {f(e_p1 - e_p0, m)} | {f(Sa - S0, m)}            | {f(Sb - S0, m)}')
a = (S1 - S0)[G['  Sep 13-14 (window has later days)']]; b = (S1 - S0)[G['  Sep 17-19 (window = past days only)']]
P(f'   D(Sep 13-14) - D(Sep 17-19) = {a.mean() - b.mean():+.5f} +- {np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)):.5f} (unpaired)')
a = S0[hd <= SEP6 + 8]; b = S0[hd >= SEP6 + 11]
P(f'   S_0 on Sep 6-14 {a.mean():+.5f} +- {a.std(ddof=1) / np.sqrt(len(a)):.5f} (n {len(a)}) vs Sep 17-19 {b.mean():+.5f} +- {b.std(ddof=1) / np.sqrt(len(b)):.5f} (n {len(b)}); difference {a.mean() - b.mean():+.5f} +- {np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)):.5f}')
# the same from the stored bf16 files (the lead's -0.00063), by the same day groups
ps = lambda mem: np.mean([z(np.load(stored_path(v))) for v in mem], 0)
SS = stat(ps(W8), ps(P4))
P('   stored bf16 statistic by group: ' + ' | '.join(f'{nm.strip()}: {f(SS, m)}' for nm, m in G.items()))
(HERE / 'logs' / 'v_extra.txt').write_text('\n'.join(OUT) + '\n')
