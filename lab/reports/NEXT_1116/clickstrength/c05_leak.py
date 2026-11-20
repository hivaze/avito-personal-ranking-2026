"""Lane A c05: gate g5 of RULE.txt (leak scans on the witness arrays; form of reports/INPUTS_1116/nx3_verify.md).
 (1) causality by real timestamps: newest source feed of the window against the row, every feed with a view of the five query sets
 (2) re-derivation by a SECOND code path (vectorised numpy + polars; none of the numba kernels of c01_build.py, not lens.feed_rk): position weights, shrinkage
     constants, c / e / imp3 and the entered value on all rows of 25 sampled feeds per query set
 (3) holdout feeds are never a source (the re-derivation with holdout feeds admitted must differ); label arrays opened by the builder (code scan)
 (4) own-feed test: mean within-feed AUC of the entered value for the click, pool rows (their feeds are sources of other rows) against holdout rows (never a source)
 (5) the checks can fail: the deliberately leaky build (*_LEAK: window up to and including the row's own day) must change values and be caught by (1) and (4)
usage: c05_leak.py   output out/c05_leak.txt, out/c05_leak.json"""
import json, re
from scipy.stats import poisson, rankdata
from cs_common import *
P = Log(OUT / 'c05_leak.txt')
P('RULE.txt md5 ' + md5(CS + '/RULE.txt') + '; c01_build.py md5 ' + md5(CS + '/c01_build.py'))
V = Views(P)
a = V.A['train']; n = a['n']; RES = {}; FAIL = []
CONST = json.load(open(OUT / 'c01_constants.json'))
# ------------------------------------------------------------------ (1) causality by timestamps
P('=== (1) causality by real timestamps ===')
src_ok = ~V.ish & (S12[a['slot']] >= 0)
maxts = np.full(24, -1, np.int64); mints = np.full(24, 1 << 62, np.int64)
np.maximum.at(maxts, a['slot'][src_ok], a['ts'][src_ok]); np.minimum.at(mints, a['slot'][src_ok], a['ts'][src_ok])


def caus(q, e_feed, label):
    sp, sel, _ = V.Q[q]; ts = V.A[sp]['ts'][sel]; sl = V.A[sp]['slot'][sel]; ok = e_feed >= 0
    newest = maxts[e_feed[ok]]; d = (ts[ok] - newest) / 86400.0; bad = int((d < 2.0).sum()); slotbad = int((e_feed[ok] > sl[ok] - 3).sum())
    P(f'  {label}: feeds with a view {int(ok.sum())}; (row time - newest source feed of its window): min {d.min():.3f} d, median {np.median(d):.3f} d; feeds with a source at or after (row time - 2 d): {bad}; feeds with newest source slot > row slot - 3: {slotbad}')
    return bad, slotbad, float(d.min())


c1 = {}
for q in QS:
    c1[q] = caus(q, V.Q[q][2], q)
    if c1[q][0] or c1[q][1]:
        FAIL.append(f'(1) {q}')
RES['causality'] = {q: dict(bad_time=v[0], bad_slot=v[1], min_days=v[2]) for q, v in c1.items()}
# ------------------------------------------------------------------ (2) second code path
P('=== (2) re-derivation by a second code path ===')
item = np.load(FLAT / 'train/cand_item_id.npy'); pos = np.load(FLAT / 'train/cand_pos.npy'); y = np.load(FLAT / 'train/cand_target.npy'); fidx = np.load(FLAT / 'train/cand_fidx.npy')
ci = {s: np.load(FLAT / s / 'cand_item_id.npy') for s in ('val', 'test')}; ci['train'] = item
praw = np.nan_to_num(np.load(ROOT / 'eda/labels_position/out/train_prandom.npz')['p_random'].astype(np.float32), nan=0.0); regX = praw > 0.5
slot_r = a['slot'][fidx].astype(np.int8); ok_r = src_ok[fidx]; rx_r = regX[fidx]
m04 = ok_r & (slot_r <= 4)
W2 = np.zeros((2, 300))
for r in (0, 1):
    m = m04 & (rx_r == bool(r)); im = np.bincount(pos[m], minlength=300).astype(np.float64); ck = np.bincount(pos[m], weights=y[m], minlength=300)
    W2[r] = (ck + 20.0 * ck.sum() / im.sum()) / (im + 20.0)
Wst = np.load(OUT / 'c01_posweights.npy'); dW = float(np.abs(W2 - Wst).max())
P(f'  position weights: max abs difference to the builder {dW:.2e}')
w_r = W2[rx_r.astype(np.int64), pos]
sw_f = np.bincount(fidx, weights=w_r, minlength=n); np_f = np.bincount(fidx, weights=y, minlength=n)
q_r = np_f[fidx] / sw_f[fidx] * w_r
del w_r
NITEM = int(max(int(item.max()), int(ci['val'].max()), int(ci['test'].max()))) + 1
cst = {}
for k, m in (('all', ok_r & (slot_r <= 2)), ('expl', ok_r & (slot_r <= 2) & rx_r)):
    c0 = np.bincount(item[m], weights=y[m], minlength=NITEM); e0 = np.bincount(item[m], weights=q_r[m], minlength=NITEM); s = e0 > 0
    tau2 = float((((c0[s] - e0[s]) ** 2) - e0[s]).sum() / (e0[s] ** 2).sum()); a_ = 50.0 if tau2 <= 0 else float(np.clip(1 / tau2, 0.5, 50.0)); cst[k] = a_
    P(f'  shrinkage {k}: tau2 {tau2:.5f} -> a {a_:.4f}; builder: tau2 {CONST["tau2_" + k]:.5f}, a {CONST["a_" + k]:.4f}')
    if abs(a_ - CONST['a_' + k]) > 1e-3 * CONST['a_' + k]:
        FAIL.append(f'(2) shrinkage {k}')
    del c0, e0
if dW > 1e-9:
    FAIL.append('(2) position weights')
rng = np.random.default_rng(20261116)
SAMP = {}
for q in QS:
    sp, sel, e = V.Q[q]; rows, fl, off, M = V.rows(q); cand = np.nonzero(e >= 0)[0]; pick = np.sort(rng.choice(cand, 25, replace=False))
    rr = np.concatenate([np.arange(off[f], off[f] + M[f]) for f in pick])
    SAMP[q] = dict(pick=pick, rr=rr, feed=np.repeat(pick, M[pick]), item=np.asarray(ci[sp][rows[rr]]).astype(np.int64), e=np.repeat(e[pick], M[pick]), ts=np.repeat(V.A[sp]['ts'][sel][pick], M[pick]))
allit = np.unique(np.concatenate([s['item'] for s in SAMP.values()]))
tab = np.zeros(NITEM, bool); tab[allit] = True
mrow = tab[item] & (S12[a['slot']] >= 0)[fidx]            # rows of the sampled items on dense slots, holdout feeds INCLUDED (flagged)
df = pl.DataFrame(dict(item=item[mrow].astype(np.int64), slot=slot_r[mrow].astype(np.int64), y=y[mrow].astype(np.float64), q=q_r[mrow], isx=rx_r[mrow].astype(np.float64), ish=V.ish[fidx[mrow]], ts=a['ts'][fidx[mrow]]))
P(f'  sampled feeds 25 per query set: rows {sum(len(s["rr"]) for s in SAMP.values())}, distinct items {len(allit)}; train rows of these items on dense slots {df.height} (of them in holdout feeds {int(df["ish"].sum())})')


def agg(d):
    return d.group_by(['item', 'slot']).agg(imp=pl.len().cast(pl.Float64), ka=pl.col('y').sum(), ea=pl.col('q').sum(), kx=(pl.col('y') * pl.col('isx')).sum(), ex=(pl.col('q') * pl.col('isx')).sum(), tsmax=pl.col('ts').max())


G = agg(df.filter(~pl.col('ish'))); GH = agg(df)


def window_sums(S, G_):
    k = len(S['item']); long = pl.DataFrame(dict(rid=np.tile(np.arange(k), 3), item=np.tile(S['item'], 3), slot=np.concatenate([S['e'] - 2, S['e'] - 1, S['e']])))
    j = long.join(G_, on=['item', 'slot'], how='left').group_by('rid').agg(imp=pl.col('imp').sum(), ka=pl.col('ka').sum(), ea=pl.col('ea').sum(), kx=pl.col('kx').sum(), ex=pl.col('ex').sum(), tsmax=pl.col('tsmax').max()).sort('rid')
    assert j.height == k
    return {c: j[c].fill_null(0).to_numpy().astype(np.float64) for c in ('imp', 'ka', 'ea', 'kx', 'ex')}, j['tsmax'].fill_null(-1).to_numpy()


def mfun(a_, e):
    u, inv = np.unique(e, return_inverse=True); out = np.zeros(len(u))
    for i, e_ in enumerate(u):
        if e_ > 0:
            K = int(np.ceil(e_ + 12 * np.sqrt(e_) + 30)); kk = np.arange(K + 1); out[i] = float((poisson.pmf(kk, e_) * np.log((kk + a_) / (e_ + a_))).sum())
    return out[inv]


def entered(x, imp_, feed):
    out = np.zeros(len(x))
    for f in np.unique(feed):
        m = feed == f; c = int(m.sum())
        rk = np.float32((rankdata(imp_[m], method='average') - 1.0) / (c - 1) - 0.5) if c > 1 else np.float32(0.0)
        b = np.minimum((NB * (np.asarray(rk, np.float32) + 0.5)).astype(np.int64), NB - 1)
        xm = x[m]; o = np.zeros(c)
        for bb in np.unique(b):
            o[b == bb] = xm[b == bb] - xm[b == bb].mean()
        out[m] = o
    return out


tot = dict(rows=0, mis_counts=0, mis_e=0, mis_entered=0, changed_by_holdout=0); mindays = []
for q in QS:
    S = SAMP[q]; w, tsm = window_sums(S, G); wh, _ = window_sums(S, GH)
    st = {nm: get(q, nm)[S['rr']] for nm in ('imp3', 'cs3a_k', 'cs3a_e', 'cs3x_k', 'cs3x_e', 'cs3a_c', 'cs3x_c')}
    mc = int((w['imp'] != st['imp3']).sum() + (w['ka'] != st['cs3a_k']).sum() + (w['kx'] != st['cs3x_k']).sum())
    me = int((np.abs(w['ea'] - st['cs3a_e']) > 1e-5 + 1e-4 * w['ea']).sum() + (np.abs(w['ex'] - st['cs3x_e']) > 1e-5 + 1e-4 * w['ex']).sum())
    xa = np.log((w['ka'] + cst['all']) / (w['ea'] + cst['all'])) - mfun(cst['all'], w['ea']); xx = np.log((w['kx'] + cst['expl']) / (w['ex'] + cst['expl'])) - mfun(cst['expl'], w['ex'])
    ea_ = entered(xa, w['imp'], S['feed']); ex_ = entered(xx, w['imp'], S['feed'])
    da = np.abs(ea_ - st['cs3a_c']); dx = np.abs(ex_ - st['cs3x_c']); men = int((da > 1e-5).sum() + (dx > 1e-6).sum())
    ch = int(((wh['imp'] != w['imp']) | (wh['ka'] != w['ka'])).sum())
    seen = tsm > 0; dd = (S['ts'][seen] - tsm[seen]) / 86400.0; mindays.append(float(dd.min()))
    P(f'  {q}: rows {len(S["rr"])} of 25 feeds; mismatches imp3 / clicks {mc}, expected clicks {me}, entered value {men} (max abs difference F1 {da.max():.2e}, F2 {dx.max():.2e}); '
      f'rows that would change if holdout feeds were sources {ch}; (row time - newest CONTRIBUTING source row): min {dd.min():.3f} d over {int(seen.sum())} seen rows')
    tot['rows'] += len(S['rr']); tot['mis_counts'] += mc; tot['mis_e'] += me; tot['mis_entered'] += men; tot['changed_by_holdout'] += ch
P(f'  TOTAL: {tot["rows"]} rows; mismatches counts {tot["mis_counts"]}, expected clicks {tot["mis_e"]}, entered value {tot["mis_entered"]}; rows that differ when holdout feeds are admitted {tot["changed_by_holdout"]} (the check can fail); min (row time - contributing source row) {min(mindays):.3f} d')
RES['rederive'] = dict(tot, min_days_contributing=min(mindays), posw_maxdiff=dW)
if tot['mis_counts'] or tot['mis_e'] or tot['mis_entered'] or min(mindays) < 2.0:
    FAIL.append('(2) re-derivation')
del df, q_r, tab, mrow
# ------------------------------------------------------------------ (3) labels opened by the builder
P('=== (3) sources and label arrays of the builder ===')
src = open(CS + '/c01_build.py').read(); opened = sorted(set(re.findall(r"FLAT / (?:s / )?'([^']+)'", src)) | set(re.findall(r"'((?:train|val|test)/cand_[a-z_0-9]+\.npy)'", src)))
P(f'  arrays opened by c01_build.py (code scan): {opened}; "cand_target" occurrences: {re.findall(r"[a-z/ ]*cand_target[.a-z]*", src)}')
P(f'  holdout feeds among the source feeds of the builder: 0 (asserted in c01_build.py; fs12 = -1 on all {len(V.hold)} holdout feeds); admitting them changes {tot["changed_by_holdout"]} of the {tot["rows"]} re-derived rows')
if any('val/cand_target' in s or 'test/cand_target' in s for s in opened) or tot['changed_by_holdout'] == 0:
    FAIL.append('(3) labels / holdout sources')
# ------------------------------------------------------------------ (4), (5) own-feed test and the leaky build


@nb.njit(parallel=True, cache=True)
def feed_auc(x, yy, off, M):
    n_ = len(M); out = np.full(n_, np.nan)
    for f in nb.prange(n_):
        o = off[f]; m = M[f]; v = x[o:o + m]
        if not (v[0] == v[0]):
            continue
        npos = 0
        for k in range(m):
            if yy[o + k] > 0:
                npos += 1
        if npos == 0 or npos == m:
            continue
        order = np.argsort(v); s = 0.0; i = 0
        while i < m:
            j = i
            while j + 1 < m and v[order[j + 1]] == v[order[i]]:
                j += 1
            r = 0.5 * (i + j) + 1.0
            for t in range(i, j + 1):
                if yy[o + order[t]] > 0:
                    s += r
            i = j + 1
        out[f] = (s - npos * (npos + 1) / 2.0) / (npos * (m - npos))
    return out


P('=== (4) own-feed test: mean within-feed AUC of the entered value for the click (feeds with a view), +- 1 SE over feeds; (5) the leaky build ===')
Y = {}; OFFM = {}
for q in ('pool', 'hold'):
    rows, fl, off, M = V.rows(q); Y[q] = np.asarray(y[rows]).astype(np.float64); OFFM[q] = (off, M)
RES['auc'] = {}
for col in ('cs3a_c', 'cs3x_c', 'cs3a_c_LEAK', 'cs3x_c_LEAK'):
    cells = []; worst = -9.0
    for lo, hi, hn in ((6, 8, 'slots 6-8'), (11, 13, 'slots 11-13')):
        st = {}
        for q in ('pool', 'hold'):
            off, M = OFFM[q]; au = feed_auc(np.load(arr_path(q, col)).astype(np.float64), Y[q], off, M)
            sl = V.A['train']['slot'][V.Q[q][1]]; m = (sl >= lo) & (sl <= hi) & ~np.isnan(au)
            st[q] = (float(au[m].mean()), float(au[m].std() / np.sqrt(m.sum())), int(m.sum()))
        d = st['pool'][0] - st['hold'][0]; se = float(np.hypot(st['pool'][1], st['hold'][1])); worst = max(worst, d / se)
        cells.append(f'{hn}: pool {st["pool"][0]:.4f} +- {st["pool"][1]:.4f} ({st["pool"][2]} feeds), holdout {st["hold"][0]:.4f} +- {st["hold"][1]:.4f} ({st["hold"][2]} feeds), pool - holdout {d:+.4f} = {d / se:+.1f} SE')
        RES['auc'][f'{col}|{hn}'] = dict(pool=st['pool'], hold=st['hold'], diff=d, diff_se=se)
    flag = worst > 3.0
    P(f'  {col:12s} | ' + ' | '.join(cells) + f' -> {"FLAGGED (pool above holdout by more than 3 SE)" if flag else "ok"}')
    if col.endswith('_LEAK'):
        if not flag:
            FAIL.append(f'(5) the own-feed test does not catch {col}')
    elif flag:
        FAIL.append(f'(4) {col}')
for q in ('pool', 'hold'):
    for k in ('cs3a_c', 'cs3x_c'):
        a0 = np.nan_to_num(np.load(arr_path(q, k))); a1 = np.nan_to_num(np.load(arr_path(q, k + '_LEAK'))); v = V.row_e(q) >= 0
        P(f'  (5) leaky build {q} {k}: rows with a different value {float((a0[v] != a1[v]).mean()):.4f}; corr with the build of record {float(np.corrcoef(a0[v], a1[v])[0, 1]):.3f}')
        if float((a0[v] != a1[v]).mean()) < 0.01:
            FAIL.append(f'(5) leaky build equals the record on {q} {k}')
sl = V.A['train']['slot']
for q in ('pool', 'hold'):
    sel = V.Q[q][1]; ok = V.Q[q][2] >= 0; eleak = np.where(ok, sl[sel], -1)
    b, sb, md = caus(q, eleak, f'(5) causality check on the leaky windows, {q}')
    if b == 0 or sb == 0:
        FAIL.append(f'(5) causality check does not catch the leaky build on {q}')
RES['fail'] = FAIL
P('NOT RUN (only after a GO): the same audit on all 246.7M train rows; the share of rows whose own user contributed an earlier impression of the same item (stated limit of RULE.txt section 2).')
P(f'=== g5: {"PASS" if not FAIL else "MISS: " + "; ".join(FAIL)} ===')
json.dump(RES, open(OUT / 'c05_leak.json', 'x'), indent=1, default=float)
P('DONE')
