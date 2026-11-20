"""INVENTORY D step 3, POST HOC follow-ups (declared as post hoc in the report; none of them can promote a signal by the rule of d03_rule.txt,
they describe the four hits): A = 20 placebo shuffles per hit; B = the three popularity/label-statistic hits rebuilt at test-like staleness on val
(train feeds with MSK date <= Sep 14 only: gap to val 7.5-9.5 days, like the 6.6-9.6 days of test rows); C = richer forms of the same-user earlier-feed
impression history (skip / click similarity, position-debiased user x category response); D = four more equality forms of the f columns.
CPU only, read-only."""
import sys, time, json
sys.path.insert(0, '<repo>/reports/INPUTS_1116/raw_fields')
from d03_lib import *
from d03_lib import _shuffle_within
t0 = time.time(); L = []
def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); L.append(s)
V = load_val(); Hd = load_hold()
ld = lambda sp, n: np.load(F / sp / f'{n}.npy', mmap_mode='r')
X = lambda n, sp: np.load(OUT / f'x_{n}_{sp}.npy').astype(np.float64)
# ---------------- A: placebo distribution, 20 shuffles
P('--- A: 20 within-feed shuffles per check (LR of the shuffled column) ---')
def placebo20(S, x, ctrl):
    C = S.ctrl[ctrl]; th0, ll0 = fit(S.off, S.M, S.y, C); out = []
    for k in range(20):
        xs = _shuffle_within(S.off, S.M, x, 77777 + 1000 * k); _, llp = fit(S.off, S.M, S.y, np.column_stack([C, xs]), th0=np.r_[th0, 0.0]); out.append(2 * (llp - ll0))
    return np.array(out)
for n, sp, S, ctrl in (('V2b_uc_ctr', 'val', V, 'TO'), ('V3_it_coec', 'val', V, 'TO'), ('V5a_it_s1share', 'val', V, 'TO'), ('V8a_f2_home', 'hold', Hd, 'H'), ('V8a_f2_home', 'val', V, 'TO'),
                       ('V2a_ui_reshown', 'val', V, 'TO'), ('V7_fresh_cnt', 'val', V, 'TO'), ('V8b_f1_tie', 'hold', Hd, 'H')):
    p = placebo20(S, X(n, sp), ctrl)
    P(f'{n:20s} {sp:5s} placebo LR over 20 shuffles: mean {p.mean():.2f} median {np.median(p):.2f} max {p.max():.2f} share >= 3.84: {np.mean(p >= 3.84):.2f}')
# ---------------- shared train arrays
tf = pl.read_parquet(F / 'train/feeds.parquet'); tts = tf['timestamp'].to_numpy().astype(np.int64); tday = (tts + 10800) // 86400
tu = tf['user_id'].to_numpy().astype(np.int64); toff = tf['cand_off'].to_numpy().astype(np.int64); tM = tf['M'].to_numpy().astype(np.int64)
D14 = int((np.datetime64('2026-09-14T00:00:00').astype('datetime64[s]').astype(np.int64)) // 86400)   # MSK date Sep 14 as a day number of (ts + 3h)
P(f'train MSK day numbers {tday.min()}..{tday.max()}; stale cut = day <= {D14} keeps {np.mean(tday <= D14):.4f} of train feeds')
fi = np.asarray(ld('train', 'cand_fidx')); y = np.asarray(ld('train', 'cand_target')).astype(np.float64); pos = np.asarray(ld('train', 'cand_pos')).astype(np.int64)
pc = np.bincount(pos, weights=y, minlength=301) / np.maximum(np.bincount(pos, minlength=301), 1); ex = pc[pos]
it = np.asarray(ld('train', 'cand_item_id')); cat = np.asarray(ld('train', 'cand_category_id')).astype(np.int64)
old = (tday <= D14)[fi]
vit = np.asarray(ld('val', 'cand_item_id')); vcat = np.asarray(ld('val', 'cand_category_id')).astype(np.int64); vu = V.user[V.fi].astype(np.int64)
# ---------------- B: test-like staleness on val
P('--- B: hits rebuilt from train feeds of MSK date <= Sep 14 (post hoc) ---')
NI = 111_400_000
clk = np.bincount(it[old], weights=y[old], minlength=NI); exs = np.bincount(it[old], weights=ex[old], minlength=NI)
x3s = np.log((clk[vit] + 1) / (exs[vit] + 1)); P(f'V3 stale: val rows whose item has impressions in the cut window {np.mean(exs[vit] > 0):.4f}')
del clk, exs
# user x category, stale and full, smoothed toward the category rate (20 pseudo-impressions), 0 for unknown pairs
def uc_form(mask):
    k = np.where(tu[fi] > 0, tu[fi] * 64 + cat, -1); m = mask & (k >= 0)
    d = pl.DataFrame({'k': k[m], 'y': y[m]}).group_by('k').agg(pl.len().alias('n'), pl.col('y').sum().alias('p'))
    cr = np.bincount(cat[mask], weights=y[mask], minlength=64) / np.maximum(np.bincount(cat[mask], minlength=64), 1)
    q = pl.DataFrame({'row': np.arange(len(vit)), 'k': np.where(vu > 0, vu * 64 + vcat, -1)}).join(d, on='k', how='inner')
    x = np.zeros(len(vit)); r = q['row'].to_numpy(); c = cr[vcat[r]]
    x[r] = np.log((q['p'].to_numpy() + 20 * c) / (q['n'].to_numpy() + 20) / c)
    return x, len(r) / len(vit)
x2s, cov = uc_form(old); P(f'V2b stale: val rows with user x category train impressions in the cut window {cov:.4f}')
x2f, covf = uc_form(np.ones(len(fi), bool)); P(f'V2b full rebuild (check of the builder): coverage {covf:.4f}, corr with the file-based column {np.corrcoef(x2f, X("V2b_uc_ctr", "val"))[0, 1]:.4f}')
# s1 share from snapshots of train feeds with date <= Sep 14 (other users only)
uq = np.unique(vit); lut = np.zeros(NI, bool); lut[uq] = True; n_e1 = np.zeros(len(uq)); n_s1 = np.zeros(len(uq)); vuu = np.unique(V.user[V.user > 0]); own = []
hi = ld('train', 'hist_item_id'); het = ld('train', 'hist_event_type'); hes = ld('train', 'hist_event_source'); hf = ld('train', 'hist_fidx'); oldf = tday <= D14
for a in range(0, len(hi), 100_000_000):
    i_ = np.asarray(hi[a:a + 100_000_000]); f_ = np.asarray(hf[a:a + 100_000_000]); m = (np.asarray(het[a:a + 100_000_000]) == 1) & lut[i_] & oldf[f_]
    i_ = i_[m]; s1 = (np.asarray(hes[a:a + 100_000_000])[m] == 1); u = tu[f_[m]]; k = np.searchsorted(uq, i_)
    n_e1 += np.bincount(k, minlength=len(uq)); n_s1 += np.bincount(k, weights=s1, minlength=len(uq)); mu = np.isin(u, vuu)
    own.append(pl.DataFrame({'u': u[mu], 'it': i_[mu].astype(np.int64), 's1': s1[mu].astype(np.int64)}))
own = pl.concat(own).group_by(['u', 'it']).agg(pl.len().alias('n'), pl.col('s1').sum().alias('ns1'))
rows = pl.DataFrame({'row': np.arange(len(vit)), 'u': vu, 'it': vit.astype(np.int64)}).join(own, on=['u', 'it'], how='inner').filter(pl.col('u') > 0)
k = np.searchsorted(uq, vit); e1 = n_e1[k].copy(); s1c = n_s1[k].copy(); e1[rows['row'].to_numpy()] -= rows['n'].to_numpy(); s1c[rows['row'].to_numpy()] -= rows['ns1'].to_numpy()
sh = (s1c + 1) / (e1 + 2); x5s = np.where(e1 > 0, np.log(sh / (1 - sh)), 0.0); P(f'V5a stale: val rows with n_e1 > 0 in the cut window {np.mean(e1 > 0):.4f}')
del lut, hi, het, hes, hf
for n, x in (('V3s_it_coec_stale', x3s), ('V2bs_uc_ctr_stale', x2s), ('V5as_it_s1share_stale', x5s)):
    np.save(OUT / f'x_{n}_val.npy', x.astype(np.float32))
    for ctrl in ('TO', 'FIN'): P(fmt(n, 'val', ctrl, lr_test(V, x, ctrl)))
# ---------------- C: same-user earlier-feed impression history, richer forms
P('--- C: earlier-feed impression history of the same user (post hoc) ---')
def table(sp):
    d = R / 'data/competition' / sp / 'item_embeddings'; meta = json.loads((d / 'data.meta').read_text())
    return np.memmap(d / 'data.memmap', dtype=np.int8, mode='r', shape=tuple(meta['shape']))
def unit(a):
    a = a.astype(np.float32); return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-3)
def gather(tab, idx):
    o = np.argsort(idx); out = np.empty((len(idx), 128), np.int8); out[o] = tab[idx[o]]; return out
@nb.njit(parallel=True, cache=True)
def simfeat(coff, M, C, soff, sn, Sv, sy):
    skip = np.zeros(len(C)); click = np.zeros(len(C))
    for f in nb.prange(len(M)):
        if sn[f] == 0: continue
        for i in range(coff[f], coff[f] + M[f]):
            ms = 0.0; mc = 0.0
            for j in range(soff[f], soff[f] + sn[f]):
                d = 0.0
                for q in range(128): d += C[i, q] * Sv[j, q]
                if sy[j] > 0:
                    if d > mc: mc = d
                elif d > ms: ms = d
            skip[i] = ms; click[i] = mc
    return skip, click
trtab = table('train'); tce = ld('train', 'cand_emb_idx'); crate = np.bincount(cat, weights=y, minlength=64) / np.maximum(np.bincount(cat, weights=ex, minlength=64), 1e-9)
def earlier(S, name, pairs, c_emb, c_cat, own_tab):
    """pairs: DataFrame (tf = target feed index in S, sf = source train feed index)."""
    pairs = pairs.sort(['tf', 'sf']); tfi = pairs['tf'].to_numpy(); sfi = pairs['sf'].to_numpy()
    srows = np.concatenate([np.arange(o, o + m) for o, m in zip(toff[sfi], tM[sfi])]); owner = np.repeat(tfi, tM[sfi])
    sn = np.bincount(owner, minlength=len(S.M)).astype(np.int64); soff = np.r_[0, np.cumsum(sn)[:-1]].astype(np.int64)
    se = np.asarray(tce[srows]); Sv = unit(gather(trtab, np.maximum(se, 0))); Sv[se < 0] = 0; sy = y[srows]
    C = unit(gather(own_tab, np.maximum(c_emb, 0))); C[c_emb < 0] = 0
    skip, click = simfeat(S.off, S.M, C, soff, sn, Sv, sy)
    # user x category response, position-debiased: log((clicks + 5 r_c)/(expected clicks + 5)) - log(r_c), r_c = category clicks / expected clicks
    d = pl.DataFrame({'k': owner.astype(np.int64) * 64 + cat[srows], 'p': sy, 'e': ex[srows]}).group_by('k').agg(pl.col('p').sum(), pl.col('e').sum())
    q = pl.DataFrame({'row': np.arange(len(c_cat)), 'k': S.fi.astype(np.int64) * 64 + c_cat}).join(d, on='k', how='inner')
    x = np.zeros(len(c_cat)); r = q['row'].to_numpy(); rc = crate[c_cat[r]]; x[r] = np.log((q['p'].to_numpy() + 5 * rc) / (q['e'].to_numpy() + 5) / rc)
    P(f'C {name}: target feeds with >= 1 earlier feed {np.mean(sn > 0):.4f} | source feeds {len(sfi)} rows {len(srows)} clicks {int(sy.sum())} | rows with skip_sim > 0 {np.mean(skip > 0):.4f} click_sim > 0 {np.mean(click > 0):.4f} uc pair known {len(r) / len(c_cat):.4f}')
    return dict(C1_skip_sim=skip, C2_click_sim=click, C3_uc_coec=x)
trf = pl.DataFrame({'sf': np.arange(len(tu)), 'u': tu, 'sday': tday}).filter(pl.col('u') > 0)
pv = pl.DataFrame({'tf': np.arange(len(V.M)), 'u': V.user.astype(np.int64)}).filter(pl.col('u') > 0).join(trf, on='u', how='inner').select('tf', 'sf')
cv = earlier(V, 'val', pv, np.asarray(ld('val', 'cand_emb_idx')), vcat, table('val'))
ph = pl.DataFrame({'tf': np.arange(len(Hd.M)), 'u': Hd.user.astype(np.int64), 'day': tday[Hd.hold]}).filter(pl.col('u') > 0).join(trf, on='u', how='inner').filter(pl.col('sday') < pl.col('day')).select('tf', 'sf')
ch = earlier(Hd, 'hold (earlier-day feeds)', ph, np.asarray(tce[Hd.rows]), cat[Hd.rows], trtab)
for n in cv:
    np.save(OUT / f'x_{n}_val.npy', cv[n].astype(np.float32)); np.save(OUT / f'x_{n}_hold.npy', ch[n].astype(np.float32))
    for ctrl in ('TO', 'FIN'): P(fmt(n, 'val', ctrl, lr_test(V, cv[n], ctrl)))
    P(fmt(n, 'hold', 'H', lr_test(Hd, ch[n], 'H')))
# ---------------- D: more equality forms of the f columns
P('--- D: equality forms of f2 / f5 / f1 (post hoc) ---')
@nb.njit(parallel=True, cache=True)
def grp(off, M, x):
    out = np.zeros(len(x))
    for f in nb.prange(len(M)):
        o, m = off[f], M[f]
        for k in range(m):
            c = 0
            for q in range(m):
                if x[o + q] == x[o + k]: c += 1
            out[o + k] = np.log(c)
    return out
def dforms(S, f1, f2, f5):
    return dict(D1_f2_zero=(f2 == 0).astype(np.float64), D2_f2_grp=grp(S.off, S.M, f2), D3_f5_tie=grp(S.off, S.M, f5), D4_f1_floor=(f1 < 0.018).astype(np.float64))
dv = dforms(V, np.asarray(ld('val', 'cand_f1')), np.asarray(ld('val', 'cand_f2')), np.asarray(ld('val', 'cand_f5')))
dh = dforms(Hd, np.asarray(ld('train', 'cand_f1')[Hd.rows]), np.asarray(ld('train', 'cand_f2')[Hd.rows]), np.asarray(ld('train', 'cand_f5')[Hd.rows]))
for n in dv:
    P(f'{n}: mean val {dv[n].mean():.4f} hold {dh[n].mean():.4f}')
    for ctrl in ('TO', 'FIN'): P(fmt(n, 'val', ctrl, lr_test(V, dv[n], ctrl)))
    P(fmt(n, 'hold', 'H', lr_test(Hd, dh[n], 'H')))
P(f'done {time.time() - t0:.0f}s')
(OUT / 'd05_followup.txt').write_text('\n'.join(L) + '\n')
