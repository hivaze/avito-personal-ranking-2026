"""CRITIC check 1 (CPU, read-only): verify the four staged files and re-derive their headline numbers with own code.
Own code: per-feed z (float64 bincount), NDCG (own sort implementation + brute-force loop on a feed sample), pool rebuild from MEMBER files in preds/,
file integrity from the parquet itself. Only src/pmix.weights() is reused (it is the definition of the public / private feed mix)."""
import sys, hashlib, json, time
sys.path.insert(0, '<repo>/src')
import numpy as np, polars as pl
from pathlib import Path
import pmix
R = Path('<repo>'); P = R / 'preds'; FLAT = R / 'data/flat'
OUT = R / 'reports/TRICKS_1115/critic_checks'
L = []
def w_(s=''):
    print(s, flush=True); L.append(s)
fv = pl.read_parquet(FLAT / 'val/feeds.parquet'); Mv = fv['M'].to_numpy().astype(np.int64); ov = fv['cand_off'].to_numpy().astype(np.int64)
ft = pl.read_parquet(FLAT / 'test/feeds.parquet'); Mt = ft['M'].to_numpy().astype(np.int64); ot = ft['cand_off'].to_numpy().astype(np.int64)
yv = np.load(FLAT / 'val/cand_target.npy').astype(np.float64)
FID = {'val': np.repeat(np.arange(len(Mv)), Mv), 'test': np.repeat(np.arange(len(Mt)), Mt)}
MM = {'val': Mv, 'test': Mt}
assert (np.diff(ov) == Mv[:-1]).all() and (np.diff(ot) == Mt[:-1]).all() and ov[0] == 0 and ot[0] == 0
def z(x, sp):
    M = MM[sp]; fid = FID[sp]; x = np.asarray(x, np.float32).astype(np.float64)
    mu = np.bincount(fid, x, len(M)) / M
    sd = np.sqrt(np.maximum(np.bincount(fid, x * x, len(M)) / M - mu * mu, 0.0)) + 1e-9
    return (x - mu[fid]) / sd[fid]
def ndcg(s):
    """own implementation: binary gain, log2 discount, pessimistic ties (negatives first among equal scores), feeds without positives -> 1"""
    s = np.asarray(s, np.float32).astype(np.float64); fid = FID['val']
    o = np.lexsort((yv, -s, fid)); pos = np.arange(len(s)) - ov[fid]            # sorted order is feed-contiguous
    d = 1.0 / np.log2(pos + 2.0)
    dcg = np.bincount(fid, yv[o] * d, len(Mv)); npos = np.bincount(fid, yv, len(Mv))
    idcg = np.bincount(fid, (pos < npos[fid]) * d, len(Mv))
    out = np.ones(len(Mv)); m = idcg > 0; out[m] = dcg[m] / idcg[m]; return out
def ndcg_brute(s, feeds):
    s = np.asarray(s, np.float32).astype(np.float64); out = []
    for f in feeds:
        a, b = ov[f], ov[f] + Mv[f]; sc = s[a:b]; y = yv[a:b]; k = int(y.sum())
        if k == 0: out.append(1.0); continue
        dcg = 0.0
        for i in np.where(y > 0)[0]:
            # pessimistic rank: every item with a higher score, every negative with an equal score, and equal-score positives before it
            r = int((sc > sc[i]).sum() + ((sc == sc[i]) & (y == 0)).sum() + ((sc == sc[i]) & (y > 0))[:i].sum())
            dcg += 1.0 / np.log2(r + 2.0)
        out.append(dcg / sum(1.0 / np.log2(j + 2.0) for j in range(k)))
    return np.array(out)
W = pmix.weights(); WPUB, WPRIV = W['public'], W['private']
def wm(d, w=None):
    if w is None: return d.mean(), d.std() / np.sqrt(len(d))
    m = np.average(d, weights=w); ne = w.sum() ** 2 / (w ** 2).sum(); return m, np.sqrt(np.average((d - m) ** 2, weights=w) / ne)
f3 = lambda t: f'{t[0]:+.5f} +- {t[1]:.5f}'
SUF = {'val': ('n2cpv3s3_val_oof', 'trainonly_val'), 'test': ('n2cpv3s3_test', 'trainonly_test')}
def pool(names, sp):
    a = np.zeros(len(FID[sp])); b = np.zeros(len(FID[sp]))
    for m in names:
        a += z(np.load(P / f'nn_{m}_{SUF[sp][0]}.npy'), sp); b += z(np.load(P / f'nn_{m}_{SUF[sp][1]}.npy'), sp)
    return a / len(names), b / len(names)
def branches(base, pop, w, sp):
    b = pool(base, sp)
    if pop:
        p = pool(pop, sp); b = ((1 - w) * z(b[0], sp) + w * z(p[0], sp), (1 - w) * z(b[1], sp) + w * z(p[1], sp))
    return b
fin = lambda br, sp, wft=0.5: wft * z(br[0], sp) + (1 - wft) * z(br[1], sp)
S8 = range(951, 959); S4 = range(951, 955)
wide = lambda pre, seeds: [f'w5_{pre}_{a}_s{s}' for a in ('kd05w192', 'kd05t25w192') for s in seeds]
MEM = {
 'W8':      (wide('e10', S4), [], 0.0),
 'W8P20':   (wide('e10', S4), [f'abl_e10_pop1kd05t25_s{s}' for s in S4], 0.2),
 'FD16P20': (wide('fd10', S8), [f'abl_fd10_pop1kd05w192_s{s}' for s in S4], 0.2),
}
def read_members(p):
    base, pop, cur = [], [], None
    for ln in open(p):
        ln = ln.strip()
        if ln.startswith('# base'): cur = base
        elif ln.startswith('# popularity-light'): cur = pop
        elif ln and not ln.startswith('#'): cur.append(ln)
    return base, pop
ST1 = R / 'nn_x/w12/ens_1115/stage'; ST2 = R / 'nn_x/w12/ft_1115/stage'
for t in ('ENSA', 'ENSB'):
    b, p = read_members(ST1 / f'{t}_members.txt'); MEM[t] = (b, p, 0.2)
t0 = time.time()
BR = {k: {sp: branches(*MEM[k], sp) for sp in ('val', 'test')} for k in MEM}
w_(f'# pools rebuilt from member files in preds/ ({sum(len(v[0]) + len(v[1]) for v in MEM.values())} member loads x 4 files, {time.time() - t0:.0f} s)')
for k, (b, p, w) in MEM.items():
    hold = sum('_e10_' in m for m in b + p); w_(f'{k}: {len(b)} base + {len(p)} popularity-light at {w}; members that exclude the holdout feeds (name _e10_): {hold} of {len(b) + len(p)}')
# stored branch files against the member rebuild
w_(); w_('## stored branch files against own rebuild from members (max abs difference, correlation)')
STORED = {'W8': (P, 'W8kd'), 'W8P20': (P, 'W8P20kd'), 'FD16P20': (P, 'FD16P20kd'), 'ENSA': (ST1, 'ENSAkd'), 'ENSB': (ST1, 'ENSBkd')}
for k, (d, tag) in STORED.items():
    for sp in ('val', 'test'):
        for i, br in enumerate(('ft', 'to')):
            s = np.load(d / f'ens_{tag}_{br}_{sp}.npy').astype(np.float64); mine = BR[k][sp][i]
            w_(f'{k} {br} {sp}: max|d| {np.abs(s - mine).max():.2e} corr {np.corrcoef(s, mine)[0, 1]:.9f} finite {bool(np.isfinite(s).all())}')
# candidate score vectors (val and test)
C = {
 'probe_W8_FTonly_wft100': {sp: fin(BR['W8'][sp], sp, 1.0) for sp in ('val', 'test')},
 'cand_FD16P20_wft060':    {sp: fin(BR['FD16P20'][sp], sp, 0.6) for sp in ('val', 'test')},
 'cand_ENSAkd_stage':      {sp: fin(BR['ENSA'][sp], sp) for sp in ('val', 'test')},
 'cand_ENSBkd_stage':      {sp: fin(BR['ENSB'][sp], sp) for sp in ('val', 'test')},
}
REF = {k: {sp: fin(BR[k][sp], sp) for sp in ('val', 'test')} for k in ('FD16P20', 'W8P20', 'W8')}
E = {k: ndcg(v['val']) for k, v in {**C, **REF}.items()}
# brute-force NDCG check
rng = np.random.default_rng(7); smp = rng.choice(len(Mv), 400, replace=False)
bf = ndcg_brute(REF['FD16P20']['val'], smp); w_(); w_(f'## NDCG implementation check: brute-force loop on 400 random val feeds, max abs difference to the sort implementation {np.abs(bf - E["FD16P20"][smp]).max():.2e}')
w_(f'feeds {len(Mv)} val rows {len(yv)} | pmix n_eff public {WPUB.sum() ** 2 / (WPUB ** 2).sum():.0f} private {WPRIV.sum() ** 2 / (WPRIV ** 2).sum():.0f}')
w_(); w_('## references, final val (own code) and record')
for k, rec, pub in (('FD16P20', 0.40271, 0.394892), ('W8P20', 0.40214, 0.3950094), ('W8', 0.40165, 0.394486)):
    w_(f'{k}: {E[k].mean():.5f} (record {rec:.5f}) | public {pub}')
d = E['FD16P20'] - E['W8P20']; w_(f'FD16P20 - W8P20: val {f3(wm(d))} | pmix-public {f3(wm(d, WPUB))} | pmix-private {f3(wm(d, WPRIV))} | paired public SE {d.std() / np.sqrt(8562):.5f}')
w_(); w_('## staged files: headline numbers re-derived (paired per feed over 14,423 val feeds)')
RES = {}
for k in C:
    r = {'final_val': float(E[k].mean())}
    w_(f'{k}: final val {E[k].mean():.5f}')
    for ref in ('FD16P20', 'W8P20') + (('W8',) if k.startswith('probe') else ()):
        d = E[k] - E[ref]; a, b, c = wm(d), wm(d, WPUB), wm(d, WPRIV)
        tc = np.corrcoef(C[k]['test'], REF[ref]['test'])[0, 1]
        w_(f'   vs {ref:8s}: val {f3(a)} (z {a[0] / a[1]:+.2f}) | pmix-public {f3(b)} (z {b[0] / b[1]:+.2f}) | pmix-private {f3(c)} (z {c[0] / c[1]:+.2f}) | paired public SE {d.std() / np.sqrt(8562):.5f} | test corr {tc:.5f}')
        r[ref] = dict(val=a, pub=b, priv=c, pse=float(d.std() / np.sqrt(8562)), corr=float(tc))
    RES[k] = r
# file integrity
w_(); w_('## file integrity (parquet read back; order compared with own rebuild of the test scores)')
ss = pl.read_parquet(R / 'data/raw/sample_submission.parquet')
tfid = ft['feed_id'].to_numpy(); cfid = tfid[np.load(FLAT / 'test/cand_fidx.npy')]; citem = np.load(FLAT / 'test/cand_item_id.npy')
refsub = pl.read_parquet(R / 'subs/cand_FD16P20kd_r4.parquet')
FILES = {'probe_W8_FTonly_wft100': ST2 / 'probe_W8_FTonly_wft100.parquet', 'cand_FD16P20_wft060': ST2 / 'cand_FD16P20_wft060.parquet',
         'cand_ENSAkd_stage': ST1 / 'cand_ENSAkd_stage.parquet', 'cand_ENSBkd_stage': ST1 / 'cand_ENSBkd_stage.parquet'}
def myrank(s):
    # within-feed ascending rank of own scores; ties (none expected in float64) by position
    o = np.lexsort((s, FID['test'])); r = np.empty(len(s)); r[o] = np.arange(len(s)) - ot[FID['test']][o] if False else 0
    pos = np.arange(len(s)) - np.repeat(ot, Mt); r[o] = pos; return r
for k, p in FILES.items():
    h = hashlib.sha256(open(p, 'rb').read()).hexdigest(); df = pl.read_parquet(p)
    sc = df['score'].to_numpy().astype(np.float64)
    ok_cols = df.columns == refsub.columns and [str(t) for t in df.dtypes] == [str(t) for t in refsub.dtypes]
    same_order = bool((df['feed_id'].to_numpy() == cfid).all() and (df['item_id'].to_numpy() == citem).all())
    nfeeds = df['feed_id'].n_unique(); dup = int(df.select('feed_id', 'item_id').is_duplicated().sum())
    j = ss.select('feed_id', 'item_id').join(df, on=['feed_id', 'item_id'], how='left'); miss = int(j['score'].null_count())
    per_feed_rows = df.group_by('feed_id').len(); cnt_ok = bool((per_feed_rows.join(ft.select('feed_id', 'M'), on='feed_id')['len'] == per_feed_rows.join(ft.select('feed_id', 'M'), on='feed_id')['M']).all())
    nan = int(np.isnan(sc).sum() + np.isinf(sc).sum())
    # score must be a permutation 0..M-1 inside every feed
    srt = np.sort(sc.reshape(-1)) if False else None
    o = np.lexsort((sc, FID['test'])); perm_ok = bool((sc[o] == (np.arange(len(sc)) - np.repeat(ot, Mt))).all()) if same_order else None
    mr = myrank(C[k]['test']); agree = float((mr == sc).mean()); top1 = float((np.bincount(FID['test'], (sc == np.repeat(Mt, Mt) - 1) & (mr == np.repeat(Mt, Mt) - 1), len(Mt)) > 0).mean())
    mad = float(np.abs(mr - sc).max())
    samefd = float((refsub['score'].to_numpy() == sc).mean())
    w_(f'{k}: sha256 {h} | rows {len(df)} (sample_submission {len(ss)}) | feeds {nfeeds} (test feeds {len(Mt)}) | duplicate keys {dup} | sample rows without a score {miss} | rows per feed = M {cnt_ok} | NaN/inf {nan}'
       f' | columns+dtypes as subs/cand_FD16P20kd_r4 {ok_cols} | row order = flat test order {same_order} | score = permutation 0..M-1 per feed {perm_ok}'
       f' | rows with the same rank as own rebuild {agree:.6f} (max rank difference {mad:.0f}) | same top-1 as own rebuild {top1:.6f} | rows with the same rank as the scored FD16P20 file {samefd:.4f}')
    RES[k]['sha256'] = h; RES[k]['rank_agree'] = agree
json.dump(RES, open(OUT / 'c1_verify_staged.json', 'w'), indent=1, default=float)
open(OUT / 'c1_verify_staged.txt', 'w').write('\n'.join(L) + '\n')
