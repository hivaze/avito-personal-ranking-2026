"""v_independent.py : independent re-check of the integrity audit of reports/FREEZE_PROTOCOL_1207.md section 5 (checks a-f).
Own code throughout: own per-feed z-score (numpy, float64), own NDCG (binary gain, log2 discount, pessimistic ties, feeds without a positive -> 1),
own parsing of the parquet files, the builder sources, the training logs and the gate outputs. It imports nothing from nn_x/w12/freeze/ and nothing from src/.
CPU only, read-only on the project; it prints to stdout (run_all.sh does not call it):
  export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 NUMBA_NUM_THREADS=12 POLARS_MAX_THREADS=12
  .venv/bin/python nn_x/w12/freeze/v_independent.py > nn_x/w12/freeze/out/v_independent.txt
The candidate list below is typed from the protocol table (section 2), not read from cands.py.
Note on (b): this script orders exact ties by row position, src/make_sub.py by a fixed random key; rows that differ are reported together with the
largest score gap between a row and the row that holds its rank in the file (0 or float noise = a tie, nothing else)."""
import os, sys, re, json, hashlib, datetime
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
import numpy as np, polars as pl
from pathlib import Path
R = Path('<repo>'); P = R / 'preds'

CANDS = [('FD16P20', 'subs/cand_FD16P20kd_r4.parquet', 'FD16P20kd', '0.40271', True),
         ('FD8P20', 'subs/cand_FD8P20kd_r4.parquet', 'FD8P20kd', '0.40240', True),
         ('FD16P20vf', 'subs/cand_FD16P20vfkd_r4.parquet', 'FD16P20vfkd', '0.40286', True),
         ('FD16P20@m0.2_0.4', 'subs/cand_FD16P20_m02_04kd_r4.parquet', 'FD16P20_m02_04kd', '0.40298', True),
         ('W8P20', 'subs/cand_W8P20kd_r3.parquet', 'W8P20kd', '0.40214', False),
         ('W16', 'subs/cand_W16kd_r3.parquet', 'W16kd', '0.40158', False),
         ('W8', 'subs/cand_W8kd_r2.parquet', 'W8kd', '0.40165', False)]

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''): h.update(b)
    return h.hexdigest()

# ---------------- meta
fv = pl.read_parquet(R / 'data/flat/val/feeds.parquet'); ft = pl.read_parquet(R / 'data/flat/test/feeds.parquet')
Mv = fv['M'].to_numpy().astype(np.int64); Mt = ft['M'].to_numpy().astype(np.int64)
ov = fv['cand_off'].to_numpy(); ot = ft['cand_off'].to_numpy()
assert (ov == np.concatenate([[0], np.cumsum(Mv)[:-1]])).all() and (ot == np.concatenate([[0], np.cumsum(Mt)[:-1]])).all()
yv = np.load(R / 'data/flat/val/cand_target.npy').astype(np.float64)
fidv = np.repeat(np.arange(len(Mv)), Mv); fidt = np.repeat(np.arange(len(Mt)), Mt)
print('v_independent.py', datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'))
print('val feeds', len(Mv), 'rows', Mv.sum(), '| test feeds', len(Mt), 'rows', Mt.sum(), '| test M<50', int((Mt < 50).sum()), 'M>=50', int((Mt >= 50).sum()))

def z(x, fid, M):
    x = np.asarray(x, np.float32).astype(np.float64); n = M.astype(np.float64)
    mu = np.bincount(fid, x) / n; d = x - mu[fid]; sd = np.sqrt(np.bincount(fid, d * d) / n) + 1e-9
    return (d / sd[fid]).astype(np.float32)

def ndcg_own(s, y, fid, M, off):
    """own implementation: binary gain, log2 discount, pessimistic ties (negatives first among equal scores), feeds without positive -> 1."""
    s = np.asarray(s, np.float64)
    order = np.lexsort((y, -s, fid)); ys = y[order]; pos = np.arange(len(s)) - np.repeat(off, M)
    disc = 1 / np.log2(pos + 2.0)
    dcg = np.bincount(fid, ys * disc, minlength=len(M)); npos = np.bincount(fid, y, minlength=len(M))
    idcg = np.bincount(fid, (pos < np.repeat(npos, M)) * disc, minlength=len(M))
    out = np.ones(len(M)); m = idcg > 0; out[m] = dcg[m] / idcg[m]; return out

fin_v = lambda a, b: 0.5 * z(a, fidv, Mv) + 0.5 * z(b, fidv, Mv)
fin_t = lambda a, b: 0.5 * z(a, fidt, Mt) + 0.5 * z(b, fidt, Mt)

# ---------------- (a) files
ss = pl.read_parquet(R / 'data/raw/sample_submission.parquet'); ref = pl.read_parquet(R / 'subs/cand_S16kd_r2.parquet')
print('sample submission', ss.shape, dict(ss.schema), '| scored S16', ref.shape, dict(ref.schema))
sskey = ss.select('feed_id', 'item_id')
print('sample submission feeds', ss['feed_id'].n_unique(), 'unique keys', sskey.is_unique().all())
item_t = np.load(R / 'data/flat/test/cand_item_id.npy'); fidx_t = np.load(R / 'data/flat/test/cand_fidx.npy')
assert (fidx_t == fidt).all()
feed_id_t = ft['feed_id'].to_numpy()[fidx_t]
FLAT = pl.DataFrame({'feed_id': feed_id_t.astype(np.int32), 'item_id': item_t.astype(np.int32), 'pos': np.arange(len(item_t))})
SC = {}
for cid, f, tag, vrec, fd in CANDS:
    p = R / f; d = pl.read_parquet(p)
    sc = d['score'].to_numpy()
    nul = sum(d[c].null_count() for c in d.columns)
    anti1 = sskey.join(d, on=['feed_id', 'item_id'], how='anti').height; anti2 = d.join(sskey, on=['feed_id', 'item_id'], how='anti').height
    g = d.group_by('feed_id').agg(pl.len().alias('n'), pl.col('item_id').n_unique().alias('u'), pl.col('score').n_unique().alias('us'), pl.col('score').min().alias('mn'), pl.col('score').max().alias('mx'))
    j = ft.select('feed_id', 'M').join(g, on='feed_id', how='left')
    okfeeds = int(((j['n'] == j['M']) & (j['u'] == j['M']) & (j['us'] == j['M']) & (j['mn'] == 0) & (j['mx'] == j['M'] - 1)).sum())
    jj = FLAT.join(d, on=['feed_id', 'item_id'], how='left').sort('pos'); SC[cid] = jj['score'].to_numpy()
    print(f'(a) {cid:18s} sha256 {sha(p)} | rows {d.height} cols {d.columns} {[str(t) for t in d.dtypes]} schema==S16 {dict(d.schema) == dict(ref.schema)} ==sample {dict(d.schema) == dict(ss.schema)}'
          f' | null {nul} nan {int(np.isnan(sc).sum())} inf {int(np.isinf(sc).sum())} | dup keys {d.height - d.select("feed_id", "item_id").n_unique()} | sample keys not in file {anti1}, file keys not in sample {anti2}'
          f' | feeds {g.height}, feeds with n==M, M distinct items, ranks 0..M-1: {okfeeds} of {ft.height} | unmatched to flat {int(jj["score"].null_count())}')

# ---------------- (b) stored branches
EV = {}; FT = {}
for cid, f, tag, vrec, fd in CANDS:
    br = {}
    for sp in ('val', 'test'):
        for b in ('ft', 'to'):
            x = np.load(P / f'ens_{tag}_{b}_{sp}.npy'); br[b, sp] = x
            assert len(x) == (Mv.sum() if sp == 'val' else Mt.sum()) and np.isfinite(x).all(), (tag, b, sp)
    e = ndcg_own(fin_v(br['ft', 'val'], br['to', 'val']), yv, fidv, Mv, ov); EV[cid] = e
    t = fin_t(br['ft', 'test'], br['to', 'test']); FT[cid] = t
    # file vs order of the stored branches: Spearman per feed between the file rank and my own final score (ties aside)
    o = np.lexsort((t, fidt)); rk = np.empty(len(t)); rk[o] = np.arange(len(t)) - np.repeat(ot, Mt)
    neq = rk != SC[cid]; ndiff = int(neq.sum())
    row_at = np.empty(len(t), np.int64); row_at[(np.repeat(ot, Mt) + rk).astype(np.int64)] = np.arange(len(t))      # row that this script puts at each within-feed rank
    i = np.flatnonzero(neq); j = row_at[(ot[fidt[i]] + SC[cid][i]).astype(np.int64)]                                  # row that this script puts where the file puts row i
    gap = float(np.abs(t[i].astype(np.float64) - t[j].astype(np.float64)).max()) if ndiff else 0.0
    print(f'(b) {cid:18s} branch files 4/4 finite | final val own-code {e.mean():.5f} record {vrec} equal {f"{e.mean():.5f}" == vrec} | file rows whose rank differs from own order of stored branches: {ndiff} in {len(np.unique(fidt[neq]))} feeds; largest score gap behind a differing row {gap:.2e}')

# ---------------- (c) members from the builder sources
src4 = (R / 'nn_x/w12/kd/build_r4.py').read_text(); ns = {}
exec('from pathlib import Path\n' + src4[src4.index('S8 = range'):src4.index('fv, Mv, ov = feed_meta')], ns)
src3 = (R / 'nn_x/w12/kd/build_r3.py').read_text(); n3 = {}
exec(src3[src3.index('W = lambda seeds'):src3.index('fv, Mv, ov = feed_meta')], n3)
MEM = {'FD16P20': ns['spec']('FD16P20'), 'FD8P20': ns['spec']('FD8P20'), 'FD16P20vf': ns['spec']('FD16P20vf'), 'FD16P20@m0.2_0.4': ns['spec']('FD16P20'),
       'W8P20': (n3['W8n'], n3['P1n'], 0.2), 'W16': (n3['W16n'], [], 0.0), 'W8': (n3['W8n'], [], 0.0)}
SUF = ('n2cpv3s3_val_oof', 'n2cpv3s3_test', 'trainonly_val', 'trainonly_test')
allm = []
for cid, (b, p, w) in MEM.items():
    ms = list(b) + list(p); miss = [f'nn_{m}_{s}.npy' for m in ms for s in SUF if not (P / f'nn_{m}_{s}.npy').exists()]
    allm += [m for m in ms if m not in allm]
    print(f'(c) {cid:18s} members {len(b)}+{len(p)} w {w} | prediction files present {4 * len(ms) - len(miss)} of {4 * len(ms)} | subset of FD16P20: {set(ms) <= set(list(MEM["FD16P20"][0]) + list(MEM["FD16P20"][1]))}')
print('distinct members', len(allm))
# rebuild two files from members with own z
def pool(names, sp, suf):
    fid, M = (fidv, Mv) if sp == 'val' else (fidt, Mt)
    return np.mean([z(np.load(P / f'nn_{m}_{suf}.npy'), fid, M) for m in names], 0)
for cid in ('FD16P20', 'W8P20', 'FD8P20'):
    b, p, w = MEM[cid]; tag = dict((c[0], c[2]) for c in CANDS)[cid]
    out = {}
    for sp, (sf, st) in (('val', ('n2cpv3s3_val_oof', 'trainonly_val')), ('test', ('n2cpv3s3_test', 'trainonly_test'))):
        fid, M = (fidv, Mv) if sp == 'val' else (fidt, Mt)
        for bn, suf in (('ft', sf), ('to', st)):
            x = pool(b, sp, suf)
            if p: x = (1 - w) * z(x, fid, M) + w * z(pool(p, sp, suf), fid, M)
            s = np.load(P / f'ens_{tag}_{bn}_{sp}.npy'); out[bn, sp] = x
            print(f'(c) {cid} rebuilt {bn}_{sp}: max|d| vs stored {np.abs(x - s).max():.2e} corr {np.corrcoef(x, s)[0, 1]:.7f}')
    e = ndcg_own(fin_v(out['ft', 'val'], out['to', 'val']), yv, fidv, Mv, ov); print(f'(c) {cid} final val from members, own code: {e.mean():.5f}')

# ---------------- (d) logs
def logp(m): return R / ('nn_x/w5' if m.startswith('w5_') else 'nn_x/abl') / 'logs' / f'{m}.log'
for cid, f, tag, vrec, fd in CANDS:
    b, p, w = MEM[cid]; ms = list(b) + list(p); nflag = 0; nline = 0; feeds = set(); kept = set(); tot = set(); pool_ = set(); nolog = []
    for m in ms:
        lp = logp(m)
        if not lp.exists(): nolog.append(m); continue
        t = lp.read_text(errors='replace'); L = t.splitlines()
        nflag += "'holdout_in_train': True" in L[0]; nline += any(l.startswith('FULLDATA --holdout_in_train') for l in L)
        mm = re.search(r'train feeds (\d+), steps/epoch (\d+), total steps (\d+)', t); feeds.add(mm.group(1) if mm else None); tot.add(mm.group(3) if mm else None)
        k = re.findall(r'kept step (\d+) = last', t); kept.add(k[-1] if k else None)
        pp = re.findall(r'exploration pool (\d+) feeds', t); pool_.add(pp[-1] if pp else None)
    print(f'(d) {cid:18s} members {len(ms)} no log {nolog} | args flag True in {nflag} | FULLDATA line in {nline} | train feeds {sorted(map(str, feeds))} total steps {sorted(map(str, tot))} kept {sorted(map(str, kept))} pool {sorted(map(str, pool_))} | expected full-data {fd} -> {"pass" if (nflag == len(ms) and nline == len(ms)) == fd and (nflag == 0) != fd and not nolog else "FAIL"}')

# ---------------- (e) correlations and val deltas
def feedcorr(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64); n = Mt.astype(np.float64)
    a0 = a - (np.bincount(fidt, a) / n)[fidt]; b0 = b - (np.bincount(fidt, b) / n)[fidt]
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.bincount(fidt, a0 * b0) / np.sqrt(np.bincount(fidt, a0 * a0) * np.bincount(fidt, b0 * b0))
def top1(x):
    o = np.lexsort((x, fidt)); return o[ot + Mt - 1]
ids = [c[0] for c in CANDS]
for a in ('FD16P20', 'W8P20'):
    for b in ids:
        if a == b: continue
        c = np.corrcoef(FT[a], FT[b])[0, 1]; fc = np.nanmean(feedcorr(FT[a], FT[b])); t1 = (top1(FT[a]) == top1(FT[b])).mean()
        d = EV[b] - EV[a]
        print(f'(e) {b:18s} vs {a:8s}: pooled Pearson {c:.4f} | feed-mean Pearson {fc:.4f} | same top-1 {100 * t1:.1f}% | val delta {d.mean():+.5f} +- {d.std() / np.sqrt(len(d)):.5f} | sd/sqrt(8562) {d.std() / np.sqrt(8562):.5f}')
# Spearman feed-mean for the two named pairs
def wr(x):
    o = np.lexsort((x, fidt)); r = np.empty(len(x)); r[o] = np.arange(len(x)) - np.repeat(ot, Mt); return r
for a, b in (('FD16P20', 'W8P20'), ('FD16P20', 'FD8P20')):
    print(f'(e) {a}|{b} feed-mean Spearman {np.nanmean(feedcorr(wr(FT[a]), wr(FT[b]))):.4f}')
for a, b in (('W8', 'W8P20'), ('W16', 'W8'), ('W8P20', 'W8'), ('FD16P20', 'W16'), ('FD8P20', 'W8P20')):
    d = EV[a] - EV[b]; print(f'    val {a} - {b}: {d.mean():+.5f} +- {d.std() / np.sqrt(len(d)):.5f} | sd/sqrt(8562) {d.std() / np.sqrt(8562):.5f}')

# ---------------- (f) gate outputs
for cid, f, tag, vrec, fd in CANDS:
    gp = R / f'nn_x/w12/kd/out/gate_{tag}.txt'; L = gp.read_text(errors='replace').splitlines()
    v = [l for l in L if l.startswith('GATE')]; dv = [l for l in L if l.startswith('D val')]; bl = [l for l in L if l.startswith('B repro')]
    nm = sum(bool(re.match(r'\s+\S+: \d+ train-stage inputs audited', l)) for l in L)
    print(f'(f) {cid:18s} {gp.relative_to(R)} | {v} | FAIL lines {sum(l.startswith("[FAIL]") for l in L)} WARN lines {sum(l.startswith("[WARN]") for l in L)} | member lines {nm} | {dv[0][:70] if dv else None} | {bl[0][:90] if bl else None}'
          f' | gate mtime - file mtime {gp.stat().st_mtime - (R / f).stat().st_mtime:.0f} s')
