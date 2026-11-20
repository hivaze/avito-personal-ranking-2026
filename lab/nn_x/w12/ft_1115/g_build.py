"""ft_1115 step G: staged files (pure re-weightings of STORED branch files preds/ens_<tag>_{ft,to}_{val,test}.npy; written with src/make_sub.py into
nn_x/w12/ft_1115/stage/ only; nothing is written to subs/ or preds/; an existing stage file is never overwritten).
  probe_W8_FTonly_wft100     = z(W8 fine-tuned pool)                      (w_ft 1.0; public points of the same pool on record: w_ft 0 = W8TO 0.394009, w_ft 0.5 = W8 0.394486)
  cand_FD16P20_wft060        = 0.6 z(FT pool) + 0.4 z(TO pool) of FD16P20 (upper end of the supported range; equal-expectation twin of the scored FD16P20, not a gain claim)
Checks: the same code path at w_ft 0.5 (and at w_ft 0 for W8) reproduces the SCORED submission files in subs/ row by row.
Read-out per file: sha256, final val, deltas against the w_ft 0.5 file (val, pmix-public, pmix-private; paired per feed), paired public SE, test correlation,
share of public / private feeds with the same top-1 item.  Output: out/g_build.txt"""
from lib import *
from make_sub import make_sub
import hashlib, os
out = []
def w_(s=''):
    print(s, flush=True); out.append(s)
ST = D / 'stage'; ST.mkdir(exist_ok=True); TMP = OUT / 'tmp_check'; TMP.mkdir(exist_ok=True)
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
def same_as(scores, ref_parquet, tag):
    p = TMP / f'{tag}.parquet'
    if p.exists(): os.remove(p)
    make_sub(scores.astype(np.float32), str(p)); a = pl.read_parquet(p); b = pl.read_parquet(ref_parquet); os.remove(p)
    j = a.join(b, on=['feed_id', 'item_id'], how='inner', suffix='_ref')
    return len(a) == len(b) == len(j) and bool((j['score'] == j['score_ref']).all()), float((j['score'] == j['score_ref']).mean())
def top1(x):
    """index (row) of the top item per test feed"""
    o = np.lexsort((-x, np.repeat(np.arange(len(Mt)), Mt))); return o[ot]
w_(f'public feeds {int(PUB.sum())} private feeds {int((~PUB).sum())} (timestamp < / >= {pmix.BOUND})')
SPECS = [('probe_W8_FTonly_wft100', 'W8kd', 1.0, 'subs/cand_W8kd_r2.parquet'), ('cand_FD16P20_wft060', 'FD16P20kd', 0.6, 'subs/cand_FD16P20kd_r4.parquet')]
for name, tag, w, ref in SPECS:
    vf, vt = stored(tag, 'val'); tf, tto = stored(tag, 'test'); zf, zo, zft, zot = zv(vf), zv(vt), zt(tf), zt(tto)
    ok, share = same_as(0.5 * zft + 0.5 * zot, R / ref, f'chk_{tag}_05')
    w_(f'## {name}: w_ft {w} on preds/ens_{tag}_*')
    w_(f'  path check: this code at w_ft 0.5 reproduces {ref} row by row: {ok} (share of equal rows {share:.6f})')
    if tag == 'W8kd':
        ok0, sh0 = same_as(zot, R / 'subs/cand_W8TOkd_r3.parquet', 'chk_W8_00'); w_(f'  path check: w_ft 0 reproduces subs/cand_W8TOkd_r3.parquet: {ok0} (share {sh0:.6f})')
    e5 = ev(0.5 * zf + 0.5 * zo); e = ev(w * zf + (1 - w) * zo); d = e - e5; t5 = 0.5 * zft + 0.5 * zot; t = w * zft + (1 - w) * zot
    a5, a = top1(t5), top1(t); s1 = a5 == a
    fc = feed_corr(t.astype(np.float64), t5.astype(np.float64), Mt, ot)
    outp = ST / f'{name}.parquet'
    if outp.exists(): w_(f'  {outp} exists: not overwritten')
    else: make_sub(t.astype(np.float32), str(outp))
    sub = pl.read_parquet(outp)
    w_(f'  file {outp} | rows {len(sub)} feeds {sub["feed_id"].n_unique()} | sha256 {sha(outp)}')
    w_(f'  final val {e.mean():.5f} (w_ft 0.5 file {e5.mean():.5f}) | delta against the w_ft 0.5 file: val {f(d)} | pmix-public {f(d, WPUB)} | pmix-private {f(d, WPRIV)} | M<50 {f(d[Mv < 50])} M>=50 {f(d[Mv >= 50])}')
    w_(f'  paired public SE of the delta (val per-feed sd / sqrt(8562)) {d.std() / np.sqrt(8562):.5f} | test correlation to the w_ft 0.5 file: pooled Pearson {np.corrcoef(t, t5)[0, 1]:.4f}, mean per-feed Pearson {fc.mean():.4f}'
       f' | same top-1 item: public {s1[PUB].mean():.4f} private {s1[~PUB].mean():.4f}')
for p in TMP.glob('*'): os.remove(p)
TMP.rmdir()
open(OUT / 'g_build.txt', 'w').write('\n'.join(out) + '\n')
