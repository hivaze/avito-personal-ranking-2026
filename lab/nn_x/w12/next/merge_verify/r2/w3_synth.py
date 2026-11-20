"""w3: synthetic PARTNER TEST INPUTS for the reader of merge_check_v2.py (r2). None is a candidate or a blend; all carry the prefix XNOTSUB_ and live in my cache.
  X1  C3 file, every feed with M < 20 negated            (the distribution check of the tool looks at feeds with M >= 20 only)
  X2  C3 file, 30% of the feeds with M < 20 negated      (4.9% of all feeds)
  X3  exp(2 z) of OUR OWN W8 scores                       (the order is identical to ours; only the scale differs: a 'probability-like' file)
  X4  per-feed softmax(1.5 z) of the C3 model scores      (a genuine partner that sends probabilities instead of logits), test and val
  X5  per-feed softmax(1.0 z) of the C3 model scores      (milder), test and val
  cd <repo> && CUDA_VISIBLE_DEVICES="" .venv/bin/python -B nn_x/w12/next/merge_verify/r2/w3_synth.py"""
import sys
sys.path.insert(0, '<repo>/nn_x/w12/next/merge_verify/r2')
from w_common import *
C = HERE / 'cache'; T, V = Split('test'), Split('val'); rng = np.random.default_rng(2026)
def write(S, s, name, dtype=np.float32):
    pl.DataFrame({'feed_id': S.feed_id[S.fid].astype(np.int32), 'item_id': np.load(FLAT / S.name / 'cand_item_id.npy'), 'score': np.asarray(s, dtype)}).write_parquet(C / name); print('written', name, flush=True)
c3 = T.read_parquet(ROOT / 'subs/cand_Q2a8_ftbag3_memeq.parquet'); small = T.M < 20
s = c3.copy(); hr = np.repeat(small, T.M); s[hr] = -s[hr]; write(T, s, 'XNOTSUB_X1_small_feeds_negated_all.parquet')
hit = small & (rng.random(T.n) < 0.30); s = c3.copy(); hr = np.repeat(hit, T.M); s[hr] = -s[hr]; write(T, s, 'XNOTSUB_X2_small_feeds_negated_30pct.parquet')
print(f'feeds with M < 20: {int(small.sum())} of {T.n} ({small.mean():.1%}); X2 negates {int(hit.sum())} feeds ({hit.mean():.2%} of all)')
w8 = T.z(fin(T, 'W8kd')); write(T, np.exp(2.0 * w8), 'XNOTSUB_X3_identical_order_exp2.parquet', np.float64)
def softmax(S, z, a):
    e = np.exp(a * z - np.maximum.reduceat(a * z, S.off)[S.fid]); return e / S.fsum(e)[S.fid]
for a, nm in ((1.5, 'X4'), (1.0, 'X5')):
    write(T, softmax(T, T.z(fin(T, 'Q2a8s3')), a), f'XNOTSUB_{nm}_c3_softmax_test.parquet', np.float64)
    write(V, softmax(V, V.z(fin(V, 'Q2a8s3')), a), f'XNOTSUB_{nm}_c3_softmax_val.parquet', np.float64)
