"""V4. (a) truth for the W8TO + C3 equal mix on val (the tool read -0.00557 with a rank-file partner); (b) per-feed Spearman distribution of genuine
partners vs W8 on test against partly corrupted files (a check the reader does not have)."""
import sys; sys.path.insert(0, '<repo>/nn_x/w12/next/merge_verify')
from v_common import *
V, T = Sp('val'), Sp('test'); yv = np.load(FLAT / 'val/cand_target.npy').astype(np.float64); ev = lambda s: ndcg_per_feed(np.asarray(s, np.float64), yv, V.off, V.M)
to = V.z(np.load(P / 'ens_W8TOkd_to_val.npy')); c3 = V.z(fin(V, 'Q2a8s3')); eo, ec = ev(to), ev(c3)
srt = to[np.lexsort((to, V.fid))]; R = srt[V.off[V.fid] + V.rank(c3).astype(np.int64)]; mix = 0.5 * to + 0.5 * R
print(f'(a) val: W8TO {eo.mean():.5f} C3 {ec.mean():.5f} | equal z-mix of model scores - W8TO {pm(ev(0.5 * to + 0.5 * c3) - eo)} | rank partner on our shape: ties left tied {pm(ev(mix) - eo)} | ties -> our order {pm(ev(V.rank(mix, to)) - eo)} | ties -> partner order {pm(ev(V.rank(mix, c3)) - eo)}')
C = partners(V, T); w8t = T.z(fin(T, 'W8kd')); rw = T.z(T.rank(w8t)); big = T.M >= 20
print('(b) per-feed Spearman vs the W8 file on test: mean, SE, share of feeds < 0, share of feeds with M >= 20 below 0 / below 0.3, 1st percentile (M >= 20)')
def desc(nm, s):
    r = T.rf(rw, T.z(T.rank(s))); print(f'   {nm:28s} mean {r.mean():.4f} ± {se(r):.4f} | < 0: {np.mean(r < 0):.3%} | M>=20: < 0 {np.mean(r[big] < 0):.3%}, < 0.3 {np.mean(r[big] < 0.3):.3%}, p1 {np.percentile(r[big], 1):+.3f} | min {r[big].min():+.3f}')
for nm in ('C3', 'GRUx6', 'CatBoost', 'XGBoost', 'LLMx2'): desc(nm, C[nm][1])
for f in ('V_partial_inversion_10pct', 'V_partial_inversion_30pct', 'V_shuffled_12pct_feeds'):
    desc(f, T.read_file(ROOT / f'nn_x/w12/next/merge_verify/cache/{f}.parquet'))
