"""F. Does the FT treat test feeds of users that also have a val feed differently (user-level memorisation that GroupKFold OOF cannot see)?"""
from lib import *
vu = set(uv[uv != 0].tolist()); inval = np.array([(u in vu) for u in ut]) & (ut != 0)
print(f'test feeds whose user has a val feed: {inval.sum()} of {len(ut)} ({inval.mean():.3f}); user0 test feeds {(ut == 0).sum()}')
pub = np.arange(len(Mt)) < 8562; print(f'  public share with a val user {inval[pub].mean():.3f}, private {inval[~pub].mean():.3f}')
u, c = np.unique(uv[uv != 0], return_counts=True); print(f'val users with >= 2 val feeds: {(c >= 2).sum()} of {len(u)}; feeds in such users {c[c >= 2].sum()}')
tot, tft = zt(pool(W8, 'to', 'test')), zt(pool(W8, 'ft', 'test')); s = tft - tot
rms = lambda x: np.sqrt((x ** 2).mean())
for nm, m in (('user in val', inval), ('user not in val (known)', ~inval & (ut != 0)), ('user 0', ut == 0)):
    r = np.repeat(m, Mt); print(f'  {nm:26s} feeds {m.sum():6d} RMS shift {rms(s[r]):.4f} corr(FT, TO) {np.corrcoef(tot[r], tft[r])[0, 1]:.5f} mean M {Mt[m].mean():.1f} mean H {Ht[m].mean():.1f}')
# matched on M and H: regress per-feed RMS shift on log M, H, inval
fr = np.sqrt(np.add.reduceat(s * s, ot) / Mt); k = ut != 0
A = np.stack([np.ones(k.sum()), np.log(Mt[k]), Ht[k] / 256.0, inval[k].astype(float)], 1); co, *_ = np.linalg.lstsq(A, fr[k], rcond=None)
res = fr[k] - A @ co; cov = np.linalg.inv(A.T @ A) * res.var()
print(f'  per-feed RMS shift ~ 1 + log M + H + in_val: coef in_val {co[3]:+.5f} +- {np.sqrt(cov[3, 3]):.5f} (mean per-feed RMS {fr[k].mean():.4f})')
# candidate-level: test candidates that the same user had as a val candidate (and as a val positive)
import polars as pl
iv = np.load(FLAT / 'val/cand_item_id.npy'); it = np.load(FLAT / 'test/cand_item_id.npy')
kv = np.repeat(uv.astype(np.int64), Mv) * (2 ** 32) + iv.astype(np.int64); kt = np.repeat(ut.astype(np.int64), Mt) * (2 ** 32) + it.astype(np.int64)
known_t = np.repeat(ut != 0, Mt); known_v = np.repeat(uv != 0, Mv)
seen = np.isin(kt, kv[known_v]) & known_t; pos = np.isin(kt, kv[known_v & (yv > 0)]) & known_t
print(f'  test candidates that were a val candidate of the same user: {seen.sum()} ({seen.mean():.5f}); that were a val POSITIVE of the same user: {pos.sum()} ({pos.mean():.6f})')
for nm, m in (('same-user val candidate (not positive)', seen & ~pos), ('same-user val positive', pos)):
    if m.sum() > 20: print(f'    {nm:40s} n {m.sum():6d} mean FT-TO shift {s[m].mean():+.4f} +- {s[m].std() / np.sqrt(m.sum()):.4f} | mean z_TO {tot[m].mean():+.3f} mean z_FT {tft[m].mean():+.3f}')
