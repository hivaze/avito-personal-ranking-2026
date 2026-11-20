"""V5. (a) exact emulation of the study's arithmetic (z with sd + 1e-9, applied repeatedly) to show which way its float residue resolves the exact ties of
the equal mix of ours and a shape-restored rank partner; (b) rebuild of one cached known-answer input of the study and comparison with its file."""
import sys; sys.path.insert(0, '<repo>/nn_x/w12/next/merge_verify')
from v_common import *
V, T = Sp('val'), Sp('test'); yv = np.load(FLAT / 'val/cand_target.npy').astype(np.float64); ev = lambda s: ndcg_per_feed(np.asarray(s, np.float64), yv, V.off, V.M)
def zf(S, x):                                                 # the study's z: population sd + 1e-9
    x = np.asarray(x, np.float64); mu = np.bincount(S.fid, x, S.n) / S.M; d = x - mu[S.fid]; sd = np.sqrt(np.bincount(S.fid, d * d, S.n) / S.M) + 1e-9; return d / sd[S.fid]
def raw(S, tag): return 0.5 * zf(S, np.load(P / f'ens_{tag}_ft_{S.name}.npy')) + 0.5 * zf(S, np.load(P / f'ens_{tag}_to_{S.name}.npy'))
def tied(S, m):
    o = np.lexsort((m, S.fid)); s = m[o]; f = S.fid[o]; eq = np.zeros(S.N, bool); eq[1:] = (f[1:] == f[:-1]) & (s[1:] == s[:-1]); t = eq.copy(); t[:-1] |= eq[1:]; return float(t.mean())
z1 = zf(V, raw(V, 'W8kd')); z2 = zf(V, z1); z2s = z2[np.lexsort((z2, V.fid))]
print(f'(a) relative size of the residue between one and two z-scorings of the W8 blend: median |z2/z1 - 1| = {np.median(np.abs(z2 / z1 - 1)):.2e}; share of rows with z2 > z1 in absolute value {np.mean(np.abs(z2) > np.abs(z1)):.3f}')
for tag, nm in (('Q2a8s3', 'C3'), ('Pall', 'Pall'), ('S16kd', 'S16kd')):
    p = zf(V, raw(V, tag)); rp = V.rank(p); R2 = z2s[V.off[V.fid] + rp.astype(np.int64)]
    exact = 0.5 * z1 + 0.5 * z1[np.lexsort((z1, V.fid))][V.off[V.fid] + rp.astype(np.int64)]      # no residue: exact ties
    tool_val = 0.5 * z1 + 0.5 * R2                             # merge_check_v2 val check and --blend: ours z once, shape z twice
    calib = 0.5 * z1 + 0.5 * zf(V, R2)                         # c1_calib.py d_sh: one more z on the restored partner
    zm = ev(0.5 * z1 + 0.5 * p); rk_p = V.rank(exact, p); rk_o = V.rank(exact, z1)
    for lab, m in (('tool arithmetic (val check, --blend)', tool_val), ('c1_calib arithmetic (table 1.2 "v2 way")', calib)):
        r = V.rank(m)
        print(f'   {nm:6s} {lab:40s}: exact ties before residue {tied(V, exact):.3f}, left after {tied(V, m):.4f} | order == [ties -> partner] {np.mean(r == rk_p):.4f}, == [ties -> ours] {np.mean(r == rk_o):.4f} | NDCG minus z-mix {pm(ev(m) - zm)}')
    print(f'   {nm:6s} reference: ties -> partner {pm(ev(rk_p) - zm)} | ties -> ours {pm(ev(rk_o) - zm)} | left tied {pm(ev(exact) - zm)}')
# (b) rebuild of the study's cached input NOTSUB_c3_val_ranks.parquet / NOTSUB_c3_test_ranks.parquet from preds/ens_Q2a8s3_*
for S in (V, T):
    mine = S.rank(S.z(fin(S, 'Q2a8s3'))); theirs = S.read_file(ROOT / f'nn_x/w12/next/merge/cache/NOTSUB_c3_{S.name}_ranks.parquet')
    print(f'(b) NOTSUB_c3_{S.name}_ranks.parquet vs my rebuild from preds/ens_Q2a8s3_*: identical rows {np.mean(mine == theirs):.6f}, max |diff| {np.abs(mine - theirs).max():.1f}')
sc = T.read_file(ROOT / 'nn_x/w12/next/merge/cache/NOTSUB_catboost_test_scores.parquet'); cb = np.load(ROOT / 'nn_x/cb26/data/e3_xm_yeti_test.npy')
print(f'(b) NOTSUB_catboost_test_scores.parquet vs nn_x/cb26/data/e3_xm_yeti_test.npy: max |diff| {np.abs(sc - cb.astype(np.float32)).max():.2e}')
