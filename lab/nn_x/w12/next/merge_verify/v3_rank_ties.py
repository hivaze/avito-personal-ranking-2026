"""V3. The 'rank-file penalty' of the merge study (verdict line 4): how much of it is the tie-break of an equal mix of two rank-derived vectors?
An equal mix of ours (rank a) and a partner restored on our shape (rank b) is EXACTLY tied for every pair of items with swapped ranks:
0.5 z[a] + 0.5 z[b] == 0.5 z[b] + 0.5 z[a]. The study's code (mc_core.zfeed, sd + 1e-9, applied one more time to the shape than to our scores)
shrinks the partner side by a factor (1 - k 1e-9), which resolves every such tie in favour of OUR order. Here the ties are kept exact and resolved four ways.
  cd <repo> && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 .venv/bin/python nn_x/w12/next/merge_verify/v3_rank_ties.py > nn_x/w12/next/merge_verify/out/v3_rank_ties.txt 2>&1"""
import sys, time; sys.path.insert(0, '<repo>/nn_x/w12/next/merge_verify')
from v_common import *
t0 = time.time(); V, T = Sp('val'), Sp('test'); yv = np.load(FLAT / 'val/cand_target.npy').astype(np.float64)
ev = lambda s: ndcg_per_feed(np.asarray(s, np.float64), yv, V.off, V.M)
w8v = V.z(fin(V, 'W8kd')); e8 = ev(w8v); C = partners(V, T); k1, k2 = np.random.default_rng(777).random(V.N), np.random.default_rng(778).random(V.N)


def restore(S, rank_p, ours_z):
    srt = ours_z[np.lexsort((ours_z, S.fid))]; return srt[S.off[S.fid] + rank_p.astype(np.int64)]


def tied_share(S, mix):
    o = np.lexsort((mix, S.fid)); s = mix[o]; f = S.fid[o]; eq = np.zeros(S.N, bool); eq[1:] = (f[1:] == f[:-1]) & (s[1:] == s[:-1]); t = eq.copy(); t[:-1] |= eq[1:]
    return float(t.mean())


def ways(ours, part, study_eps=False):
    """equal mix of OURS (model-score z) and a PARTNER rank file restored on our shape; per-feed NDCG under each tie-break, and the z-mix."""
    R = restore(V, V.rank(part), ours); mix = 0.5 * ours + 0.5 * R
    out = dict(zmix=ev(0.5 * ours + 0.5 * part), pess=ev(mix), ours=ev(V.rank(mix, ours)), rand=ev(V.rank(mix, k1)), rand2=ev(V.rank(mix, k2)), partner=ev(V.rank(mix, part)), tied=tied_share(V, mix))
    if study_eps:                                             # the study's arithmetic: partner side shrunk by (1 - 2e-9) relative to ours
        out['eps'] = ev(0.5 * ours + 0.5 * R * (1 - 2e-9))
    return out


print(f'W8 val {e8.mean():.5f}. Equal mix, OURS = W8 model scores, PARTNER = rank file restored on the W8 shape. All columns: mix minus the z-mix of the model scores, paired per feed, {V.n} val feeds')
print(f'  {"partner":9s} {"gap":>8s} {"tied rows":>9s} | {"ties -> OUR order":>19s} {"study arithmetic":>19s} | {"ties at random":>19s} {"(2nd key)":>10s} | {"ties -> PARTNER order":>21s} | {"ties left (metric)":>19s} | zmix - better')
S = {}
for nm, (pv, _) in C.items():
    e = ev(pv); g = (e8 - e).mean(); better = e8 if g >= 0 else e; w = ways(w8v, pv, True); S[nm] = (g, w, better)
    print(f'  {nm:9s} {g:+8.5f} {w["tied"]:9.3f} | {pm(w["ours"] - w["zmix"]):>19s} {pm(w["eps"] - w["zmix"]):>19s} | {pm(w["rand"] - w["zmix"]):>19s} {(w["rand2"] - w["zmix"]).mean():+10.5f} | {pm(w["partner"] - w["zmix"]):>21s} | {pm(w["pess"] - w["zmix"]):>19s} | {pm(w["zmix"] - better)}', flush=True)
for k in ('ours', 'eps', 'rand', 'partner', 'pess'):
    v = np.array([(w[k] - w['zmix']).mean() for _, w, _ in S.values()]); print(f'  mean over 10 partners, {k:8s}: {v.mean():+.5f} (min {v.min():+.5f}, max {v.max():+.5f})')
print('  study table 1.2, "v2 way" minus "measured z-mix": C3 +0.00005, G3 -0.00035, Pall -0.00107, Cd -0.00108, S16 -0.00021, GRU -0.00102, base -0.00069, CatBoost -0.00182, XGBoost -0.00193, LLM -0.00305 (mean -0.0011)')
print('\n-- vs the better file (the quantity a merge decision reads), by tie-break')
for nm, (g, w, better) in S.items():
    print(f'  {nm:9s} zmix {pm(w["zmix"] - better)} | ties -> ours {pm(w["ours"] - better)} | random {pm(w["rand"] - better)} | -> partner {pm(w["partner"] - better)} | left tied {pm(w["pess"] - better)}')
print('\n-- roles swapped (OURS = the weaker model, PARTNER = W8 as a rank file) and equal-strength pairs: is the penalty a property of the rank file or of the tie-break / partner strength?')
def line(lab, ours, part):
    w = ways(ours, part); print(f'  {lab:46s} tied {w["tied"]:.3f} | ties -> ours {pm(w["ours"] - w["zmix"])} | random {pm(w["rand"] - w["zmix"])} | -> partner {pm(w["partner"] - w["zmix"])}')
for nm in ('Cd', 'GRUx6', 'CatBoost'):
    line(f'ours {nm}, partner W8 ranks', C[nm][0], w8v)
for a, b in (('Pall', 'Cd'), ('GRUx6', 'basex4'), ('C3', 'G3')):
    line(f'ours {a}, partner {b} ranks', C[a][0], C[b][0]); line(f'ours {b}, partner {a} ranks', C[b][0], C[a][0])
line('ours S16kd, partner W8 ranks', C['S16kd'][0], w8v)
print('\n-- the val check of the tool at a weight other than 0.5 has no exact ties: v2 way at w = 0.4 (partner), minus the z-mix at the same weight')
for nm in ('C3', 'Pall', 'S16kd', 'CatBoost'):
    pv = C[nm][0]; R = restore(V, V.rank(pv), w8v); m = 0.6 * w8v + 0.4 * R; print(f'  {nm:9s} tied rows {tied_share(V, m):.4f} | {pm(ev(m) - ev(0.6 * w8v + 0.4 * pv))}')
print(f'\n[done in {time.time() - t0:.0f} s]')
