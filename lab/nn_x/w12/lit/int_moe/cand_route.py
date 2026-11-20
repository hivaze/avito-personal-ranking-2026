"""int_moe / cand_route.py: candidate-level cold / warm gate between two existing pools, nested by user, with placebos,
plus in-sample per-feed oracles with a same-recipe null. CPU only, stored val predictions only. Rules: DECLARATION.txt block 1.
Reads: preds/nn_*_{n2cpv3s3_val_oof,trainonly_val}.npy, data flat val, nn/feat_cache/val/it_hist_all_luo.npy. Writes: this directory only."""
import os, sys, json, time
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
sys.dont_write_bytecode = True
sys.path.insert(0, '<repo>/src')
import numpy as np, numba as nb
from pathlib import Path
from common import feed_meta, FLAT
from metrics import ndcg_per_feed
from stack_val import feed_rank_z

ROOT = Path('<repo>'); P = ROOT / 'preds'; HERE = Path(__file__).resolve().parent
T0 = time.time()
LOG = open(HERE / 'cand_route.log', 'w')


def say(*a):
    s = f'[{time.time() - T0:6.1f}s] ' + ' '.join(str(x) for x in a)
    print(s, flush=True); LOG.write(s + '\n'); LOG.flush()


S8 = range(951, 959); S4 = range(951, 955)
MEM = {
    'FD16': [f'w5_fd10_{a}_s{s}' for a in ('kd05w192', 'kd05t25w192') for s in S8],
    'FDP': [f'abl_fd10_pop1kd05w192_s{s}' for s in S4],
    'W16': [f'w5_e10_{a}_s{s}' for a in ('kd05w192', 'kd05t25w192') for s in S8],
    'P1w': [f'abl_e10_pop1kd05w192_s{s}' for s in S4],
}
SUF = ('n2cpv3s3_val_oof', 'trainonly_val')
feeds, M, off = feed_meta('val'); M = M.astype(np.int64); off = off.astype(np.int64); NF = len(M)
y = np.load(FLAT / 'val/cand_target.npy')
fid = np.repeat(np.arange(NF), M)
luo = np.asarray(np.load(ROOT / 'nn/feat_cache/val/it_hist_all_luo.npy', mmap_mode='r'), np.float32)
assert len(luo) == M.sum() == len(y)
COLD = (luo == 0)
Mrow = np.repeat(M, M)


@nb.njit(parallel=True, cache=False)
def ndcg_fast(s, y, off, M):
    n = len(M); out = np.ones(n)
    for f in nb.prange(n):
        o = off[f]; m = M[f]; npos = 0
        for k in range(m):
            if y[o + k] > 0: npos += 1
        if npos == 0: continue
        idcg = 0.0
        for r in range(npos): idcg += 1.0 / np.log2(r + 2.0)
        dcg = 0.0
        for k in range(m):
            if y[o + k] > 0:
                sk = s[o + k]; r = 0
                for j in range(m):
                    sj = s[o + j]
                    if sj > sk: r += 1
                    elif sj == sk and j != k and (y[o + j] <= 0 or j < k): r += 1
                dcg += 1.0 / np.log2(r + 2.0)
        out[f] = dcg / idcg
    return out


@nb.njit(parallel=True, cache=False)
def zfeed(x, off, M):
    out = np.empty(len(x), np.float32)
    for f in nb.prange(len(M)):
        o = off[f]; m = M[f]; v = x[o:o + m]; mu = v.mean(); sd = v.std() + 1e-9
        for k in range(m): out[o + k] = (v[k] - mu) / sd
    return out


def Zp(x):
    return feed_rank_z(off, M, np.asarray(x, np.float32))[:, 1]


_MZ = {}


def member_z(name, br):
    k = (name, br)
    if k not in _MZ:
        _MZ[k] = Zp(np.load(P / f'nn_{name}_{SUF[br]}.npy'))
    return _MZ[k]


def pools(base, pop):
    out = {}
    for br, tag in ((0, 'ft'), (1, 'to')):
        out['B' + tag] = Zp(np.mean([member_z(m, br) for m in base], 0))
        out['P' + tag] = Zp(np.mean([member_z(m, br) for m in pop], 0))
    return out


def final(q, wrow):
    wrow = np.float32(wrow) if np.ndim(wrow) == 0 else wrow.astype(np.float32)
    bf = zfeed(((1 - wrow) * q['Bft'] + wrow * q['Pft']).astype(np.float32), off, M)
    bt = zfeed(((1 - wrow) * q['Bto'] + wrow * q['Pto']).astype(np.float32), off, M)
    return (0.5 * bf + 0.5 * bt).astype(np.float32)


def ev(s):
    return ndcg_fast(np.asarray(s, np.float32), y, off, M)


# user-grouped folds, user 0 = singleton groups
u = feeds['user_id'].to_numpy().astype(np.int64); g = u.copy(); z0 = u == 0; g[z0] = -(np.arange(z0.sum()) + 1)
GRP = np.unique(g, return_inverse=True)[1]; NG = GRP.max() + 1


def cse(d):
    d = np.asarray(d, np.float64); m = d.mean(); s = np.bincount(GRP, d - m, NG)
    return m, np.sqrt((s ** 2).sum()) / len(d)


def fmt(d):
    m, s = cse(d); return f'{m:+.5f} +- {s:.5f} (z {m / s:+.2f})' if s > 0 else f'{m:+.5f}'


def folds(seed, k=5):
    r = np.random.default_rng(seed); gf = r.integers(0, k, NG)
    return gf[GRP]


WG = np.array([0, .1, .2, .3, .4, .5, .6, .8], np.float32)
CW_RULES = [(a, b) for a in WG for b in WG]                       # (w_cold, w_warm)
CONST_IDX = np.array([i for i, (a, b) in enumerate(CW_RULES) if a == b])
G4 = np.array([0, .2, .4, .6], np.float32)
CWM_RULES = [(a, b, c, d) for a in G4 for b in G4 for c in G4 for d in G4]   # cold&short, warm&short, cold&long, warm&long
LONG = Mrow >= 50


def tensor_cw(q, cold):
    E = np.empty((len(CW_RULES), NF))
    for i, (wc, ww) in enumerate(CW_RULES):
        E[i] = ev(final(q, np.where(cold, wc, ww)))
    return E


def tensor_cwm(q, cold):
    E = np.empty((len(CWM_RULES), NF))
    for i, (a, b, c, d) in enumerate(CWM_RULES):
        w = np.where(LONG, np.where(cold, c, d), np.where(cold, a, b))
        E[i] = ev(final(q, w))
    return E


def nested(E, idx=None, seeds=range(5)):
    """per-feed nested value (mean over fold seeds) of 'best rule on the training folds'; also the picks."""
    idx = np.arange(E.shape[0]) if idx is None else np.asarray(idx)
    out = np.zeros(NF); picks = {}
    for sd in seeds:
        fo = folds(1000 + sd)
        for k in range(5):
            te = fo == k
            r = idx[np.argmax(E[idx][:, ~te].mean(1))]
            out[te] += E[r, te]; picks[int(r)] = picks.get(int(r), 0) + 1
    return out / len(list(seeds)), picks


def perm_within_feed(flag, seed):
    r = np.random.default_rng(seed); key = r.random(len(flag))
    order = np.lexsort((key, fid))            # fid is sorted, so each feed's block of `order` is a random permutation of its own rows
    return flag[order]


def main():
    res = {}
    say(f'val feeds {NF}, rows {len(y)}, cold rows (luo == 0) {COLD.mean():.4f}, feeds with M >= 50 {np.mean(M >= 50):.4f}, groups {NG}')
    # ---- step 0: reproduction
    chk = np.random.default_rng(0).normal(size=len(y)).astype(np.float32)
    d0 = np.abs(ndcg_fast(chk, y, off, M) - ndcg_per_feed(chk, y, off, M)).max()
    say(f'step 0: own NDCG vs src/metrics on a random score: max abs per-feed diff {d0:.2e}')
    Q = {'C1': pools(MEM['FD16'], MEM['FDP']), 'C3': pools(MEM['W16'], MEM['P1w'])}
    rec = {'C1': (0.402714, 0.402975), 'C3': (0.402141, 0.402847)}
    wm = np.where(LONG, 0.4, 0.2).astype(np.float32)
    PL = {}; MR = {}
    ok = d0 < 1e-9
    for c in Q:
        PL[c] = ev(final(Q[c], 0.2)); MR[c] = ev(final(Q[c], wm))
        d1 = abs(ndcg_per_feed(final(Q[c], 0.2), y, off, M).mean() - PL[c].mean())
        good = abs(PL[c].mean() - rec[c][0]) < 1e-5 and abs(MR[c].mean() - rec[c][1]) < 1e-5 and d1 < 1e-9
        ok &= good
        say(f'step 0 {c}: PLAIN {PL[c].mean():.6f} (record {rec[c][0]}) | M-RULE {MR[c].mean():.6f} (record {rec[c][1]}) | rule - plain {fmt(MR[c] - PL[c])} | {"OK" if good else "MISMATCH"}')
    res['step0_ok'] = bool(ok)
    if not ok:
        say('STEP 0 FAIL -> STOP'); json.dump(res, open(HERE / 'cand_route.json', 'w'), indent=1); return
    say('STEP 0 PASS')
    import pmix
    W = pmix.weights()

    def pm(d, which):
        m, s = pmix.wmean_se(np.asarray(d, np.float64), W[which]); return f'{m:+.5f} +- {s:.5f}'

    # ---- descriptive: how the two pools differ on cold vs warm rows (label-free and labelled)
    for c in Q:
        q = Q[c]
        dz = q['Pto'] - q['Bto']
        say(f'{c} descriptive: mean (z_pop - z_base) on cold rows {dz[COLD].mean():+.4f}, warm rows {dz[~COLD].mean():+.4f} (train-only pools); '
            f'positives cold share {COLD[y > 0].mean():.4f} vs all rows {COLD.mean():.4f}')
    # ---- step 1: tensors and nested reads
    E = {};
    for c in Q:
        E[c] = tensor_cw(Q[c], COLD); say(f'{c}: CW tensor done ({E[c].shape})')
    out_rows = {}
    for c in Q:
        nC, pC = nested(E[c], CONST_IDX); nW, pW = nested(E[c])
        ins_c = E[c][CONST_IDX].mean(1).max() - PL[c].mean(); ins_w = E[c].mean(1).max() - PL[c].mean()
        best = CW_RULES[int(np.argmax(E[c].mean(1)))]
        say(f'{c} CONST: in-sample best - PLAIN {ins_c:+.5f}; nested - PLAIN {fmt(nC - PL[c])}; picks {[(tuple(float(x) for x in CW_RULES[k]), v) for k, v in sorted(pC.items(), key=lambda t: -t[1])][:4]}')
        say(f'{c} CW   : in-sample best - PLAIN {ins_w:+.5f} at (w_cold, w_warm) = {tuple(float(x) for x in best)}; nested - PLAIN {fmt(nW - PL[c])}; nested - M-RULE {fmt(nW - MR[c])}')
        say(f'{c} CW picks: {[(tuple(float(x) for x in CW_RULES[k]), v) for k, v in sorted(pW.items(), key=lambda t: -t[1])][:6]}')
        G = nW - nC
        say(f'{c} G = nested CW - nested CONST: {fmt(G)} | public mix {pm(G, "public")} | private mix {pm(G, "private")}')
        # fixed diagnostic contrasts (not selected): cold weight 0.4 / warm 0.2 and the reverse, minus PLAIN
        i_cw = CW_RULES.index((np.float32(.4), np.float32(.2))); i_wc = CW_RULES.index((np.float32(.2), np.float32(.4)))
        say(f'{c} fixed contrasts minus PLAIN: (cold .4, warm .2) {fmt(E[c][i_cw] - PL[c])} | (cold .2, warm .4) {fmt(E[c][i_wc] - PL[c])} | '
            f'(cold .0, warm .2) {fmt(E[c][CW_RULES.index((np.float32(0), np.float32(.2)))] - PL[c])} | (cold .2, warm .0) {fmt(E[c][CW_RULES.index((np.float32(.2), np.float32(0)))] - PL[c])}')
        out_rows[c] = {'G': cse(G), 'nested_cw_minus_plain': cse(nW - PL[c]), 'nested_const_minus_plain': cse(nC - PL[c]), 'nested_cw_minus_mrule': cse(nW - MR[c]),
                       'insample_cw': float(ins_w), 'insample_const': float(ins_c), 'best_rule': [float(x) for x in best]}
        np.save(HERE / f'G_{c}.npy', G.astype(np.float32))
    res['real'] = {c: {k: (list(map(float, v)) if isinstance(v, tuple) else v) for k, v in out_rows[c].items()} for c in out_rows}
    # in-sample surface of C1 and C3 (mean NDCG minus PLAIN, 1e-5)
    for c in Q:
        S = (E[c].mean(1) - PL[c].mean()).reshape(len(WG), len(WG)) * 1e5
        say(f'{c} in-sample surface (rows w_cold, cols w_warm; 1e-5 vs PLAIN):')
        say('        ' + ' '.join(f'{w:6.1f}' for w in WG))
        for i, w in enumerate(WG):
            say(f'  {w:4.1f}  ' + ' '.join(f'{v:+6.0f}' for v in S[i]))
    # ---- placebos on C1
    Gp = []
    for s in range(5):
        cp = perm_within_feed(COLD, 500 + s)
        assert np.array_equal(np.bincount(fid, cp, NF), np.bincount(fid, COLD, NF))
        Ep = tensor_cw(Q['C1'], cp); nC, _ = nested(Ep, CONST_IDX); nW, _ = nested(Ep)
        m, se = cse(nW - nC); Gp.append(m)
        say(f'P-perm {s}: G {fmt(nW - nC)}; in-sample best CW - PLAIN {Ep.mean(1).max() - PL["C1"].mean():+.5f}; share of rows whose flag changed {np.mean(cp != COLD):.3f}')
    res['G_perm'] = [float(x) for x in Gp]
    blocks = [[f'w5_fd10_{a}_s{s}' for s in ss] for a in ('kd05w192', 'kd05t25w192') for ss in (range(951, 955), range(955, 959))]
    Gq = []; ONULL = []
    for b, blk in enumerate(blocks):
        rest = [m for m in MEM['FD16'] if m not in blk]
        qn = pools(rest, blk); qr = pools(rest, MEM['FDP'])
        En = tensor_cw(qn, COLD); nC, _ = nested(En, CONST_IDX); nW, _ = nested(En)
        Gq.append(cse(nW - nC)[0])
        say(f'P-pool {b} (12 base + 4 same-recipe as "popularity-light"): G {fmt(nW - nC)}; in-sample best CW - plain(0.2) {En.mean(1).max() - ev(final(qn, 0.2)).mean():+.5f}')
        # oracles: real 12 + 4 popularity-light vs null 12 + 4 same-recipe
        row = {}
        for tag, q in (('real', qr), ('null', qn)):
            a0 = ev(final(q, 0.0)); a1 = ev(final(q, 1.0)); a2 = ev(final(q, 0.2)); a4 = ev(final(q, 0.4))
            Et = En if tag == 'null' else tensor_cw(q, COLD)
            row[tag] = {'lvl_base': a0.mean(), 'lvl_alt': a1.mean(), 'O1': np.maximum(a0, a1).mean() - a0.mean(), 'O1_differ': float(np.mean(a0 != a1)),
                        'O2': np.maximum(a2, a4).mean() - a2.mean(), 'O2_differ': float(np.mean(a2 != a4)), 'd42': a4.mean() - a2.mean(),
                        'O3': Et.max(0).mean() - a2.mean()}
        ONULL.append(row)
        say(f'  oracles block {b}: ' + ' | '.join(f'{t}: base {r["lvl_base"]:.5f} alt-only {r["lvl_alt"]:.5f} O1 {r["O1"]:+.5f} (differ {r["O1_differ"]:.3f}) O2(.2 vs .4) {r["O2"]:+.5f} (differ {r["O2_differ"]:.3f}; .4-.2 {r["d42"]:+.5f}) O3(64 CW rules) {r["O3"]:+.5f}' for t, r in row.items()))
    res['G_pool'] = [float(x) for x in Gq]
    res['oracles_12p4'] = [{t: {k: float(v) for k, v in r.items()} for t, r in row.items()} for row in ONULL]
    for k in ('O1', 'O2', 'O3'):
        say(f'ORACLE {k} (12 + 4, mean of 4 blocks): real {np.mean([r["real"][k] for r in ONULL]):+.5f} | same-recipe null {np.mean([r["null"][k] for r in ONULL]):+.5f} | ratio {np.mean([r["real"][k] for r in ONULL]) / np.mean([r["null"][k] for r in ONULL]):.2f}')
    # full 16 + 4 oracles on both compositions
    for c in Q:
        a0 = ev(final(Q[c], 0.0)); a1 = ev(final(Q[c], 1.0)); a2 = PL[c]; a4 = ev(final(Q[c], 0.4))
        say(f'ORACLES {c} (16 + 4): base-only {a0.mean():.5f}, popularity-light-only {a1.mean():.5f}, PLAIN {a2.mean():.5f}; O1 per-feed best of (base-only, pop-only) - base-only {np.maximum(a0, a1).mean() - a0.mean():+.5f}; '
            f'O2 per-feed best of (0.2, 0.4) - PLAIN {np.maximum(a2, a4).mean() - a2.mean():+.5f} (M-rule gets {MR[c].mean() - a2.mean():+.5f}); O3 per-feed best of 64 CW rules - PLAIN {E[c].max(0).mean() - a2.mean():+.5f}')
        res[f'oracle16_{c}'] = {'O1': float(np.maximum(a0, a1).mean() - a0.mean()), 'O2': float(np.maximum(a2, a4).mean() - a2.mean()), 'O3': float(E[c].max(0).mean() - a2.mean()),
                                'mrule_minus_plain': float(MR[c].mean() - a2.mean()), 'base_only': float(a0.mean()), 'pop_only': float(a1.mean()), 'plain': float(a2.mean())}
    # ---- CWM information family
    for c in Q:
        E4 = tensor_cwm(Q[c], COLD); n4, p4 = nested(E4)
        nC, _ = nested(E[c], CONST_IDX)
        b = CWM_RULES[int(np.argmax(E4.mean(1)))]
        say(f'{c} CWM (information): in-sample best - PLAIN {E4.mean(1).max() - PL[c].mean():+.5f} at (cold&short, warm&short, cold&long, warm&long) = {tuple(float(x) for x in b)}; nested - PLAIN {fmt(n4 - PL[c])}; nested - M-RULE {fmt(n4 - MR[c])}; nested - nested CONST {fmt(n4 - nC)}')
        res[f'cwm_{c}'] = {'nested_minus_plain': list(map(float, cse(n4 - PL[c]))), 'nested_minus_mrule': list(map(float, cse(n4 - MR[c]))), 'best': [float(x) for x in b]}
    # ---- criterion
    g1, s1 = out_rows['C1']['G']; g3, s3 = out_rows['C3']['G']
    gpriv = pmix.wmean_se(np.load(HERE / 'G_C1.npy').astype(np.float64), W['private'])[0]
    c_a = g1 >= 2 * s1; c_b = g3 >= s3; c_c = g1 > max(Gp); c_d = gpriv >= 0
    say(f'CRITERION: G(C1) {g1:+.5f} +- {s1:.5f} >= +2 SE: {c_a} | G(C3) {g3:+.5f} +- {s3:.5f} >= +1 SE: {c_b} | G(C1) > max P-perm ({max(Gp):+.5f}): {c_c} | private-mix G(C1) {gpriv:+.5f} >= 0: {c_d}')
    say('VERDICT: ' + ('SIGNAL' if (c_a and c_b and c_c and c_d) else 'NO SIGNAL at pool level'))
    res['criterion'] = {'a': bool(c_a), 'b': bool(c_b), 'c': bool(c_c), 'd': bool(c_d), 'verdict': 'SIGNAL' if (c_a and c_b and c_c and c_d) else 'NO SIGNAL'}
    json.dump(res, open(HERE / 'cand_route.json', 'w'), indent=1, default=float)
    say('done')


if __name__ == '__main__':
    main()
