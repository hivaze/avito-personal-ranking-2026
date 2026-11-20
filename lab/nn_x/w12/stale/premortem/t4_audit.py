"""premortem t4_audit: CPU tests for the assumption audit (rules R4 w_ft scan, R5 / R5b fold variance + SIMEX, R6 pool size).
Usage: t4_audit.py <wft|fold|size>[,..]      Writes out/t4_<part>.md"""
import sys, time
from pm_common import *

parts = sys.argv[1].split(',')
m = Meta(test=('fold' in parts), hold=('size' in parts))
Wt = m.pmix(); wpriv, wpub = Wt['private'], Wt['public']
big = m.M['val'] >= 50; n = len(big)
L = lambda k, p: np.load(CACHE / f'pool_{k}_{p}.npy')
Zv = lambda x: m.z('val', x)


def emit(name, lines):
    open(OUT / f't4_{name}.md', 'w').write('\n'.join(lines) + '\n')


# =========================================================================================== R4: w_ft scan
if 'wft' in parts:
    out = []
    def P_(s=''):
        print(s, flush=True); out.append(s)
    WF = (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0)
    def branches(spec):
        base, pop, w = spec
        r = {}
        for kd in ('val_ft', 'val_to', 'sv'):
            r[kd] = Zv(L(kd, base)) if pop is None else Zv((1 - w) * Zv(L(kd, base)) + w * Zv(L(kd, pop)))
        return r
    POOLS = {'W8': ('W8', None, 0), 'W16new (s955-958)': ('W8b', None, 0), 'W16': ('W16', None, 0), 'W8P20': ('W8', 'Pd', 0.2), 'W16P20x8': ('W16', 'Px8', 0.2), 'S16': ('S16', None, 0)}
    ND = {}
    for pn, spec in POOLS.items():
        b = branches(spec)
        ND[pn] = np.stack([m.ndcg('val', w * b['val_ft'] + (1 - w) * b['val_to']) for w in WF])
        ND[pn + '|svf'] = np.stack([m.ndcg('val', w * b['val_ft'] + (1 - w) * b['sv']) for w in WF])
    P_('## T4a. Fine-tune weight w_ft (FINAL = w_ft z(FT) + (1 - w_ft) z(TO)); delta against w_ft = 0.5, paired per feed (rule R4)')
    for view, lab in (('val', 'plain val'), ('ppub', 'pmix-public'), ('ppriv', 'pmix-private'), ('m50p', 'val M>=50'), ('m50m', 'val M<50'), ('svf', 'half-stale (TO branch from stale-val files; information)')):
        P_(); P_(f'### {lab}')
        P_('| pool | ' + ' | '.join(f'w_ft {w:.1f}' for w in WF if w != 0.5) + ' |')
        P_('|---|' + '---|' * (len(WF) - 1))
        for pn in POOLS:
            A = ND[pn + '|svf'] if view == 'svf' else ND[pn]
            cells = []
            for i, w in enumerate(WF):
                if w == 0.5: continue
                d = A[i] - A[4]
                t = wms(d) if view in ('val', 'svf') else wms(d, wpub) if view == 'ppub' else wms(d, wpriv) if view == 'ppriv' else wms(d[big]) if view == 'm50p' else wms(d[~big])
                cells.append(f5(t))
            P_(f'| {pn} | ' + ' | '.join(cells) + ' |')
    # R4 verdict
    def chk(pn, i):
        d = ND[pn][i] - ND[pn][4]; pp = wms(d, wpriv); v = wms(d); return pp[0] >= 1.5 * pp[1] and v[0] >= 0
    def chk_low(pn, i):
        d = ND[pn][i] - ND[pn][4]; v = wms(d); return v[0] >= 1.5 * v[1]
    hi = [WF[i] for i in range(5, len(WF)) if chk('W8', i) and chk('W16new (s955-958)', i)]
    lo = [WF[i] for i in range(0, 4) if chk_low('W8', i) and chk_low('W16new (s955-958)', i)]
    P_(); P_(f'**R4:** w_ft > 0.5 passing (pmix-private >= +1.5 SE and val >= 0 in BOTH W8 and W16new): {hi if hi else "none"} -> "w_ft = 0.5" is {"WEAKENED" if hi else "not weakened by this scan"}. '
             f'w_ft < 0.5 with val >= +1.5 SE in both: {lo if lo else "none"}.')
    # nested pick of w_ft on user halves (information; ft_redesign did 20 splits)
    uu, uinv = np.unique(m.vuser, return_inverse=True); rng = np.random.default_rng(7)
    P_(); P_('Nested pick of w_ft (grid above; picked on one user half by plain val or by pmix-private-weighted val, scored on the other; 200 splits, both directions; delta against 0.5):')
    P_('| pool | pick by plain val: mean picked w_ft | out-of-sample val | out-of-sample pmix-private | pick by pmix-private weights: mean picked w_ft | out-of-sample val | out-of-sample pmix-private |')
    P_('|---|---|---|---|---|---|---|')
    for pn in POOLS:
        A = ND[pn]; res = {}
        for mode in ('val', 'ppriv'):
            o, op, pk = [], [], []
            for it in range(200):
                hu = rng.random(len(uu)) < 0.5; Am = hu[uinv]; d = np.zeros(n)
                for tr, te in ((Am, ~Am), (~Am, Am)):
                    sc = A[:, tr].mean(1) if mode == 'val' else (A[:, tr] * wpriv[tr]).sum(1) / wpriv[tr].sum()
                    i = int(np.argmax(sc)); pk.append(WF[i]); d[te] = (A[i] - A[4])[te]
                o.append(d.mean()); op.append(np.average(d, weights=wpriv))
            res[mode] = (np.mean(pk), np.mean(o), np.std(o), np.mean(op), np.std(op))
        P_(f'| {pn} | {res["val"][0]:.2f} | {res["val"][1]:+.5f} (sd {res["val"][2]:.5f}) | {res["val"][3]:+.5f} (sd {res["val"][4]:.5f}) | {res["ppriv"][0]:.2f} | {res["ppriv"][1]:+.5f} (sd {res["ppriv"][2]:.5f}) | {res["ppriv"][3]:+.5f} (sd {res["ppriv"][4]:.5f}) |')
    # FT branch value by feed cold share (the mix-shift reading of the public gap)
    P_(); P_('FT half value (FINAL w_ft 0.5 minus TO only) by feed cold share and by M, val:')
    P_('| pool | cold < 0.15 | 0.15-0.30 | 0.30-0.45 | >= 0.45 | M < 50 | M >= 50 | Tue | Thu |')
    P_('|---|---|---|---|---|---|---|---|---|')
    c = m.vcold; cg = [(c < 0.15), (c >= 0.15) & (c < 0.30), (c >= 0.30) & (c < 0.45), c >= 0.45, ~big, big, m.tue, m.thu]
    for pn in POOLS:
        d = ND[pn][4] - ND[pn][0]
        P_(f'| {pn} | ' + ' | '.join(f5(wms(d[g])) for g in cg) + ' |')
    P_('| feeds | ' + ' | '.join(str(int(g.sum())) for g in cg) + ' |')
    emit('wft', out)

# =========================================================================================== R5: fold variance + SIMEX
if 'fold' in parts:
    out = []
    def P_(s=''):
        print(s, flush=True); out.append(s)
    t0 = time.time()
    off, M = m.off['test'], m.M['test']; fid = np.repeat(np.arange(len(M)), M); nt = len(M)
    def pf_corr(a, b):
        a = a.astype(np.float64); b = b.astype(np.float64)
        da = a - (np.bincount(fid, a, nt) / M)[fid]; db = b - (np.bincount(fid, b, nt) / M)[fid]
        num = np.bincount(fid, da * db, nt); den = np.sqrt(np.bincount(fid, da * da, nt) * np.bincount(fid, db * db, nt))
        return num / np.maximum(den, 1e-12)
    def pf_var(a):
        a = a.astype(np.float64); da = a - (np.bincount(fid, a, nt) / M)[fid]; return np.bincount(fid, da * da, nt) / M
    folds = {}
    for v in G['W16'] + G['Pd']:
        folds[v] = np.load(path(v, 'folds')).astype(np.float32)          # (5, N): per-fold mean over 3 seeds of the per-feed z
    P_('## T4b. Fold-to-fold agreement of the fine-tune branch on TEST rows (label-free; rule R5)')
    P_('| unit | feed-mean Pearson r between two fold predictions (10 pairs) | v1 = 1 - r | r(fold k, 5-fold mean) | public feeds r / private feeds r |')
    P_('|---|---|---|---|---|')
    pairs = [(i, j) for i in range(5) for j in range(i + 1, 5)]
    stat = {}
    for lab, names in (('single member (mean of the 16 wide students)', None), ('W8 FT pool (8 members)', G['W8']), ('W16 FT pool (16 members)', G['W16']), ('Pd FT pool (4 pop1 d128 students)', G['Pd'])):
        if names is None:
            rs, rm, rp, rq = [], [], [], []
            for v in G['W16'][:6]:
                F = folds[v]; mean5 = F.mean(0)
                for (i, j) in pairs[:4]:
                    r = pf_corr(F[i], F[j]); rs.append(r.mean()); rp.append(r[m.public].mean()); rq.append(r[~m.public].mean())
                rm.append(pf_corr(F[0], mean5).mean())
            q1 = float(np.mean(rs)); stat[lab] = q1
            P_(f'| {lab} | {q1:.4f} | {1 - q1:.4f} | {np.mean(rm):.4f} | {np.mean(rp):.4f} / {np.mean(rq):.4f} |')
        else:
            Pk = np.stack([np.mean([folds[v][k] for v in names], 0) for k in range(5)]); mean5 = Pk.mean(0)
            rs, rp, rq = [], [], []
            for (i, j) in pairs:
                r = pf_corr(Pk[i], Pk[j]); rs.append(r.mean()); rp.append(r[m.public].mean()); rq.append(r[~m.public].mean())
            rm = np.mean([pf_corr(Pk[k], mean5).mean() for k in range(5)])
            q1 = float(np.mean(rs)); stat[lab] = q1
            # top-1 agreement fold pool vs 5-fold mean
            def top1(s):
                o = np.lexsort((-s.astype(np.float64), fid)); return o[off]
            t1 = np.mean([(top1(Pk[k]) == top1(mean5)).mean() for k in range(5)])
            P_(f'| {lab} | {q1:.4f} | {1 - q1:.4f} | {rm:.4f} (same top-1 item {t1:.3f}) | {np.mean(rp):.4f} / {np.mean(rq):.4f} |')
            stat[lab + '|var'] = float(pf_var(Pk[0]).mean())
    q_m = stat['single member (mean of the 16 wide students)']; q8 = stat['W8 FT pool (8 members)']; q16 = stat['W16 FT pool (16 members)']
    P_(); P_(f'Fold noise of a single member v1 = {1 - q_m:.4f}; of the 8-pool {1 - q8:.4f}; of the 16-pool {1 - q16:.4f}. If the fold noise were independent across members the 8-pool would have about {(1 - q_m) / 8:.4f} '
             f'and the 16-pool {(1 - q_m) / 16:.4f}: the part common to all members (same GroupKFold split) is what a val row keeps and a test row averages over 5 folds.')
    log_t = time.time() - t0
    # ---- SIMEX on val
    rng = np.random.default_rng(11)
    LAM = (0.5, 1.0, 2.0); voff, vM = m.off['val'], m.M['val']
    P_(); P_('## T4c. SIMEX on val: what removing 4/5 of the fold noise of the out-of-fold FT pool would be worth (rule R5 white noise, R5b structured noise)')
    P_('| pool | noise | FINAL val at lambda 0 | lambda 0.5 | lambda 1 | lambda 2 | extrapolated to lambda -0.8 (quadratic) | same, linear fit | gain FINAL at -0.8 (quadratic / linear) | FT branch alone: gain at -0.8 (quadratic / linear) |')
    P_('|---|---|---|---|---|---|---|---|---|---|')
    def simex(pn, q1, mode):
        ft = Zv(L('val_ft', pn)); to = Zv(L('val_to', pn)); raw = L('val_ft', pn).astype(np.float32)
        v1 = 1 - q1
        base_fin = m.ndcg('val', 0.5 * ft + 0.5 * to).mean(); base_ft = m.ndcg('val', ft).mean()
        fin, fto = {0.0: base_fin}, {0.0: base_ft}
        if mode == 'struct':
            names = G['W16']; zs = [mz(m, v, 'val_ft') for v in names]
        for lam in LAM:
            a, b = [], []
            for it in range(20 if mode == 'white' else 30):
                if mode == 'white':
                    noisy = ft + np.sqrt(lam * v1) * rng.standard_normal(len(ft)).astype(np.float32)      # ft has within-feed variance 1
                else:
                    perm = rng.permutation(16); A = np.mean([zs[i] for i in perm[:8]], 0); Bm = np.mean([zs[i] for i in perm[8:]], 0); D = (A - Bm) / 2
                    # scale D so that its mean within-feed variance = lam * v1 * var(pool mean)
                    vD = float(np.mean((D.astype(np.float64)) ** 2)); vP = float(np.mean(raw.astype(np.float64) ** 2))
                    noisy = Zv(raw + np.sqrt(lam * v1 * vP / vD) * D)
                zn = Zv(noisy)
                a.append(m.ndcg('val', 0.5 * zn + 0.5 * to).mean()); b.append(m.ndcg('val', zn).mean())
            fin[lam] = float(np.mean(a)); fto[lam] = float(np.mean(b))
        xs = np.array([0.0] + list(LAM))
        def ext(dv):
            ys = np.array([dv[x] for x in xs]); q = np.polyfit(xs, ys, 2); l = np.polyfit(xs, ys, 1); return float(np.polyval(q, -0.8)), float(np.polyval(l, -0.8))
        eq, el = ext(fin); fq, fl = ext(fto)
        P_(f'| {pn} | {mode} | {fin[0.0]:.5f} | {fin[0.5]:.5f} | {fin[1.0]:.5f} | {fin[2.0]:.5f} | {eq:.5f} | {el:.5f} | **{eq - base_fin:+.5f}** / {el - base_fin:+.5f} | {fq - base_ft:+.5f} / {fl - base_ft:+.5f} |')
        return eq - base_fin, el - base_fin
    r_w8 = simex('W8', q8, 'white'); r_w16 = simex('W16', q16, 'white'); r_s16 = simex('W16', q16, 'struct')
    g = r_s16[0]
    P_(); P_(f'**R5:** structured-noise SIMEX gain at FINAL level for W16 = {g:+.5f} (white noise: W8 {r_w8[0]:+.5f}, W16 {r_w16[0]:+.5f}). Bars: SUPPORTED iff >= 0.0002; NOT SUPPORTED AS THE MAIN CAUSE iff < 0.0001. '
             f'-> {"SUPPORTED" if g >= 0.0002 else "NOT SUPPORTED AS THE MAIN CAUSE" if g < 0.0001 else "IN BETWEEN (partly)"}. '
             f'For comparison: public reads the FT half of W8 at +0.00048 ± 0.00055, plain val at +0.00009 ± 0.00043, pmix-public at +0.00066 ± 0.00054.')
    emit('fold', out)

# =========================================================================================== R6: pool size
if 'size' in parts:
    out = []
    def P_(s=''):
        print(s, flush=True); out.append(s)
    names = G['W16']; rng = np.random.default_rng(3)
    Zm = {kd: np.stack([mz(m, v, kd) for v in names]) for kd in ('val_ft', 'val_to', 'sv')}
    Zh = np.stack([m.z('hold', np.load(path(v, 'hold')).astype(np.float32)) for v in names])
    NS = (1, 2, 4, 8, 16); res = {k: {} for k in ('val FINAL', 'pmix-public FINAL', 'pmix-private FINAL', 'val TO', 'stale-val TO', 'val FT', 'holdout TO')}
    for k_ in NS:
        acc = {k: [] for k in res}
        subs = [np.arange(16)] if k_ == 16 else [rng.choice(16, k_, replace=False) for _ in range(60)]
        for s in subs:
            ft = Zv(Zm['val_ft'][s].mean(0)); to = Zv(Zm['val_to'][s].mean(0)); sv = Zv(Zm['sv'][s].mean(0))
            f = m.ndcg('val', 0.5 * ft + 0.5 * to)
            acc['val FINAL'].append(f.mean()); acc['pmix-public FINAL'].append(np.average(f, weights=wpub)); acc['pmix-private FINAL'].append(np.average(f, weights=wpriv))
            acc['val TO'].append(m.ndcg('val', to).mean()); acc['stale-val TO'].append(m.ndcg('val', sv).mean()); acc['val FT'].append(m.ndcg('val', ft).mean())
            acc['holdout TO'].append(m.ndcg('hold', m.z('hold', Zh[s].mean(0))).mean())
        for k in res: res[k][k_] = (float(np.mean(acc[k])), float(np.std(acc[k])))
    P_('## T4d. Pool size: n wide students (mean over 60 random subsets of the 16; sd over subsets in brackets) and the fit a - b / n (rule R6)')
    P_('| column | n = 1 | n = 2 | n = 4 | n = 8 | n = 16 | a (n = infinity) | b | expected gain of 16 -> 32 (b / 32) | 8 -> 16 measured | 8 -> 16 by the fit (b / 16) |')
    P_('|---|---|---|---|---|---|---|---|---|---|---|')
    worst = 0
    for k in res:
        x = np.array([1 / k_ for k_ in NS]); y = np.array([res[k][k_][0] for k_ in NS]); sl, ic = np.polyfit(x, y, 1); b = -sl; worst = max(worst, b / 32)
        P_(f'| {k} | ' + ' | '.join(f'{res[k][k_][0]:.5f} ({res[k][k_][1]:.5f})' for k_ in NS) + f' | {ic:.5f} | {b:.5f} | {b / 32:+.5f} | {res[k][16][0] - res[k][8][0]:+.5f} | {b / 16:+.5f} |')
    P_(); P_(f'**R6:** largest b / 32 over the columns = {worst:+.5f} -> "saturated at 16" is {"CONFIRMED (< 0.0002 on every column)" if worst < 0.0002 else "NOT confirmed by the size law"}.')
    emit('size', out)
