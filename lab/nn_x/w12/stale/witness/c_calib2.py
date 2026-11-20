"""witness task C (v2 of c_calib.py; rule: DECLARATION.txt block 1 (3) + block 4 (6)-(8), applied as written).
Calibration of the staleness witnesses on the public readings. Train-only pools; A = arm, B = reference; +- = 1 SE paired per feed.
Public deltas are FINAL files (0.5 FT + 0.5 TO): a known mismatch, shown by the two extra columns val_final / val_to_file (lead's stored halves).
Pair (iv): free statistics on the FULL pools (C3 28, G3 20), CPU statistics on the reduced pools C3h / G3h (13 members).

  cd <repo>/nn_x/w12/stale/witness && WT_THREADS=4 CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 ../../../../.venv/bin/python c_calib2.py [aa]
"""
import os
os.environ.setdefault('WT_THREADS', '4')
from wt_common import *
import lagged_score as LS
import stale_val_cpu as SVC
import wt_read as WR

OUT = []


def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); OUT.append(s)


PAIRS = [  # key, A full, B full, A cpu, B cpu (None = the full spec), public delta, scored, lead's file tags (A, B), note
    ('i   W8P20 - W8', '0.8*W8+0.2*pop1kd', 'W8', None, None, +0.000523, True, ('W8P20kd', 'W8kd'), 'staleness family'),
    ('ii  W16 - W8', 'W16', 'W8', None, None, -0.000003, True, ('W16kd', 'W8kd'), 'same recipe, more seeds'),
    ('iii S16 - W8', 'S16', 'W8', None, None, -0.000408, True, ('S16kd', 'W8kd'), 'narrow students added'),
    ('iv  C3 - G3', 'C3', 'G3', C3H, G3H, +0.000634, True, ('Q2a8s3', 'G3'), 'staleness family: 8 hard-label pop-light members added to 20'),
    ('v   W8 - base4', 'W8', 'base4', None, None, None, False, None, 'scale reference, no public reading'),
    ('vi  W8 - C3', 'W8', 'C3', 'W8', C3H, +0.001032, False, ('W8kd', 'Q2a8s3'), 'information (not a pair of the task)'),
    ('x1  W16P20 - W16', '0.8*W16+0.2*pop1kd', 'W16', None, None, None, False, None, 'information'),
    ('x2  W8Pw20 - W8', '0.8*W8+0.2*pop1kdw', 'W8', None, None, None, False, None, 'information (wide pop-light students)'),
]
FREE = ['val', 'val_M>=50', 'val_M<50', 'pmix_pub', 'pmix_priv', 'coldpos', 'warmpos', 'hold', 'hold_subset', 'HOLD-SHORT']
SINGLE = ['val', 'pmix_pub', 'pmix_priv', 'stale', 'hold', 'lag_lag0', 'lag_priv', 'lag_pub', 'SV2']
COMBO = {'ivw(val, lag_pub)': ('val', 'lag_pub'), 'ivw(val, lag_priv)': ('val', 'lag_priv'), 'ivw(stale, lag_priv)': ('stale', 'lag_priv'),
         'ivw(SV2, lag_priv)': ('SV2', 'lag_priv'), 'ivw(pmix_pub, lag_pub)': ('pmix_pub', 'lag_pub')}
NPUB = 8562

wt = WR.Witness()
V = wt.V
P(f'=== witness calibration v2 {now()} (train-only pools; +- = 1 SE paired per feed; val 14,423 feeds, holdout 30,000, lagged subset 13,424) ===')

# ---- pool identity (label-free): my train-only pools against the lead's stored train-only halves of the scored files
P('\n##### POOL IDENTITY: feed-mean Pearson r of per-feed z, my train-only pool vs preds/ens_<tag>_to_val.npy #####')
fidv = np.repeat(np.arange(len(V.M)), V.M)


def feedcorr(a, b):
    za, zb = feed_z(V.off, V.M, a).astype(np.float64), feed_z(V.off, V.M, b).astype(np.float64)
    return float((np.bincount(fidv, za * zb, len(V.M)) / V.M).mean())


IDENT = {}
for tag, spec in (('W8kd', 'W8'), ('W16kd', 'W16'), ('S16kd', 'S16'), ('G3', 'G3'), ('Q2a8s3', 'C3'), ('W8P20kd', '0.8*W8+0.2*pop1kd')):
    f = ROOT / f'preds/ens_{tag}_to_val.npy'
    if not f.exists():
        P(f' {tag}: no stored train-only half'); continue
    r = feedcorr(V.pool(WR.parse(spec), 'val'), np.load(f)); IDENT[tag] = r
    P(f' {tag:8s} vs my [{spec}]: r = {r:.5f} {"ok" if r >= 0.999 else "FLAG (< 0.999): my pool is not the train-only half of the scored file"}')
r = feedcorr(V.pool(WR.parse(G3H), 'val'), V.pool(WR.parse('G3'), 'val')); P(f' reduced G3h vs full G3: r = {r:.5f}')
r = feedcorr(V.pool(WR.parse(C3H), 'val'), V.pool(WR.parse('C3'), 'val')); P(f' reduced C3h vs full C3: r = {r:.5f}')


def lead_val(tagA, tagB):
    """val delta of the lead's stored files: FINAL = 0.5 z(FT) + 0.5 z(TO), and the TO half alone."""
    z = lambda x: feed_z(V.off, V.M, np.load(x))
    out = {}
    for name, fn in (('val_final', lambda t: 0.5 * z(ROOT / f'preds/ens_{t}_ft_val.npy') + 0.5 * z(ROOT / f'preds/ens_{t}_to_val.npy')),
                     ('val_to_file', lambda t: z(ROOT / f'preds/ens_{t}_to_val.npy'))):
        d = ndcg_fast(fn(tagA), V.y, V.off, V.M) - ndcg_fast(fn(tagB), V.y, V.off, V.M)
        out[name] = ms(d) + (len(d),)
        if name == 'val_final':
            out['pmix_pub_final'] = ms(d, wt.W['public']) + (len(d),); out['pmix_priv_final'] = ms(d, wt.W['private']) + (len(d),)
    return out


def read_pair(a, b, ac, bc):
    A, B = WR.parse(a), WR.parse(b)
    if ac is None:
        return wt.read(A, B), None
    rf = wt.read(A, B, lag_views=())                       # free statistics AND stale-val / SV2 on the full pools (block 5)
    rc = wt.read(WR.parse(ac), WR.parse(bc), short=False)  # lagged columns on the reduced pools (reported only when the representativeness check fails)
    r = {k: v for k, v in rf.items() if k != '_pf'}
    r.update({k: v for k, v in rc.items() if k.startswith(('lag_', 'D_', 'drop', 'A_lag', 'B_lag'))})
    for k in ('stale', 'SV2'):
        if k in rc:
            r[f'{k}_reduced'] = rc[k]
    r['_pf'] = dict(rf['_pf']); r['_pf'].update({k: v for k, v in rc['_pf'].items() if k.startswith('lag_')})
    rep = {}
    for k in ('val', 'hold'):
        if k in rc['_pf'] and k in rf['_pf']:
            rep[k] = (rc[k], rf[k], ms(rc['_pf'][k] - rf['_pf'][k]))
    return r, rep


R = {}
for key, a, b, ac, bc, pub, scored, tags, note in PAIRS:
    r, rep = read_pair(a, b, ac, bc)
    if 'val' not in r:
        P(f'\n{key}: val predictions missing, skipped'); continue
    r['public'] = pub; r['se_public'] = float(r['_pf']['val'].std(ddof=1) / np.sqrt(NPUB)); r['scored'] = scored
    if tags and all((ROOT / f'preds/ens_{t}_{h}_val.npy').exists() for t in tags for h in ('ft', 'to')):
        r.update(lead_val(*tags))
    for cn, (x, y) in COMBO.items():
        if x in r and y in r:
            r[cn] = WR.ivw(r[x], r[y]) + (0,)
    R[key] = r
    P('\n' + WR.show(r, f'{key}: [{a}] vs [{b}]  ({note}); PUBLIC {"n/a" if pub is None else f"{pub:+.6f} +- {r['se_public']:.5f}"}'))
    for k in ('val_final', 'pmix_pub_final', 'pmix_priv_final', 'val_to_file'):
        if k in r:
            P(f'   {k:16s} {r[k][0]:+.5f} +- {r[k][1]:.5f}  (lead\'s stored files)')
    for cn in COMBO:
        if cn in r:
            P(f'   {cn:24s} {r[cn][0]:+.5f} +- {r[cn][1]:.5f}')
    if rep is not None:
        P(f'   LAGGED columns on the reduced pools [{ac}] vs [{bc}] (stale-val / SV2 on the full pools)')
        for k in ('stale_reduced', 'SV2_reduced'):
            if k in r:
                P(f'   {k:16s} {r[k][0]:+.5f} +- {r[k][1]:.5f}  (reduced pools, information)')
        okrep = True
        for k, (red, full, dd) in rep.items():
            ok = abs(dd[0]) <= 2 * dd[1]; okrep &= ok
            P(f'   representativeness {k}: reduced {red[0]:+.5f} +- {red[1]:.5f} | full {full[0]:+.5f} +- {full[1]:.5f} | reduced - full {dd[0]:+.5f} +- {dd[1]:.5f} ({dd[0] / dd[1]:+.2f} SE) -> {"ok" if ok else "OUTSIDE 2 SE"}')
        r['rep_ok'] = bool(okrep)
        # member controls of the reduced pools (block 2 / block 4 (6)): CPU lag0 on the subset vs the stored GPU holdout file, same feeds
        L = wt.L; okm = True
        mem = sorted(set(WR.members(WR.parse(ac)) + WR.members(WR.parse(bc))))
        for v in mem:
            if LS.pred_path('lag0', v).exists() and not v.startswith('w5_e10_kd05'):
                c_, g_ = L.member_level(v, 'lag0'), L.member_level(v, 'stored'); ok = abs(c_.mean() - g_.mean()) <= 1e-3; okm &= ok
                P(f'   member control lag0 {v:26s}: CPU {c_.mean():.5f} vs stored GPU {g_.mean():.5f} ({fmt(c_ - g_)}) -> {"ok" if ok else "FAIL"}')
        ev = lambda x: ndcg_fast(feed_z(V.off, V.M, x), V.y, V.off, V.M)
        for v in ('v23_gru_pre2s_s501', 'r6_cd_m02_r101', 'r6_cd_m02_p70k_r101', 'abl_e10_pop1_s951'):
            if v in mem and SVC.sv_path(v, 'base').exists():
                c_ = np.load(SVC.sv_path(v, 'base')); g_ = np.load(stored_val(v)); a_, b_ = ev(c_), ev(g_); cr = float(np.corrcoef(c_, g_)[0, 1])
                ok = abs(a_.mean() - b_.mean()) <= 1e-3 and cr >= 0.9995; okm &= ok
                P(f'   family control val  {v:26s}: CPU no-override {a_.mean():.5f} vs stored GPU {b_.mean():.5f} ({fmt(a_ - b_)}), corr {cr:.6f} -> {"ok" if ok else "FAIL"}')
        r['members_ok'] = bool(okm)

# ---- harness control with every witness: ref re-cools vs the original wide students (same snapshots)
orig = lambda st, s: f'w5_e10_{"kd05w192" if st == "k" else "kd05t25w192"}_s{s}'
rcn = lambda a, st, s: f'w5_{st}cd_{a}_s{s}'
ps = [(st, s) for st in 'kl' for s in S8 if done_json(rcn('ref', st, s)).exists() and SVC.sv_path(rcn('ref', st, s)).exists() and SVC.sv_path(orig(st, s)).exists()]
if ps:
    r = wt.read([(1.0, [rcn('ref', st, s) for st, s in ps])], [(1.0, [orig(st, s) for st, s in ps])])
    r['public'] = None; r['scored'] = False; R[f'hc  ref - orig ({len(ps)})'] = r
    P('\n' + WR.show(r, f'hc: ref re-cools vs the ORIGINAL wide students, {len(ps)} snapshot pairs (harness control with every witness; expected 0)'))

P('\n\n##### SIDE BY SIDE (delta A - B) #####')
cols = ['val_final', 'pmix_pub_final', 'val', 'pmix_pub', 'pmix_priv', 'stale', 'SV2', 'hold', 'hold_subset', 'lag_lag0', 'lag_priv', 'lag_pub', 'D_priv', 'D_pub'] + list(COMBO)
P('pair | PUBLIC | ' + ' | '.join(cols))
for key, r in R.items():
    pub = 'n/a' if r['public'] is None else f"{r['public']:+.5f} +- {r['se_public']:.5f}"
    P(f'{key} | {pub} | ' + ' | '.join((f'{r[c][0]:+.5f} +- {r[c][1]:.5f}' if c in r else 'n/a') for c in cols))

P('\n##### SCORES of the candidate statistics over the SCORED pairs (rule block 1 (3)) #####')
scored = [k for k, r in R.items() if r.get('scored') and r['public'] is not None]
for k in list(scored):
    if R[k].get('rep_ok') is False or R[k].get('members_ok') is False:
        P(f'pair {k}: representativeness / member control of the reduced pools not met -> the LAGGED candidates are NOT scored on it (block 5)')
cpu_ok = lambda k: R[k].get('rep_ok', True) and R[k].get('members_ok', True)
is_cpu = lambda c: 'lag' in c
P(f'scored pairs: {scored}; pairs counted for SIGN (|public| >= 1 public SE): {[k for k in scored if abs(R[k]["public"]) >= R[k]["se_public"]]}')
P('candidate | pairs available | SIGN right / counted | X2 (mean squared standardised distance to public) | slope of public on candidate through the origin | mean SE over its pairs')
S = {}
for c in SINGLE + list(COMBO) + ['val_final', 'pmix_pub_final', 'pmix_priv_final']:
    ks = [k for k in scored if c in R[k] and (cpu_ok(k) or not is_cpu(c))]
    if not ks:
        continue
    cnt = [k for k in ks if abs(R[k]['public']) >= R[k]['se_public']]
    right = sum(np.sign(R[k][c][0]) == np.sign(R[k]['public']) for k in cnt)
    x2 = float(np.mean([(R[k][c][0] - R[k]['public']) ** 2 / (R[k][c][1] ** 2 + R[k]['se_public'] ** 2) for k in ks]))
    x = np.array([R[k][c][0] for k in ks]); y = np.array([R[k]['public'] for k in ks]); slope = float((x * y).sum() / (x * x).sum())
    S[c] = dict(n=len(ks), right=int(right), counted=len(cnt), x2=x2, slope=slope, se=float(np.mean([R[k][c][1] for k in ks])), pairs=ks)
    P(f'{c:24s} | {len(ks)} | {right} / {len(cnt)} | {x2:.2f} | {slope:+.2f} | {S[c]["se"]:.5f}' + ('   (information: FINAL-level, not a declared candidate)' if 'final' in c else ''))
cand = {c: v for c, v in S.items() if 'final' not in c}
full = max(v['n'] for v in cand.values())
elig = {c: v for c, v in cand.items() if v['right'] == v['counted'] and v['n'] == full}
P(f'\ncandidates available on all {full} scored pairs: {[c for c, v in cand.items() if v["n"] == full]}')
P(f'of these, with no sign error: {sorted(elig, key=lambda c: elig[c]["x2"])}')
if elig:
    best = min(elig, key=lambda c: elig[c]['x2'])
    free = [c for c in elig if c in ('val', 'pmix_pub', 'pmix_priv')]
    bf = min(free, key=lambda c: elig[c]['x2']) if free else None
    P(f'smallest X2: {best} ({elig[best]["x2"]:.2f})' + (f'; best free candidate (val / pmix): {bf} ({elig[bf]["x2"]:.2f})' if bf else '; no free candidate without a sign error'))
    near = [c for c in elig if elig[c]['x2'] - elig[best]['x2'] <= 0.25]
    P(f'within 0.25 of the smallest X2 (tie by the rule -> smaller SE on the 16-vs-16 contrast decides): {near}')
    if bf and best not in ('val', 'pmix_pub', 'pmix_priv') and elig[bf]['x2'] - elig[best]['x2'] <= 0.25:
        P(f'RULE: the CPU-inference candidate {best} is not better than the free candidate {bf} by more than 0.25 in X2 -> recommended by the rule: {bf}')
    else:
        P(f'RULE (before the SE tie-break among {near}): {best}')
# ---- joint test on the two staleness-family pairs (information; not one of the declared scores)
ki = next((k for k in R if k.startswith('i ')), None); kiv = next((k for k in R if k.startswith('iv ')), None)
JOINT = {}
if ki and kiv:
    P('\n##### JOINT TEST on the two staleness-family pairs (i) + (iv): sum of the two deltas, candidate against public (information) #####')
    dv = R[ki]['_pf']['val'] + R[kiv]['_pf']['val']
    pub = R[ki]['public'] + R[kiv]['public']; sep = float(dv.std(ddof=1) / np.sqrt(NPUB))
    cpair = float(np.corrcoef(R[ki]['_pf']['val'], R[kiv]['_pf']['val'])[0, 1])
    P(f' public sum {pub:+.5f} +- {sep:.5f} (SE from the val per-feed sum of the two train-only deltas; per-feed correlation of the two pairs on val {cpair:+.2f})')
    P(' candidate | sum of the two deltas | public - candidate | z | candidate / public')
    for c, w in (('val', None), ('pmix_pub', wt.W['public']), ('pmix_priv', wt.W['private']), ('stale', None), ('SV2', None), ('hold', None), ('lag_lag0', None), ('lag_priv', None)):
        src = 'val' if c.startswith('pmix') else c
        if src in R[ki]['_pf'] and src in R[kiv]['_pf'] and len(R[ki]['_pf'][src]) == len(R[kiv]['_pf'][src]):
            m, se_ = ms(R[ki]['_pf'][src] + R[kiv]['_pf'][src], w)
            z = (pub - m) / np.hypot(sep, se_); JOINT[c] = dict(sum=m, se=se_, z=float(z), ratio=float(m / pub))
            P(f'   {c:10s} | {m:+.5f} +- {se_:.5f} | {pub - m:+.5f} +- {np.hypot(sep, se_):.5f} | {z:+.2f} | {m / pub:+.2f}')
(CACHE / 'readout').mkdir(exist_ok=True)
json.dump(JOINT, open(CACHE / 'readout/calib2_joint.json', 'w'), indent=1)
json.dump({k: WR.dump(r) for k, r in R.items()}, open(CACHE / 'readout/calib2.json', 'w'), indent=1)
json.dump(S, open(CACHE / 'readout/calib2_scores.json', 'w'), indent=1)

if len(sys.argv) > 1 and sys.argv[1] == 'aa':
    STATS = ('val', 'val_M<50', 'stale', 'SV2', 'pmix_pub', 'pmix_priv', 'hold', 'lag_lag0', 'lag_priv', 'lag_pub', 'D_priv', 'D_pub')
    P('\n\n##### A/A (same-recipe null): seed noise of each statistic against its per-feed SE #####')
    W16 = G['W16']; rng = np.random.default_rng(20261115); AA = {}
    lag16 = all(LS.pred_path('lag0', v).exists() and LS.pred_path('priv', v).exists() for v in W16)
    for N, K in ((4, 40), (8, 40)):
        acc = {c: [] for c in STATS}
        for _ in range(K):
            ix = rng.permutation(16); A = [(1.0, [W16[i] for i in ix[:N]])]; B = [(1.0, [W16[i] for i in ix[N:2 * N]])]
            r = wt.read(A, B, short=False, lag_views=('lag0', 'priv') if lag16 else ())
            for c in acc:
                if c in r:
                    acc[c].append((r[c][0], r[c][1]))
        P(f' 16 wide students, N = {N} vs {N} ({K} random disjoint splits): statistic | sd of the delta over splits | mean per-feed SE | sd / SE')
        for c, v in acc.items():
            if v:
                v = np.array(v); AA[f'{c}|{N}'] = (float(v[:, 0].std(ddof=1)), float(v[:, 1].mean()))
                P(f'   {c:10s} | {v[:, 0].std(ddof=1):.5f} | {v[:, 1].mean():.5f} | {v[:, 0].std(ddof=1) / v[:, 1].mean():.2f}')
    if not lag16:
        import itertools
        W8 = G['W8']; acc = {c: [] for c in STATS}
        for comb in itertools.combinations(range(8), 4):
            if 0 not in comb:
                continue                                    # each complementary split once: 35 splits
            A = [(1.0, [W8[i] for i in comb])]; B = [(1.0, [W8[i] for i in range(8) if i not in comb])]
            r = wt.read(A, B, short=False)
            for c in acc:
                if c in r:
                    acc[c].append((r[c][0], r[c][1]))
        P(' 8 wide students of W8 (the only ones with lagged predictions), all 35 complementary 4 vs 4 splits: statistic | sd over splits | mean per-feed SE | sd / SE')
        for c, v in acc.items():
            if v:
                v = np.array(v); AA[f'{c}|4w8'] = (float(v[:, 0].std(ddof=1)), float(v[:, 1].mean()))
                P(f'   {c:10s} | {v[:, 0].std(ddof=1):.5f} | {v[:, 1].mean():.5f} | {v[:, 0].std(ddof=1) / v[:, 1].mean():.2f}')
    json.dump(AA, open(CACHE / 'readout/calib2_aa.json', 'w'), indent=1)
    P('\n 16-vs-16 SE. Declared (block 4 (7)): seed sd(16 vs 16) = sd(8 vs 8) * sqrt(1/2) [= sd(4 vs 4) / 2], total = sqrt(SE_feed^2 + sd_seed^2).')
    P(' Corrected (block 7): SE_16v16 = sqrt(SE_feed^2 + max(0, sd_splits^2 - SE_feed,splits^2) * N_split / 16) (the per-feed SE already holds the feed-wise seed noise).')
    hc = next((r for k, r in R.items() if k.startswith('hc')), None)
    P(f' statistic | per-feed SE of a 16-vs-16 contrast (source) | A/A sd over splits / per-feed SE | declared total | corrected SE_16v16')
    SE16 = {}
    for c in STATS:
        sd8 = AA.get(f'{c}|8'); sd4 = AA.get(f'{c}|4') or AA.get(f'{c}|4w8')
        if not (sd8 or sd4):
            continue
        base, nsp = (sd8, 8) if sd8 else (sd4, 4)
        seed16 = base[0] * np.sqrt(nsp / 16)
        if hc is not None and c in hc and len(ps) >= 12:
            sef, src = hc[c][1], f'ref re-cools vs originals, {len(ps)} snapshot pairs'
        else:
            sef, src = base[1] * np.sqrt(nsp / 16), f'A/A {nsp} vs {nsp} splits scaled by sqrt({nsp}/16)'
        exc = max(0.0, base[0] ** 2 - base[1] ** 2) * nsp / 16
        SE16[c] = dict(se_feed=float(sef), ratio=float(base[0] / base[1]), declared=float(np.hypot(sef, seed16)), corrected=float(np.sqrt(sef ** 2 + exc)), src=src)
        P(f'   {c:10s} | {sef:.5f} ({src}) | {base[0] / base[1]:.2f} | {np.hypot(sef, seed16):.5f} | {np.sqrt(sef ** 2 + exc):.5f}')
    json.dump(SE16, open(CACHE / 'readout/calib2_se16.json', 'w'), indent=1)
(LOGS / 'calib2.txt').write_text('\n'.join(OUT) + '\n')
(LOGS / f'calib2_{time.strftime("%H%MZ", time.gmtime())}.txt').write_text('\n'.join(OUT) + '\n')
