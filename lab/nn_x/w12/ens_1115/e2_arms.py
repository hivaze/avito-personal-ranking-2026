"""e2 part B: how much a SECOND ARM adds at equal pool size, against same-recipe wide students (train-only branch; val and in-period holdout).
For every arm A with >= 4 members (h = 4; also h = 8 when A has >= 8 members):
  level     : E[pool of h members of A] - E[pool of h wide students]                        (random subsets, per-feed means, paired)
  corr      : mean pairwise correlation of per-feed z: within A | A x wide (different seed / root) | A x wide (same seed number or same root) | within wide
  replace   : pool(A_h + W_h) - pool(W'_h + W_h): half of a 2h wide pool replaced by the arm, seeds / roots of A_h and W_h disjoint, W'_h drawn from the other wide students
              (re-cool arms: W'_h = the ORIGINAL runs of the roots of A_h, so the contrast is the arm's recipe at equal independence)
  add       : pool(W16 + all of A) - pool(W16)
Part C: two-level law for cooldown branches of one plateau snapshot (branches orig / ref / ref2 = same recipe, 16 roots):  N(R, B) = a - c1 / R - c2 / (R B).
Output out/arms.json."""
from ens_common import *
import itertools, math, time, collections
R = int(sys.argv[1]) if len(sys.argv) > 1 else 60
ONLY = sys.argv[2] if len(sys.argv) > 2 else None; OUTF = OUT / (f'arms_{sys.argv[3]}.json' if ONLY else 'arms.json')   # optional: regex filter on arm names + output tag (late arms)
m = Meta(); st = Store(m); rng = np.random.default_rng(1116); t0 = time.time(); inv = json.load(open(OUT / 'inventory.json'))['members']
import pmix; WTS = pmix.weights()
seed_of = lambda n: re.sub(r'^.*_(s|r)(\d+)(@last)?$', r'\2', n)
def root_of(n):   # plateau snapshot identity of a wide student or of its re-cool
    k = re.match(r'^w5_([kl])cd_.*_s(\d+)', n)
    if k: return k.group(1) + k.group(2)
    k = re.match(r'^w5_e10_(kd05w192|kd05t25w192)_s(\d+)$', n)
    return ('k' if k.group(1) == 'kd05w192' else 'l') + k.group(2) if k else None
WV = st.mat('to_val', W16).astype(np.float64); WH = st.mat('hold', W16).astype(np.float64); wroot = [root_of(n) for n in W16]; wseed = [seed_of(n) for n in W16]
ev2 = lambda v, h: (m.ev(v), m.eh(h))

def level(XV, XH, h):
    N = len(XV); cs = [np.array(c) for c in itertools.combinations(range(N), h)] if math.comb(N, h) <= R else [rng.choice(N, h, replace=False) for _ in range(R)]
    a = np.zeros(len(m.Mv)); b = np.zeros(len(m.Mh))
    for c in cs:
        x, y = ev2(XV[c].mean(0), XH[c].mean(0)); a += x; b += y
    return a / len(cs), b / len(cs)
LW = {h: level(WV, WH, h) for h in (4, 8)}
print('wide level', {h: (round(float(v[0].mean()), 5), round(float(v[1].mean()), 5)) for h, v in LW.items()}, flush=True)
CW = np.corrcoef(WV); cw_within = float(CW[np.triu_indices(16, 1)].mean())

arms = collections.defaultdict(list)
for n in sorted(inv):
    if st.has('hold', n): arms[inv[n]['arm']].append(n)
# re-cool arms: K and L lineages are one arm of 16 (decision checkpoints @last; selected checkpoints as a second row)
for a in ('ref', 'ref2', 'hz', 'th', 'xd30', 'hz2', 'hzt', 'hznFS', 'hz2nF'):
    for suf in ('@last', ''):
        ms = rc(a, suf)
        if all(st.has('hold', x) for x in ms): arms[f'RC:{a}{suf}'] = ms
        for st_ in 'kl': arms.pop(f'w5_{st_}cd_{a}{suf}', None)
skip = ('w5_e10_kd05w192', 'w5_e10_kd05t25w192')
res = {}
for a, ns in arms.items():
    k = len(ns)
    if k < 4 or a in skip or a.endswith('np') or (ONLY and not re.search(ONLY, a)): continue
    XV = st.mat('to_val', ns).astype(np.float64); XH = st.mat('hold', ns).astype(np.float64); recool = a.startswith('RC:')
    aseed = [seed_of(n) for n in ns]; aroot = [root_of(n) for n in ns]
    C = np.corrcoef(np.vstack([XV, WV])); cx = C[:k, k:]
    same = np.array([[(aroot[i] == wroot[j]) if recool else (aseed[i] == wseed[j]) for j in range(16)] for i in range(k)])
    r = dict(n=k, member_val=float(np.mean([inv[n]['to_val']['ndcg'] for n in ns])), member_hold=float(np.mean([inv[n]['hold']['ndcg'] for n in ns])),
             corr_within=float(C[:k, :k][np.triu_indices(k, 1)].mean()), corr_cross_diff=float(cx[~same].mean()), corr_cross_same=(float(cx[same].mean()) if same.any() else None), corr_wide=cw_within)
    for h in ((4, 8) if k >= 8 else (4,)):
        lv, lh = level(XV, XH, h); dv = lv - LW[h][0]; dh = lh - LW[h][1]
        q = dict(level_val=(float(dv.mean()), se(dv)), level_hold=(float(dh.mean()), se(dh)), pool_val=float(lv.mean()), pool_hold=float(lh.mean()))
        # replace design
        dV = np.zeros(len(m.Mv)); dH = np.zeros(len(m.Mh)); draws = []; nd = 0
        for _ in range(R):
            ai = rng.choice(k, h, replace=False)
            if recool:
                ar = {aroot[i] for i in ai}
                if len(ar) < h: continue
                rest = [j for j in range(16) if wroot[j] not in ar]; wi = rng.choice(rest, h, replace=False) if len(rest) >= h else None
                wj = np.array([j for j in range(16) if wroot[j] in ar])
            else:
                asd = {aseed[i] for i in ai}; rest = [j for j in range(16) if wseed[j] not in asd]
                if len(rest) < h: rest = list(range(16))     # 8-seed arms at h = 8: seed numbers cannot be disjoint
                wi = rng.choice(rest, h, replace=False); oth = [j for j in range(16) if j not in set(wi)]; wj = rng.choice(oth, h, replace=False)
            if wi is None: continue
            mv, mh = ev2((XV[ai].sum(0) + WV[wi].sum(0)) / (2 * h), (XH[ai].sum(0) + WH[wi].sum(0)) / (2 * h))
            bv, bh = ev2((WV[wj].sum(0) + WV[wi].sum(0)) / (2 * h), (WH[wj].sum(0) + WH[wi].sum(0)) / (2 * h))
            dV += mv - bv; dH += mh - bh; draws.append((float((mv - bv).mean()), float((mh - bh).mean()))); nd += 1
        dV /= nd; dH /= nd; dr = np.array(draws)
        pp, sp = pmix.wmean_se(dV, WTS['public']); pr, sr = pmix.wmean_se(dV, WTS['private'])
        q.update(replace_val=(float(dV.mean()), se(dV)), replace_hold=(float(dH.mean()), se(dH)), replace_pub=(float(pp), float(sp)), replace_priv=(float(pr), float(sr)),
                 draws=nd, draw_sd_val=float(dr[:, 0].std()), draw_sd_hold=float(dr[:, 1].std()))
        r[f'h{h}'] = q
    av, ah = ev2((XV.sum(0) + WV.sum(0)) / (k + 16), (XH.sum(0) + WH.sum(0)) / (k + 16)); bv, bh = ev2(WV.mean(0), WH.mean(0))
    r['add_val'] = (float((av - bv).mean()), se(av - bv)); r['add_hold'] = (float((ah - bh).mean()), se(ah - bh))
    res[a] = r; q = r['h8'] if 'h8' in r else r['h4']; hh = 8 if 'h8' in r else 4
    print(f'{a:28s} n {k:2d} member {r["member_val"]:.5f}|{r["member_hold"]:.5f} corr within {r["corr_within"]:.4f} cross {r["corr_cross_diff"]:.4f} same {r["corr_cross_same"] if r["corr_cross_same"] is None else round(r["corr_cross_same"], 4)} '
          f'| h{hh} level {q["level_val"][0]:+.5f}+-{q["level_val"][1]:.5f}|{q["level_hold"][0]:+.5f}+-{q["level_hold"][1]:.5f} replace {q["replace_val"][0]:+.5f}+-{q["replace_val"][1]:.5f}|{q["replace_hold"][0]:+.5f}+-{q["replace_hold"][1]:.5f} '
          f'pmix {q["replace_pub"][0]:+.5f}/{q["replace_priv"][0]:+.5f} | add to W16 {r["add_val"][0]:+.5f}+-{r["add_val"][1]:.5f}|{r["add_hold"][0]:+.5f}+-{r["add_hold"][1]:.5f} [{time.time() - t0:.0f}s]', flush=True)
    json.dump(dict(wide_within=cw_within, arms=res), open(OUTF, 'w'), indent=0)

if ONLY: raise SystemExit(0)
# ---- Part C: roots x branches law (same recipe: original, ref@last, ref2@last)
BR = {'orig': W16, 'ref': rc('ref'), 'ref2': rc('ref2'), 'hz': rc('hz'), 'th': rc('th'), 'hzt': rc('hzt')}
roots = wroot
def by_root(names): d = {root_of(n): n for n in names}; return [d[r_] for r_ in roots]
BV = {b: st.mat('to_val', by_root(ns)).astype(np.float64) for b, ns in BR.items()}; BH = {b: st.mat('hold', by_root(ns)).astype(np.float64) for b, ns in BR.items()}
def two_level(branches, tag):
    nb_ = len(branches); pts = []
    for Rn in (1, 2, 3, 4, 6, 8, 12, 16):
        for B in range(1, nb_ + 1):
            a = np.zeros(len(m.Mv)); b = np.zeros(len(m.Mh)); K = 1 if (Rn == 16 and B == nb_) else min(R, 40)
            for _ in range(K):
                rs = rng.choice(16, Rn, replace=False); sv = 0; sh = 0
                for r_ in rs:
                    for bi in rng.choice(nb_, B, replace=False): sv = sv + BV[branches[bi]][r_]; sh = sh + BH[branches[bi]][r_]
                x, y = ev2(sv / (Rn * B), sh / (Rn * B)); a += x; b += y
            pts.append((Rn, B, a / K, b / K))
    A = np.array([[1.0, -1.0 / Rn, -1.0 / (Rn * B)] for Rn, B, _, _ in pts]); G = np.linalg.pinv(A); out = {}
    for nm, i in (('val', 2), ('hold', 3)):
        Y = np.stack([p[i] for p in pts]); pf = G @ Y; co = pf.mean(1); s_ = pf.std(1) / np.sqrt(pf.shape[1]); fitv = A @ co; obs = Y.mean(1)
        f = lambda Rn, B: co[0] - co[1] / Rn - co[2] / (Rn * B)
        # per-feed SE of a forecast contrast: linear in pf
        def con(R1, B1, R2, B2):
            v = np.array([0.0, -(1 / R2 - 1 / R1), -(1 / (R2 * B2) - 1 / (R1 * B1))]); d = v @ pf; return float(d.mean()), se(d)
        out[nm] = dict(a=float(co[0]), c_root=float(co[1]), c_branch=float(co[2]), se=[float(x) for x in s_], max_resid=float(np.abs(fitv - obs).max()),
                       obs={f'{Rn}x{B}': float(o) for (Rn, B, _, _), o in zip(pts, obs)},
                       forecast={k_: con(*v_) for k_, v_ in {'16x1->16x2': (16, 1, 16, 2), '16x1->16x3': (16, 1, 16, 3), '16x1->16x6': (16, 1, 16, 6), '16x1->16xinf': (16, 1, 16, 1e9), '16x1->32x1': (16, 1, 32, 1),
                                                           '16x1->32x3': (16, 1, 32, 3), '16x1->64x1': (16, 1, 64, 1), '16x1->infx1': (16, 1, 1e9, 1)}.items()})
        print(f'TWO-LEVEL {tag} {nm}: a {co[0]:.5f} c_root {co[1]:.5f}+-{s_[1]:.5f} c_branch {co[2]:.5f}+-{s_[2]:.5f} max resid {np.abs(fitv - obs).max():.5f} | observed 16x1 {out[nm]["obs"]["16x1"]:.5f} 16x{nb_} {out[nm]["obs"][f"16x{nb_}"]:.5f} | '
              + ' '.join(f'{k_} {v_[0]:+.5f}+-{v_[1]:.5f}' for k_, v_ in out[nm]['forecast'].items()), flush=True)
    return out
tl = {'same3': two_level(['orig', 'ref', 'ref2'], 'orig/ref/ref2'), 'six': two_level(['orig', 'ref', 'ref2', 'hz', 'th', 'hzt'], 'orig/ref/ref2/hz/th/hzt')}
json.dump(dict(wide_within=cw_within, arms=res, two_level=tl), open(OUT / 'arms.json', 'w'), indent=0)
print(f'done {time.time() - t0:.0f}s')
