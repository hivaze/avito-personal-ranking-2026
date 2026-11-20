"""e7: view table. Which on-disk input / structure views are DIFFERENT from the standard view beyond seed noise?  (descriptive; no selection)
10-epoch hard-label arms against w5_e10_base (6 seeds): member level gap, cross-arm correlation of per-feed z against base (different seeds) minus the base-base correlation
(= decorrelation in excess of seed noise), and the matched 2 + 2 mix against 4 base members.  2-epoch ablation arms (seeds 23, 24; val + holdout files only) against abl_base.
Student arms against the wide students are in out/arms.json.  Output out/views.json."""
from ens_common import *
import collections, itertools
m = Meta(); st = Store(m); inv = json.load(open(OUT / 'inventory.json'))['members']; res = {}
arms = collections.defaultdict(list)
for n in sorted(inv):
    if '@last' not in n: arms[inv[n]['arm']].append(n)
B = e10('base', (951, 952, 953, 954, 955, 956)); BV = st.mat('to_val', B).astype(np.float64); BH = st.mat('hold', B).astype(np.float64); CB = np.corrcoef(BV); bb = float(CB[np.triu_indices(6, 1)].mean())
bl = float(np.mean([inv[n]['to_val']['ndcg'] for n in B])); bh = float(np.mean([inv[n]['hold']['ndcg'] for n in B]))
print(f'10-epoch hard-label base: member {bl:.5f}|{bh:.5f}, base-base corr {bb:.4f}')
seed = lambda n: re.sub(r'^.*_s(\d+).*$', r'\1', n)
for a in ('abl_e10_pop1', 'abl_e10_pop2', 'abl_e10_xdrop_pop', 'abl_e10_xdrop_pre2', 'abl_e10_xrel_more', 'abl_e10_retr32', 'abl_e10_retr32u', 'abl_e10_blocks3', 'abl_e10_latedrop', 'w5_e10_trunc', 'w5_e10_trunc2', 'w5_e10_e20',
          'w5_e10_ap1', 'w5_e10_bs4', 'w5_e10_pseudo', 'w5_e10_pseudo_xpre2', 'w5_e10_xf0', 'v23_gru_pre2s', 'v26_d192_pre2s', 'v22_pre2s_lr1', 'r6_cd_m02'):
    ns = arms.get(a, [])
    if len(ns) < 2 or not all(st.has('hold', x) for x in ns): continue
    XV = st.mat('to_val', ns).astype(np.float64); XH = st.mat('hold', ns).astype(np.float64); k = len(ns); C = np.corrcoef(np.vstack([XV, BV]))[:k, k:]
    diff = np.array([[seed(ns[i]) != seed(B[j]) for j in range(6)] for i in range(k)]); cx = float(C[diff].mean())
    # matched mix: 2 arm + 2 base (different seeds) against 4 base, averaged over draws
    dv = np.zeros(len(m.Mv)); dh = np.zeros(len(m.Mh)); nd = 0; rng = np.random.default_rng(7)
    for _ in range(30):
        ai = rng.choice(k, 2, replace=False); rest = [j for j in range(6) if seed(B[j]) not in {seed(ns[i]) for i in ai}]
        if len(rest) < 4: rest = list(range(6))
        bi = rng.choice(rest, 4, replace=False)
        dv += m.ev((XV[ai].sum(0) + BV[bi[:2]].sum(0)) / 4) - m.ev(BV[bi].mean(0)); dh += m.eh((XH[ai].sum(0) + BH[bi[:2]].sum(0)) / 4) - m.eh(BH[bi].mean(0)); nd += 1
    dv /= nd; dh /= nd
    r = dict(n=k, level_val=float(np.mean([inv[n]['to_val']['ndcg'] for n in ns]) - bl), level_hold=float(np.mean([inv[n]['hold']['ndcg'] for n in ns]) - bh), cross=cx, excess_decorr=bb - cx,
             within=float(np.corrcoef(XV)[np.triu_indices(k, 1)].mean()), mix_val=(float(dv.mean()), se(dv)), mix_hold=(float(dh.mean()), se(dh)))
    res[a] = r
    print(f'{a:24s} n {k} member gap {r["level_val"]:+.5f}|{r["level_hold"]:+.5f} within {r["within"]:.4f} cross-to-base {cx:.4f} excess decorrelation {bb - cx:+.4f} | 2 + 2 mix vs 4 base {fmt(dv)}|{fmt(dh)}', flush=True)
# 2-epoch ablation arms (val + holdout only)
def ld(n, kind):
    f = find(n, kind); return m.z(SPLIT[kind], np.load(f[0])) if f else None
b2 = {s: (ld(f'abl_base_s{s}', 'to_val'), ld(f'abl_base_s{s}', 'hold')) for s in (23, 24)}; bb2 = float(np.corrcoef(b2[23][0], b2[24][0])[0, 1]); bl2 = np.mean([m.ev(b2[s][0]).mean() for s in (23, 24)]); bh2 = np.mean([m.eh(b2[s][1]).mean() for s in (23, 24)])
eb = m.ev((b2[23][0] + b2[24][0]) / 2); ebh = m.eh((b2[23][1] + b2[24][1]) / 2)
print(f'2-epoch base: member {bl2:.5f}|{bh2:.5f}, base-base corr {bb2:.4f}')
names = sorted({f.name[3:-len('_s23_trainonly_val.npy')] for f in AB.glob('nn_abl_*_s23_trainonly_val.npy') if '@' not in f.name and 'e10' not in f.name and 'fd' not in f.name and 'smoke' not in f.name})
for a in names:
    if a == 'abl_base': continue
    x = {s: (ld(f'{a}_s{s}', 'to_val'), ld(f'{a}_s{s}', 'hold')) for s in (23, 24)}
    if any(v[0] is None or v[1] is None for v in x.values()): continue
    cx = float(np.mean([np.corrcoef(x[23][0], b2[24][0])[0, 1], np.corrcoef(x[24][0], b2[23][0])[0, 1]])); lv = np.mean([m.ev(x[s][0]).mean() for s in (23, 24)]) - bl2; lh = np.mean([m.eh(x[s][1]).mean() for s in (23, 24)]) - bh2
    dv = 0.5 * (m.ev((x[23][0] + b2[24][0]) / 2) + m.ev((x[24][0] + b2[23][0]) / 2)) - eb; dh = 0.5 * (m.eh((x[23][1] + b2[24][1]) / 2) + m.eh((x[24][1] + b2[23][1]) / 2)) - ebh
    res['2ep:' + a] = dict(level_val=float(lv), level_hold=float(lh), cross=cx, excess_decorr=bb2 - cx, mix_val=(float(dv.mean()), se(dv)), mix_hold=(float(dh.mean()), se(dh)))
    print(f'2ep {a:24s} member gap {lv:+.5f}|{lh:+.5f} cross-to-base {cx:.4f} excess decorrelation {bb2 - cx:+.4f} | 1 + 1 mix vs 2 base {fmt(dv)}|{fmt(dh)}', flush=True)
json.dump(res, open(OUT / 'views.json', 'w'), indent=0)
