"""INVENTORY D, POST HOC: joint read of the two families that the lens flags (V2 same-user earlier feeds, V8 equality structure of f columns).
Bundle P = the four PRE-DECLARED forms of the two families (chosen by family, not by result): ui_reshown, uc_ctr, f2_home, f1_tie.
Bundle B = the six best forms (chosen after their LRs were seen; in-sample choice of columns): uc_coec, click_sim, ui_reshown, f1_floor, f2_home, f1_tie.
Reads: joint LR (k dof) and the 5-fold user-grouped cross-fitted NDCG delta of the linear residual (the residual form has 10-25% of the sensitivity of
a trained input, so this is a floor, not a forecast); val also reweighted to the public / private feed mix (src/pmix.py). CPU only, read-only."""
import sys
sys.path.insert(0, '<repo>/reports/INPUTS_1116/raw_fields'); sys.path.insert(0, '<repo>/src')
from d03_lib import *
import pmix
from scipy.stats import chi2
L = []
def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); L.append(s)
V = load_val(); Hd = load_hold(); wts = pmix.weights()
X = lambda n, sp: np.load(OUT / f'x_{n}_{sp}.npy').astype(np.float64)
ld = lambda sp, n: np.load(F / sp / f'{n}.npy', mmap_mode='r')
fl = {'val': (np.asarray(ld('val', 'cand_f1')) < 0.018).astype(np.float64), 'hold': (np.asarray(ld('train', 'cand_f1')[Hd.rows]) < 0.018).astype(np.float64)}
bund = {'P (4 pre-declared forms)': ['V2a_ui_reshown', 'V2b_uc_ctr', 'V8a_f2_home', 'V8b_f1_tie'],
        'B (6 best forms)': ['C3_uc_coec', 'C2_click_sim', 'V2a_ui_reshown', 'D4_f1_floor', 'V8a_f2_home', 'V8b_f1_tie'],
        'U (same-user earlier feeds only: uc_coec, click_sim, ui_reshown)': ['C3_uc_coec', 'C2_click_sim', 'V2a_ui_reshown'],
        'E (f equality only: f1_floor, f2_home, f1_tie)': ['D4_f1_floor', 'V8a_f2_home', 'V8b_f1_tie']}
def col(n, sp): return fl[sp] if n == 'D4_f1_floor' else X(n, sp)
def run(S, sp, ctrl, names):
    C = S.ctrl[ctrl]; Xb = np.column_stack([col(n, sp) for n in names]); k = Xb.shape[1]
    th0, ll0 = fit(S.off, S.M, S.y, C); th, ll1 = fit(S.off, S.M, S.y, np.column_stack([C, Xb]), th0=np.r_[th0, np.zeros(k)]); lr = 2 * (ll1 - ll0)
    u = S.user.astype(np.int64).copy(); z0 = u == 0; u[z0] = -(np.arange(len(u))[z0] + 1); fold = (np.abs(u * 2654435761) % 1000003) % 5
    sc = np.empty(len(S.y)); sb = np.empty(len(S.y)); XX = np.column_stack([C, Xb])
    for q in range(5):
        tr = fold != q; tb, _ = fit(S.off, S.M, S.y, C, feeds=tr, th0=th0); tk, _ = fit(S.off, S.M, S.y, XX, feeds=tr, th0=th); te = np.repeat(~tr, S.M)
        sc[te] = (XX @ tk)[te]; sb[te] = (C @ tb)[te]
    d = ndcg_pf(sc, S.y, S.off, S.M) - ndcg_pf(sb, S.y, S.off, S.M)
    s = f'{sp:5s} ctrl {ctrl:3s}: joint LR({k} dof) {lr:7.2f} (p {chi2.sf(lr, k):.1e}) | coefficients ' + ' '.join(f'{n.split("_", 1)[1]} {b:+.3f}' for n, b in zip(names, th[4:])) + f' | cross-fit dNDCG {d.mean():+.5f}+-{d.std() / np.sqrt(len(d)):.5f}'
    if sp == 'val':
        for w in ('public', 'private'):
            m, se = pmix.wmean_se(d, wts[w]); s += f' | pmix {w} {m:+.5f}+-{se:.5f}'
        linked = np.isin(S.user, np.unique(pl.read_parquet(F / 'train/feeds.parquet')['user_id'].to_numpy())) & (S.user > 0)
        s += f' | feeds with a train feed of the user ({linked.mean():.3f}): {d[linked].mean():+.5f}+-{d[linked].std() / np.sqrt(linked.sum()):.5f}; other feeds: {d[~linked].mean():+.5f}+-{d[~linked].std() / np.sqrt((~linked).sum()):.5f}'
    P(s)
for bn, names in bund.items():
    P(f'--- bundle {bn} ---')
    run(V, 'val', 'TO', names); run(V, 'val', 'FIN', names); run(Hd, 'hold', 'H', names)
(OUT / 'd09_bundle.txt').write_text('\n'.join(L) + '\n')
