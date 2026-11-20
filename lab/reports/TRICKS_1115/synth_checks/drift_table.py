"""SYNTHESIS check (CPU, LABEL-FREE: no label file is opened): do the all-data wide students promote cold rows differently from their held-out twins on test days?
Idea I6 of reports/RECHECK_1114/SYNTHESIS.md 4.6 / reports/TRICKS_1115/int_weak.md list B item 3. Purpose: the one unexplained val / public disagreement among current files
(all-data files sit below the level line, held-out files above it).
Pools (member-equal mean of per-feed z; no popularity-light members): H16 = w5_e10_{kd05w192,kd05t25w192}_s951-958 (held-out, holdout-selected checkpoint),
A16 = w5_fd10_* same seeds (all data, last step). Halves a = s951-954, b = s955-958. Branches: train-only (TO), fine-tuned (FT, test = 15-model mean).
Per test feed with cold and warm rows (cold = it_hist_all_luo <= 0): offset = mean z of its cold rows - mean z of its warm rows.
Per day and branch: D_AH = mean over feeds of offset(A16) - offset(H16), paired SE over feeds. Twin-to-twin null of the same statistic: D_HH = offset(H8a) - offset(H8b),
D_AA = offset(A8a) - offset(A8b) (8-member pools, so their spread is larger than that of 16-member pools: a conservative null).
RULE (written 2026-11-15T22:2xZ before the run; from int_weak.md): FLAG iff |D_AH| > 2 x max(SE of D_AH, |D_HH|, |D_AA|) on two or more test days with the same sign, in either branch.
Also printed (information): top-1 agreement of A16 with H16 against the agreement of the two halves; correlation of the per-feed z."""
import sys; sys.path.insert(0, '<repo>/src')
from common import *
from stack_val import feed_rank_z
from pathlib import Path
R = Path('<repo>'); P = R / 'preds'
ft_, Mt, ot = feed_meta('test'); ts = ft_['timestamp'].to_numpy().astype(np.int64)
zt = lambda x: feed_rank_z(ot, Mt, np.ascontiguousarray(np.asarray(x, np.float32)))[:, 1]
cold = (np.load(R / 'data/feats/test/it_hist_all_luo.npy') <= 0)
fid = np.repeat(np.arange(len(Mt)), Mt)
nc = np.bincount(fid, cold, len(Mt)); nw = Mt - nc; both = (nc > 0) & (nw > 0)
day = (ts + 3 * 3600) // 86400; days = [d for d in np.unique(day) if (day == d).sum() > 500]
def offset(z):
    return np.bincount(fid, z * cold, len(Mt)) / np.maximum(nc, 1) - np.bincount(fid, z * (~cold), len(Mt)) / np.maximum(nw, 1)
def top1(z):
    o = np.full(len(Mt), -1, np.int64); 
    order = np.lexsort((-z, fid)); first = np.r_[0, np.cumsum(Mt)[:-1]]; return order[first]
def pool(pre, seeds, suf):
    return np.mean([zt(np.load(P / f'nn_{pre}_{a}_s{s}_{suf}.npy')) for a in ('kd05w192', 'kd05t25w192') for s in seeds], 0)
out = []
def w_(s=''):
    print(s, flush=True); out.append(s)
w_(f'test feeds {len(Mt)}, with cold and warm rows {int(both.sum())}; cold row share {cold.mean():.3f}; days {[int(d) for d in days]} sizes {[int((day == d).sum()) for d in days]}')
flags = {}
for br, suf in (('TO', 'trainonly_test'), ('FT', 'n2cpv3s3_test')):
    Ha, Hb, Aa, Ab = pool('w5_e10', range(951, 955), suf), pool('w5_e10', range(955, 959), suf), pool('w5_fd10', range(951, 955), suf), pool('w5_fd10', range(955, 959), suf)
    H16, A16 = (Ha + Hb) / 2, (Aa + Ab) / 2
    oH, oA, oHa, oHb, oAa, oAb = map(offset, (H16, A16, Ha, Hb, Aa, Ab))
    tH, tA, tHa, tHb, tAa, tAb = map(top1, (H16, A16, Ha, Hb, Aa, Ab))
    w_(); w_(f'## branch {br}: corr of per-feed z A16 vs H16 {np.corrcoef(A16, H16)[0, 1]:.5f} | H8a vs H8b {np.corrcoef(Ha, Hb)[0, 1]:.5f} | A8a vs A8b {np.corrcoef(Aa, Ab)[0, 1]:.5f} | A8a vs H8a (same seeds) {np.corrcoef(Aa, Ha)[0, 1]:.5f}')
    w_('day | feeds | cold row share | offset H16 | offset A16 | D_AH +- SE | D_HH +- SE | D_AA +- SE | flag cell | top-1 agreement A16-H16 | H8a-H8b | A8a-A8b')
    sg = []
    for d in days:
        m = both & (day == d); n = int(m.sum())
        f = lambda x: (x[m].mean(), x[m].std() / np.sqrt(n))
        (dah, sah), (dhh, shh), (daa, saa) = f(oA - oH), f(oHa - oHb), f(oAa - oAb)
        fl = abs(dah) > 2 * max(sah, abs(dhh), abs(daa)); sg.append(np.sign(dah) if fl else 0)
        md = day == d
        w_(f'{int(d)} | {n} | {cold[np.isin(fid, np.where(md)[0])].mean():.3f} | {oH[m].mean():+.4f} | {oA[m].mean():+.4f} | {dah:+.5f} +- {sah:.5f} | {dhh:+.5f} +- {shh:.5f} | {daa:+.5f} +- {saa:.5f} | {"YES" if fl else "no"} | {(tA[md] == tH[md]).mean():.4f} | {(tHa[md] == tHb[md]).mean():.4f} | {(tAa[md] == tAb[md]).mean():.4f}')
    sg = np.array(sg); flags[br] = bool((sg > 0).sum() >= 2 or (sg < 0).sum() >= 2)
    m = both; w_(f'all days | {int(m.sum())} | | {oH[m].mean():+.4f} | {oA[m].mean():+.4f} | {(oA - oH)[m].mean():+.5f} +- {(oA - oH)[m].std() / np.sqrt(m.sum()):.5f} | {(oHa - oHb)[m].mean():+.5f} | {(oAa - oAb)[m].mean():+.5f}')
w_(); w_(f'RULE: FLAG in TO {flags["TO"]} | in FT {flags["FT"]}')
open(R / 'reports/TRICKS_1115/synth_checks/drift_table.txt', 'w').write('\n'.join(out) + '\n')
