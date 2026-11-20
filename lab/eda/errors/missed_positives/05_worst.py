"""Print 30 concrete worst feeds (leak-free, every positive in the bottom 40% of the v5 ranking, 30 <= M <= 300).
Sampled at random (seed 0) from all such feeds so that the sample is not dominated by the largest feeds.
"""
import sys
sys.path.insert(0, "<repo>/src")
from common import *

D = Path("<repo>/eda/errors/missed_positives")


def main():
    fr = np.load(D / "frame.npz")
    y, best, M, off, leakf, fid, r, q = (fr[k] for k in ["y", "best", "M", "off", "leakf", "fid", "r", "q"])
    f = pl.read_parquet(FLAT / "val/feeds.parquet")
    H = f["H"].to_numpy(); hoff = f["hist_off"].to_numpy()
    d = FLAT / "val"
    ccat = np.load(d / "cand_category_id.npy"); hcat = np.load(d / "hist_category_id.npy")
    het = np.load(d / "hist_event_type.npy"); hsrc = np.load(d / "hist_event_source.npy"); hpos = np.load(d / "hist_hpos.npy")
    F = {n: np.load(FEATS / "val" / f"{n}.npy") for n in ["f1", "f2", "f3", "f4", "f5", "f1_rk", "f3_rk", "maxsim32", "cat_min_hpos",
                                                         "cat_cnt_all", "cos_u", "it_hist_all_luo", "ctx_nd_c90", "pre_v3_fe",
                                                         "clu_ccos1024", "trd_skde_s6h"]}
    worst_q = np.full(len(M), np.inf)
    np.minimum.at(worst_q, fid[y > 0], q[y > 0])  # best (lowest) relative rank of a positive
    cand = np.where((~leakf) & (worst_q >= 0.6) & (M >= 30) & (M <= 300))[0]
    print(f"eligible feeds (all positives in bottom 40%): {len(cand)} of {(~leakf).sum()}")
    rng = np.random.default_rng(0)
    pick = np.sort(rng.choice(cand, 30, replace=False))
    stats = []
    for fi in pick:
        o, m, ho, h = off[fi], M[fi], hoff[fi], H[fi]
        idx = np.arange(o, o + m)
        hc = hcat[ho:ho + h]; hp = hpos[ho:ho + h]; he = het[ho:ho + h]; hs = hsrc[ho:ho + h]
        order = np.argsort(hp)
        last10 = " ".join(f"{hc[j]}:{he[j]}/{hs[j]}" for j in order[:10])
        vals, cnts = np.unique(hc, return_counts=True)
        top = sorted(zip(cnts, vals), reverse=True)[:6]
        fc_vals, fc_cnts = np.unique(ccat[idx], return_counts=True)
        ftop = sorted(zip(fc_cnts, fc_vals), reverse=True)[:6]
        print(f"\n### feed {fi} user {int(f['user_id'][int(fi)])} M={m} H={h} hist cats(count): {', '.join(f'{c}({n})' for n, c in top)}")
        print(f"    last10 hist (cat:etype/src): {last10}")
        print(f"    feed cats(count): {', '.join(f'{c}({n})' for n, c in ftop)}")
        show = list(idx[y[idx] > 0]) + list(idx[np.argsort(r[idx])[:3]])
        for i in show:
            tag = "POS" if y[i] > 0 else "top"
            print(f"    {tag} rank {r[i]:3d}/{m} cat {ccat[i]:2d} f1 {F['f1'][i]:.3f}(rk {F['f1_rk'][i]:.2f}) f3 {F['f3'][i]:.3f}(rk {F['f3_rk'][i]:.2f}) "
                  f"f4 {F['f4'][i]:.2f} f5 {F['f5'][i]:.2f} maxsim {F['maxsim32'][i]:.2f} cos_u {F['cos_u'][i]:.2f} catmin {F['cat_min_hpos'][i]:.0f} "
                  f"catcnt {F['cat_cnt_all'][i]:.0f} pop {F['it_hist_all_luo'][i]:.1f} nd90 {F['ctx_nd_c90'][i]:.0f} pre {F['pre_v3_fe'][i]:.1f} "
                  f"ccos {F['clu_ccos1024'][i]:.2f} skde {F['trd_skde_s6h'][i]:.1f}")
            if y[i] > 0:
                stats.append((F['cat_min_hpos'][i], F['cat_cnt_all'][i], F['f3_rk'][i], F['maxsim32'][i]))
    s = np.array(stats)
    print(f"\nsummary over the {len(s)} positives shown: cat never in history {np.mean(s[:, 1] == 0):.2f}; cat_min_hpos>=32 {np.mean(s[:, 0] >= 32):.2f}; "
          f"median f3_rk {np.median(s[:, 2]):.2f}; median maxsim32 {np.median(s[:, 3]):.2f}")


if __name__ == "__main__":
    main()
