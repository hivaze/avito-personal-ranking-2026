"""INVENTORY B, step 13: robustness of the short-list columns to a richer null model, and where the walk's information sits.
(1) null = pool + the ranker's own inputs + NON-LINEAR forms of the collaborative inputs: rank forms of all 8 popularity / co-visitation inputs,
    squares of their z forms, the non-zero flags and frozen quantile transforms of seq_cv_invd20 / seq_cv_norm (AUTOFE recipe columns), and
    the pairwise products z(seq_cv_invd20) x z(it_hist_all_luo), z(seq_cv_invd20) x z(seq_max_all).
(2) the RP3 column split by whether the windowed co-visitation input is zero on the row (rp3 x 1[seq_cv_invd20 = 0], rp3 x 1[seq_cv_invd20 > 0]).
Val (train-only pool) and holdout.  Output: out/robust.txt.  CPU only, read-only."""
import sys
sys.path.insert(0, "<repo>/reports/INPUTS_1116/residual_rows")
sys.argv = [sys.argv[0], "val", "TO16"]
from lens import *
import cand_specs as cs, condlib
SP = {f"{g}:{n}": s for g, n, s in cs.specs()}
POP8 = "seq_cv_invd20 seq_cv_pop seq_cv_norm it_hist_all_luo it_hist_et1_luo it_hist_et2_luo it_hist_et3_luo it_hist_et4_luo".split()
COLS = ["ccb:c09_p3a", "ccb:c09_rp3b05", "cbB_c:kn_cos1", "cbB_c:kn_cos20", "codes:ex_X3_s1s2_lf", "codes:ex_n_s2_s1s2_lf", "ccb:c06_theta_l3", "timegap:tg_gap_first", "store:tte_it_coec"]
out = open(OUT / "robust.txt", "w")
def P(s): print(s, flush=True); out.write(s + "\n")
for sp in ("val", "hold"):
    D = load_split(sp, "TO16"); condlib.sp = sp; nm, cl = condlib.control_block(D); N0 = condlib.Null(D, nm, cl)
    F = {n: forms(cs.load(SP["store:" + n][sp], D), D) for n in POP8 + ["seq_max_all"]}
    extra = []; en = []
    for n in POP8:
        extra += [F[n]["rk"], (F[n]["z"] ** 2).astype(np.float32)]; en += [n + ":rk", n + ":z2"]
    for n in ("afe_nz_seq_cv_invd20", "afe_qn_seq_cv_invd20", "afe_qn_seq_cv_norm"):
        a = np.load(ROOT / "nn_x/w12/autofe/feats" / ("val" if sp == "val" else "train") / f"{n}.npy", mmap_mode="r")
        x = np.asarray(a if D["rows"] is None else a[D["rows"]], np.float64); extra.append(forms(x, D)["z"]); en.append(n)
    extra += [(F["seq_cv_invd20"]["z"] * F["it_hist_all_luo"]["z"]).astype(np.float32), (F["seq_cv_invd20"]["z"] * F["seq_max_all"]["z"]).astype(np.float32)]; en += ["cv x luo", "cv x smax"]
    keep_n, keep_c = list(nm), list(cl)
    for n_, c_ in zip(en, extra):   # drop exact duplicates of columns already in the block
        if all(abs(np.corrcoef(c_, c2)[0, 1]) < 0.999 for c2 in keep_c[-40:]): keep_n.append(n_); keep_c.append(c_)
    N1 = condlib.Null(D, keep_n, keep_c)
    P(f"[{sp}] null A = pool + {len(nm)} own-input columns; null B = A + {len(keep_n) - len(nm)} non-linear forms of the collaborative inputs (LR of the added forms given A: {2 * (N1.r['L'] - N0.r['L']):.1f})")
    P(f"{'column':26s} cz given A (z / rank)   cz given B (z / rank)")
    for k in COLS:
        f = forms(cs.load(SP[k][sp], D), D)
        P(f"{k:26s} {N0.score(f['z'])['cz']:+6.2f} / {N0.score(f['rk'])['cz']:+6.2f}        {N1.score(f['z'])['cz']:+6.2f} / {N1.score(f['rk'])['cz']:+6.2f}")
    x = cs.load(SP["ccb:c09_rp3b05"][sp], D); cv = np.nan_to_num(cs.load(SP["store:seq_cv_invd20"][sp], D)); z = forms(x, D)["z"]
    P(f"   rows: rp3 > 0 {float((x > 0).mean()):.3f}; seq_cv_invd20 > 0 {float((cv > 0).mean()):.3f}; rp3 > 0 and seq_cv_invd20 = 0 {float(((x > 0) & (cv == 0)).mean()):.3f}; rp3 = 0 and seq_cv_invd20 > 0 {float(((x == 0) & (cv > 0)).mean()):.3f}")
    a = (z * (cv == 0)).astype(np.float32); b = (z * (cv > 0)).astype(np.float32)
    P(f"   rp3b05 z on rows without a windowed co-visitation hit: cz given A {N0.score(a)['cz']:+.2f}, given B {N1.score(a)['cz']:+.2f}; on rows with a hit: {N0.score(b)['cz']:+.2f}, {N1.score(b)['cz']:+.2f}")
