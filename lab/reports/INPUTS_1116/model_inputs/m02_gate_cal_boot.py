"""INVENTORY A, measurement M2 (CPU only, read-only on predictions / features): the residual gate against REAL positive
controls, with a feed-level bootstrap of the recovered share.

Re-implements nn_x/w12/recheck/duplicates/d02_gate_cal.py (the positive control of COVERAGE_MAP l.323) on top of the same
helper module (nn_x/w12/recheck/forensics_skeptic/hx.py, imported read-only), keeps the per-feed deltas and adds:
  * a bootstrap (2000 draws over feeds, numerator and denominator drawn together) of  recovered / true ;
  * for the linear lens the optimism of the post hoc "best of k columns" pick (the max is re-taken inside every draw);
  * three more controls that the record does not have:
      pre2 block on the HOLDOUT (the record has it on val only, 3 of 5 columns; here all 5 from nn/feat_cache),
      ONE ADDED column, xp_imp_c50  (twin = abl_e10_pop1, base = abl_e10_xdrop_pop, seeds 951-954),
      ONE ADDED label statistic, xp_ctr_c (twin = abl_e10_pop2, same base).
All members are hard-label 10-epoch d128 members (the only era with stored with / without twins).
Usage: m02_gate_cal_boot.py val|hold      -> out/m02_<which>.txt, out/m02_<which>.npz
"""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ[k] = "12"
sys.dont_write_bytecode = True
sys.path.insert(0, "<repo>/nn_x/w12/recheck/forensics_skeptic")
from hx import *  # noqa: E402,F401  (load_val, load_hold, nested_lin, resid_cv, ctx_cols, zfeed, ndcg_per_feed, fmt, se, R, W, FEATS)

which = sys.argv[1]
OUTD = R / "reports/INPUTS_1116/model_inputs/out"
A = R / "nn_x/abl/preds"
LOG = open(OUTD / f"m02_{which}.txt", "w")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + "\n")
    LOG.flush()


POP8 = "seq_cv_invd20,seq_cv_pop,seq_cv_norm,it_hist_all_luo,it_hist_et1_luo,it_hist_et2_luo,it_hist_et3_luo,it_hist_et4_luo".split(",")
# name -> (base arm dir/prefix, twin arm dir/prefix, seeds, columns offered to the gate, kind)
BLK = {
    "pop8_drop": (("abl", "abl_e10_xdrop_pop"), ("w5", "w5_e10_base"), (951, 952, 953, 954), POP8, "8 popularity / co-visitation inputs removed (the record's control)"),
    "pre2_drop": (("abl", "abl_e10_xdrop_pre2"), ("w5", "w5_e10_base"), (951, 952), "pre2_fe_z,pre2_new_z,pre2_fe_s,pre2_new_s,pre2_hmax32".split(","), "encoder scalars removed"),
    "xp_imp_add": (("abl", "abl_e10_xdrop_pop"), ("abl", "abl_e10_pop1"), (951, 952, 953, 954), ["xp_imp_c50"], "ONE column added to the popularity-free view"),
    "xp_ctr_add": (("abl", "abl_e10_xdrop_pop"), ("abl", "abl_e10_pop2"), (951, 952, 953, 954), ["xp_ctr_c"], "ONE label statistic added to the popularity-free view"),
}
D = load_val() if which == "val" else load_hold()
y, M, off = D["y"], D["M"], D["off"]
n = len(M)
say(f"== M2 gate calibration on {which}: {n} feeds, {len(y)} rows")


def pfile(h, arm, s, last=False):
    d = A if h == "abl" else W
    if which == "val":
        return d / f"nn_{arm}_s{s}{'@last' if last else ''}_trainonly_val.npy"
    return d / f"holdout/nn_{arm}_s{s}{'@last' if last else ''}_holdout.npy"


def pool(files):
    return zfeed(np.mean([zfeed(np.load(f), off, M) for f in files], 0), off, M)


def col(nm):
    sp = "val" if which == "val" else "train"
    for d in (FEATS / sp, R / "nn/feat_cache" / sp, R / "data/feats_new" / sp):
        p = d / f"{nm}.npy"
        if p.exists():
            x = np.load(p, mmap_mode="r")
            x = np.asarray(x if D["rows"] is None else x[D["rows"]], np.float32)
            return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), str(p.relative_to(R))
    raise FileNotFoundError(nm)


def boot_ratio(num, den, B=2000, seed=7):
    """num [k, n] or [n], den [n]; returns point ratio(s), 2.5 / 97.5 percentiles, and the same for max over k (re-picked per draw)."""
    rng = np.random.default_rng(seed)
    num = np.atleast_2d(num)
    rr = np.empty((B, num.shape[0]))
    for b in range(B):
        ix = rng.integers(0, len(den), len(den))
        rr[b] = num[:, ix].mean(1) / den[ix].mean()
    pt = num.mean(1) / den.mean()
    lo, hi = np.percentile(rr, [2.5, 97.5], 0)
    mx = rr.max(1)
    return pt, lo, hi, (pt.max(), *np.percentile(mx, [2.5, 97.5]), float(np.median(mx)))


save = {}
ctx = ctx_cols.__call__  # resolved below per block (the base changes)
for name, ((hb, ab), (ht, at), seeds, cols, kind) in BLK.items():
    say(f"\n--- block {name}: {kind}; base {ab} / twin {at}, seeds {seeds}")
    base = pool([pfile(hb, ab, s) for s in seeds])
    twin = pool([pfile(ht, at, s) for s in seeds])
    e0, e1 = ndcg_per_feed(base, y, off, M), ndcg_per_feed(twin, y, off, M)
    true = e1 - e0
    say(f"   base pool {e0.mean():.5f}, twin pool {e1.mean():.5f}; TRUE trained-input value {fmt(true)} (z {true.mean() / se(true):+.1f}); selected checkpoints")
    try:  # the same contrast on the last-step checkpoints (no checkpoint selection on the holdout)
        bl = pool([pfile(hb, ab, s, True) for s in seeds]); tl = pool([pfile(ht, at, s, True) for s in seeds])
        tr_last = ndcg_per_feed(tl, y, off, M) - ndcg_per_feed(bl, y, off, M)
        say(f"   last-step checkpoints: TRUE {fmt(tr_last)}")
    except FileNotFoundError as e:
        say(f"   last-step files missing ({e})")
    per_seed = [float((ndcg_per_feed(zfeed(np.load(pfile(ht, at, s)), off, M), y, off, M) - ndcg_per_feed(zfeed(np.load(pfile(hb, ab, s)), off, M), y, off, M)).mean()) for s in seeds]
    say("   member-level deltas per seed:", [f"{v:+.5f}" for v in per_seed], f"mean {np.mean(per_seed):+.5f}")
    D["base"] = base
    X, names = [], []
    for c in cols:
        x, src = col(c)
        X.append(x); names.append(c)
        X.append(zfeed(x, off, M).astype(np.float32)); names.append(c + "_fz")
    # (i) linear nested, one column at a time (per-feed z of the column; weight picked on the other user half)
    dl = []
    for nm, x in zip(names, X):
        if not nm.endswith("_fz"):
            continue
        d, wc = nested_lin(D, x.astype(np.float64))
        dl.append(d)
        say(f"   linear nested {nm:22s} {fmt(d)}  w {wc}  = {d.mean() / true.mean():+.2f} of true")
    dl = np.stack(dl)
    pt, lo, hi, (mpt, mlo, mhi, mmed) = boot_ratio(dl, true)
    for i, c in enumerate(cols):
        say(f"   LINEAR share {c:20s} {pt[i]:+.3f}  [95% {lo[i]:+.3f}, {hi[i]:+.3f}]")
    say(f"   LINEAR best-of-{len(cols)} (post hoc): point {mpt:+.3f}; the max re-picked inside each bootstrap draw: median {mmed:+.3f}, 95% [{mlo:+.3f}, {mhi:+.3f}]")
    # (ii) monotone GBDT residual: context + block columns (raw and per-feed z)
    cx = ctx_cols(D)
    Xm = np.stack([cx[k] for k in cx] + X, 1).astype(np.float32)
    dg, det, _ = resid_cv(D, Xm, list(cx) + names, name, log=say)
    gpt, glo, ghi, _ = boot_ratio(dg, true)
    say(f"   GBDT residual {fmt(dg)} (z {dg.mean() / se(dg):+.2f}); share {gpt[0]:+.3f}  [95% {glo[0]:+.3f}, {ghi[0]:+.3f}]")
    say(f"   detectability: linear best z {max(d.mean() / se(d) for d in dl):+.2f}; GBDT z {dg.mean() / se(dg):+.2f}; true z {true.mean() / se(true):+.2f}")
    save[f"{name}_true"] = true.astype(np.float32); save[f"{name}_lin"] = dl.astype(np.float32); save[f"{name}_gbdt"] = dg.astype(np.float32)
np.savez_compressed(OUTD / f"m02_{which}.npz", **save)
say("\nsaved per-feed deltas ->", f"out/m02_{which}.npz")
