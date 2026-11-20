"""Adversarial review of W25 (independent code): recompute the headline val / half / trH numbers, paired deltas,
and the ensemble-value test (G3 pool, nested user-grouped w selection, holdout) from the saved predictions.
Own NDCG and own per-feed z; src/metrics is only used as a cross-check. CPU only."""
import os, sys, json, time
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ[k] = "12"
import numpy as np, polars as pl
from sklearn.model_selection import GroupKFold

ROOT = "<repo>"
FLAT = f"{ROOT}/data/flat"; P = f"{ROOT}/preds"; OUT = f"{ROOT}/nn_x/w25/out"
sys.path.insert(0, f"{ROOT}/src")
from metrics import ndcg_per_feed as ndcg_ref  # cross-check only
import pmix

T0 = time.time()
def log(*a): print(f"[{time.time()-T0:6.1f}s]", *a, flush=True)


def ndcg_own(s, y, off, M):
    """independent per-feed NDCG, binary gain, log2 discount, pessimistic ties (negatives first on equal score), no-positive feed -> 1."""
    s = np.asarray(s, np.float64); y = np.asarray(y, np.float64)
    out = np.ones(len(M))
    for f in range(len(M)):
        o, m = off[f], M[f]
        yy = y[o:o + m]; ss = s[o:o + m]
        npos = int(yy.sum())
        if npos == 0:
            continue
        # stable sort by score desc; among equal scores, negatives (y=0) first -> pessimistic
        order = np.lexsort((yy, -ss))
        ranks = np.arange(m)
        dcg = (yy[order] / np.log2(ranks + 2.0)).sum()
        idcg = (1.0 / np.log2(np.arange(npos) + 2.0)).sum()
        out[f] = dcg / idcg
    return out


def zfeed(x, off, M):
    """per-feed standardisation (x - mean) / (std + 1e-9), population std, as src/stack_val.feed_rank_z[:,1]."""
    x = np.asarray(x, np.float64)
    fid = np.repeat(np.arange(len(M)), M)
    mu = np.bincount(fid, x, len(M)) / M
    var = np.bincount(fid, (x - mu[fid]) ** 2, len(M)) / M
    return ((x - mu[fid]) / (np.sqrt(var)[fid] + 1e-9)).astype(np.float64)


def pse(d):
    d = np.asarray(d, np.float64); return float(d.mean()), float(d.std() / np.sqrt(len(d)))


def corr_feed(za, zb, M):
    """mean over feeds of the per-feed Pearson correlation of two per-feed-standardised vectors."""
    fid = np.repeat(np.arange(len(M)), M)
    return float((np.bincount(fid, za * zb, len(M)) / M).mean())


# ---------------- val meta ----------------
fv = pl.read_parquet(f"{FLAT}/val/feeds.parquet")
Mv = fv["M"].to_numpy().astype(np.int64); ov = fv["cand_off"].to_numpy().astype(np.int64)
assert (ov == np.r_[0, np.cumsum(Mv)[:-1]]).all()
yv = np.load(f"{FLAT}/val/cand_target.npy").astype(np.float64)
uv = fv["user_id"].to_numpy().astype(np.int64)
gv = np.where(uv == 0, -1 - np.arange(len(uv)), uv)
hA, hB = [np.sort(te) for _, te in GroupKFold(n_splits=2).split(np.arange(len(Mv)), groups=gv)]
log(f"val feeds {len(Mv)} rows {Mv.sum()} halfA {len(hA)} halfB {len(hB)} user0 {(uv==0).sum()}")

# ---------------- trH meta (holdout), rebuilt independently from the documented rule ----------------
ftr = pl.read_parquet(f"{FLAT}/train/feeds.parquet", columns=["M", "cand_off", "user_id"])
Mtr = ftr["M"].to_numpy().astype(np.int64)
pr = np.load(f"{ROOT}/eda/labels_position/out/train_prandom.npz")
p = np.nan_to_num(pr["p_random"].astype(np.float32), nan=0.0); p[Mtr < 50] = 0.0
cand = np.where((p > 0.9) & (Mtr >= 50))[0]
hold = np.sort(np.random.default_rng(12345).choice(cand, 30000, replace=False))
hm = np.load(f"{P}/holdout/holdout_meta.npy")
Mh = Mtr[hold]; oh = np.r_[0, np.cumsum(Mh)[:-1]]
assert (hm[0] == Mh).all() and (hm[1] == oh).all(), "holdout meta mismatch"
yh = np.load(f"{P}/holdout/holdout_y.npy").astype(np.float64)
# cross-check holdout labels against the flat train labels (same rows)
ytr = np.load(f"{FLAT}/train/cand_target.npy", mmap_mode="r")
otr = ftr["cand_off"].to_numpy().astype(np.int64)
rows_h = np.concatenate([np.arange(otr[i], otr[i] + Mtr[i]) for i in hold])
assert (np.asarray(ytr[rows_h]).astype(np.float64) == yh).all(), "holdout_y != flat train labels"
log(f"trH feeds {len(Mh)} rows {Mh.sum()} (meta and labels reproduced from the documented rule)")

# ---------------- 1. single-model numbers ----------------
arms = ["A0_cb26repro", "A1_base_xm", "A1_base_xm_s1", "A2_all_xm", "A2c_cb26", "abl_loo_G1", "abl_loo_G3", "abl_add_G1"]
PV, PH = {}, {}
print("\n== single arms (own NDCG | ref NDCG max abs diff) ==")
print(f"{'arm':16s} {'val':>8s} {'halfA':>8s} {'halfB':>8s} {'trH':>8s} {'M<50':>7s} {'M>=50':>7s} | json val / trH")
for t in arms:
    sv = np.load(f"{OUT}/{t}_val.npy").astype(np.float64); sh = np.load(f"{OUT}/{t}_trH.npy").astype(np.float64)
    assert len(sv) == Mv.sum() and len(sh) == Mh.sum()
    pv = ndcg_own(sv, yv, ov, Mv); ph = ndcg_own(sh, yh, oh, Mh)
    dv = np.abs(pv - ndcg_ref(sv, yv, ov, Mv)).max(); dh = np.abs(ph - ndcg_ref(sh, yh, oh, Mh)).max()
    j = json.load(open(f"{OUT}/{t}.json"))
    PV[t], PH[t] = pv, ph
    print(f"{t:16s} {pv.mean():8.5f} {pv[hA].mean():8.5f} {pv[hB].mean():8.5f} {ph.mean():8.5f} {pv[Mv<50].mean():7.4f} {pv[Mv>=50].mean():7.4f} | "
          f"{j['val_full']:.5f} / {j['trH']:.5f}  (ref-diff {dv:.1e}/{dh:.1e}, iter {j['best_iter']})")

print("\n== paired deltas (first - second), val and trH, with SE ==")
pairs = [("A2c_cb26", "A0_cb26repro", "A2c-A0: W25 features on cb26 rows"), ("A2_all_xm", "A1_base_xm", "A2-A1: W25 features on XM rows"),
         ("A1_base_xm_s1", "A1_base_xm", "seed1-seed0 NOISE"), ("A1_base_xm", "A0_cb26repro", "A1-A0 rows+lr"),
         ("A2_all_xm", "abl_loo_G1", "G1 on top"), ("A2_all_xm", "abl_loo_G3", "G3 on top"), ("abl_add_G1", "A1_base_xm", "G1 alone")]
wpriv = pmix.weights()["private"]
for a, b, lab in pairs:
    dv = PV[a] - PV[b]; dh = PH[a] - PH[b]
    m, s = pse(dv); hm_, hs = pse(dh); pm, ps = pmix.wmean_se(dv, wpriv)
    print(f"{lab:36s} val {m:+.5f} +- {s:.5f} (z {m/s:+.2f}) | halfB-only {pse(dv[hB])[0]:+.5f} +- {pse(dv[hB])[1]:.5f} | trH {hm_:+.5f} +- {hs:.5f} (z {hm_/hs:+.2f}) | pmix {pm:+.5f} +- {ps:.5f}")

# ---------------- 2. ensemble value vs G3 ----------------
FAM = {"F10": [f"v23_gru_pre2s_s{i}" for i in range(501, 507)], "F7x": [f"r6_cd_m02_r{i}" for i in range(101, 108)],
       "F7x70": [f"r6_cd_m02_p70k_r{i}" for i in range(101, 108)]}
members = sum(FAM.values(), [])
ft = np.load(f"{P}/ens_G3_ft_val.npy").astype(np.float64); to = np.load(f"{P}/ens_G3_to_val.npy").astype(np.float64)
# rebuild the 'to' branch from the member files to make sure the pool arrays are what they claim
to_re = np.mean([zfeed(np.load(f"{P}/nn_{m}_trainonly_val.npy"), ov, Mv) for m in members], 0)
ft_re = np.mean([zfeed(np.load(f"{P}/nn_{m}_n2cpv3_val_oof.npy"), ov, Mv) for m in members], 0)
log(f"G3 val pool rebuild: |to - to_re| max {np.abs(to - to_re).max():.2e}, |ft - ft_re| max {np.abs(ft - ft_re).max():.2e}")
pool_v = zfeed(0.5 * zfeed(ft, ov, Mv) + 0.5 * zfeed(to, ov, Mv), ov, Mv)
base_v = ndcg_own(pool_v, yv, ov, Mv)
pool_h = np.mean([zfeed(np.load(f"{P}/holdout/nn_{m}_holdout.npy"), oh, Mh) for m in members], 0)
base_h = ndcg_own(pool_h, yh, oh, Mh)
print(f"\n== G3 pool: val {base_v.mean():.5f} (claimed 0.40032), trH {base_h.mean():.5f} (claimed 0.32379) ==")
ws = (0.05, 0.1, 0.2, 0.3)
folds5 = [np.sort(te) for _, te in GroupKFold(n_splits=5).split(np.arange(len(Mv)), groups=gv)]
fidv = np.repeat(np.arange(len(Mv)), Mv); fidh = np.repeat(np.arange(len(Mh)), Mh)
for t in ["A2c_cb26", "A2_all_xm", "A1_base_xm", "A0_cb26repro"]:
    zc = zfeed(np.load(f"{OUT}/{t}_val.npy"), ov, Mv); zch = zfeed(np.load(f"{OUT}/{t}_trH.npy"), oh, Mh)
    rho = corr_feed(zc, pool_v, Mv)
    rho_h = corr_feed(zch, pool_h, Mh)
    bar = base_v.mean() - 0.049 * (1 - rho)
    dv = {w: ndcg_own(pool_v + w * zc, yv, ov, Mv) - base_v for w in ws}
    dh = {w: ndcg_own(pool_h + w * zch, yh, oh, Mh) - base_h for w in ws}
    print(f"\n[{t}] member val {PV[t].mean():.5f}; rho val {rho:.3f} trH {rho_h:.3f}; law bar val >= {bar:.4f} -> {'MET' if PV[t].mean() >= bar else 'NOT met'}")
    for w in ws:
        m, s = pse(dv[w]); hm_, hs = pse(dh[w])
        print(f"   w={w:.2f}: val {m:+.5f} +- {s:.5f} (z {m/s:+.2f}) | trH {hm_:+.5f} +- {hs:.5f}")
    oof = np.zeros(len(Mv)); chosen = []
    for te in folds5:
        tr = np.setdiff1d(np.arange(len(Mv)), te)
        wb = max(ws, key=lambda w: dv[w][tr].mean()); chosen.append(wb); oof[te] = dv[wb][te]
    m, s = pse(oof); pm, ps = pmix.wmean_se(oof, wpriv)
    wall = max(ws, key=lambda w: dv[w].mean())
    print(f"   nested w {chosen}; nested val delta {m:+.5f} +- {s:.5f} (z {m/s:+.2f}); holdout at w={wall}: {pse(dh[wall])[0]:+.5f} +- {pse(dh[wall])[1]:.5f}; pmix {pm:+.5f} +- {ps:.5f}")
    # also: a free least-squares weight instead of the grid (what is the most the pool can take from this member?)
    # per-feed-z linear stacker, user-grouped OOF on val (2 inputs, no intercept needed on z scale)
    best_oof = np.zeros(len(Mv))
    grid = np.linspace(-0.1, 0.5, 25)
    pers = {w: ndcg_own(pool_v + w * zc, yv, ov, Mv) for w in grid}
    for te in folds5:
        tr = np.setdiff1d(np.arange(len(Mv)), te)
        wb = max(grid, key=lambda w: pers[w][tr].mean()); best_oof[te] = pers[wb][te] - base_v[te]
    print(f"   fine-grid nested (w in [-0.1,0.5]): {pse(best_oof)[0]:+.5f} +- {pse(best_oof)[1]:.5f}")

# ---------------- 3. final file check ----------------
for s in ("val", "test"):
    a = np.load(f"{OUT}/w25_{s}.npy"); b = np.load(f"{OUT}/A2c_cb26_{s}.npy")
    print(f"w25_{s}.npy == A2c_cb26_{s}.npy: {np.array_equal(a, b)} (n {len(a)})")
ft_, Mt = None, pl.read_parquet(f"{FLAT}/test/feeds.parquet")["M"].to_numpy().sum()
print(f"test rows {Mt}; w25_test rows {len(np.load(f'{OUT}/w25_test.npy'))}")
log("done")
