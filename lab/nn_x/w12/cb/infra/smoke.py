"""Smoke test of every cbx data path. CPU only; light (reads caches, 1,000-row samples; one tiny CatBoost fit).
usage: CUDA_VISIBLE_DEVICES="" python smoke.py [--big]      (--big also checks the xs / prod200k sets)
Every check prints PASS / FAIL; exit code 1 on any FAIL."""
import json, os, sys, time, inspect
import numpy as np
sys.path.insert(0, "<repo>/nn_x/w12/cb/infra")

# ---- record every np.load path (check 6: the training path never opens an evaluation label file) ----
_OPENED = []
_np_load = np.load


def _spy(file, *a, **k):
    _OPENED.append(str(file))
    return _np_load(file, *a, **k)


np.load = _spy
import cbx  # noqa: E402

BIG = "--big" in sys.argv
R = cbx.ROOT
FAILS = []
rng = np.random.default_rng(0)


def check(name, ok, info=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {info}", flush=True)
    if not ok:
        FAILS.append(name)


t0 = time.time()
# ---------------------------------------------------------------- 6a. training path first (so the spy sees only that path)
tr = cbx.load("xm")
fit, sel = cbx.trainside_split(tr)
Xtr = tr.X; ytr = tr.y; t11 = tr.teacher("mean11"); toof = tr.teacher("oof")
small = np.zeros(tr.n_feeds, bool); small[rng.choice(tr.n_feeds, 1500, replace=False)] = True
pool_small = cbx.make_pool(tr, feeds=small & fit, label=tr.y)
va, te = cbx.load("val"), cbx.load("test")
Xv, Xt = va.X, te.X  # prediction matrices (label-free)
bad = [p for p in _OPENED if ("/val/" in p and "target" in p) or "/test/cand_target" in p or "holdout_y" in p
       or "/hold/y" in p or "/val/y" in p]
check("6  training path opens no val / test / holdout label file", not bad, f"{len(_OPENED)} np.load calls; offending: {bad}")
for nm in ("val", "hold", "test"):
    try:
        cbx.load(nm).y
        check(f"6  ds.y guarded for {nm}", False)
    except PermissionError:
        check(f"6  ds.y guarded for {nm}", True, "PermissionError as designed")

# ---------------------------------------------------------------- 1. feature list
sys.path.insert(0, str(R / "nn_x/cb26")); sys.path.insert(0, str(R / "src"))
from cb_member import member_features  # noqa: E402
from cbdata import load_tab  # noqa: E402
names = cbx.feature_names()
rec = json.loads((R / "nn_x/cb26/data/e3_xm_yeti.json").read_text())["features"]
check("1  177 features == cb26 member_features() == record json", names == member_features() == rec and len(names) == 177)

# ---------------------------------------------------------------- 2. shapes
EXPECT = dict(xm=(134118, 19741607), hold=(30000, 4630966), val=(14423, 1389971), test=(28595, 2904806))
if BIG:
    EXPECT.update(xs=(350905, 5361945), prod200k=(200000, None))
for nm, (nf, nr) in EXPECT.items():
    d = cbx.load(nm)
    ok = d.n_feeds == nf and (nr is None or d.n_rows == nr) and d.X.shape == (d.n_rows, 177) and d.X.dtype == np.float32
    ok &= len(d.gid) == d.n_rows and len(d.item_id) == d.n_rows and len(d.cat) == d.n_rows and int(d.M.sum()) == d.n_rows
    ok &= (np.diff(d.fidx) > 0).all() and (np.diff(d.gid) >= 0).all() and d.gid[-1] == d.n_feeds - 1
    check(f"2  shapes {nm}", ok, f"feeds {d.n_feeds} rows {d.n_rows} X {d.X.shape} M range {d.M.min()}..{d.M.max()}")

# ---------------------------------------------------------------- 3. alignment on 1,000 random rows per set
for nm in EXPECT:
    d = cbx.load(nm)
    ii = np.sort(rng.choice(d.n_rows, 1000, replace=False))
    flat = ii if d.rows is None else d.rows[ii]
    Xs = np.asarray(d.X[ii])
    col = lambda c: Xs[:, names.index(c)]
    raw = lambda c: np.asarray(_np_load(cbx.FLAT / d.split / f"cand_{c}.npy", mmap_mode="r")[flat])
    f = cbx.feeds(d.split)
    ok = np.array_equal(col("f1"), raw("f1"), equal_nan=True) and np.array_equal(col("f5"), raw("f5"), equal_nan=True)
    ok &= np.array_equal(col("category_id"), raw("category_id").astype(np.float32)) and np.array_equal(np.asarray(d.cat[ii]), raw("category_id"))
    ok &= np.array_equal(col("M"), d.M[d.gid[ii]].astype(np.float32)) and np.array_equal(col("H"), d.H[d.gid[ii]].astype(np.float32))
    ok &= np.array_equal(col("emb_missing"), (raw("emb_idx") < 0).astype(np.float32)) and np.array_equal(np.asarray(d.item_id[ii]), raw("item_id"))
    # raw row -> feed mapping: the flat fidx of the row equals the set's feed index
    ok &= np.array_equal(raw("fidx"), d.fidx[d.gid[ii]])
    check(f"3a alignment {nm}: f1, f5, category_id, M, H, emb_missing, item_id, fidx recomputed from raw (1,000 rows)", ok)
    # the same rows through the cb26 loader
    ref = load_tab("train" if d.split == "train" else d.split, names, flat)
    check(f"3b alignment {nm}: X rows == cb26 cbdata.load_tab (1,000 rows x 177)", np.array_equal(Xs, ref, equal_nan=True))
    # embedding-derived features recomputed from the cbx raw embeddings: dot_u, cos_u, item_norm (src/features.py:_emb_feats)
    v = np.asarray(d.emb_cand_i8[ii], np.float32); u = np.asarray(d.emb_feed("user"), np.float32)[d.gid[ii]]
    has = raw("emb_idx") >= 0
    vn = np.sqrt((v * v).sum(1)) + 1e-6; un = np.sqrt((u * u).sum(1)) + 1e-6; dot = (v * u).sum(1)
    e1 = np.abs(dot[has] - col("dot_u")[has]).max(); e2 = np.abs(dot[has] / (vn[has] * un[has]) - col("cos_u")[has]).max()
    e3 = np.abs(vn[has] - col("item_norm")[has]).max()
    ok = e1 < 0.51 and e2 < 2e-4 and e3 < 1e-2 and np.isnan(col("cos_u")[~has]).all() and (v[~has] == 0).all()
    check(f"3c alignment {nm}: dot_u / cos_u / item_norm recomputed from cbx raw embeddings", ok,
          f"max abs err {e1:.3g} / {e2:.3g} / {e3:.3g}; rows without embedding in sample {int((~has).sum())}")
    # history means recomputed in numpy for 100 random feeds
    tab = cbx.emb_table(d.split); hidx = _np_load(cbx.FLAT / d.split / "hist_emb_idx.npy", mmap_mode="r")
    hpos = _np_load(cbx.FLAT / d.split / "hist_hpos.npy", mmap_mode="r")
    ff = rng.choice(d.n_feeds, 100, replace=False); ok = True; okp = True
    hm, h5, h20 = d.emb_feed("hmean"), d.emb_feed("h5"), d.emb_feed("h20")
    na, n5, n20 = d.emb_feed_n("hmean"), d.emb_feed_n("h5"), d.emb_feed_n("h20")
    for k in ff:
        o = f["hoff"][d.fidx[k]]; h = d.H[k]
        e = np.asarray(hidx[o:o + h]); okp &= np.array_equal(np.asarray(hpos[o:o + h]), np.arange(h))
        val = e[e >= 0]
        E = np.asarray(tab[val], np.float64) if len(val) else np.zeros((0, 128))
        for arr, cnt, kk in ((hm, na, len(val)), (h5, n5, min(5, len(val))), (h20, n20, min(20, len(val)))):
            ref_ = E[:kk].mean(0) if kk else np.zeros(128)
            ok &= np.abs(np.asarray(arr[k]) - ref_).max() < 1e-3 and cnt[k] == kk
    check(f"3d embeddings {nm}: hmean / h5 / h20 and their counts recomputed in numpy (100 feeds)", ok)
    check(f"3e history order {nm}: hist_hpos == 0..H-1 (newest first) on 100 feeds", okp)
    ue = _np_load(cbx.FLAT / d.split / "user_emb.npy", mmap_mode="r")
    check(f"3f user embedding {nm} == user_emb.npy rows of the set's feeds", np.array_equal(np.asarray(d.emb_feed("user")[ff]), np.asarray(ue[d.fidx[ff]], np.float32)))

# ---------------------------------------------------------------- 4. teacher targets
for nm in (["xm", "hold"] + (["xs", "prod200k"] if BIG else [])):
    d = cbx.load(nm)
    ii = np.sort(rng.choice(d.n_rows, 1000, replace=False)); flat = d.rows[ii]
    a11 = np.asarray(_np_load(cbx.KD / "teacher_mean11.npy", mmap_mode="r")[flat]); aoof = np.asarray(_np_load(cbx.KD / "teacher_oof.npy", mmap_mode="r")[flat])
    ok = np.array_equal(np.asarray(d.teacher("mean11")[ii]), a11) and np.array_equal(np.asarray(d.teacher("oof")[ii]), aoof)
    check(f"4a teacher {nm}: stored value == teacher_mean11.npy / teacher_oof.npy at the row (1,000 rows)", ok)
    # fold alignment: teacher_oof row == teacher_xf<fold>.npy / 3 with fold = cbx.xfold_of(user)
    fold = d.xfold[d.gid[ii]]; okf = True
    for k in range(5):
        mk = fold == k
        if mk.any():
            okf &= np.allclose(np.asarray(_np_load(cbx.KD / f"teacher_xf{k}.npy", mmap_mode="r")[flat[mk]]) / 3.0, aoof[mk], rtol=0, atol=1e-4)
    check(f"4b teacher {nm}: teacher_oof == teacher_xf<k>/3 with k = cbx.xfold_of(user_id)", okf)
    s = d.teacher("mean11", "soft"); sums = np.add.reduceat(np.asarray(s, np.float64), d.off)
    c = d.teacher("oof", "centred"); cm = np.add.reduceat(np.asarray(c, np.float64), d.off) / d.M
    m11 = np.add.reduceat(np.asarray(d.teacher("mean11"), np.float64), d.off) / d.M
    check(f"4c teacher {nm}: softmax sums to 1 per feed; centred oof has zero feed mean; mean11 is stored centred",
          np.abs(sums - 1).max() < 1e-3 and np.abs(cm).max() < 1e-3 and np.abs(m11).max() < 1e-3,
          f"within-feed sd mean11 {float(np.asarray(d.teacher('mean11', 'centred')).std()):.3f} oof {float(np.asarray(c).std()):.3f}")

# ---------------------------------------------------------------- 5. holdout exclusion
ts = cbx.train_sets(); hold = ts["hold"]
for nm in (["xm"] + (["xs", "prod200k"] if BIG else [])):
    d = cbx.load(nm)
    check(f"5  no holdout feed in {nm}", not np.isin(d.fidx, hold).any(),
          f"shared users with the holdout (feed-level holdout, user 0 excluded): {len(np.intersect1d(d.uid_feed[d.uid_feed != 0], ts and cbx.load('hold').uid_feed))}")
check("5  holdout == nn_x/cb26 cbdata.hold_index()", np.array_equal(hold, __import__("cbdata").hold_index()[0]))
check("5  train-side split: fit and sel share no known user", len(np.intersect1d(tr.uid_feed[fit & (tr.uid_feed != 0)], tr.uid_feed[sel])) == 0,
      f"fit {int(fit.sum())} feeds / sel {int(sel.sum())} feeds ({int(tr.M[sel].sum())} rows)")
if BIG:
    c = cbx.concat(cbx.load("xm"), cbx.load("xs"))
    check("5  concat(xm, xs) = 485,023 feeds, no holdout feed", c.n_feeds == 485023 and not np.isin(c.fidx, hold).any(), repr(c))

# ---------------------------------------------------------------- 7. metric and z
from metrics import ndcg_per_feed  # noqa: E402
from stack_val import feed_rank_z  # noqa: E402
ev = cbx.Evaluator()
src_here = inspect.getsource(cbx._ndcg.py_func).split("n = len(M)")[1]
pc_txt = (R / "nn_x/w12/recheck/pool_composition/pc_common.py").read_text()
check("7a cbx._ndcg body is a verbatim copy of pc_common._ndcg", src_here.strip() in pc_txt)
for label, s in (("W16 pool", ev.w16["val"]), ("heavy ties (1 decimal)", np.round(ev.w16["val"], 1)), ("constant", np.zeros(va.n_rows, np.float32)),
                 ("random", rng.standard_normal(va.n_rows).astype(np.float32))):
    a = cbx.ndcg_fast(s, ev.yv, va.off, va.M); b = ndcg_per_feed(s, ev.yv, va.off, va.M)
    check(f"7b fast NDCG == src/metrics.ndcg_per_feed on val, {label}", np.abs(a - b).max() < 1e-12, f"mean {a.mean():.5f} max abs diff {np.abs(a - b).max():.2e}")
s = rng.standard_normal(ev.h.n_rows).astype(np.float32)
a = cbx.ndcg_fast(np.round(s, 1), ev.yh, ev.h.off, ev.h.M); b = ndcg_per_feed(np.round(s, 1), ev.yh, ev.h.off, ev.h.M)
check("7c fast NDCG == src/metrics.ndcg_per_feed on the holdout (tied random scores)", np.abs(a - b).max() < 1e-12)
x = rng.standard_normal(va.n_rows).astype(np.float32)
check("7d cbx.feed_z == src/stack_val.feed_rank_z[:, 1]", np.abs(cbx.feed_z(va.off, va.M, x) - feed_rank_z(va.off, va.M, x)[:, 1]).max() < 1e-5)
check("7e W16 train-only pool reproduces the numbers on record (0.40144 / 0.32460)",
      abs(ev.nd16["val"].mean() - 0.40144) < 6e-6 and abs(ev.nd16["hold"].mean() - 0.32460) < 6e-6,
      f"val {ev.nd16['val'].mean():.5f} holdout {ev.nd16['hold'].mean():.5f}")

# ---------------------------------------------------------------- 9. evaluator on the models of record (row-order check)
D = R / "nn_x/cb26/data"
r = ev.evaluate("record_e3_xm_yeti_GPU", _np_load(D / "e3_xm_yeti_val.npy"), _np_load(D / "e3_xm_yeti_trH.npy"), _np_load(D / "e3_xm_yeti_test.npy"),
                out_dir=None)
check("9a evaluator reproduces the record: val 0.38201, holdout 0.30424", abs(r["val"] - 0.38201) < 6e-6 and abs(r["hold"] - 0.30424) < 6e-6)
json.dump(r, open(cbx.HERE / "out/record_e3_xm_yeti_GPU.eval.json", "w"), indent=1)
LB = R / "nn_x/w12/recheck/long_bets/out"
for tag in ("gbdt_K_xm_134118", "gbdt_H_xm_134118", "gbdt_X_xm_60000"):
    if (LB / f"{tag}_val.npy").exists():
        r2 = ev.evaluate(f"record_{tag}", _np_load(LB / f"{tag}_val.npy"), _np_load(LB / f"{tag}_trH.npy"), None, out_dir=None)
        json.dump(r2, open(cbx.HERE / f"out/record_{tag}.eval.json", "w"), indent=1)

# ---------------------------------------------------------------- 11. CatBoost paths (tiny fits, 12 threads)
from catboost import CatBoostRanker, CatBoostRegressor  # noqa: E402
m = CatBoostRanker(loss_function="YetiRank", iterations=5, depth=6, thread_count=cbx.THREADS, allow_writing_files=False, verbose=0).fit(pool_small)
pv = m.predict(cbx.make_pool(va), thread_count=cbx.THREADS)
check("11a tiny YetiRank fit on make_pool(xm subset) and prediction on val", pv.shape == (va.n_rows,) and np.isfinite(pv).all(),
      f"val NDCG after 5 trees on {pool_small.num_row()} rows: {ev.ndcg('val', pv).mean():.4f}")
EMB_KINDS = ["cand", "user", "h5"]
tiny = np.zeros(tr.n_feeds, bool); tiny[np.where(small & fit)[0][:300]] = True      # 300 feeds: the embedding calcers are slow (see timing)
tsel = np.zeros(tr.n_feeds, bool); tsel[np.where(small & sel)[0][:100]] = True
for label, loss, cls, lab in (("click", "YetiRank", CatBoostRanker, tr.y), ("teacher logit", "RMSE", CatBoostRegressor, tr.teacher("mean11", "centred")),
                              ("soft teacher", "QuerySoftMax", CatBoostRanker, tr.teacher("mean11", "soft"))):
    try:
        t1 = time.time()
        pe = cbx.make_pool(tr, feeds=tiny, label=lab, embeddings=EMB_KINDS)
        t2 = time.time()
        me = cls(loss_function=loss, iterations=5, depth=6, thread_count=cbx.THREADS, allow_writing_files=False, verbose=0).fit(pe)
        t3 = time.time()
        pp = me.predict(cbx.make_pool(tr, feeds=tsel, embeddings=EMB_KINDS), thread_count=cbx.THREADS)
        imp = dict(zip(me.feature_names_, me.get_feature_importance(type="PredictionValuesChange", thread_count=cbx.THREADS)))
        check(f"11b embedding_features path ({loss}, label = {label})", np.isfinite(pp).all(),
              f"{pe.num_row()} rows, pool cols {pe.num_col()}, pool {t2 - t1:.1f}s, fit(5 trees) {t3 - t2:.1f}s with {cbx.THREADS} threads; importance: "
              + ", ".join(f"{k} {imp.get(k, float('nan')):.2f}" for k in ("emb_cand", "emb_user", "emb_h5")))
    except Exception as e:  # noqa: BLE001
        check(f"11b embedding_features path ({loss}, label = {label})", False, f"{type(e).__name__}: {str(e)[:400]}")
try:  # baseline path: boosting on top of the centred out-of-fold teacher logit
    pb = cbx.make_pool(tr, feeds=tiny, label=tr.y, baseline=tr.teacher("oof", "centred"))
    mb = CatBoostRanker(loss_function="YetiRank", iterations=5, depth=6, thread_count=cbx.THREADS, allow_writing_files=False, verbose=0).fit(pb)
    check("11c baseline path (YetiRank on top of the centred OOF teacher logit)", mb.tree_count_ == 5)
except Exception as e:  # noqa: BLE001
    check("11c baseline path", False, f"{type(e).__name__}: {str(e)[:400]}")

try:  # extra columns: item_id / user_id as CatBoost categorical features (CTRs on ids), OOF NN logit as a numeric input
    kw_x = dict(extra={"item_id": None, "user_id": None, "nn": cbx.nn_input}, extra_cat=["item_id", "user_id"])
    px = cbx.make_pool(tr, feeds=tiny, label=tr.y, **kw_x)
    mx = CatBoostRanker(loss_function="YetiRank", iterations=5, depth=6, thread_count=cbx.THREADS, allow_writing_files=False, verbose=0).fit(px)
    pxv = mx.predict(cbx.make_pool(va, **kw_x), thread_count=cbx.THREADS)
    check("11d extra columns: item_id / user_id native categorical + OOF NN logit as a feature (train: oof, val: xf15)",
          px.num_col() == 180 and np.isfinite(pxv).all(), f"val NDCG after 5 trees on 300 feeds: {ev.ndcg('val', pxv).mean():.4f}")
except Exception as e:  # noqa: BLE001
    check("11d extra columns", False, f"{type(e).__name__}: {str(e)[:400]}")

# ---------------------------------------------------------------- 12. NN inputs for the evaluation sets
ho = cbx.load("hold")
a11 = np.asarray(cbx.nn_input(ho, "mean11")); b11 = np.asarray(ho.teacher("mean11", "centred"))
c11 = float(np.corrcoef(a11, b11)[0, 1])
check("12a holdout: mean of the 11 members' holdout prediction files == teacher_mean11 on the holdout rows (centred)", c11 > 0.9999 and np.abs(a11 - b11).max() < 0.05,
      f"corr {c11:.6f}, max abs diff {np.abs(a11 - b11).max():.4f}")
for sp in ("val", "hold"):
    d = cbx.load(sp); r_ = {w: float(ev.ndcg(sp, np.asarray(cbx.nn_input(d, w))).mean()) for w in ("xf0", "xf15", "mean11")}
    check(f"12b nn_input on {sp}: NDCG of the NN inputs (sanity)", r_["mean11"] > r_["xf0"] > 0.3, str({k: round(v, 5) for k, v in r_.items()}))
check("12c nn_input on test: shape", np.asarray(cbx.nn_input(te, "xf15")).shape == (te.n_rows,))
oo = np.asarray(ho.teacher("oof", "centred")); x15 = np.asarray(cbx.nn_input(ho, "xf15"))
check("12d holdout: OOF teacher (3 fold models) vs xf15 input agree", np.corrcoef(oo, x15)[0, 1] > 0.97, f"corr {np.corrcoef(oo, x15)[0, 1]:.4f}; "
      f"OOF-teacher NDCG on the holdout {ev.ndcg('hold', oo).mean():.5f}")

print(f"\n{'ALL PASS' if not FAILS else 'FAILED: ' + '; '.join(FAILS)}  ({time.time() - t0:.0f}s)")
sys.exit(1 if FAILS else 0)
