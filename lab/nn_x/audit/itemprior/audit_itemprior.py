"""Item-level exploration-label prior as a residual on G3 (see plan.txt, pre-declared).

Prior per candidate row: x = log(p_i/p0), p_i = (s_i + a*p0)/(n_i + a), x = 0 if n_i = 0, with s_i/n_i the clicks /
impressions of the item on SOURCE rows of OTHER users (leave-user-out; user 0 = anonymous bucket, treated as distinct
users). Score: z(G3) + beta * x. (a, beta) chosen nested over the FT folds (GroupKFold(5) by user, user-0 singletons).
CPU only. Writes out.txt / summary.json / arrays into this directory; never touches preds/ or subs/ (the lead script
writes submission files only if the pre-declared acceptance rule passes).
"""
import os, sys, json, time
os.environ.setdefault("NUMBA_NUM_THREADS", "12")
sys.path.insert(0, "<repo>/src")
import numpy as np
from sklearn.model_selection import GroupKFold
from common import feed_meta, FLAT, FEATS
from metrics import ndcg_per_feed
from stack_val import feed_rank_z

OUT = "<repo>/nn_x/audit/itemprior"
P = "<repo>/preds"
T0 = time.time()
LOG = open(f"{OUT}/out.txt", "w")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


def se(d):
    return float(np.std(d, ddof=1) / np.sqrt(len(d)))


def fmt(d):
    return f"{np.mean(d):+.5f} +- {se(d):.5f} (n={len(d)})"


# ----------------------------------------------------------------------------------------------------------- data
fv, Mv, ov = feed_meta("val"); Mv = Mv.astype(np.int64)
ftst, Mt, ot = feed_meta("test"); Mt = Mt.astype(np.int64)
ftr, Mtr, otr = feed_meta("train"); Mtr = Mtr.astype(np.int64)
y = np.load(FLAT / "val/cand_target.npy").astype(np.float64)
uid_v = fv["user_id"].to_numpy().astype(np.int64); ts_v = fv["timestamp"].to_numpy().astype(np.int64)
uid_t = ftst["user_id"].to_numpy().astype(np.int64); ts_t = ftst["timestamp"].to_numpy().astype(np.int64)
feed_v = np.repeat(np.arange(len(Mv)), Mv); feed_t = np.repeat(np.arange(len(Mt)), Mt)
urow_v = uid_v[feed_v]; urow_t = uid_t[feed_t]
tue_f = ((ts_v + 10800) // 86400) < 20720; tue_r = tue_f[feed_v]
pub_f = ts_t < 1790497920; pub_r = pub_f[feed_t]
it_v = np.load(FLAT / "val/cand_item_id.npy"); it_t = np.load(FLAT / "test/cand_item_id.npy")
cat_v = np.load(FLAT / "val/cand_category_id.npy").astype(np.int64)
pr = np.load("<repo>/eda/labels_position/out/train_prandom.npz")
assert (pr["fidx"] == np.arange(len(Mtr))).all()
sel_tr = pr["p_random"] > 0.5  # exploration-like train feeds (NaN -> False)
rows_tr = np.repeat(sel_tr, Mtr)
it_tr = np.asarray(np.load(FLAT / "train/cand_item_id.npy", mmap_mode="r"))[rows_tr]
y_tr = np.asarray(np.load(FLAT / "train/cand_target.npy", mmap_mode="r"))[rows_tr].astype(np.float64)
u_tr = ftr["user_id"].to_numpy().astype(np.int64)[np.repeat(np.arange(len(Mtr)), Mtr)[rows_tr]]
del rows_tr
say(f"data loaded {time.time()-T0:.0f}s: val rows {len(y)}, test rows {len(it_t)}, train-expl feeds {sel_tr.sum()} rows {len(it_tr)}")

items = np.unique(np.concatenate([it_v, it_t, it_tr]))
K = len(items)
ix_v = np.searchsorted(items, it_v); ix_t = np.searchsorted(items, it_t); ix_tr = np.searchsorted(items, it_tr)
say(f"distinct items (val+test+train-expl) {K}")

# FT folds (nn/finetune.py --u0_singleton)
grp = np.where(uid_v == 0, -1 - np.arange(len(uid_v)), uid_v)
fold_f = np.empty(len(Mv), np.int64)
for k, (_, ho) in enumerate(GroupKFold(n_splits=5).split(np.arange(len(Mv)), groups=grp)):
    fold_f[ho] = k
fold_r = fold_f[feed_v]
say("fold sizes (feeds):", np.bincount(fold_f).tolist())

base = feed_rank_z(ov, Mv, np.load(f"{P}/ens_G3_final_val.npy").astype(np.float32))[:, 1].astype(np.float64)
nd_base = ndcg_per_feed(base, y, ov, Mv)
say(f"G3 val NDCG {nd_base.mean():.5f}")

A_GRID = [5.0, 20.0, 50.0]
B_GRID = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]


# -------------------------------------------------------------------------------------------------------- helpers
def luo_counts(src_i, src_u, src_y, tgt_i, tgt_u, w=None):
    """(n, s) per target row from source rows, minus the contribution of the target row's own user (u > 0)."""
    w = np.ones(len(src_i)) if w is None else w
    N = np.bincount(src_i, weights=w, minlength=K); S = np.bincount(src_i, weights=src_y * w, minlength=K)
    n = N[tgt_i].copy(); s = S[tgt_i].copy()
    m = src_u > 0
    key = src_u[m] * K + src_i[m]
    uk, inv = np.unique(key, return_inverse=True)
    Nu = np.bincount(inv, weights=w[m]); Su = np.bincount(inv, weights=(src_y * w)[m])
    tm = tgt_u > 0
    tkey = tgt_u[tm] * K + tgt_i[tm]
    pos = np.clip(np.searchsorted(uk, tkey), 0, len(uk) - 1)
    hit = uk[pos] == tkey
    n[tm] -= np.where(hit, Nu[pos], 0.0); s[tm] -= np.where(hit, Su[pos], 0.0)
    p0 = S.sum() / N.sum()
    return n, s, p0


def prior_x(n, s, p0, a):
    p = (s + a * p0) / (n + a)
    return np.where(n > 0, np.log(p / p0), 0.0)


def nd_sub(scores, fmask):
    rows = np.repeat(fmask, Mv)
    Ms = Mv[fmask]; offs = np.concatenate([[0], np.cumsum(Ms)[:-1]])
    return ndcg_per_feed(scores[rows], y[rows], offs, Ms)


def counts_val(src_rows):
    """Item counts from val rows in src_rows (bool over rows); no explicit LUO needed when src/target folds differ."""
    N = np.bincount(ix_v[src_rows], minlength=K).astype(np.float64)
    S = np.bincount(ix_v[src_rows], weights=y[src_rows], minlength=K)
    return N, S, S.sum() / N.sum()


def strict_nested(make_x, grid, target_f=None, label=""):
    """make_x(src_folds:set, setting) -> x over all val rows (prior from val rows of those folds, plus any fold-free part).
    Strict nesting: for outer k, inner fold j uses the prior from folds not in {j, k}; the setting maximising the mean
    inner delta is applied to fold k with the prior from the 4 folds != k. Returns per-feed nested delta (NaN outside
    target_f) and the chosen settings."""
    target_f = np.ones(len(Mv), bool) if target_f is None else target_f
    nested = np.full(len(Mv), np.nan); chosen = []
    full_tab = {}
    for k in range(5):
        tot = {g: 0.0 for g in grid}; cnt = 0
        for j in range(5):
            if j == k:
                continue
            src = set(range(5)) - {j, k}
            fm = (fold_f == j) & target_f
            if fm.sum() == 0:
                continue
            xs = make_x(src, None)  # returns dict setting -> x  (settings share the counts)
            for g in grid:
                d = nd_sub(base + xs[g], fm) - nd_base[fm]
                tot[g] += d.sum()
            cnt += fm.sum()
        best = max(grid, key=lambda g: tot[g] / cnt)
        chosen.append(best)
        src = set(range(5)) - {k}
        fm = (fold_f == k) & target_f
        xs = make_x(src, None)
        nested[fm] = nd_sub(base + xs[best], fm) - nd_base[fm]
        for g in grid:  # non-nested OOF table (for the full-val argmax used on test)
            full_tab.setdefault(g, np.full(len(Mv), np.nan))[fm] = nd_sub(base + xs[g], fm) - nd_base[fm]
        say(f"  [{label}] outer fold {k}: chosen {best}, inner mean {tot[best]/cnt:+.5f}, outer {fmt(nested[fm])}  {time.time()-T0:.0f}s")
    return nested, chosen, full_tab


def report(nested, label, extra_slices=None):
    m = ~np.isnan(nested)
    d = nested[m]
    say(f"{label}: nested delta {fmt(d)}  z={np.mean(d)/max(se(d),1e-12):.2f}")
    sl = {"M<50": Mv < 50, "M>=50": Mv >= 50, "Tue": tue_f, "Thu": ~tue_f}
    if extra_slices:
        sl.update(extra_slices)
    worst = 0.0
    for nm, sm in sl.items():
        dd = nested[m & sm]
        if len(dd) < 50:
            continue
        zz = np.mean(dd) / max(se(dd), 1e-12); worst = min(worst, zz)
        say(f"   {nm:8s} {fmt(dd)}  z={zz:.2f}")
    return float(np.mean(d)), se(d), worst


def full_argmax(full_tab):
    best = max(full_tab, key=lambda g: np.nanmean(full_tab[g]))
    d = full_tab[best][~np.isnan(full_tab[best])]
    return best, float(np.mean(d)), se(d)


# ---------------------------------------------------------------------------------------- (V) val-label item prior
say("\n=== (V) item prior from VAL labels, OOF by FT folds, log-ratio form ===")
_cache = {}


def counts_folds(src):
    key = tuple(sorted(src))
    if key not in _cache:
        _cache[key] = counts_val(np.isin(fold_r, list(key)))
    return _cache[key]


def make_x_V(src, _):
    N, S, p0 = counts_folds(src)
    n, s = N[ix_v], S[ix_v]
    out = {}
    for a in A_GRID:
        xa = prior_x(n, s, p0, a)
        for b in B_GRID:
            out[(a, b)] = b * xa
    return out


GRID_AB = [(a, b) for a in A_GRID for b in B_GRID]
nested_V, chosen_V, tab_V = strict_nested(make_x_V, GRID_AB, label="V")
say("chosen (a, beta) per outer fold:", chosen_V)
mV, seV, worstV = report(nested_V, "V log-ratio")
bestV, dV_full, seV_full = full_argmax(tab_V)
say(f"full-val OOF table argmax (a, beta) = {bestV}: {dV_full:+.5f} +- {seV_full:.5f} (non-nested, used for the test setting)")
say("full-val OOF table (a, beta -> delta):")
for a in A_GRID:
    say("   a=%4.0f  " % a + "  ".join(f"b{b:.2f}:{np.nanmean(tab_V[(a, b)]):+.5f}" for b in B_GRID))
# OOF prior at the full-val best setting (for correlation / files)
N4 = {k: counts_folds(set(range(5)) - {k}) for k in range(5)}
n_oof = np.zeros(len(y)); s_oof = np.zeros(len(y)); p0_oof = np.zeros(len(y))
for k in range(5):
    r = fold_r == k
    N, S, p0 = N4[k]; n_oof[r] = N[ix_v[r]]; s_oof[r] = S[ix_v[r]]; p0_oof[r] = p0
x_oof = prior_x(n_oof, s_oof, p0_oof, bestV[0])
cov_val_rows = float((n_oof > 0).mean()); cov_val_rows3 = float((n_oof >= 3).mean())
cov_val_feeds = float(np.bincount(feed_v, weights=(n_oof > 0), minlength=len(Mv)).astype(bool).mean())
cov_val_pos = float((n_oof[y > 0] > 0).mean())
say(f"val OOF coverage: rows>=1 {cov_val_rows:.4f}, rows>=3 {cov_val_rows3:.4f}, feeds with any covered row {cov_val_feeds:.4f}, positives covered {cov_val_pos:.4f}")
say(f"x_oof on covered rows: mean {x_oof[n_oof>0].mean():+.3f} sd {x_oof[n_oof>0].std():.3f}; Tue-row coverage {(n_oof[tue_r]>0).mean():.4f} Thu-row coverage {(n_oof[~tue_r]>0).mean():.4f}")
np.save(f"{OUT}/x_val_oof.npy", x_oof.astype(np.float32)); np.save(f"{OUT}/n_val_oof.npy", n_oof.astype(np.float32))

# secondary: within-feed z form
say("\n--- (V) secondary: within-feed z(x) form ---")


def make_x_Vz(src, _):
    N, S, p0 = counts_folds(src)
    n, s = N[ix_v], S[ix_v]
    out = {}
    for a in A_GRID:
        xa = feed_rank_z(ov, Mv, prior_x(n, s, p0, a).astype(np.float32))[:, 1].astype(np.float64)
        for b in B_GRID:
            out[(a, b)] = b * xa
    return out


nested_Vz, chosen_Vz, tab_Vz = strict_nested(make_x_Vz, GRID_AB, label="Vz")
say("chosen:", chosen_Vz); report(nested_Vz, "V z-form")

# ------------------------------------------------------------------------------------------- (C) category prior
say("\n=== (C) category-level prior from VAL labels (OOF by FT folds) ===")
cats = np.unique(cat_v); cx = np.searchsorted(cats, cat_v)


def make_x_C(src, _):
    r = np.isin(fold_r, list(src))
    N = np.bincount(cx[r], minlength=len(cats)).astype(np.float64); S = np.bincount(cx[r], weights=y[r], minlength=len(cats))
    p0 = S.sum() / N.sum(); n, s = N[cx], S[cx]
    out = {}
    for a in A_GRID:
        xa = prior_x(n, s, p0, a)
        for b in B_GRID:
            out[(a, b)] = b * xa
    return out


nested_C, chosen_C, tab_C = strict_nested(make_x_C, GRID_AB, label="C")
say("chosen:", chosen_C); report(nested_C, "C category")

# ------------------------------------------------------------------------------- time-forward Tue -> Thu (and back)
say("\n=== time-forward: Tue-only source (other users) -> Thu feeds, nested over FT folds restricted to Thu ===")


def make_x_fwd(src, _):
    r = np.isin(fold_r, list(src)) & tue_r
    N, S, p0 = counts_val(r)
    n, s = N[ix_v], S[ix_v]
    out = {}
    for a in A_GRID:
        xa = prior_x(n, s, p0, a)
        for b in B_GRID:
            out[(a, b)] = b * xa
    return out


nested_F, chosen_F, tab_F = strict_nested(make_x_fwd, GRID_AB, target_f=~tue_f, label="Tue->Thu")
say("chosen:", chosen_F); mF, seF, _ = report(nested_F, "Tue->Thu time-forward")
# fixed-setting time-forward at the full-val best (a, beta): no selection at all
xf = {}
for k in range(5):
    r = (fold_r != k) & tue_r
    N, S, p0 = counts_val(r); rk = (fold_r == k) & ~tue_r
    xf[k] = (N[ix_v], S[ix_v], p0)
d_fixed = np.full(len(Mv), np.nan)
for k in range(5):
    N_, S_, p0 = xf[k]; fm = (fold_f == k) & ~tue_f
    d_fixed[fm] = nd_sub(base + bestV[1] * prior_x(N_, S_, p0, bestV[0]), fm) - nd_base[fm]
say(f"Tue->Thu at the fixed full-val setting {bestV}: {fmt(d_fixed[~np.isnan(d_fixed)])}")
# coverage of Thu rows by Tue items
r = tue_r; N, S, p0 = counts_val(r); nn_ = N[ix_v]
say(f"Thu rows covered by Tue items (not LUO, descriptive): {(nn_[~tue_r]>0).mean():.4f}")


def make_x_bwd(src, _):
    r = np.isin(fold_r, list(src)) & ~tue_r
    N, S, p0 = counts_val(r)
    n, s = N[ix_v], S[ix_v]
    out = {}
    for a in A_GRID:
        xa = prior_x(n, s, p0, a)
        for b in B_GRID:
            out[(a, b)] = b * xa
    return out


nested_B, chosen_B, _ = strict_nested(make_x_bwd, GRID_AB, target_f=tue_f, label="Thu->Tue")
say("chosen:", chosen_B); report(nested_B, "Thu->Tue (backward, information)")

# ------------------------------------------------------------------------------------- (T) train-exploration prior
say("\n=== (T) item prior from TRAIN exploration-like feeds (p_random > 0.5), leave-user-out ===")
n_T, s_T, p0_T = luo_counts(ix_tr, u_tr, y_tr, ix_v, urow_v)
say(f"train-expl source p0 {p0_T:.4f}; val coverage rows>=1 {(n_T>0).mean():.4f} rows>=3 {(n_T>=3).mean():.4f}, "
    f"feeds any {np.bincount(feed_v, weights=(n_T>0), minlength=len(Mv)).astype(bool).mean():.4f}, positives {(n_T[y>0]>0).mean():.4f}  {time.time()-T0:.0f}s")
xT = {a: prior_x(n_T, s_T, p0_T, a) for a in A_GRID}


def make_x_T(src, _):
    return {(a, b): b * xT[a] for a in A_GRID for b in B_GRID}


nested_T, chosen_T, tab_T = strict_nested(make_x_T, GRID_AB, label="T")
say("chosen:", chosen_T); mT, seT, worstT = report(nested_T, "T train-expl")
bestT, dT_full, seT_full = full_argmax(tab_T)
say(f"full-val argmax (a, beta) = {bestT}: {dT_full:+.5f} +- {seT_full:.5f}")
for a in A_GRID:
    say("   a=%4.0f  " % a + "  ".join(f"b{b:.2f}:{np.nanmean(tab_T[(a, b)]):+.5f}" for b in B_GRID))
both = (n_T > 0) & (n_oof > 0)
say(f"rows covered by both V and T: {both.mean():.4f}; corr(x_V, x_T) on them: {np.corrcoef(x_oof[both], xT[bestT[0]][both])[0,1]:.3f}")

# ------------------------------------------------------------------------- (U) union: two-feature form V + T
say("\n=== (U) union: z(G3) + b1 * x_V + b2 * x_T  (a_V, a_T fixed at the single-source full-val argmax) ===")
B2 = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]
aV, aT = bestV[0], bestT[0]


def make_x_U(src, _):
    N, S, p0 = counts_folds(src)
    xv = prior_x(N[ix_v], S[ix_v], p0, aV); xt = xT[aT]
    return {(b1, b2): b1 * xv + b2 * xt for b1 in B2 for b2 in B2}


GRID_U = [(b1, b2) for b1 in B2 for b2 in B2]
nested_U, chosen_U, tab_U = strict_nested(make_x_U, GRID_U, label="U")
say("chosen (b1, b2):", chosen_U); mU, seU, worstU = report(nested_U, "U two-feature")
bestU, dU_full, seU_full = full_argmax(tab_U)
say(f"full-val argmax (b1, b2) = {bestU}: {dU_full:+.5f} +- {seU_full:.5f}")
# count-union with weight w on train rows (single x), information
say("--- count-union (s_V + w s_T, n_V + w n_T), pooled p0 ---")
for w in (0.3, 1.0):
    def make_x_Uc(src, _, w=w):
        N, S, p0v = counts_folds(src)
        n = N[ix_v] + w * n_T; s = S[ix_v] + w * s_T
        p0 = (S.sum() + w * y_tr.sum()) / (N.sum() + w * len(y_tr))  # pooled source click rate
        return {(a, b): b * prior_x(n, s, p0, a) for a in A_GRID for b in B_GRID}
    nested_Uc, chosen_Uc, _ = strict_nested(make_x_Uc, GRID_AB, label=f"Uc w={w}")
    say("chosen:", chosen_Uc); report(nested_Uc, f"count-union w={w}")

# ------------------------------------------------------------- duplication check vs the NN's popularity inputs
say("\n=== duplication check: x_V (OOF, a=%g) vs NN popularity inputs (data/feats/val) ===" % aV)
pop_names = ["it_hist_all_luo", "it_hist_all_luo_z", "it_hist_all_luo_rk", "it_hist_et1_luo", "it_tr_ctr", "it_tr_coec",
             "it_cand_cnt_split_rk", "it_cand_cnt_train", "seq_cv_pop"]
feats = {}
for nm in pop_names:
    fp = FEATS / "val" / f"{nm}.npy"
    if fp.exists():
        feats[nm] = np.load(fp).astype(np.float64)
cov = n_oof > 0


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return np.corrcoef(ra, rb)[0, 1]


for nm, v in feats.items():
    v = np.nan_to_num(v)
    tr = np.log1p(np.maximum(v, 0)) if nm in ("it_hist_all_luo", "it_hist_et1_luo", "it_cand_cnt_train", "seq_cv_pop") else v
    say(f"   {nm:22s} covered rows: pearson(x_V, f) {np.corrcoef(x_oof[cov], tr[cov])[0,1]:+.3f}  spearman {spearman(x_oof[cov], v[cov]):+.3f}"
        f" | all rows pearson {np.corrcoef(x_oof, tr)[0,1]:+.3f}")
# popularity-only residual, and x_V on top of it (simple nested: OOF x fixed, choice over the other 4 folds)
zpop = feed_rank_z(ov, Mv, np.log1p(np.maximum(feats["it_hist_all_luo"], 0)).astype(np.float32))[:, 1].astype(np.float64)
G_GRID = [0.0, 0.05, 0.1, 0.2, 0.3]
BB = [0.0] + B_GRID
tab2 = {}
for g in G_GRID:
    for b in BB:
        tab2[(g, b)] = ndcg_per_feed(base + g * zpop + b * x_oof, y, ov, Mv) - nd_base
say(f"   full-val table (gamma on z(log1p it_hist_all_luo), beta on x_V): best {max(tab2, key=lambda k: tab2[k].mean())}")
for g in G_GRID:
    say("   gamma=%.2f  " % g + "  ".join(f"b{b:.2f}:{tab2[(g, b)].mean():+.5f}" for b in BB))


def simple_nested(tab, keys):
    out = np.zeros(len(Mv)); ch = []
    for k in range(5):
        fm = fold_f == k
        best = max(keys, key=lambda kk: tab[kk][~fm].mean()); ch.append(best)
        out[fm] = tab[best][fm]
    return out, ch


d_pop, ch = simple_nested(tab2, [(g, 0.0) for g in G_GRID]); say(f"   popularity term only (nested): {fmt(d_pop)} chosen {ch}")
d_x, ch = simple_nested(tab2, [(0.0, b) for b in BB]); say(f"   x_V only (simple nested):        {fmt(d_x)} chosen {ch}")
d_both, ch = simple_nested(tab2, list(tab2)); say(f"   both (nested):                   {fmt(d_both)} chosen {ch}")
say(f"   increment of x_V given the popularity term: {fmt(d_both - d_pop)}")
# OLS residualisation of x_V on popularity features (covered rows)
cols = [np.ones(cov.sum())]
for nm in ("it_hist_all_luo", "it_hist_et1_luo", "it_cand_cnt_train"):
    if nm in feats:
        cols.append(np.log1p(np.maximum(np.nan_to_num(feats[nm]), 0))[cov])
if "it_tr_ctr" in feats:
    c = np.clip(np.nan_to_num(feats["it_tr_ctr"]), 1e-4, 1 - 1e-4)[cov]; cols.append(np.log(c / (1 - c)))
if "it_tr_coec" in feats:
    cols.append(np.log(np.maximum(np.nan_to_num(feats["it_tr_coec"]), 1e-4))[cov])
Xd = np.stack(cols, 1)
coef, *_ = np.linalg.lstsq(Xd, x_oof[cov], rcond=None)
fit = Xd @ coef; r2 = 1 - ((x_oof[cov] - fit) ** 2).sum() / ((x_oof[cov] - x_oof[cov].mean()) ** 2).sum()
x_res = np.zeros(len(y)); x_res[cov] = x_oof[cov] - fit
say(f"   OLS of x_V on popularity features (covered rows): R2 {r2:.3f}")
tab3 = {b: ndcg_per_feed(base + b * x_res, y, ov, Mv) - nd_base for b in BB}
d_res, ch = simple_nested(tab3, BB); say(f"   residualised x_V (simple nested): {fmt(d_res)} chosen {ch}")

# ---------------------------------------------------------------------------------------- test coverage, projection
say("\n=== test coverage (leave-user-out) and projection ===")
n_Vt, s_Vt, p0_Vt = luo_counts(ix_v, urow_v, y, ix_t, urow_t)
n_Tt, s_Tt, p0_Tt = luo_counts(ix_tr, u_tr, y_tr, ix_t, urow_t)
cover = {}
for nm, n_ in (("V(all val)", n_Vt), ("T(train-expl)", n_Tt), ("U(V or T)", np.maximum(n_Vt, n_Tt))):
    for part, pm in (("public", pub_r), ("private", ~pub_r)):
        c1 = float((n_[pm] > 0).mean()); c3 = float((n_[pm] >= 3).mean())
        feed_any = np.bincount(feed_t, weights=(n_ > 0), minlength=len(Mt)) > 0
        fa = float(feed_any[pub_f if part == "public" else ~pub_f].mean())
        cover[(nm, part)] = (c1, c3, fa)
        say(f"   {nm:14s} {part:7s}: rows>=1 {c1:.4f}  rows>=3 {c3:.4f}  feeds with any covered row {fa:.4f}")
# val-side coverage for the ratios
covT_val = float((n_T > 0).mean())
say(f"   val OOF coverage V rows>=1 {cov_val_rows:.4f}; val T coverage rows>=1 {covT_val:.4f}")
proj = {}
for nm, m_, cv in (("V", mV, cov_val_rows), ("T", mT, covT_val), ("U", mU, float(((n_oof > 0) | (n_T > 0)).mean()))):
    for part in ("public", "private"):
        key = {"V": "V(all val)", "T": "T(train-expl)", "U": "U(V or T)"}[nm]
        ratio = cover[(key, part)][0] / cv
        proj[(nm, part)] = m_ * ratio
        say(f"   projected test gain {nm} {part}: nested val {m_:+.5f} x coverage ratio {ratio:.3f} = {m_*ratio:+.5f}")
say("   ASSUMPTION: gain proportional to the covered share of candidate rows, same per-covered-row effect on test as on val "
    "(ignores colder test items and the longer val->test gap).")

# ------------------------------------------------------------------------------------------------------- verdict
say("\n=== pre-declared acceptance (plan.txt) ===")
tf_ok = mF >= 0
say(f"V: nested {mV:+.5f} +- {seV:.5f} (z {mV/seV:.2f}) | worst slice z {worstV:.2f} | Tue->Thu {mF:+.5f} +- {seF:.5f} | projected private {proj[('V','private')]:+.5f}")
acc_V = (mV > 2 * seV) and (worstV >= -2) and tf_ok and (proj[("V", "private")] >= 0.0003)
say(f"V verdict: {'ACCEPT' if acc_V else 'REJECT'}")
say(f"T: nested {mT:+.5f} +- {seT:.5f} (z {mT/seT:.2f}) | worst slice z {worstT:.2f} | projected private {proj[('T','private')]:+.5f}")
say(f"U: nested {mU:+.5f} +- {seU:.5f} (z {mU/seU:.2f}) | worst slice z {worstU:.2f} | projected private {proj[('U','private')]:+.5f}")
acc_U = (mU > 2 * seU) and (worstU >= -2) and tf_ok and (proj[("U", "private")] >= 0.0003)
say(f"U verdict (same rule, time-forward taken from V): {'ACCEPT' if acc_U else 'REJECT'}")

# arrays for a possible submission (written by the lead script only if accepted)
np.save(f"{OUT}/x_test_V.npy", prior_x(n_Vt, s_Vt, p0_Vt, aV).astype(np.float32))
np.save(f"{OUT}/x_test_T.npy", prior_x(n_Tt, s_Tt, p0_Tt, aT).astype(np.float32))
np.save(f"{OUT}/x_val_T.npy", xT[aT].astype(np.float32))
np.save(f"{OUT}/n_test_V.npy", n_Vt.astype(np.float32)); np.save(f"{OUT}/n_test_T.npy", n_Tt.astype(np.float32))
np.savez(f"{OUT}/nested_deltas.npz", V=nested_V, Vz=nested_Vz, C=nested_C, F=nested_F, T=nested_T, U=nested_U, fold_f=fold_f)
summary = {"V": {"nested": mV, "se": seV, "worst_slice_z": worstV, "chosen": [list(c) for c in chosen_V], "full_best": list(bestV),
                 "full_delta": dV_full, "cov_val_rows": cov_val_rows, "cov_val_rows3": cov_val_rows3, "cov_val_feeds": cov_val_feeds,
                 "accept": bool(acc_V)},
           "T": {"nested": mT, "se": seT, "worst_slice_z": worstT, "chosen": [list(c) for c in chosen_T], "full_best": list(bestT), "full_delta": dT_full},
           "U": {"nested": mU, "se": seU, "worst_slice_z": worstU, "chosen": [list(c) for c in chosen_U], "full_best": list(bestU), "full_delta": dU_full, "accept": bool(acc_U)},
           "time_forward": {"nested": mF, "se": seF, "chosen": [list(c) for c in chosen_F]},
           "coverage_test": {f"{k[0]}|{k[1]}": v for k, v in cover.items()}, "projection": {f"{k[0]}|{k[1]}": v for k, v in proj.items()}}
json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1)
say(f"done {time.time()-T0:.0f}s")
