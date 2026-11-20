"""C10 user aggregates close-out (F26 + F27): feed-constant aggregates a_g enter only as interactions a_g * zf(q_k).
Control = LIN[b, zf(q_1..q_10)]; primary = control + all a_g * zf(q_k). Delta = primary - control (nested and holdout).
Aggregates (val-standardised at the feed level; the same constants on the holdout; NaN -> 0 after standardising):
 F26 log1p(H), u0, u_het{1,2,3,4,10}_share, tte_u_ctr, tte_u_imp, u_contact32, u_fav32; F27 hist_n_uniq_cat, ent32, ent32_top,
 entall, entall_top. Holdout tte_u_* rebuilt here as-of days < d from data/flat/train labels of the same user's other feeds
 (expanding window, = src/features_tte.py's train rule) and checked against data/feats/train."""
from h import *
H = Harness()
O = Path(HERE) / "out"
XC = Path("/tmp/scratchpad/ext/xcheck193/user_feats_val.npz")
CUES = ["f1_z", "f2_z", "f3_z", "f4_z", "f5_z", "cos_u_z", "seq_max_all_z", "cat_cnt_all_z", "it_hist_all_luo_z", "cat_min_hpos"]
AGG = ["log1pH", "u0", "u_het1_share", "u_het2_share", "u_het3_share", "u_het4_share", "u_het10_share", "tte_u_ctr", "tte_u_imp",
       "u_contact32", "u_fav32", "hist_n_uniq_cat", "ent32", "ent32_top", "entall", "entall_top"]


def hist_aggs(split, feeds_idx):
    """u_contact32, u_fav32, ent32, ent32_top, entall, entall_top per feed (build_user.py definitions), label-free."""
    f = pl.read_parquet(FLAT / split / "feeds.parquet")
    Hh = f["H"].to_numpy().astype(np.int64); ho = f["hist_off"].to_numpy().astype(np.int64)
    if feeds_idx is None:
        feeds_idx = np.arange(len(f))
    rows = np.concatenate([np.arange(ho[i], ho[i] + Hh[i]) for i in feeds_idx])
    loc = np.repeat(np.arange(len(feeds_idx)), Hh[feeds_idx])
    hpos = np.asarray(np.load(FLAT / split / "hist_hpos.npy", mmap_mode="r")[rows])
    hcat = np.asarray(np.load(FLAT / split / "hist_category_id.npy", mmap_mode="r")[rows])
    het = np.asarray(np.load(FLAT / split / "hist_event_type.npy", mmap_mode="r")[rows])
    D = pl.DataFrame({"fidx": loc, "hpos": hpos, "cat": hcat, "et": het})
    L32 = D.filter(pl.col("hpos") < 32)
    U = L32.group_by("fidx").agg((pl.col("et").is_in([2, 3])).mean().alias("u_contact32"), (pl.col("et") == 4).mean().alias("u_fav32"))

    def ent(df, name):
        c = df.group_by(["fidx", "cat"]).len().with_columns((pl.col("len") / pl.col("len").sum().over("fidx")).alias("p"))
        return c.group_by("fidx").agg((-(pl.col("p") * pl.col("p").log())).sum().alias(name), pl.col("p").max().alias(name + "_top"))
    U = U.join(ent(L32, "ent32"), on="fidx", how="left").join(ent(D, "entall"), on="fidx", how="left")
    U = pl.DataFrame({"fidx": np.arange(len(feeds_idx))}).join(U, on="fidx", how="left").sort("fidx")
    return {c: U[c].cast(pl.Float64).fill_null(np.nan).to_numpy() for c in ["u_contact32", "u_fav32", "ent32", "ent32_top", "entall", "entall_top"]}


# ---------------- val feed-level aggregates
fv = H.fv; Mv = H.M; offv = H.off
first_v = offv  # first row of each feed
ldv = lambda n: np.load(feat_file("val", n)).astype(np.float64)
A_v = {}
A_v["log1pH"] = np.log1p(fv["H"].to_numpy().astype(np.float64))
A_v["u0"] = (fv["user_id"].to_numpy() == 0).astype(np.float64)
for n in ["u_het1_share", "u_het2_share", "u_het3_share", "u_het4_share", "u_het10_share", "tte_u_ctr", "tte_u_imp", "hist_n_uniq_cat"]:
    x = ldv(n)
    xf = x[first_v]
    # assert feed-constant
    xr = np.repeat(xf, Mv)
    assert np.allclose(np.nan_to_num(x, nan=-9), np.nan_to_num(xr, nan=-9)), n
    A_v[n] = xf
ha = hist_aggs("val", None)
XCv = dict(np.load(XC))
for n in ["u_contact32", "u_fav32", "ent32", "ent32_top", "entall", "entall_top"]:
    A_v[n] = ha[n]
    chk = XCv[n].astype(np.float64)[first_v]
    assert np.allclose(np.nan_to_num(chk, nan=-9), np.nan_to_num(ha[n], nan=-9), atol=1e-5), n
log("val aggregates built; hist aggregates match scratchpad/ext/xcheck193/user_feats_val.npz")

# ---------------- holdout feed-level aggregates
hold = H.hold
ftr = pl.read_parquet(FLAT / "train/feeds.parquet")
Mtr = ftr["M"].to_numpy().astype(np.int64); otr = ftr["cand_off"].to_numpy().astype(np.int64)
ldt = lambda n: np.load(feat_file("train", n), mmap_mode="r")
A_h = {}
A_h["log1pH"] = np.log1p(ftr["H"].to_numpy()[hold].astype(np.float64))
A_h["u0"] = (ftr["user_id"].to_numpy()[hold] == 0).astype(np.float64)
for n in ["u_het1_share", "u_het2_share", "u_het3_share", "u_het4_share", "u_het10_share", "hist_n_uniq_cat"]:
    A_h[n] = np.asarray(ldt(n)[otr[hold]]).astype(np.float64)
# tte_u rebuild as-of days < d (same user's other train feeds; labels of days < d only)
uid_f = ftr["user_id"].to_numpy().astype(np.int64); day_f = (ftr["timestamp"].to_numpy().astype(np.int64) + 10800) // 86400
ytr = np.load(FLAT / "train/cand_target.npy", mmap_mode="r")
fid_tr = np.repeat(np.arange(len(Mtr)), Mtr)
pos_f = np.bincount(fid_tr, weights=np.asarray(ytr).astype(np.float64), minlength=len(Mtr))
FD = pl.DataFrame({"u": uid_f, "d": day_f, "imp": Mtr.astype(np.float64), "pos": pos_f}).filter(pl.col("u") > 0)
UD = FD.group_by(["u", "d"]).agg(pl.col("imp").sum(), pl.col("pos").sum())
Q = pl.DataFrame({"i": np.arange(len(hold)), "u": uid_f[hold], "d": day_f[hold]})
J = Q.join(UD, on="u", how="left").filter(pl.col("d_right") < pl.col("d")).group_by("i").agg(pl.col("imp").sum(), pl.col("pos").sum())
J = Q.join(J, on="i", how="left").sort("i")
u_imp = J["imp"].fill_null(0).to_numpy(); u_pos = J["pos"].fill_null(0).to_numpy()
valid = uid_f[hold] > 0
A_h["tte_u_imp"] = np.where(valid, u_imp, np.nan)
A_h["tte_u_ctr"] = np.where(valid, (u_pos + 0.0159 * 50) / (u_imp + 50), np.nan)
for n in ["tte_u_imp", "tte_u_ctr"]:
    ref = np.asarray(ldt(n)[otr[hold]]).astype(np.float64)
    ok = np.isclose(np.nan_to_num(ref, nan=-9), np.nan_to_num(A_h[n], nan=-9), rtol=1e-4, atol=1e-4)
    log(f"holdout {n} rebuilt as-of days < d: matches data/feats/train on {ok.mean():.4f} of holdout feeds")
ha = hist_aggs("train", hold)
for n in ["u_contact32", "u_fav32", "ent32", "ent32_top", "entall", "entall_top"]:
    A_h[n] = ha[n]
assert set(A_v) == set(AGG) == set(A_h), (set(AGG) ^ set(A_v))
log(f"{len(AGG)} aggregates (the predeclared text lists these 16 names; its count '15' / '150' is an arithmetic slip)")

# standardise with val feed-level mean/sd, NaN -> 0
Sv, Sh = [], []
for n in AGG:
    mu, sd = np.nanmean(A_v[n]), np.nanstd(A_v[n])
    sd = sd if sd > 0 else 1.0
    a_v = np.nan_to_num((A_v[n] - mu) / sd, nan=0.0); a_h = np.nan_to_num((A_h[n] - mu) / sd, nan=0.0)
    log(f"agg {n:16s} val mean {mu:.4g} sd {sd:.4g} NaN val {np.isnan(A_v[n]).mean():.3f} hold {np.isnan(A_h[n]).mean():.3f} | holdout std-mean {a_h.mean():+.3f} sd {a_h.std():.3f}")
    Sv.append(np.repeat(a_v, Mv)); Sh.append(np.repeat(a_h, H.Mh))
Qv = [H.zv(ldv(c)) for c in CUES]
Qh = [H.zh(np.asarray(ldt(c)[H.hrows]).astype(np.float64)) for c in CUES]
Cv = np.stack([H.b] + Qv, 1); Ch = np.stack([H.bh] + Qh, 1)
Iv = [a * q for a in Sv for q in Qv]; Ih = [a * q for a in Sh for q in Qh]
Gv = np.concatenate([Cv, np.stack(Iv, 1)], 1); Gh = np.concatenate([Ch, np.stack(Ih, 1)], 1)
del Iv, Ih
log(f"design: control {Cv.shape[1]} cols, primary {Gv.shape[1]} cols")
res, d = H.evaluate("C10_user_agg_interactions", Gv, Gh, Cv, Ch, primary=True)
with open(O / "results.jsonl", "a") as fh:
    fh.write(json.dumps(res) + "\n")
# information: the cue control vs LIN[b]
res2, _ = H.evaluate("C10_info_cue_control_vs_b", Cv, Ch, primary=False)
with open(O / "results.jsonl", "a") as fh:
    fh.write(json.dumps(res2) + "\n")
