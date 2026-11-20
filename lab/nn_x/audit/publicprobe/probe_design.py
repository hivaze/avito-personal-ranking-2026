"""Public-leaderboard ablation programme: val predictions, public/private paired SEs and slice decompositions of every
candidate probe built from EXISTING preds/ arrays (CPU only). Writes out/probe_design.txt + out/probe_design.json and the
val/test score arrays of the probes (out/cand_<name>_{val,test}.npy).

Conventions: G3 = preds/ens_G3_final_* = z(0.5 z(ft) + 0.5 z(to)), member-equal per-feed z-mean per branch (verified).
  d_i = per-feed val NDCG(probe) - NDCG(G3).  'pubmix' = d reweighted to the public feed-size (M) mix (src/calib.py),
  'privmix' = same for private.  Public paired SE = sqrt(Var_pubmix(d) / n_public), private = sqrt(Var_privmix(d) / n_private)
  (the 'Mw' model of nn_x/pubdiag/noise: it matched the realised public residuals, chi2 p 0.19).
  Slice-restricted probe (A on slice S, G3 elsewhere): public delta = (1/N_pub) * sum_{i in pub and S} d_i, expected
  (n_S_pub / N_pub) * mean_S(d) with M-reweighting inside S, SE = sqrt(n_S_pub * Var_S(d)) / N_pub.
"""
import sys, json, re
from pathlib import Path
import numpy as np, polars as pl
sys.path.insert(0, "<repo>/src")
from common import feed_meta, FLAT
from metrics import ndcg_per_feed
from stack_val import feed_rank_z
import calib
import pmix

ROOT = Path("<repo>"); P = ROOT / "preds"; OUT = Path(__file__).resolve().parent / "out"; OUT.mkdir(exist_ok=True)
BOUND = 1790497920; N_PUB, N_PRIV = 8562, 20033
fv, Mv, ov = feed_meta("val"); ft_, Mt, ot = feed_meta("test"); y = np.load(FLAT / "val/cand_target.npy")
tst = ft_["timestamp"].to_numpy(); pub = tst < BOUND; assert pub.sum() == N_PUB and (~pub).sum() == N_PRIV
hour_t = ((tst + 10800) // 3600) % 24; hour_v = ((fv["timestamp"].to_numpy() + 10800) // 3600) % 24
tue = (fv["timestamp"].to_numpy() + 10800) // 86400 < 20720
segv = pl.read_parquet(ROOT / "eda/errors/segments/feed_table.parquet"); segt = pl.read_parquet(ROOT / "eda/errors/segments/test_segments.parquet")
assert (segv["fidx"].to_numpy() == np.arange(len(Mv))).all() and (segt["fidx"].to_numpy() == np.arange(len(Mt))).all()
csv_, cst_ = segv["cold_share"].to_numpy(), segt["cold_share"].to_numpy()
ntr_v, ntr_t = segv["n_train_feeds"].to_numpy(), segt["n_train_feeds"].to_numpy()
Wpub, Wpriv = calib.weights("public"), calib.weights("private")
_pm = pmix.weights(); Wpub2, Wpriv2 = _pm["public"], _pm["private"]  # M x cold-share reweighting (report-only; level-miscalibrated, deltas informative)
zv = lambda x: feed_rank_z(ov, Mv, np.asarray(x, np.float32))[:, 1]
zt = lambda x: feed_rank_z(ot, Mt, np.asarray(x, np.float32))[:, 1]
ev = lambda s: ndcg_per_feed(np.asarray(s, np.float32), y, ov, Mv)
fidv = np.repeat(np.arange(len(Mv)), Mv); fidt = np.repeat(np.arange(len(Mt)), Mt)

FAM = {"F10": [f"v23_gru_pre2s_s{s}" for s in range(501, 507)], "F7x": [f"r6_cd_m02_r{s}" for s in range(101, 108)],
       "F7x70": [f"r6_cd_m02_p70k_r{s}" for s in range(101, 108)],
       "NEW": ["v27_gru_m02_s511", "v27_gru_m02_s513", "v27_gru_m02_s515", "v28_lr1_m02_s411", "v28_lr1_m02_s413", "v28_lr1_m02_s414",
               "v28_lr1_m02_s415", "v28_lr1_m02_s417", "v29_d192_m02_s711", "v29_d192_m02_s712", "v30_f7_m02_s901", "v30_f7_m02_s902"]}
V, T = {}, {}
def load(m):
    if (m, "ft") not in V:
        V[(m, "ft")] = zv(np.load(P / f"nn_{m}_n2cpv3_val_oof.npy")); V[(m, "to")] = zv(np.load(P / f"nn_{m}_trainonly_val.npy"))
        T[(m, "ft")] = zt(np.load(P / f"nn_{m}_n2cpv3_test.npy")); T[(m, "to")] = zt(np.load(P / f"nn_{m}_trainonly_test.npy"))
def pool(ms, w_ft=0.5):
    for m in ms: load(m)
    fv_ = zv(np.mean([V[(m, "ft")] for m in ms], 0)); tv_ = zv(np.mean([V[(m, "to")] for m in ms], 0))
    ft__ = zt(np.mean([T[(m, "ft")] for m in ms], 0)); tt_ = zt(np.mean([T[(m, "to")] for m in ms], 0))
    return w_ft * fv_ + (1 - w_ft) * tv_, w_ft * ft__ + (1 - w_ft) * tt_
fam = lambda *fs: [m for f in fs for m in FAM[f]]

# ---- G3 reference and its branches
g3v, g3t = np.load(P / "ens_G3_final_val.npy"), np.load(P / "ens_G3_final_test.npy")
G3ftv, G3ftt, G3tov, G3tot = (np.load(P / f"ens_G3_{h}_{s}.npy") for h in ["ft", "to"] for s in ["val", "test"])
base = ev(g3v); print(f"G3 val {base.mean():.5f}")
rec_v, rec_t = pool(fam("F10", "F7x", "F7x70"))
print(f"G3 rebuilt from members: corr val {np.corrcoef(rec_v, g3v)[0,1]:.6f} test {np.corrcoef(rec_t, g3t)[0,1]:.6f}; val NDCG {ev(rec_v).mean():.5f}")

def wblend(w):  # w_ft variants from the stored branch means
    return w * zv(G3ftv) + (1 - w) * zv(G3tov), w * zt(G3ftt) + (1 - w) * zt(G3tot)

def slice_file(av, at, sv_mask, st_mask):  # probe = A on slice feeds, G3 elsewhere (within-feed z keeps ranks)
    out_v, out_t = zv(g3v).copy(), zt(g3t).copy()
    rv, rt = sv_mask[fidv], st_mask[fidt]
    out_v[rv] = zv(av)[rv]; out_t[rt] = zt(at)[rt]
    return out_v, out_t

CANDS = {}  # name -> (val scores, test scores, description, kind)
CANDS["wft100_FTonly"] = (*wblend(1.0), "G3 FT branch only (w_ft = 1.0); twin of the scored TO-only (w_ft = 0)", "full")
CANDS["wft75"] = (*wblend(0.75), "G3 with w_ft 0.75", "full")
CANDS["wft25"] = (*wblend(0.25), "G3 with w_ft 0.25", "full")
CANDS["msplit_M50"] = (*slice_file(zv(G3tov), zt(G3tot), Mv < 50, Mt < 50), "G3 on M>=50, TO-only on M<50 (slice-restricted FT contrast)", "slice:M<50")
CANDS["NEW12"] = (*pool(FAM["NEW"]), "12 from-scratch m02 members (v27/v28/v29/v30, gate-a passing), no re-cool", "full")
CANDS["N1_G3+NEW"] = (*pool(fam("F10", "F7x", "F7x70", "NEW")), "G3 + NEW (32)", "full")
CANDS["noF7x70"] = (*pool(fam("F10", "F7x")), "G3 minus F7x70 (leave-one-family-out, 13)", "full")
CANDS["noF7x"] = (*pool(fam("F10", "F7x70")), "G3 minus F7x (leave-one-family-out, 13)", "full")
CANDS["F10only"] = (*pool(fam("F10")), "F10 alone (6 GRU pre2s, no m02)", "full")
CANDS["M2_check"] = (*pool(fam("F7x", "F7x70")), "M2 = G3 minus F10 (ALREADY SCORED 0.392434)", "scored")
CANDS["TOonly_check"] = (*wblend(0.0), "TO-only (ALREADY SCORED 0.392346)", "scored")
for nm, lab in [("Pall", "Pall (6 families, no m02, 38) = PICK 2 hedge, never on public"), ("PF7x", "PF7x (ALREADY SCORED 0.39272)"), ("Cd", "Cd (ALREADY SCORED 0.39208)")]:
    CANDS[nm if nm == "Pall" else nm + "_check"] = (0.5 * zv(np.load(P / f"ens_{nm}_ft_val.npy")) + 0.5 * zv(np.load(P / f"ens_{nm}_to_val.npy")),
                                                      0.5 * zt(np.load(P / f"ens_{nm}_ft_test.npy")) + 0.5 * zt(np.load(P / f"ens_{nm}_to_test.npy")), lab, "full" if nm == "Pall" else "scored")
# hour-restricted twins of the two most pick-relevant contrasts (private is evening-heavy: 37% of feeds after 18h vs public 18%)
h14v, h14t = hour_v >= 14, hour_t >= 14
CANDS["NEW12_h14"] = (*slice_file(*pool(FAM["NEW"]), h14v, h14t), "NEW12 on feeds at 14-24h MSK, G3 elsewhere (hour-slice twin)", "slice:h>=14")
CANDS["FTonly_h14"] = (*slice_file(*wblend(1.0), h14v, h14t), "FT-only on feeds at 14-24h MSK, G3 elsewhere (hour-slice twin)", "slice:h>=14")
# cold-share twins: probe on the feeds with cold share <= 0.3 (26% of public, 16% of private), G3 elsewhere; the private-dominant
# cold>.3 part of a contrast is then (full probe delta) - (twin delta), both exact on public
cle_v, cle_t = csv_ <= 0.3, cst_ <= 0.3
CANDS["NEW12_coldle3"] = (*slice_file(*pool(FAM["NEW"]), cle_v, cle_t), "NEW12 on cold-share<=0.3 feeds, G3 elsewhere (cold-slice twin)", "slice:cold<=.3")
CANDS["FTonly_coldle3"] = (*slice_file(*wblend(1.0), cle_v, cle_t), "FT-only on cold-share<=0.3 feeds, G3 elsewhere (cold-slice twin)", "slice:cold<=.3")

SL = {"M<50": (Mv < 50, Mt < 50), "M>=50": (Mv >= 50, Mt >= 50), "h00-05": (hour_v < 6, hour_t < 6), "h06-09": ((hour_v >= 6) & (hour_v < 10), (hour_t >= 6) & (hour_t < 10)),
      "h10-13": ((hour_v >= 10) & (hour_v < 14), (hour_t >= 10) & (hour_t < 14)), "h14-17": ((hour_v >= 14) & (hour_v < 18), (hour_t >= 14) & (hour_t < 18)),
      "h18-23": (hour_v >= 18, hour_t >= 18), "cold0": (csv_ == 0, cst_ == 0), "cold0-.3": ((csv_ > 0) & (csv_ <= 0.3), (cst_ > 0) & (cst_ <= 0.3)),
      "cold>.3": (csv_ > 0.3, cst_ > 0.3), "newuser": (ntr_v == 0, ntr_t == 0), "trainuser": (ntr_v > 0, ntr_t > 0), "Tue": (tue, None), "Thu": (~tue, None)}

def wstat(d, w):
    m = np.average(d, weights=w); v = np.average((d - m) ** 2, weights=w); return m, v
def se(x): return x.std() / np.sqrt(len(x))

lines, J = [], {}
lines.append(f"G3 val {base.mean():.5f}; public 0.392820 (exact), n_pub {N_PUB}, n_priv {N_PRIV}; public share by slice (label-free):")
for k, (mv, mt) in SL.items():
    if mt is not None:
        lines.append(f"   {k:10s} val {mv.mean():.3f}  public {mt[pub].mean():.3f}  private {mt[~pub].mean():.3f}")
lines.append("")
hdr = f"{'probe':16s} {'val':>8s} {'d_val':>9s} {'SEval':>8s} | {'d_pubmix':>9s} {'SEpub':>8s} | {'d_privmix':>9s} {'SEpriv':>8s} | {'pmixPub':>8s} {'pmixPrv':>8s} | {'M<50':>8s} {'M>=50':>8s} | {'h<14':>8s} {'h>=14':>8s} | {'cold0':>8s} {'cold>.3':>8s} | {'Tue':>8s} {'Thu':>8s}"
lines.append(hdr)
for name, (sv, st, desc, kind) in CANDS.items():
    pf = ev(sv); d = pf - base
    r = {"desc": desc, "kind": kind, "val": float(pf.mean()), "d_val": float(d.mean()), "se_val": float(se(d))}
    if kind.startswith("slice:"):
        key = kind.split(":")[1]; mv, mt = {"M<50": SL["M<50"], "h>=14": (h14v, h14t), "cold<=.3": (cle_v, cle_t)}[key]
        nS = int(mt[pub].sum()); nSp = int(mt[~pub].sum())
        dS = d[mv]; wS = Wpub[mv]; mS, vS = wstat(dS, wS); mSp, vSp = wstat(dS, Wpriv[mv])
        r.update(slice=key, n_slice_pub=nS, n_slice_priv=nSp, d_pub=float(nS / N_PUB * mS), se_pub=float(np.sqrt(nS * vS) / N_PUB),
                 d_priv=float(nSp / N_PRIV * mSp), se_priv=float(np.sqrt(nSp * vSp) / N_PRIV), slice_mean_pubmix=float(mS))
        assert np.allclose(d[~mv], 0)
    else:
        mP, vP = wstat(d, Wpub); mQ, vQ = wstat(d, Wpriv)
        r.update(d_pub=float(mP), se_pub=float(np.sqrt(vP / N_PUB)), d_priv=float(mQ), se_priv=float(np.sqrt(vQ / N_PRIV)))
    mP2, vP2 = wstat(d, Wpub2); mQ2, vQ2 = wstat(d, Wpriv2)
    if kind.startswith("slice:"):
        mP2 = float(np.sum(Wpub2[mv] * d[mv]) / Wpub2.sum()); mQ2 = float(np.sum(Wpriv2[mv] * d[mv]) / Wpriv2.sum())
    r.update(d_pub_pmix=float(mP2), d_priv_pmix=float(mQ2))
    sl = {}
    for k, (mv, mt) in SL.items():
        sl[k] = (float(d[mv].mean()), float(se(d[mv])))
    r["slices"] = sl
    # contribution view: how much of the public delta comes from each slice under the public mix
    r["pub_contrib"] = {k: float(np.sum(Wpub[mv] * d[mv]) / Wpub.sum()) for k, (mv, mt) in SL.items() if mt is not None}
    J[name] = r
    f = lambda k: f"{sl[k][0]:+8.5f}"
    lines.append(f"{name:16s} {pf.mean():8.5f} {d.mean():+9.5f} {se(d):8.5f} | {r['d_pub']:+9.5f} {r['se_pub']:8.5f} | {r['d_priv']:+9.5f} {r['se_priv']:8.5f} | {r['d_pub_pmix']:+8.5f} {r['d_priv_pmix']:+8.5f} | {f('M<50')} {f('M>=50')} | "
                 f"{(sl['h00-05'][0]*SL['h00-05'][0].sum()+sl['h06-09'][0]*SL['h06-09'][0].sum()+sl['h10-13'][0]*SL['h10-13'][0].sum())/(hour_v<14).sum():+8.5f} {d[h14v].mean():+8.5f} | {f('cold0')} {f('cold>.3')} | {f('Tue')} {f('Thu')}")
    np.save(OUT / f"cand_{name}_val.npy", np.asarray(sv, np.float32)); np.save(OUT / f"cand_{name}_test.npy", np.asarray(st, np.float32))
lines.append("")
lines.append("Realised public deltas vs G3 of the scored files, for the transfer check (val d -> public d): "
             "M2 val +0.00062 pub -0.00039 | TOonly val +0.00003 pub -0.00047 | PF7x val -0.00058 pub -0.00010 | Cd val -0.00131 pub -0.00074 | S2 val +0.00056 pub -0.00035")
# transfer check of the scored pairs against the Mw SE
for name, pubd in [("M2_check", 0.392434 - 0.392820), ("TOonly_check", 0.392346 - 0.392820), ("PF7x_check", 0.39272 - 0.39282), ("Cd_check", 0.39208 - 0.39282)]:
    r = J[name]; z = (pubd - r["d_pub"]) / np.sqrt(r["se_pub"] ** 2 + r["se_val"] ** 2)
    lines.append(f"   {name:13s} expected (pubmix) {r['d_pub']:+.5f} realised {pubd:+.5f} resid {pubd - r['d_pub']:+.5f} z {z:+.2f}")
    r["realised_pub"] = pubd
# per-feed sd of the deltas (for the SE scaling of other contrasts)
lines.append("")
lines.append("Pairwise within-feed corr of the probe test scores with G3 (public feeds; label-free disagreement):")
zg = zt(g3t)
for name, (sv, st, desc, kind) in CANDS.items():
    zs = zt(st); c = np.bincount(fidt, zg * zs, len(Mt)) / Mt
    lines.append(f"   {name:16s} mean within-feed corr public {c[pub].mean():.4f} private {c[~pub].mean():.4f}")
(OUT / "probe_design.txt").write_text("\n".join(lines) + "\n"); json.dump(J, open(OUT / "probe_design.json", "w"), indent=1)
print("\n".join(lines))
