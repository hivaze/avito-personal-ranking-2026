"""Part 2: checkpoint-selection optimism on the holdout, from the training-log histories (step, val_ndcg, holdout_ndcg).
Every member keeps the argmax-holdout checkpoint (select_on holdout, keep best); the SAME 30K holdout is then used to
score pools, so the pool holdout numbers carry the selection luck. Writes p2_ckpt.txt and p2_runs.csv."""
import json
from lib import *

rng = np.random.default_rng(0)
lines = []
P_ = lambda s="": (print(s, flush=True), lines.append(s))


def load_hist(v):
    for d in (ROOT / "nn_x/r6/logs", ROOT / "nn/logs"):
        f = d / f"{v}.json"
        if f.exists():
            j = json.load(open(f))
            h = j["history"]
            return j["args"], j.get("best_step"), np.array([x["step"] for x in h]), \
                np.array([x["val_ndcg"] for x in h]), np.array([x["holdout_ndcg"] for x in h])
    raise FileNotFoundError(v)


def segments(a, s):
    """Trend segments: the whole run for re-cools (cooldown_frac 1); for 10-epoch WSD runs the stable phase from 25%
    of training to the cooldown start, and the cooldown phase, fitted separately (the cooldown is a regime change)."""
    if a.get("cooldown_frac", 0.2) >= 1.0:
        return [np.ones(len(s), bool)]
    t_cool = (1 - a["cooldown_frac"]) * s[-1]
    return [(s >= 0.25 * s[-1]) & (s < t_cool), s >= t_cool]


def trend(x, y, deg):
    xs = (x - x.mean()) / (x.std() + 1e-9)
    return np.polyval(np.polyfit(xs, y, deg), xs)


def detrend(a, s, y):
    """Residual and trend on the union of segments (NaN outside)."""
    res, tr = np.full(len(y), np.nan), np.full(len(y), np.nan)
    for w in segments(a, s):
        deg = 3 if w.sum() >= 12 else 2
        t = trend(s[w], y[w], deg)
        tr[w], res[w] = t, y[w] - t
    return res, tr


rows = []
for fam, ms in FAM.items():
    for m in ms:
        a, best, s, v, h = load_hist(m)
        assert a["select_on"] == "holdout" and a["keep"] == "best", (m, a["select_on"], a["keep"])
        k = int(np.argmax(h))  # train.py keeps the first strict max
        assert s[k] == best, (m, s[k], best)
        rh, trh = detrend(a, s, h)
        rv, _ = detrend(a, s, v)
        ok = ~np.isnan(rh)
        assert ok[k], (m, "selected step outside the trend segments")
        nseg = len(segments(a, s))
        dof = ok.sum() - nseg * 4
        sd_h = np.sqrt(np.nansum(rh ** 2) / max(dof, 1))
        cool = segments(a, s)[-1]
        rows.append(dict(fam=fam, m=m, grid=a["eval_every"], n_eval=len(s), n_cool=int(cool.sum()), sel=int(s[k]),
                         sel_in_cool=bool(cool[k]), sel_frac=float((s[k] - s[cool][0]) / max(s[cool][-1] - s[cool][0], 1)),
                         last=int(s[-1]), h_best=h[k], h_last=h[-1], v_sel=v[k], v_last=v[-1], v_max=v.max(),
                         v_argmax=int(s[np.argmax(v)]), sd_h=sd_h, sd_v=np.sqrt(np.nansum(rv ** 2) / max(dof, 1)),
                         r_hv=float(np.corrcoef(rh[ok], rv[ok])[0, 1]), res_h_sel=rh[k], res_v_sel=rv[k],
                         near=int((h[ok] >= h[k] - sd_h).sum()), tr_h=trh[ok], ws=s[ok]))

# Monte-Carlo inflation: if the residuals were pure holdout sampling noise (iid N(0, sd_h^2)) around the fitted trend,
# E[noise at the argmax] = selection inflation. Upper bound (real checkpoint fluctuation is not inflation).
for r in rows:
    f = r["tr_h"]
    E = rng.standard_normal((4000, len(f))) * r["sd_h"]
    kk = np.argmax(f[None, :] + E, 1)
    r["infl_mc"] = float(E[np.arange(len(kk)), kk].mean())
    r["infl_mc_pooled"] = None

# pooled residual SD per grid type is more stable than per-run (few points); recompute MC with the family-pooled SD
P_("== Part 2: checkpoint selection on the holdout (all members: select_on=holdout, keep=best; verified from json args)")
P_("Per family (means over members). sel_frac = position of the selected eval in the cooldown (0 = start, 1 = end; <0 = selected in the stable phase).")
P_("res_h/res_v = holdout/val residual vs a smooth per-phase trend (cubic, quadratic if < 12 evals); res_v@sel = does the holdout luck replicate on val?")
P_("fam    grid n_eval n_cool sel_frac  best-last(h)  vsel-vlast  vmax-vsel  sd_res_h sd_res_v r(res_h,res_v) res_h@sel res_v@sel near-ties  infl_MC(upper)  infl_MC(pooled sd)")
fam_infl = {}
for fam in FAM:
    R = [r for r in rows if r["fam"] == fam]
    sdp = np.sqrt(np.nanmean([r["sd_h"] ** 2 for r in R]))
    for r in R:
        f = r["tr_h"]
        E = rng.standard_normal((4000, len(f))) * sdp
        kk = np.argmax(f[None, :] + E, 1)
        r["infl_mc_pooled"] = float(E[np.arange(len(kk)), kk].mean())
    g = lambda k: np.nanmean([r[k] for r in R])
    fam_infl[fam] = (g("infl_mc"), g("infl_mc_pooled"), g("res_h_sel"), g("res_v_sel"), g("r_hv"))
    P_(f"{fam:6s} {R[0]['grid']:5d} {g('n_eval'):5.0f} {g('n_cool'):5.1f}  {g('sel_frac'):6.2f}   {g('h_best') - g('h_last'):+.5f}   "
       f"{g('v_sel') - g('v_last'):+.5f}   {g('v_max') - g('v_sel'):+.5f}  {g('sd_h'):.5f}  {g('sd_v'):.5f}   {g('r_hv'):+.2f}"
       f"       {g('res_h_sel'):+.5f}  {g('res_v_sel'):+.5f}   {g('near'):4.1f}     {g('infl_mc'):+.5f}        {g('infl_mc_pooled'):+.5f}")

P_()
P_("Pooled over all runs: regression of the val residual on the holdout residual (same checkpoint, detrended):")
rh_all = np.concatenate([[r["res_h_sel"]] for r in rows]); rv_all = np.concatenate([[r["res_v_sel"]] for r in rows])
P_(f"  mean holdout residual at the argmax {rh_all.mean():+.5f} +- {se(rh_all):.5f}; val residual at the same step "
   f"{rv_all.mean():+.5f} +- {se(rv_all):.5f} (n={len(rows)} runs)")
P_("  -> if the holdout luck at the argmax were real model quality it would show up on val at the same checkpoint.")
for grp, keys in [("r6 re-cools (F7x, F7x70, F7b)", ("F7x", "F7x70", "F7b")), ("F5cd re-cools", ("F5cd",)),
                  ("10-epoch nn/ runs (F6 F7 F9 F10 F12)", ("F6", "F7", "F9", "F10", "F12"))]:
    R = [r for r in rows if r["fam"] in keys]
    a = np.array([r["res_h_sel"] for r in R]); b = np.array([r["res_v_sel"] for r in R])
    P_(f"  {grp:40s} res_h@sel {a.mean():+.5f}+-{se(a):.5f}  res_v@sel {b.mean():+.5f}+-{se(b):.5f}  "
       f"mean r(res_h,res_v) {np.nanmean([r['r_hv'] for r in R]):+.2f}")

P_()
P_("Pool-level holdout inflation, upper bound = member-equal mean of the member MC inflation (pooled sd):")
for p in POOLS:
    ms = pool_members(p)
    mi = np.mean([r["infl_mc_pooled"] for r in rows if r["m"] in ms])
    P_(f"  {p:5s} {mi:+.5f}   (vs Cd {mi - np.mean([r['infl_mc_pooled'] for r in rows if r['m'] in pool_members('Cd')]):+.5f})")

P_()
P_("Per run: selected step / last step, holdout best-last, val at selected vs val max (val in log is never used to select):")
for r in rows:
    P_(f"  {r['m']:22s} grid {r['grid']:4d} sel {r['sel']:6d}/{r['last']:6d} h_best {r['h_best']:.5f} best-last {r['h_best'] - r['h_last']:+.5f}"
       f" | v_sel {r['v_sel']:.5f} v_last {r['v_last']:.5f} v_max {r['v_max']:.5f}@{r['v_argmax']} | res_h@sel {r['res_h_sel']:+.5f}"
       f" res_v@sel {r['res_v_sel']:+.5f} MC {r['infl_mc_pooled']:+.5f}")
pl.DataFrame([{k: v for k, v in r.items() if k not in ("tr_h", "ws")} for r in rows]).write_csv(OUT / "p2_runs.csv")
(OUT / "p2_ckpt.txt").write_text("\n".join(lines) + "\n")
