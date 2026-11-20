"""Candidate-level follow-ups (val vs private):
  fixid:  notime set with the parity-fixed pre_id_contrib_v2 in place of the live pre_id_contrib (P5)
  ftonly: the fine-tune extras alone (n2c_pre minus hour), fixed id_contrib
  nnonly: the NN train-stage inputs alone (BASE of cand_adv.py), no extras
Also per-feature PSI / univariate AUC of every extra on cold vs warm candidates separately (does drift survive
conditioning on coldness, i.e. is it more than the cold-share change?)."""
import json, time
from common_av import *
from adv_lib import *
import cand_adv as C

t0 = time.time()
def log(*a): print(f"[{time.time() - t0:7.1f}s]", *a, flush=True)

fv, Mv, _ = meta("val"); ft, Mt, _ = meta("test")
tt = ft["timestamp"].to_numpy(); pri = np.where(tt >= BOUND)[0]
EXTRA2 = [("pre_id_contrib_v2" if n == "pre_id_contrib" else n) for n in C.EXTRA]
C.EXTRA = EXTRA2
ca, ua, fa = C.load_block("val", np.arange(len(fv)), True, 11)
cb, ub, fb = C.load_block("test", pri, True, 13)
first_seen = np.load(f"{SCR}/item_first_seen_day.npy")
# coldness of the sampled rows (same sampling as load_block)
ra, _ = C.sample_rows("val", np.arange(len(fv)), 11); rb, _ = C.sample_rows("test", pri, 13)
cold_a = first_seen[np.load(f"{FLAT}/val/cand_item_id.npy", mmap_mode="r")[ra]] >= 10 ** 9
cold_b = first_seen[np.load(f"{FLAT}/test/cand_item_id.npy", mmap_mode="r")[rb]] >= 10 ** 9
y = np.r_[np.zeros(len(ua)), np.ones(len(ub))].astype(np.int8)
g = groups_of(np.r_[ua, ub], np.r_[fa, fb + 10 ** 8])
out = {}
sets = {"fixid": C.BASE + EXTRA2, "ftonly": EXTRA2, "nnonly": C.BASE,
        "fixid_nocold": C.BASE + EXTRA2}
for sname, fl in sets.items():
    X = np.stack([np.r_[np.asarray(ca[n], np.float32), np.asarray(cb[n], np.float32)] for n in fl], 1)
    cat = ["category"] if "category" in fl else None
    if sname == "fixid_nocold":  # warm candidates only: drift beyond the cold-share change
        keep = ~np.r_[cold_a, cold_b]
        res = run_adv(X[keep], y[keep], g[keep], fl, P_CAND, 300, cat=cat, shap_rows=30000)
        uni = univariate(X[keep], y[keep], fl)
    else:
        res = run_adv(X, y, g, fl, P_CAND, 300, cat=cat, shap_rows=30000)
        uni = univariate(X, y, fl)
    out[sname] = dict(auc=res["auc"], fold_auc=res["fold_auc"].tolist(), table=top_table(res, uni, 15),
                      imp={n: dict(shap=float(res["shap"][j]), gain=float(res["gain"][j]), **uni[n]) for j, n in enumerate(fl)})
    log(sname, f"AUC {res['auc']:.4f}")
    print(out[sname]["table"], flush=True)
# per-feature PSI within cold and within warm strata (val -> private)
strat = {}
for n in C.BASE + EXTRA2:
    if n == "category":
        continue
    a = np.asarray(ca[n], float); b = np.asarray(cb[n], float)
    lo, hi = np.nanquantile(a, [0.001, 0.999]) if np.isfinite(a).any() else (np.nan, np.nan)
    oor = float(np.mean((b < lo) | (b > hi))) if np.isfinite(lo) else np.nan
    strat[n] = dict(all=psi(a, b), cold=psi(a[cold_a], b[cold_b]), warm=psi(a[~cold_a], b[~cold_b]),
                    nan_val=float(np.isnan(a).mean()), nan_pri=float(np.isnan(b).mean()), oor_pri=oor,
                    mean_val=float(np.nanmean(a)) if np.isfinite(a).any() else np.nan,
                    mean_pri=float(np.nanmean(b)) if np.isfinite(b).any() else np.nan)
out["strat_psi"] = strat
out["cold_share"] = dict(val=float(cold_a.mean()), private=float(cold_b.mean()))
json.dump(out, open(f"{OUT}/cand_adv2.json", "w"), indent=1, default=float)
print("cold share rows", out["cold_share"])
print("| feature | PSI all | PSI cold | PSI warm | NaN val | NaN pri | pri outside val q0.1-99.9% | mean val | mean pri |")
for n, d in sorted(strat.items(), key=lambda kv: -kv[1]["all"])[:40]:
    print(f"| {n} | {d['all']:.4f} | {d['cold']:.4f} | {d['warm']:.4f} | {d['nan_val']:.3f} | {d['nan_pri']:.3f} | "
          f"{d['oor_pri']:.4f} | {d['mean_val']:.4g} | {d['mean_pri']:.4g} |")
