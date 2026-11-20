"""Follow-ups (1) s4 graph and (2) region-normalised popularity: EVALUATION under the pre-declared gates
(nn_x/codes/followup/PREDECLARED.txt, 05:47:47 UTC). Features from f01_build.py. CPU only.
Readouts per feature: tail win rate (tail.py), popularity-matched tail win rate, Spearman with popularity, val nested residual
(x02), holdout nested residual (G3 holdout pool), buried-cell pair AUC (stage0), G3 act/exp by quintile.
Controls: it_hist_all_luo (popularity), maxsim32 and cos_hall (cosine). Output: out/f02_eval.log, out/f02_eval.json."""
import sys, json, time
sys.path.insert(0, "<repo>/nn_x/codes/followup")
from fu_common import *
from scipy.stats import spearmanr
from qt import leak_mask

T0 = time.time()
LOGF = open(OUT / "f02_eval.log", "w")


def log(s=""):
    print(s, flush=True); LOGF.write(s + "\n"); LOGF.flush()


NEW = ["s4_link", "s4_link_any", "s4_sym", "s4_sum", "rp_lift", "rp_cnt"]
PRIMARY = {"s4_link": 2.0, "s4_link_any": 2.5, "s4_sym": 2.5, "s4_sum": 2.5, "rp_lift": 2.0, "rp_cnt": 2.5}
TASK = {"s4_link": 1, "s4_link_any": 1, "s4_sym": 1, "s4_sum": 1, "rp_lift": 2, "rp_cnt": 2}
RES = {}

# ----------------------------------------------------------------- val
V = Split("val")
g = g3_score(V); gz = V.fz(g); base = V.nd(gz)
log(f"G3 val NDCG {base.mean():.5f} (recorded 0.40032)")
assert abs(base.mean() - 0.40032) < 2e-5
X = {k: np.load(OUT / f"val_{k}.npy").astype(np.float64) for k in NEW}
pop = np.nan_to_num(np.load(FEATS / "val/it_hist_all_luo.npy").astype(np.float64), nan=0.0)
CTRL = {"pop it_hist_all_luo": pop, "cos maxsim32": np.load(FEATS / "val/maxsim32.npy").astype(np.float64),
        "cos cos_hall": np.load(FEATS / "val/cos_hall.npy").astype(np.float64)}
TL = Tail(V, g)
lp = np.log1p(pop)
popmatch = np.abs(lp[TL.fp][V.fid] - lp) <= 0.25
keep = ~leak_mask()
B = buried_cell(V, keep)
pi, ni = B["pi"], B["ni"]
log(f"tail feeds {TL.tail.sum()} ({TL.tail.mean():.4f}); tail beater pairs {int((TL.above & TL.tail[V.fid]).sum()):,}; "
    f"pop-matched pairs {int((TL.above & TL.tail[V.fid] & popmatch).sum()):,}")
log(f"buried cell: positives {int(B['cellpos'].sum()):,} (all feeds), leak-free pairs {len(pi):,} in {len(np.unique(V.fid[pi])):,} feeds")
log("")
log("feature            | coverage | tail win (+-SE) [untied share, win|untied] | pop-matched win | rho(pop) | val nested (SE, z) [fold w] | buried AUC | G3 act/exp by quintile")
for name, x in list(X.items()) + list(CTRL.items()):
    t0 = time.time()
    cov = float(np.mean(x != 0))
    tw = TL.win(x); twm = TL.win(x, popmatch)
    rho = spearmanr(x, pop).statistic if np.std(x) > 0 else np.nan
    dl, ws = nested(V, gz, x); se = dl.std() / np.sqrt(V.n)
    au, ause, npair = pair_win(V, x, pi, ni)
    ae = actexp_bins(V, gz, x, V.y)
    z = dl.mean() / max(se, 1e-12)
    rec = dict(coverage=cov, tail=tw, tail_popmatched=twm, rho_pop=float(rho), val_nested=float(dl.mean()), val_se=float(se), val_z=float(z),
               fold_w=ws, buried_auc=float(au), buried_se=float(ause), buried_pairs=npair, actexp=ae)
    log(f"{name:18s} | {cov:.4f} | {tw['win']:.4f} +- {tw['se']:.4f} [{tw['untied_share']:.3f}, {tw['win_untied']:.3f}] | "
        f"{twm['win']:.4f} +- {twm['se']:.4f} | {rho:+.3f} | {dl.mean():+.5f} ({se:.5f}, z {z:+.2f}) {ws} | {au:.4f} +- {ause:.4f} | {ae}")
    RES[name] = rec
log("")

# ----------------------------------------------------------------- holdout
H = Split("hold")
gh = g3_score(H); ghz = H.fz(gh); bh = H.nd(ghz)
log(f"G3 holdout NDCG {bh.mean():.5f} (recorded 0.32379)")
popH = np.load(FEATS / "train/it_hist_all_luo.npy", mmap_mode="r")[np.load(OUT / "hold_rows.npy")]
popH = np.nan_to_num(np.asarray(popH, np.float64), nan=0.0)
TH = Tail(H, gh)
for name in NEW + ["pop it_hist_all_luo"]:
    x = np.load(OUT / f"hold_{name}.npy").astype(np.float64) if name in NEW else popH
    dl, ws = nested(H, ghz, x); se = dl.std() / np.sqrt(H.n)
    tw = TH.win(x)
    log(f"HOLDOUT {name:18s}: coverage {np.mean(x != 0):.4f} | nested {dl.mean():+.5f} ({se:.5f}, z {dl.mean()/max(se,1e-12):+.2f}) {ws} | "
        f"tail win {tw['win']:.4f} +- {tw['se']:.4f} [untied {tw['untied_share']:.3f}, {tw['win_untied']:.3f}] | G3 act/exp {actexp_bins(H, ghz, x, H.y)}")
    RES.setdefault(name, {}).update(hold_nested=float(dl.mean()), hold_se=float(se), hold_fold_w=ws, hold_tail=tw, hold_coverage=float(np.mean(x != 0)))
    # holdout by day-of-graph depth (s4 / region need earlier days): first two days carry little or no graph
    if name in NEW:
        d = ((H.ts.astype(np.int64) + 10800) // 86400 - 20702)
        late = d >= 4
        log(f"         {'':18s}  holdout days >= 4 only ({late.sum()} feeds): nested-delta mean {dl[late].mean():+.5f} ({dl[late].std()/np.sqrt(late.sum()):.5f})")

# ----------------------------------------------------------------- gates
log("")
verdict = {1: [], 2: []}
for name in NEW:
    r = RES[name]
    ok_val = r["val_nested"] > 0.0003 and r["val_z"] > PRIMARY[name]
    ok_tail = r["tail"]["win"] > 0.5
    passed = ok_val and ok_tail
    hold_contra = passed and r["hold_nested"] < -2 * r["hold_se"]
    v = "PASS" if passed and not hold_contra else ("HOLD" if passed else "FAIL")
    verdict[TASK[name]].append(v)
    r["gate"] = v
    log(f"GATE task {TASK[name]} {name:12s}: val nested {r['val_nested']:+.5f} (z {r['val_z']:+.2f}, need > +0.0003 and z > {PRIMARY[name]}) -> {ok_val}; "
        f"tail win {r['tail']['win']:.4f} (> 0.5) -> {ok_tail}; holdout {r['hold_nested']:+.5f} ({r['hold_se']:.5f}) -> {v}")
for t in (1, 2):
    vv = verdict[t]
    tv = "PASS" if "PASS" in vv else ("HOLD" if "HOLD" in vv else "CLOSED")
    RES[f"task{t}_verdict"] = tv
    log(f"TASK {t} VERDICT: {tv}")
json.dump(RES, open(OUT / "f02_eval.json", "w"), indent=1, default=float)
log(f"done in {time.time()-T0:.0f}s")
