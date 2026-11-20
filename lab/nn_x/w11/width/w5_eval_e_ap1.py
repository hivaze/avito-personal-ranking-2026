"""Stage-E read-out of the AP1 CONFIRMATION pair (reports/CURRENT_STATUS.md 2026-11-13 07:28 UTC; nn_x/w5/logs/E_ap1_predeclared.txt):
w5_e10_ap1_s951 / s952 (E_BASE + HYG + --aux_pos 1.0) paired per seed vs w5_e10_base_s951 / s952, train-only members, selected
checkpoints primary, @last reported. The code is the scratchpad e_readout.py (the stage-E machinery used for the TRUNC pairs) with the
seed list / arm / output path as parameters; the rule is unchanged:
  PASS iff ALL: pooled per-seed paired member val delta > 0; pooled holdout delta > +1 SE; no val slice (M<50, M>=50, Tue, Thu) < -2 SE;
  pmix private-mix >= -1 SE.  Also reported: the 2-member pool ap1 vs base (val, holdout), per-seed deltas, @last deltas.
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 .venv/bin/python nn_x/w5/w5_eval_e_ap1.py [arm=ap1] [seeds=951,952]
      -> nn_x/w5/logs/w5_eval_E_<arm>.{txt,json}
"""
import json
import os
import sys
import time

assert os.environ.get("CUDA_VISIBLE_DEVICES", "x") == "", "CPU only (read-out of saved predictions)"
sys.path.insert(0, "<repo>/src")
from common import *  # noqa: E402,F401,F403
from metrics import ndcg_per_feed  # noqa: E402
from stack_val import feed_rank_z  # noqa: E402
import pmix  # noqa: E402

ARM = sys.argv[1] if len(sys.argv) > 1 else "ap1"
SEEDS = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "951,952").split(",")]
W = Path("<repo>/nn_x/w5/preds")
OUT_TXT = os.environ.get("OUT") or f"<repo>/nn_x/w5/logs/w5_eval_E_{ARM}.txt"
LINES = []


def P(s):
    print(s, flush=True)
    LINES.append(s)


fv, Mv, ov = feed_meta("val")
yv = np.load(FLAT / "val/cand_target.npy")
Mh, oh = np.load(W / "holdout/holdout_meta.npy")
Mh = Mh.astype(np.int64)
oh = oh.astype(np.int64)
yh = np.load(W / "holdout/holdout_y.npy")
ev = lambda s: ndcg_per_feed(np.asarray(s, np.float32), yv, ov, Mv)  # noqa: E731
eh = lambda s: ndcg_per_feed(np.asarray(s, np.float32), yh, oh, Mh)  # noqa: E731
se = lambda d: d.std() / np.sqrt(len(d))  # noqa: E731
zv = lambda x: feed_rank_z(ov, Mv, np.asarray(x, np.float32))[:, 1]  # noqa: E731
zh = lambda x: feed_rank_z(oh, Mh, np.asarray(x, np.float32))[:, 1]  # noqa: E731
tue = (fv["timestamp"].to_numpy() + 10800) // 86400 < 20720
SL = [("M<50", Mv < 50), ("M>=50", Mv >= 50), ("Tue", tue), ("Thu", ~tue)]
P(f"== STAGE E AP1 CONFIRMATION read-out ({time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}): w5_e10_{ARM}_s{SEEDS} vs w5_e10_base_s{SEEDS}, "
  f"selected checkpoints primary (@last info). Rule (logs/E_{ARM}_predeclared.txt): PASS iff pooled val > 0 AND holdout > +1 SE AND no val slice "
  f"(M<50, M>=50, Tue, Thu) < -2 SE AND pmix >= -1 SE")
PV, PH, dv, dh, dvl, dhl, per_seed = {}, {}, [], [], [], [], {}
for s in SEEDS:
    b, t = f"w5_e10_base_s{s}", f"w5_e10_{ARM}_s{s}"
    for v in (b, t):
        PV[v] = np.load(W / f"nn_{v}_trainonly_val.npy")
        PH[v] = np.load(W / f"holdout/nn_{v}_holdout.npy")
    eb, et, hb, ht = ev(PV[b]), ev(PV[t]), eh(PH[b]), eh(PH[t])
    dv.append(et - eb)
    dh.append(ht - hb)
    per_seed[s] = dict(base_val=float(eb.mean()), arm_val=float(et.mean()), base_hold=float(hb.mean()), arm_hold=float(ht.mean()),
                       d_val=float((et - eb).mean()), d_hold=float((ht - hb).mean()))
    P(f"seed {s}: base val {eb.mean():.5f} hold {hb.mean():.5f} | {ARM} val {et.mean():.5f} hold {ht.mean():.5f} | d val {(et - eb).mean():+.5f} "
      f"d hold {(ht - hb).mean():+.5f}")
    try:  # last-checkpoint view (reported only)
        lb, lt = ev(np.load(W / f"nn_{b}@last_trainonly_val.npy")), ev(np.load(W / f"nn_{t}@last_trainonly_val.npy"))
        hlb, hlt = eh(np.load(W / f"holdout/nn_{b}@last_holdout.npy")), eh(np.load(W / f"holdout/nn_{t}@last_holdout.npy"))
        dvl.append(lt - lb)
        dhl.append(hlt - hlb)
        per_seed[s].update(d_val_last=float((lt - lb).mean()), d_hold_last=float((hlt - hlb).mean()))
        P(f"         last ckpt: d val {(lt - lb).mean():+.5f} d hold {(hlt - hlb).mean():+.5f}")
    except FileNotFoundError:
        pass
dv = np.mean(dv, 0)
dh = np.mean(dh, 0)
zs = {n: float(dv[m].mean() / max(se(dv[m]), 1e-9)) for n, m in SL}
ds = {n: float(dv[m].mean()) for n, m in SL}
Wt = pmix.weights()
pm, pse = pmix.wmean_se(dv, Wt["private"])
P(f"POOLED member (selected): val {dv.mean():+.5f}+-{se(dv):.5f} (z {dv.mean() / se(dv):+.2f}) | holdout {dh.mean():+.5f}+-{se(dh):.5f} "
  f"(z {dh.mean() / se(dh):+.2f}) | pmix {pm:+.5f}+-{pse:.5f} (z {pm / pse:+.2f}) | slices " + " ".join(f"{n} {ds[n]:+.5f} (z {zs[n]:+.1f})" for n in zs))
if dvl:
    dvl_, dhl_ = np.mean(dvl, 0), np.mean(dhl, 0)
    P(f"POOLED member (@last, info): val {dvl_.mean():+.5f}+-{se(dvl_):.5f} (z {dvl_.mean() / se(dvl_):+.2f}) | holdout {dhl_.mean():+.5f}+-{se(dhl_):.5f} "
      f"(z {dhl_.mean() / se(dhl_):+.2f})")
pb = ev(zv(np.mean([zv(PV[f"w5_e10_base_s{s}"]) for s in SEEDS], 0)))
pt = ev(zv(np.mean([zv(PV[f"w5_e10_{ARM}_s{s}"]) for s in SEEDS], 0)))
hb_ = eh(zh(np.mean([zh(PH[f"w5_e10_base_s{s}"]) for s in SEEDS], 0)))
ht_ = eh(zh(np.mean([zh(PH[f"w5_e10_{ARM}_s{s}"]) for s in SEEDS], 0)))
P(f"{len(SEEDS)}-member POOL (info): base val {pb.mean():.5f} hold {hb_.mean():.5f} | {ARM} val {pt.mean():.5f} hold {ht_.mean():.5f} | d val "
  f"{(pt - pb).mean():+.5f}+-{se(pt - pb):.5f} d hold {(ht_ - hb_).mean():+.5f}+-{se(ht_ - hb_):.5f}")
gates = {"val > 0": bool(dv.mean() > 0), "holdout > +1 SE": bool(dh.mean() > se(dh)), "no val slice < -2 SE": bool(min(zs.values()) >= -2),
         "pmix >= -1 SE": bool(pm >= -pse)}
ok = all(gates.values())
P("gates: " + "; ".join(f"{k} {'OK' if v else 'NO'}" for k, v in gates.items()))
P(f"VERDICT {ARM}: " + ("PASS" if ok else "FAIL") + f" (val {dv.mean():+.5f}, holdout {dh.mean():+.5f} +- {se(dh):.5f}, pmix {pm:+.5f} +- {pse:.5f})")
P(f"registry: w5_e10_{ARM}_s{'/'.join(map(str, SEEDS))} vs w5_e10_base (10-epoch from-scratch pair, E_BASE + HYG, selected ckpt, pre-declared E rule 07:28 UTC): "
  f"val {dv.mean():+.5f} +- {se(dv):.5f} (z {dv.mean() / se(dv):+.1f}), holdout {dh.mean():+.5f} +- {se(dh):.5f} (z {dh.mean() / se(dh):+.1f}), pmix {pm:+.5f}, "
  + " ".join(f"{n} {ds[n]:+.5f}" for n in zs) + f", pool val {(pt - pb).mean():+.5f} / hold {(ht_ - hb_).mean():+.5f}"
  + (f", @last val {np.mean(dvl, 0).mean():+.5f} / hold {np.mean(dhl, 0).mean():+.5f}" if dvl else "") + f" -> {'PASS' if ok else 'FAIL'} ({OUT_TXT})")
open(OUT_TXT, "w").write("\n".join(LINES) + "\n")
json.dump({"arm": ARM, "seeds": SEEDS, "pass": bool(ok), "gates": gates, "val": float(dv.mean()), "val_se": float(se(dv)), "hold": float(dh.mean()),
           "hold_se": float(se(dh)), "pmix": float(pm), "pmix_se": float(pse), "slices_z": zs, "slices_d": ds, "per_seed": per_seed,
           "pool": dict(d_val=float((pt - pb).mean()), d_hold=float((ht_ - hb_).mean()))},
          open(OUT_TXT[:-4] + ".json", "w"), indent=1)
print(f"wrote {OUT_TXT}")
