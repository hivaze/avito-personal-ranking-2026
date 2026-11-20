"""STAGE S read-out of the WAVE-9 HS1 2-epoch futility screen (nn_x/w5/logs/E_hs1_predeclared.txt, stage S).
CPU only, read-only on every prediction file.

  cd <repo> && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 .venv/bin/python nn_x/w9/hs1/s_readout.py [arm=hs1] [seeds=23,24]
      -> nn_x/w9/hs1/logs/s_readout_<arm>.{txt,json}  (copied by hand to nn_x/w5/logs/w9_s_readout_<arm>.txt)

Arm runs: nn_x/w9/hs1 w9_<arm>_s<seed> (A_BASE + HYG + the arm flag; selected checkpoint primary, @last information).
Base pair: nn_x/abl abl_base_s<seed> (A_BASE + HYG, identical stored args; tests/test_hs1.py C0(h)).
Delta = arm - base on the SAME seed, per val feed (and per holdout feed); pooled = per-feed mean over the two seeds;
SE = sd / sqrt(n_feeds). Diversity at 2 epochs (member level, within-feed rank-z of the val preds, Pearson):
  rho_bb = corr(z base_s23, z base_s24);  rho_x = mean(corr(z arm_s23, z base_s24), corr(z arm_s24, z base_s23));
  margin = rho_bb - rho_x.
PROCEED to stage E iff dval >= -0.0015 AND margin >= +0.005 AND (dval >= 0 OR margin / |dval| >= 5). Otherwise CLOSED
('removal on the frontier'). Information: holdout delta, slices (M<50, M>=50, Tue, Thu), pmix, @last deltas, same-seed and
arm-arm member correlations, 2-member pool (arm{23,24} vs base{23,24}) on val and holdout.
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

ARM = sys.argv[1] if len(sys.argv) > 1 else "hs1"
SEEDS = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "23,24").split(",")]
W9 = Path("<repo>/nn_x/w9/hs1/preds")
AB = Path("<repo>/nn_x/abl/preds")
OUT_TXT = os.environ.get("OUT") or f"<repo>/nn_x/w9/hs1/logs/s_readout_{ARM}.txt"
LINES = []


def P(s):
    print(s, flush=True)
    LINES.append(s)


fv, Mv, ov = feed_meta("val")
yv = np.load(FLAT / "val/cand_target.npy")
Mh, oh = np.load(AB / "holdout/holdout_meta.npy")
Mh, oh = Mh.astype(np.int64), oh.astype(np.int64)
yh = np.load(AB / "holdout/holdout_y.npy")
assert np.array_equal(np.load(W9 / "holdout/holdout_meta.npy"), np.stack([Mh, oh])), "holdout meta mismatch w9 vs abl"
ev = lambda s: ndcg_per_feed(np.asarray(s, np.float32), yv, ov, Mv)  # noqa: E731
eh = lambda s: ndcg_per_feed(np.asarray(s, np.float32), yh, oh, Mh)  # noqa: E731
se = lambda d: d.std() / np.sqrt(len(d))  # noqa: E731
zv = lambda x: feed_rank_z(ov, Mv, np.asarray(x, np.float32))[:, 1]  # noqa: E731
zh = lambda x: feed_rank_z(oh, Mh, np.asarray(x, np.float32))[:, 1]  # noqa: E731
corr = lambda a, b: float(np.corrcoef(a, b)[0, 1])  # noqa: E731
tue = (fv["timestamp"].to_numpy() + 10800) // 86400 < 20720
SL = [("M<50", Mv < 50), ("M>=50", Mv >= 50), ("Tue", tue), ("Thu", ~tue)]
A = lambda s, suf="": W9 / f"nn_w9_{ARM}_s{s}{suf}_trainonly_val.npy"  # noqa: E731
B = lambda s, suf="": AB / f"nn_abl_base_s{s}{suf}_trainonly_val.npy"  # noqa: E731
AH = lambda s, suf="": W9 / f"holdout/nn_w9_{ARM}_s{s}{suf}_holdout.npy"  # noqa: E731
BH = lambda s, suf="": AB / f"holdout/nn_abl_base_s{s}{suf}_holdout.npy"  # noqa: E731
P(f"== STAGE S read-out {ARM} ({time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}): w9_{ARM}_s{SEEDS} (nn_x/w9/hs1) vs abl_base_s{SEEDS} "
  f"(nn_x/abl), selected checkpoints primary (@last info). Rule (nn_x/w5/logs/E_hs1_predeclared.txt stage S): PROCEED iff dval >= -0.0015 "
  f"AND rho_bb - rho_x >= +0.005 AND (dval >= 0 OR margin / |dval| >= 5)")
PV, PH, dv, dh, dvl, dhl, per_seed = {}, {}, [], [], [], [], {}
for s in SEEDS:
    PV[("a", s)], PV[("b", s)] = np.load(A(s)), np.load(B(s))
    PH[("a", s)], PH[("b", s)] = np.load(AH(s)), np.load(BH(s))
    eb, et, hb, ht = ev(PV[("b", s)]), ev(PV[("a", s)]), eh(PH[("b", s)]), eh(PH[("a", s)])
    dv.append(et - eb)
    dh.append(ht - hb)
    per_seed[s] = dict(base_val=float(eb.mean()), arm_val=float(et.mean()), base_hold=float(hb.mean()), arm_hold=float(ht.mean()),
                       d_val=float((et - eb).mean()), d_hold=float((ht - hb).mean()))
    P(f"seed {s}: base val {eb.mean():.5f} hold {hb.mean():.5f} | {ARM} val {et.mean():.5f} hold {ht.mean():.5f} | d val {(et - eb).mean():+.5f} "
      f"d hold {(ht - hb).mean():+.5f}")
    try:
        lb, lt = ev(np.load(B(s, "@last"))), ev(np.load(A(s, "@last")))
        hlb, hlt = eh(np.load(BH(s, "@last"))), eh(np.load(AH(s, "@last")))
        dvl.append(lt - lb)
        dhl.append(hlt - hlb)
        per_seed[s].update(d_val_last=float((lt - lb).mean()), d_hold_last=float((hlt - hlb).mean()))
        P(f"         last ckpt: d val {(lt - lb).mean():+.5f} d hold {(hlt - hlb).mean():+.5f}")
    except FileNotFoundError:
        pass
dv, dh = np.mean(dv, 0), np.mean(dh, 0)
zs = {n: float(dv[m].mean() / max(se(dv[m]), 1e-9)) for n, m in SL}
ds = {n: float(dv[m].mean()) for n, m in SL}
Wt = pmix.weights()
pm, pse = pmix.wmean_se(dv, Wt["private"])
P(f"POOLED member (selected): val {dv.mean():+.5f}+-{se(dv):.5f} (z {dv.mean() / se(dv):+.2f}) | holdout {dh.mean():+.5f}+-{se(dh):.5f} "
  f"(z {dh.mean() / se(dh):+.2f}) | pmix {pm:+.5f}+-{pse:.5f} (z {pm / pse:+.2f}) | slices " + " ".join(f"{n} {ds[n]:+.5f} (z {zs[n]:+.1f})" for n in zs))
if dvl:
    dvl_, dhl_ = np.mean(dvl, 0), np.mean(dhl, 0)
    P(f"POOLED member (@last, info): val {dvl_.mean():+.5f}+-{se(dvl_):.5f} | holdout {dhl_.mean():+.5f}+-{se(dhl_):.5f}")
# diversity (member level, val, selected)
Z = {k: zv(v) for k, v in PV.items()}
s1, s2 = SEEDS
rho_bb = corr(Z[("b", s1)], Z[("b", s2)])
rho_x = float(np.mean([corr(Z[("a", s1)], Z[("b", s2)]), corr(Z[("a", s2)], Z[("b", s1)])]))
rho_aa = corr(Z[("a", s1)], Z[("a", s2)])
rho_same = float(np.mean([corr(Z[("a", s)], Z[("b", s)]) for s in SEEDS]))
margin = rho_bb - rho_x
dval = float(dv.mean())
ratio = float("inf") if dval >= 0 else margin / abs(dval)
ZH = {k: zh(v) for k, v in PH.items()}
rho_bbh = corr(ZH[("b", s1)], ZH[("b", s2)])
rho_xh = float(np.mean([corr(ZH[("a", s1)], ZH[("b", s2)]), corr(ZH[("a", s2)], ZH[("b", s1)])]))
P(f"DIVERSITY (val, member level): rho_bb = corr(base_s{s1}, base_s{s2}) {rho_bb:.5f} | rho_x = mean cross-seed corr(arm, base) {rho_x:.5f} | "
  f"margin {margin:+.5f} | margin / |dval| {ratio:.1f} | arm-arm {rho_aa:.5f} | same-seed arm-base {rho_same:.5f}")
P(f"     (info) holdout: rho_bb {rho_bbh:.5f} rho_x {rho_xh:.5f} margin {rho_bbh - rho_xh:+.5f}")
pb = ev(zv(np.mean([Z[("b", s)] for s in SEEDS], 0)))
pa = ev(zv(np.mean([Z[("a", s)] for s in SEEDS], 0)))
hb_ = eh(zh(np.mean([ZH[("b", s)] for s in SEEDS], 0)))
ha_ = eh(zh(np.mean([ZH[("a", s)] for s in SEEDS], 0)))
P(f"2-member POOL (info): base val {pb.mean():.5f} hold {hb_.mean():.5f} | {ARM} val {pa.mean():.5f} hold {ha_.mean():.5f} | d val "
  f"{(pa - pb).mean():+.5f}+-{se(pa - pb):.5f} d hold {(ha_ - hb_).mean():+.5f}+-{se(ha_ - hb_):.5f}")
gates = {"dval >= -0.0015": bool(dval >= -0.0015), "margin >= +0.005": bool(margin >= 0.005),
         "dval >= 0 or margin/|dval| >= 5": bool(dval >= 0 or ratio >= 5)}
ok = all(gates.values())
P("gates: " + "; ".join(f"{k} {'OK' if v else 'NO'}" for k, v in gates.items()))
P(f"VERDICT stage S {ARM}: " + ("PROCEED to stage E" if ok else "CLOSED (removal on the frontier)")
  + f" (dval {dval:+.5f} +- {se(dv):.5f}, margin {margin:+.5f}, ratio {ratio:.1f}, holdout {dh.mean():+.5f} +- {se(dh):.5f})")
P(f"registry: w9_{ARM}_s{'/'.join(map(str, SEEDS))} vs abl_base (2-epoch screen, A_BASE + HYG, selected ckpt, pre-declared stage-S rule): val "
  f"{dval:+.5f} +- {se(dv):.5f} (z {dval / se(dv):+.1f}), holdout {dh.mean():+.5f} +- {se(dh):.5f}, pmix {pm:+.5f}, "
  + " ".join(f"{n} {ds[n]:+.5f}" for n in zs) + f", rho_bb {rho_bb:.4f} rho_x {rho_x:.4f} margin {margin:+.4f} ratio {ratio:.1f}, "
  f"pool val {(pa - pb).mean():+.5f} / hold {(ha_ - hb_).mean():+.5f}"
  + (f", @last val {np.mean(dvl, 0).mean():+.5f} / hold {np.mean(dhl, 0).mean():+.5f}" if dvl else "")
  + f" -> {'PROCEED' if ok else 'CLOSED'} ({OUT_TXT})")
open(OUT_TXT, "w").write("\n".join(LINES) + "\n")
json.dump({"arm": ARM, "seeds": SEEDS, "proceed": bool(ok), "gates": gates, "val": dval, "val_se": float(se(dv)), "hold": float(dh.mean()),
           "hold_se": float(se(dh)), "pmix": float(pm), "pmix_se": float(pse), "slices_z": zs, "slices_d": ds, "per_seed": per_seed,
           "rho_bb": rho_bb, "rho_x": rho_x, "margin": margin, "ratio": ratio, "rho_aa": rho_aa, "rho_same": rho_same,
           "rho_bb_hold": rho_bbh, "rho_x_hold": rho_xh, "pool": dict(d_val=float((pa - pb).mean()), d_hold=float((ha_ - hb_).mean()))},
          open(OUT_TXT[:-4] + ".json", "w"), indent=1)
print(f"wrote {OUT_TXT}")
