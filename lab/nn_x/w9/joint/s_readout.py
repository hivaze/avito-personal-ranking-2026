"""STAGE S read-out of the WAVE 9 JOINT arm (nn_x/w5/logs/E_joint_predeclared.txt, 2026-11-13T13:25:13Z): the 2-epoch futility
screen w9_<arm>_s23 / s24 (nn_x/w9/joint/preds) paired per seed vs the nn_x/abl abl_base_s23 / s24 (A_BASE + HYG exactly).
CPU only, read-only on every prediction / log file.
  cd <repo> && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 .venv/bin/python nn_x/w9/joint/s_readout.py [arm=joint] [seeds=23,24]
      -> nn_x/w9/joint/logs/s_readout_<arm>.{txt,json} (+ a copy nn_x/w5/logs/w9_s_readout_<arm>.txt)
Primary read = the LAST checkpoint (@last, step 19,670); the selected checkpoint is reported. Delta = arm - base on the same seed,
per feed; pooled = per-feed mean over the seeds; SE_feed = sd / sqrt(n_feeds).
  GO iff (s1) pooled member val >= -0.0015 AND (s2) pooled member holdout >= -0.0010 AND
         (s3) rho_bb - rho_x >= +0.008 on val, rho_bb = corr(z(base_s23), z(base_s24)),
              rho_x = mean(corr(z(arm_s23), z(base_s24)), corr(z(arm_s24), z(base_s23))) (within-feed rank-z, Pearson, @last).
Reported (never gates): slices M<50 / M>=50 / Tue / Thu, pmix, holdout rho gap, arm-arm / same-seed corr, selected-ckpt reads,
train minutes vs base, the jointhq0 information pair when its files exist."""
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

ARM = sys.argv[1] if len(sys.argv) > 1 else "joint"
SEEDS = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "23,24").split(",")]
W9 = Path("<repo>/nn_x/w9/joint")
ABL = Path("<repo>/nn_x/abl")
OUT = W9 / "logs" / f"s_readout_{ARM}.txt"
LINES = []


def P(s):
    print(s, flush=True)
    LINES.append(s)


fv, Mv, ov = feed_meta("val")
yv = np.load(FLAT / "val/cand_target.npy")
Mh, oh = np.load(W9 / "preds/holdout/holdout_meta.npy") if (W9 / "preds/holdout/holdout_meta.npy").exists() else np.load(ABL / "preds/holdout/holdout_meta.npy")
Mh, oh = Mh.astype(np.int64), oh.astype(np.int64)
yh = np.load(ABL / "preds/holdout/holdout_y.npy")
assert np.array_equal(np.load(ABL / "preds/holdout/holdout_meta.npy").astype(np.int64), np.stack([Mh, oh])), "holdout meta mismatch"
ev = lambda s: ndcg_per_feed(np.asarray(s, np.float32), yv, ov, Mv)  # noqa: E731
eh = lambda s: ndcg_per_feed(np.asarray(s, np.float32), yh, oh, Mh)  # noqa: E731
se = lambda d: d.std() / np.sqrt(len(d))  # noqa: E731
zv = lambda x: feed_rank_z(ov, Mv, np.asarray(x, np.float32))[:, 1]  # noqa: E731
zh = lambda x: feed_rank_z(oh, Mh, np.asarray(x, np.float32))[:, 1]  # noqa: E731
corr = lambda a, b: float(np.corrcoef(a, b)[0, 1])  # noqa: E731
tue = (fv["timestamp"].to_numpy() + 10800) // 86400 < 20720
SL = [("M<50", Mv < 50), ("M>=50", Mv >= 50), ("Tue", tue), ("Thu", ~tue)]
Wt = pmix.weights()


def arm_files(arm, s, ck):
    return W9 / f"preds/nn_w9_{arm}_s{s}{ck}_trainonly_val.npy", W9 / f"preds/holdout/nn_w9_{arm}_s{s}{ck}_holdout.npy"


def base_files(s, ck):
    return ABL / f"preds/nn_abl_base_s{s}{ck}_trainonly_val.npy", ABL / f"preds/holdout/nn_abl_base_s{s}{ck}_holdout.npy"


def read_pair(arm, ck, label):
    """Per-seed + pooled paired deltas and the correlation structure of arm vs base at checkpoint ck ('@last' or '')."""
    ZV, ZH, dv, dh, per = {}, {}, [], [], {}
    for s in SEEDS:
        av, ah = arm_files(arm, s, ck)
        bv, bh = base_files(s, ck)
        A, B = np.load(av), np.load(bv)
        AH, BH = np.load(ah), np.load(bh)
        ZV[("a", s)], ZV[("b", s)], ZH[("a", s)], ZH[("b", s)] = zv(A), zv(B), zh(AH), zh(BH)
        ea, eb, ha, hb = ev(A), ev(B), eh(AH), eh(BH)
        dv.append(ea - eb)
        dh.append(ha - hb)
        per[s] = dict(arm_val=float(ea.mean()), base_val=float(eb.mean()), d_val=float((ea - eb).mean()),
                      arm_hold=float(ha.mean()), base_hold=float(hb.mean()), d_hold=float((ha - hb).mean()))
        P(f"  seed {s}: base val {eb.mean():.5f} hold {hb.mean():.5f} | {arm} val {ea.mean():.5f} hold {ha.mean():.5f} | "
          f"d val {(ea - eb).mean():+.5f} d hold {(ha - hb).mean():+.5f}")
    dv, dh = np.mean(dv, 0), np.mean(dh, 0)
    zs = {n: float(dv[m].mean() / max(se(dv[m]), 1e-9)) for n, m in SL}
    ds = {n: float(dv[m].mean()) for n, m in SL}
    pm, pse = pmix.wmean_se(dv, Wt["private"])
    s1, s2 = SEEDS[0], SEEDS[1]
    rho_bb = corr(ZV[("b", s1)], ZV[("b", s2)])
    rho_x = float(np.mean([corr(ZV[("a", s1)], ZV[("b", s2)]), corr(ZV[("a", s2)], ZV[("b", s1)])]))
    rho_aa = corr(ZV[("a", s1)], ZV[("a", s2)])
    rho_same = float(np.mean([corr(ZV[("a", s)], ZV[("b", s)]) for s in SEEDS]))
    rho_bbh = corr(ZH[("b", s1)], ZH[("b", s2)])
    rho_xh = float(np.mean([corr(ZH[("a", s1)], ZH[("b", s2)]), corr(ZH[("a", s2)], ZH[("b", s1)])]))
    # 2-member partner value (info): pair(arm_s, base_other) vs pair(base, base)
    pbb = ev(zv((ZV[("b", s1)] + ZV[("b", s2)]) / 2))
    pv = np.mean([ev(zv((ZV[("a", s1)] + ZV[("b", s2)]) / 2)), ev(zv((ZV[("a", s2)] + ZV[("b", s1)]) / 2))], 0) - pbb
    pbbh = eh(zh((ZH[("b", s1)] + ZH[("b", s2)]) / 2))
    ph = np.mean([eh(zh((ZH[("a", s1)] + ZH[("b", s2)]) / 2)), eh(zh((ZH[("a", s2)] + ZH[("b", s1)]) / 2))], 0) - pbbh
    paa = ev(zv((ZV[("a", s1)] + ZV[("a", s2)]) / 2)) - pbb
    paah = eh(zh((ZH[("a", s1)] + ZH[("a", s2)]) / 2)) - pbbh
    P(f"  POOLED member [{label}]: val {dv.mean():+.5f}+-{se(dv):.5f} (z {dv.mean() / se(dv):+.2f}) | holdout {dh.mean():+.5f}+-{se(dh):.5f} "
      f"(z {dh.mean() / se(dh):+.2f}) | pmix {pm:+.5f}+-{pse:.5f} (z {pm / pse:+.2f}) | slices "
      + " ".join(f"{n} {ds[n]:+.5f} (z {zs[n]:+.1f})" for n in zs))
    P(f"  corr (val, within-feed rank-z) [{label}]: rho_bb {rho_bb:.5f} | rho_x {rho_x:.5f} | gap rho_bb - rho_x {rho_bb - rho_x:+.5f} | "
      f"arm-arm {rho_aa:.5f} | same-seed arm-base {rho_same:.5f} || holdout rho_bb {rho_bbh:.5f} rho_x {rho_xh:.5f} gap {rho_bbh - rho_xh:+.5f}")
    P(f"  2-member pools [{label}] (info): partner pair(arm_s, base_other) - pair(base, base): val {pv.mean():+.5f}+-{se(pv):.5f} hold "
      f"{ph.mean():+.5f}+-{se(ph):.5f} | arm pair - base pair: val {paa.mean():+.5f}+-{se(paa):.5f} hold {paah.mean():+.5f}+-{se(paah):.5f}")
    return dict(per_seed=per, val=float(dv.mean()), val_se=float(se(dv)), hold=float(dh.mean()), hold_se=float(se(dh)), pmix=float(pm),
                pmix_se=float(pse), slices_d=ds, slices_z=zs, rho_bb=rho_bb, rho_x=rho_x, gap=rho_bb - rho_x, rho_aa=rho_aa,
                rho_same=rho_same, rho_bb_hold=rho_bbh, rho_x_hold=rho_xh, gap_hold=rho_bbh - rho_xh,
                partner_val=float(pv.mean()), partner_hold=float(ph.mean()), armpair_val=float(paa.mean()), armpair_hold=float(paah.mean()))


def minutes(path):
    try:
        j = json.load(open(path))
        return float(j["train_minutes"]), int(j["best_step"])
    except Exception:
        return float("nan"), -1


P(f"== STAGE S read-out ({time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}): w9_{ARM}_s{SEEDS} vs nn_x/abl abl_base_s{SEEDS} "
  f"(A_BASE + HYG; 2 epochs, 19,670 steps). Rule nn_x/w5/logs/E_joint_predeclared.txt: GO iff @last pooled val >= -0.0015 AND "
  f"holdout >= -0.0010 AND val rho_bb - rho_x >= +0.008")
P("-- PRIMARY: LAST checkpoint (@last)")
last = read_pair(ARM, "@last", "@last")
P("-- SECONDARY (info): selected checkpoint (holdout-best among evals at step >= 12000)")
sel = read_pair(ARM, "", "selected")
mins = {s: (minutes(W9 / f"logs/w9_{ARM}_s{s}.json"), minutes(ABL / f"logs/abl_base_s{s}.json")) for s in SEEDS}
P("  train minutes (arm / base; selected step): " + "; ".join(f"s{s}: {a[0]:.1f} / {b[0]:.1f} (steps {a[1]} / {b[1]})" for s, (a, b) in mins.items()))
info = {}
if all(arm_files("jointhq0", s, "@last")[0].exists() for s in SEEDS):
    P("-- INFORMATION arm jointhq0 (joint_hq 0: user / history queries blocked from candidate keys), @last, same reads, no gate")
    info["jointhq0_last"] = read_pair("jointhq0", "@last", "hq0 @last")
gates = {"s1 val >= -0.0015": bool(last["val"] >= -0.0015), "s2 holdout >= -0.0010": bool(last["hold"] >= -0.0010),
         "s3 rho gap >= +0.008": bool(last["gap"] >= 0.008)}
ok = all(gates.values())
P("gates (@last): " + "; ".join(f"{k} {'OK' if v else 'NO'}" for k, v in gates.items()))
P(f"VERDICT stage S {ARM}: " + ("GO -> stage E (10-epoch pair w5_e10_joint_s951 / s952)" if ok else "NO-GO -> arm CLOSED at stage S")
  + f" (val {last['val']:+.5f} +- {last['val_se']:.5f}, holdout {last['hold']:+.5f} +- {last['hold_se']:.5f}, rho gap {last['gap']:+.5f}; "
  f"selected: val {sel['val']:+.5f}, holdout {sel['hold']:+.5f}, gap {sel['gap']:+.5f})")
P(f"registry: w9_{ARM}_s23/24 vs abl_base_s23/24 (2-epoch futility screen, A_BASE + HYG + --joint_layers 4 --blocks 0 --hist_layers 0, "
  f"@last; rule E_joint_predeclared.txt 13:25:13Z): val {last['val']:+.5f} +- {last['val_se']:.5f} (z {last['val'] / last['val_se']:+.1f}), "
  f"holdout {last['hold']:+.5f} +- {last['hold_se']:.5f} (z {last['hold'] / last['hold_se']:+.1f}), pmix {last['pmix']:+.5f}, "
  + " ".join(f"{n} {last['slices_d'][n]:+.5f}" for n in last["slices_d"]) + f", rho_x {last['rho_x']:.4f} vs rho_bb {last['rho_bb']:.4f} "
  f"(gap {last['gap']:+.4f}), partner pool val {last['partner_val']:+.5f} / hold {last['partner_hold']:+.5f} -> {'GO' if ok else 'NO-GO'} ({OUT})")
OUT.write_text("\n".join(LINES) + "\n")
json.dump(dict(arm=ARM, seeds=SEEDS, go=bool(ok), gates=gates, last=last, selected=sel, minutes={str(s): v for s, v in mins.items()}, info=info),
          open(str(OUT)[:-4] + ".json", "w"), indent=1)
try:
    Path(f"<repo>/nn_x/w5/logs/w9_s_readout_{ARM}.txt").write_text("\n".join(LINES) + "\n")
except Exception as e:  # noqa: BLE001
    print(f"copy into nn_x/w5/logs failed: {e}")
print(f"wrote {OUT}")
