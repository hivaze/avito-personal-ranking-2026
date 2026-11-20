"""ENCODER-DOSE (B) PRE2-10EP read-out (ledger nn_x/abl/ledger/ledger.md section 10.B, pre-declared 2026-11-14T00:03:07Z):
abl_e10_xdrop_pre2_s951 / s952 (nn_x/abl/preds; E_BASE + HYG + --xt_exclude <6 pre2>) paired per seed vs w5_e10_base_s951 / s952
(nn_x/w5/preds), train-only members. Mechanics = nn_x/w5/w5_eval_e_ap1.py (selected checkpoints PRIMARY, @last information; per seed;
per-feed paired deltas pooled over the seeds; val slices M<50, M>=50, Tue, Thu; pmix; the 2-member pool), with the arm path
redirected and the sign turned into the pre2 channel's WORTH:
    Wv = mean(base - xdrop_pre2) on val, Wh on the holdout, W_M50 on the val M>=50 slice.
Plus (CURRENT_STATUS 11-13 16:10 amendment (b)) the holdout late (Sep 17-19) / early (Sep 6-8) columns as information.
SEQUENTIAL GATE (SOLVER_POWER 6.3 (B), verbatim ">= +0.0015 val, holdout > 0, M>=50 > 0"), in this order:
    (1) Wv >= +0.0015   if FALSE -> STOP, FAIL ((2) / (3) printed, not counted)
    (2) Wh > 0          if FALSE -> FAIL
    (3) W_M50 > 0       if FALSE -> FAIL
    PASS iff (1), (2), (3) hold in sequence.

  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 .venv/bin/python nn_x/abl/encdose/readout_pre2_10ep.py [--arm abl_e10_xdrop_pre2] [--arm_dir nn_x/abl/preds]
        [--seeds 951,952] [--plumb]      -> nn_x/abl/logs/readout_pre2_10ep.txt (+ .json)
  --plumb: plumbing check = arm w5_e10_ap1 from nn_x/w5/preds; its member numbers must equal nn_x/w5/logs/w5_eval_E_ap1.txt with the sign
           flipped (worth = base - arm); written to the scratchpad, never to logs/.
"""
import argparse
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

ap = argparse.ArgumentParser()
ap.add_argument("--arm", default="abl_e10_xdrop_pre2")
ap.add_argument("--arm_dir", default="<repo>/nn_x/abl/preds")
ap.add_argument("--seeds", default="951,952")
ap.add_argument("--plumb", action="store_true")
a = ap.parse_args()
if a.plumb:
    a.arm, a.arm_dir = "w5_e10_ap1", "<repo>/nn_x/w5/preds"
ARM, SEEDS = a.arm, [int(x) for x in a.seeds.split(",")]
W5 = Path("<repo>/nn_x/w5/preds")
AD = Path(a.arm_dir)
OUT_TXT = os.environ.get("OUT") or ("/tmp/scratchpad/encdose/readout_plumb_ap1.txt"
                                    if a.plumb else "<repo>/nn_x/abl/logs/readout_pre2_10ep.txt")
Path(OUT_TXT).parent.mkdir(parents=True, exist_ok=True)
LINES = []
GATE_VAL = 0.0015


def P(s):
    print(s, flush=True)
    LINES.append(s)


fv, Mv, ov = feed_meta("val")
yv = np.load(FLAT / "val/cand_target.npy")
Mh, oh = np.load(W5 / "holdout/holdout_meta.npy")
Mh = Mh.astype(np.int64)
oh = oh.astype(np.int64)
assert np.array_equal(np.load(AD / "holdout/holdout_meta.npy"), np.stack([Mh, oh])), "arm / base holdout meta differ"
yh = np.load(W5 / "holdout/holdout_y.npy")
ev = lambda s: ndcg_per_feed(np.asarray(s, np.float32), yv, ov, Mv)  # noqa: E731
eh = lambda s: ndcg_per_feed(np.asarray(s, np.float32), yh, oh, Mh)  # noqa: E731
se = lambda d: d.std() / np.sqrt(len(d))  # noqa: E731
zv = lambda x: feed_rank_z(ov, Mv, np.asarray(x, np.float32))[:, 1]  # noqa: E731
zh = lambda x: feed_rank_z(oh, Mh, np.asarray(x, np.float32))[:, 1]  # noqa: E731
tue = (fv["timestamp"].to_numpy() + 10800) // 86400 < 20720
SL = [("M<50", Mv < 50), ("M>=50", Mv >= 50), ("Tue", tue), ("Thu", ~tue)]
# holdout late / early columns (amendment (b)): the rng-12345 holdout feed set, exactly as nn_x/r6/r6_eval.setup('hold') derives it
_tr = pl.read_parquet(FLAT / "train/feeds.parquet", columns=["timestamp", "M"])
_pr = np.nan_to_num(np.load("<repo>/eda/labels_position/out/train_prandom.npz")["p_random"].astype(np.float32), nan=0.0)
_trM = _tr["M"].to_numpy()
_pr[_trM < 50] = 0.0
_hold = np.sort(np.random.default_rng(12345).choice(np.where((_pr > 0.9) & (_trM >= 50))[0], 30000, replace=False))
assert (_trM[_hold] == Mh).all(), "holdout feed set mismatch"
_day = (_tr["timestamp"].to_numpy().astype(np.int64)[_hold] + 10800) // 86400
HL = [("late(Sep17-19)", _day >= 20713), ("early(Sep6-8)", _day <= 20704)]
assert int(HL[0][1].sum()) == 8279, int(HL[0][1].sum())
P(f"== ENCODER-DOSE (B) PRE2-10EP read-out ({time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}): {ARM}_s{SEEDS} ({AD}) vs w5_e10_base_s{SEEDS} "
  f"(nn_x/w5/preds), selected checkpoints primary (@last info). WORTH W = base - arm. SEQUENTIAL GATE (ledger 10.B): (1) Wv >= +0.0015 "
  f"-> (2) Wh > 0 -> (3) W_M50 > 0; PASS iff all three in sequence" + (" [PLUMBING CHECK on w5_e10_ap1: numbers = w5_eval_E_ap1.txt, sign flipped]" if a.plumb else ""))
PV, PH, dv, dh, dvl, dhl, per_seed = {}, {}, [], [], [], [], {}
for s in SEEDS:
    b, t = f"w5_e10_base_s{s}", f"{ARM}_s{s}"
    PV[b] = np.load(W5 / f"nn_{b}_trainonly_val.npy"); PH[b] = np.load(W5 / f"holdout/nn_{b}_holdout.npy")
    PV[t] = np.load(AD / f"nn_{t}_trainonly_val.npy"); PH[t] = np.load(AD / f"holdout/nn_{t}_holdout.npy")
    eb, et, hb, ht = ev(PV[b]), ev(PV[t]), eh(PH[b]), eh(PH[t])
    dv.append(eb - et)
    dh.append(hb - ht)
    per_seed[s] = dict(base_val=float(eb.mean()), arm_val=float(et.mean()), base_hold=float(hb.mean()), arm_hold=float(ht.mean()),
                       W_val=float((eb - et).mean()), W_hold=float((hb - ht).mean()))
    P(f"seed {s}: base val {eb.mean():.5f} hold {hb.mean():.5f} | {ARM} val {et.mean():.5f} hold {ht.mean():.5f} | W val {(eb - et).mean():+.5f} "
      f"W hold {(hb - ht).mean():+.5f}")
    try:  # last-checkpoint view (reported only)
        lb, lt = ev(np.load(W5 / f"nn_{b}@last_trainonly_val.npy")), ev(np.load(AD / f"nn_{t}@last_trainonly_val.npy"))
        hlb, hlt = eh(np.load(W5 / f"holdout/nn_{b}@last_holdout.npy")), eh(np.load(AD / f"holdout/nn_{t}@last_holdout.npy"))
        dvl.append(lb - lt)
        dhl.append(hlb - hlt)
        per_seed[s].update(W_val_last=float((lb - lt).mean()), W_hold_last=float((hlb - hlt).mean()))
        P(f"         last ckpt: W val {(lb - lt).mean():+.5f} W hold {(hlb - hlt).mean():+.5f}")
    except FileNotFoundError:
        pass
dv = np.mean(dv, 0)
dh = np.mean(dh, 0)
zs = {n: float(dv[m].mean() / max(se(dv[m]), 1e-9)) for n, m in SL}
ds = {n: float(dv[m].mean()) for n, m in SL}
hs = {n: dict(W=float(dh[m].mean()), se=float(se(dh[m])), n=int(m.sum())) for n, m in HL}
Wt = pmix.weights()
pm, pse = pmix.wmean_se(dv, Wt["private"])
P(f"POOLED member WORTH (selected): val {dv.mean():+.5f}+-{se(dv):.5f} (z {dv.mean() / se(dv):+.2f}) | holdout {dh.mean():+.5f}+-{se(dh):.5f} "
  f"(z {dh.mean() / se(dh):+.2f}) | pmix {pm:+.5f}+-{pse:.5f} (z {pm / pse:+.2f}) | slices " + " ".join(f"{n} {ds[n]:+.5f} (z {zs[n]:+.1f})" for n in zs))
P("holdout columns (information): " + "; ".join(f"{n} {h_['W']:+.5f}+-{h_['se']:.5f} (n {h_['n']})" for n, h_ in hs.items()))
if dvl:
    dvl_, dhl_ = np.mean(dvl, 0), np.mean(dhl, 0)
    P(f"POOLED member WORTH (@last, info): val {dvl_.mean():+.5f}+-{se(dvl_):.5f} (z {dvl_.mean() / se(dvl_):+.2f}) | holdout {dhl_.mean():+.5f}+-{se(dhl_):.5f} "
      f"(z {dhl_.mean() / se(dhl_):+.2f})")
pb = ev(zv(np.mean([zv(PV[f"w5_e10_base_s{s}"]) for s in SEEDS], 0)))
pt = ev(zv(np.mean([zv(PV[f"{ARM}_s{s}"]) for s in SEEDS], 0)))
hb_ = eh(zh(np.mean([zh(PH[f"w5_e10_base_s{s}"]) for s in SEEDS], 0)))
ht_ = eh(zh(np.mean([zh(PH[f"{ARM}_s{s}"]) for s in SEEDS], 0)))
P(f"{len(SEEDS)}-member POOL (info): base val {pb.mean():.5f} hold {hb_.mean():.5f} | {ARM} val {pt.mean():.5f} hold {ht_.mean():.5f} | W val "
  f"{(pb - pt).mean():+.5f}+-{se(pb - pt):.5f} W hold {(hb_ - ht_).mean():+.5f}+-{se(hb_ - ht_):.5f}")
Wv, Wh, Wm = float(dv.mean()), float(dh.mean()), ds["M>=50"]
g1, g2, g3 = Wv >= GATE_VAL, Wh > 0, Wm > 0
if not g1:
    stop, ok = 1, False
elif not g2:
    stop, ok = 2, False
elif not g3:
    stop, ok = 3, False
else:
    stop, ok = 0, True
P(f"SEQUENTIAL GATE: (1) Wv {Wv:+.5f} >= +0.0015: {'OK' if g1 else 'NO -> STOP'}; (2) Wh {Wh:+.5f} > 0: {'OK' if g2 else 'NO'}"
  f"{'' if g1 else ' (printed, not counted)'}; (3) W_M50 {Wm:+.5f} > 0: {'OK' if g3 else 'NO'}{'' if g1 and g2 else ' (printed, not counted)'}")
P(f"VERDICT (B) PRE2-10EP {ARM}: " + ("PASS" if ok else f"FAIL (stopped at clause {stop})")
  + f" (worth val {Wv:+.5f} +- {se(dv):.5f}, holdout {Wh:+.5f} +- {se(dh):.5f}, M>=50 {Wm:+.5f}; 2-epoch worth was +0.00128 val / +0.00054 holdout; F9 - F6 +0.0009 val)")
P(f"registry: {ARM}_s{'/'.join(map(str, SEEDS))} vs w5_e10_base (10-epoch from-scratch pair, E_BASE + HYG, selected ckpt; WORTH of the pre2 channel = base - xdrop_pre2; "
  f"ledger 10.B sequential gate): val {Wv:+.5f} +- {se(dv):.5f} (z {Wv / se(dv):+.1f}), holdout {Wh:+.5f} +- {se(dh):.5f} (z {Wh / se(dh):+.1f}), pmix {pm:+.5f}, "
  + " ".join(f"{n} {ds[n]:+.5f}" for n in zs) + f", holdout late {hs['late(Sep17-19)']['W']:+.5f} / early {hs['early(Sep6-8)']['W']:+.5f}, pool val {(pb - pt).mean():+.5f} / hold {(hb_ - ht_).mean():+.5f}"
  + (f", @last val {np.mean(dvl, 0).mean():+.5f} / hold {np.mean(dhl, 0).mean():+.5f}" if dvl else "") + f" -> {'PASS' if ok else 'FAIL'} ({OUT_TXT})")
open(OUT_TXT, "w").write("\n".join(LINES) + "\n")
json.dump({"arm": ARM, "arm_dir": str(AD), "seeds": SEEDS, "pass": bool(ok), "stopped_at_clause": stop, "gates": dict(val_ge_0015=bool(g1), hold_gt_0=bool(g2), m50_gt_0=bool(g3)),
           "W_val": Wv, "W_val_se": float(se(dv)), "W_hold": Wh, "W_hold_se": float(se(dh)), "pmix": float(pm), "pmix_se": float(pse), "slices_z": zs, "slices_W": ds,
           "holdout_cols": hs, "per_seed": per_seed, "pool": dict(W_val=float((pb - pt).mean()), W_hold=float((hb_ - ht_).mean())),
           "last": (dict(W_val=float(np.mean(dvl, 0).mean()), W_hold=float(np.mean(dhl, 0).mean())) if dvl else None)},
          open(OUT_TXT[:-4] + ".json", "w"), indent=1)
print(f"wrote {OUT_TXT}")
