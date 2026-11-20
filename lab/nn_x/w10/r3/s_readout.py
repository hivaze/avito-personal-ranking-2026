"""STAGE-S (2-epoch futility screen) read-out of the WAVE-10 arm R3 (nn_x/w10/r3; rule nn_x/w5/logs/E_r3_predeclared.txt).
w10_<arm>_s23 / s24 (A_BASE + HYG + --bce_w W) paired per feed vs the REFERENCE 2-epoch base pair w9_base_s23 / s24 of nn_x/w9/dw3
(A_BASE + HYG, code bit-identical to nn_x/w5 at flag off, stored args == the w5 parser's A_BASE + HYG; read in place, read-only),
with the nn_x/w9/catgrad w9_base_s23 / s24 pair (an independent run of the same recipe; run-to-run corr 0.99967) as the
REPLICATE reference and nn_x/abl abl_base_s23 / s24 as a secondary information reference. CPU only.
  cd <repo> && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 .venv/bin/python nn_x/w10/r3/s_readout.py [arm=r3] [seeds=23,24]
      -> nn_x/w10/r3/logs/s_readout_<arm>.{txt,json}
Primary read = the LAST checkpoint (step 19,670; the 2-epoch screen convention of nn_x/w5 stage A / nn_x/abl / wave 9); the selected
checkpoint (holdout-best at step >= 12k) is reported. Pooled = per-feed mean of the paired deltas over the seeds; SE_feed = sd / sqrt(n);
SE_cal = sqrt(SE_feed^2 + sigma_run^2 / 2), sigma_run 0.00035 val / 0.00015 holdout (information). Slices: M<50, M>=50, Tue, Thu
(own SE_feed). pmix = src/pmix.py private-mix delta (veto only). Holdout columns (audit 11-13 16:10 amendment b): late = train day
Sep 17-19 (MSK day index >= 11 of the holdout feeds' timestamps; 8,279 feeds), early = Sep 6-8 (index <= 2; 7,823 feeds).
Diversity at the member level (val, within-feed rank-z, Pearson): rho_x = mean(corr(arm_s23, base_s24), corr(arm_s24, base_s23)),
rho_bb = corr(base_s23, base_s24); margin = rho_bb - rho_x; also arm-arm, same-seed arm-base, the holdout margin, the replicate null
(the same margin computed with the catgrad base pair in the arm's place: ~0 by construction) and the 2-member pool arm vs base.
STAGE-S RULE (pre-declared): CONTINUE to stage E iff (s1) pooled val (last) >= -0.0015 AND (s2) pooled holdout (last) >= -0.0010 AND
(s3) no val slice < -2 SE_feed AND (s4) member margin rho_bb - rho_x >= +0.008 on val AND (s5) pmix >= -1 SE (veto). Otherwise STOP.
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
import polars as pl  # noqa: E402

ARM = sys.argv[1] if len(sys.argv) > 1 else "r3"
SEEDS = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "23,24").split(",")]
D = Path("<repo>/nn_x/w10/r3")
W = D / "preds"
REF = Path("<repo>/nn_x/w9/dw3/preds")       # primary reference pair w9_base_s23/24
REP = Path("<repo>/nn_x/w9/catgrad/preds")   # replicate reference pair w9_base_s23/24
ABL = Path("<repo>/nn_x/abl/preds")          # secondary information reference abl_base_s23/24
W5H = Path("<repo>/nn_x/w5/preds/holdout")
OUT = D / "logs" / f"s_readout_{ARM}.txt"
SIG_V, SIG_H = 0.00035, 0.00015
LINES = []


def P(s):
    print(s, flush=True)
    LINES.append(s)


fv, Mv, ov = feed_meta("val")
yv = np.load(FLAT / "val/cand_target.npy")
Mh, oh = np.load(W5H / "holdout_meta.npy")
for q in (W / "holdout/holdout_meta.npy", REF / "holdout/holdout_meta.npy", REP / "holdout/holdout_meta.npy"):
    assert (np.load(q) == np.load(W5H / "holdout_meta.npy")).all(), f"holdout meta != nn_x/w5: {q}"
Mh = Mh.astype(np.int64)
oh = oh.astype(np.int64)
yh = np.load(W5H / "holdout_y.npy")
assert (np.load(W / "holdout/holdout_y.npy") == yh).all()
# holdout feeds' train day (train.py's holdout construction replicated; day index 0 = Sep 6 MSK)
_f = pl.read_parquet(FLAT / "train/feeds.parquet", columns=["M", "timestamp"])
_M = _f["M"].to_numpy().astype(np.int64)
_p = np.nan_to_num(np.load("<repo>/eda/labels_position/out/train_prandom.npz")["p_random"].astype(np.float32), nan=0.0)
_p[_M < 50] = 0.0
_hold = np.sort(np.random.default_rng(12345).choice(np.where((_p > 0.9) & (_M >= 50))[0], 30000, replace=False))
assert np.array_equal(_M[_hold], Mh), "holdout feed ids do not reproduce holdout_meta"
_day = (_f["timestamp"].to_numpy()[_hold].astype(np.int64) + 10800) // 86400
_day = _day - _day.min()
H_LATE, H_EARLY = _day >= 11, _day <= 2
assert H_LATE.sum() == 8279 and H_EARLY.sum() == 7823, (H_LATE.sum(), H_EARLY.sum())
ev = lambda s: ndcg_per_feed(np.asarray(s, np.float32), yv, ov, Mv)  # noqa: E731
eh = lambda s: ndcg_per_feed(np.asarray(s, np.float32), yh, oh, Mh)  # noqa: E731
se = lambda d: d.std() / np.sqrt(len(d))  # noqa: E731
zv = lambda x: feed_rank_z(ov, Mv, np.asarray(x, np.float32))[:, 1]  # noqa: E731
zh = lambda x: feed_rank_z(oh, Mh, np.asarray(x, np.float32))[:, 1]  # noqa: E731
corr = lambda a, b: float(np.corrcoef(a, b)[0, 1])  # noqa: E731
tue = (fv["timestamp"].to_numpy() + 10800) // 86400 < 20720
SL = [("M<50", Mv < 50), ("M>=50", Mv >= 50), ("Tue", tue), ("Thu", ~tue)]
A = lambda s: f"w10_{ARM}_s{s}"  # noqa: E731
B = lambda s: f"w9_base_s{s}"  # noqa: E731
P(f"== STAGE S read-out {ARM} ({time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}): {[A(s) for s in SEEDS]} (nn_x/w10/r3) vs reference "
  f"{[B(s) for s in SEEDS]} (nn_x/w9/dw3, 2-epoch A_BASE + HYG; replicate pair nn_x/w9/catgrad; secondary nn_x/abl abl_base); LAST checkpoint "
  f"primary, selected reported; rule nn_x/w5/logs/E_r3_predeclared.txt stage S")


def load(root, v, suf, split):
    return np.load(root / (f"holdout/nn_{v}{suf}_holdout.npy" if split == "h" else f"nn_{v}{suf}_trainonly_val.npy"))


PV, PH = {}, {}
for s in SEEDS:
    for tag, suf in (("last", "@last"), ("sel", "")):
        PV[(A(s), tag)], PH[(A(s), tag)] = load(W, A(s), suf, "v"), load(W, A(s), suf, "h")
        PV[(B(s), tag)], PH[(B(s), tag)] = load(REF, B(s), suf, "v"), load(REF, B(s), suf, "h")
        PV[(f"rep{s}", tag)], PH[(f"rep{s}", tag)] = load(REP, B(s), suf, "v"), load(REP, B(s), suf, "h")
        PV[(f"abl{s}", tag)], PH[(f"abl{s}", tag)] = load(ABL, f"abl_base_s{s}", suf, "v"), load(ABL, f"abl_base_s{s}", suf, "h")
RES = {}
for tag in ("last", "sel"):
    dv, dh, per = [], [], {}
    for s in SEEDS:
        eb, et, hb, ht = ev(PV[(B(s), tag)]), ev(PV[(A(s), tag)]), eh(PH[(B(s), tag)]), eh(PH[(A(s), tag)])
        dv.append(et - eb)
        dh.append(ht - hb)
        per[s] = dict(base_val=float(eb.mean()), arm_val=float(et.mean()), base_hold=float(hb.mean()), arm_hold=float(ht.mean()),
                      d_val=float((et - eb).mean()), d_hold=float((ht - hb).mean()), d_hold_late=float((ht - hb)[H_LATE].mean()),
                      d_hold_early=float((ht - hb)[H_EARLY].mean()))
        P(f"[{tag}] seed {s}: base val {eb.mean():.5f} hold {hb.mean():.5f} | {ARM} val {et.mean():.5f} hold {ht.mean():.5f} | d val {(et - eb).mean():+.5f} "
          f"d hold {(ht - hb).mean():+.5f} (late {(ht - hb)[H_LATE].mean():+.5f} early {(ht - hb)[H_EARLY].mean():+.5f})")
    dv, dh = np.mean(dv, 0), np.mean(dh, 0)
    zs = {n: float(dv[m].mean() / max(se(dv[m]), 1e-9)) for n, m in SL}
    ds = {n: float(dv[m].mean()) for n, m in SL}
    pm, pse = pmix.wmean_se(dv, pmix.weights()["private"])
    sec_v, sec_h = float(np.sqrt(se(dv) ** 2 + SIG_V ** 2 / 2)), float(np.sqrt(se(dh) ** 2 + SIG_H ** 2 / 2))
    hl, he = dh[H_LATE], dh[H_EARLY]
    P(f"[{tag}] POOLED member: val {dv.mean():+.5f}+-{se(dv):.5f} (z {dv.mean() / se(dv):+.2f}; SE_cal {sec_v:.5f}) | holdout {dh.mean():+.5f}+-{se(dh):.5f} "
      f"(z {dh.mean() / se(dh):+.2f}; SE_cal {sec_h:.5f}) | holdout late (Sep 17-19) {hl.mean():+.5f}+-{se(hl):.5f} (z {hl.mean() / se(hl):+.2f}) | "
      f"holdout early (Sep 6-8) {he.mean():+.5f}+-{se(he):.5f} (z {he.mean() / se(he):+.2f}) | pmix {pm:+.5f}+-{pse:.5f} (z {pm / pse:+.2f}) | slices "
      + " ".join(f"{n} {ds[n]:+.5f} (z {zs[n]:+.1f})" for n in zs))
    RES[tag] = dict(val=float(dv.mean()), val_se=float(se(dv)), hold=float(dh.mean()), hold_se=float(se(dh)), hold_late=float(hl.mean()),
                    hold_late_se=float(se(hl)), hold_early=float(he.mean()), hold_early_se=float(se(he)), pmix=float(pm), pmix_se=float(pse),
                    slices_d=ds, slices_z=zs, per_seed=per, se_cal_val=sec_v, se_cal_hold=sec_h)
    # replicate / secondary references (information): the same pooled member deltas vs the catgrad base pair and vs abl_base
    for nm, key in (("replicate (catgrad base pair)", "rep"), ("secondary (abl_base)", "abl")):
        dv2 = np.mean([ev(PV[(A(s), tag)]) - ev(PV[(f"{key}{s}", tag)]) for s in SEEDS], 0)
        dh2 = np.mean([eh(PH[(A(s), tag)]) - eh(PH[(f"{key}{s}", tag)]) for s in SEEDS], 0)
        P(f"[{tag}] vs {nm}: val {dv2.mean():+.5f}+-{se(dv2):.5f} | holdout {dh2.mean():+.5f}+-{se(dh2):.5f} | late {dh2[H_LATE].mean():+.5f} early {dh2[H_EARLY].mean():+.5f}")
        RES[tag][f"val_{key}"], RES[tag][f"hold_{key}"] = float(dv2.mean()), float(dh2.mean())
    # diversity (member level) + 2-member pool
    ZV = {v: zv(PV[(v, tag)]) for v in [A(s) for s in SEEDS] + [B(s) for s in SEEDS] + [f"rep{s}" for s in SEEDS]}
    ZH = {v: zh(PH[(v, tag)]) for v in ZV}
    rho_x = float(np.mean([corr(ZV[A(SEEDS[0])], ZV[B(SEEDS[1])]), corr(ZV[A(SEEDS[1])], ZV[B(SEEDS[0])])]))
    rho_bb = corr(ZV[B(SEEDS[0])], ZV[B(SEEDS[1])])
    rho_aa = corr(ZV[A(SEEDS[0])], ZV[A(SEEDS[1])])
    rho_same = float(np.mean([corr(ZV[A(s)], ZV[B(s)]) for s in SEEDS]))
    rho_xh = float(np.mean([corr(ZH[A(SEEDS[0])], ZH[B(SEEDS[1])]), corr(ZH[A(SEEDS[1])], ZH[B(SEEDS[0])])]))
    rho_bbh = corr(ZH[B(SEEDS[0])], ZH[B(SEEDS[1])])
    # replicate null: the catgrad base pair in the arm's place (same recipe, independent run)
    rho_x_rep = float(np.mean([corr(ZV[f"rep{SEEDS[0]}"], ZV[B(SEEDS[1])]), corr(ZV[f"rep{SEEDS[1]}"], ZV[B(SEEDS[0])])]))
    rho_same_rep = float(np.mean([corr(ZV[f"rep{s}"], ZV[B(s)]) for s in SEEDS]))
    pA, pB = np.mean([ZV[A(s)] for s in SEEDS], 0), np.mean([ZV[B(s)] for s in SEEDS], 0)
    hA, hB = np.mean([ZH[A(s)] for s in SEEDS], 0), np.mean([ZH[B(s)] for s in SEEDS], 0)
    pv, ph = ev(zv(pA)) - ev(zv(pB)), eh(zh(hA)) - eh(zh(hB))
    # mixed 2-member pools (information): arm_s + base_s' vs base_s + base_s'
    mx = np.mean([ev(zv(np.mean([ZV[A(SEEDS[i])], ZV[B(SEEDS[1 - i])]], 0))) - ev(zv(pB)) for i in (0, 1)], 0)
    mxh = np.mean([eh(zh(np.mean([ZH[A(SEEDS[i])], ZH[B(SEEDS[1 - i])]], 0))) - eh(zh(hB)) for i in (0, 1)], 0)
    P(f"[{tag}] DIVERSITY (member, val rank-z): rho_x (arm-base cross-seed) {rho_x:.5f} | rho_bb (base-base) {rho_bb:.5f} | margin {rho_bb - rho_x:+.5f} | "
      f"arm-arm {rho_aa:.5f} | same-seed arm-base {rho_same:.5f} | holdout: rho_x {rho_xh:.5f} rho_bb {rho_bbh:.5f} margin {rho_bbh - rho_xh:+.5f} | "
      f"replicate null: rho_x(rep) {rho_x_rep:.5f} margin {rho_bb - rho_x_rep:+.5f}, same-seed rep-base {rho_same_rep:.5f}")
    P(f"[{tag}] 2-member POOL {ARM} vs base: val {pv.mean():+.5f}+-{se(pv):.5f} | holdout {ph.mean():+.5f}+-{se(ph):.5f} (late {ph[H_LATE].mean():+.5f} early {ph[H_EARLY].mean():+.5f}) | "
      f"mixed pool (arm_s + base_s') vs base pair: val {mx.mean():+.5f}+-{se(mx):.5f} holdout {mxh.mean():+.5f}+-{se(mxh):.5f}")
    RES[tag].update(rho_x=rho_x, rho_bb=rho_bb, margin=rho_bb - rho_x, rho_aa=rho_aa, rho_same=rho_same, rho_xh=rho_xh, rho_bbh=rho_bbh,
                    margin_h=rho_bbh - rho_xh, rho_x_rep=rho_x_rep, margin_rep=rho_bb - rho_x_rep, rho_same_rep=rho_same_rep,
                    pool_val=float(pv.mean()), pool_hold=float(ph.mean()), pool_hold_late=float(ph[H_LATE].mean()), pool_hold_early=float(ph[H_EARLY].mean()),
                    mixed_val=float(mx.mean()), mixed_hold=float(mxh.mean()))
r = RES["last"]
gates = {"(s1) val >= -0.0015": r["val"] >= -0.0015, "(s2) holdout >= -0.0010": r["hold"] >= -0.0010,
         "(s3) no val slice < -2 SE": min(r["slices_z"].values()) >= -2, "(s4) member margin >= +0.008": r["margin"] >= 0.008,
         "(s5) pmix >= -1 SE (veto)": r["pmix"] >= -r["pmix_se"]}
ok = all(gates.values())
P("gates (LAST ckpt): " + "; ".join(f"{k} {'OK' if v else 'NO'}" for k, v in gates.items()))
P(f"VERDICT stage S {ARM}: " + ("CONTINUE to stage E (10-epoch pair s951/952)" if ok else "STOP (futility)")
  + f" (val {r['val']:+.5f} +- {r['val_se']:.5f}, holdout {r['hold']:+.5f} +- {r['hold_se']:.5f}, margin {r['margin']:+.5f}, pmix {r['pmix']:+.5f})")
P(f"registry: w10_{ARM}_s{'/'.join(map(str, SEEDS))} vs w9_base_s23/24 (2-epoch A_BASE + HYG reference pair of nn_x/w9/dw3, LAST ckpt, pre-declared stage-S rule): "
  f"val {r['val']:+.5f} +- {r['val_se']:.5f} (z {r['val'] / r['val_se']:+.1f}), holdout {r['hold']:+.5f} +- {r['hold_se']:.5f} (z {r['hold'] / r['hold_se']:+.1f}), "
  f"late {r['hold_late']:+.5f} early {r['hold_early']:+.5f}, pmix {r['pmix']:+.5f}, " + " ".join(f"{n} {r['slices_d'][n]:+.5f}" for n in r["slices_d"])
  + f", member margin {r['margin']:+.5f} (rho_x {r['rho_x']:.4f} rho_bb {r['rho_bb']:.4f}; replicate null {r['margin_rep']:+.4f}), "
  f"pool val {r['pool_val']:+.5f} / hold {r['pool_hold']:+.5f}; selected ckpt val {RES['sel']['val']:+.5f} / hold {RES['sel']['hold']:+.5f} -> {'CONTINUE' if ok else 'STOP'} ({OUT})")
open(OUT, "w").write("\n".join(LINES) + "\n")
json.dump(dict(arm=ARM, seeds=SEEDS, pass_=bool(ok), gates={k: bool(v) for k, v in gates.items()}, res=RES), open(str(OUT)[:-4] + ".json", "w"), indent=1)
print(f"wrote {OUT}")
