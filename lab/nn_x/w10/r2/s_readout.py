"""STAGE-S (2-epoch screen) read-out of the WAVE-10 arm R2 XOFF (nn_x/w10/r2; rule nn_x/w5/logs/E_r2_predeclared.txt, frozen 16:57:46Z).
w10_<arm>_s23 / s24 paired per feed on the same seed vs the PRIMARY reference abl_base_s23 / s24 (nn_x/abl/preds; A_BASE + HYG, args verified
by tests/test_r2.py C0c) and the REPLICATION reference w9_base_s23 / s24 (nn_x/w9/dw3/preds; information + run-noise read, never a gate).
CPU only, read-only on the preds.
  cd <repo> && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 .venv/bin/python nn_x/w10/r2/s_readout.py [arm=r2] [seeds=23,24]
      -> nn_x/w10/r2/logs/s_readout_<arm>.{txt,json}
Primary read = the LAST checkpoint (step 19,670; the 2-epoch screen convention of nn_x/abl / w5 stage A: the holdout is an unselected judge);
the selected checkpoint (holdout-best at step >= 12k) is reported. Pooled = per-feed mean of the paired deltas over the seeds;
SE_feed = sd / sqrt(n_feeds); SE_cal = sqrt(SE_feed^2 + sigma_run^2 / 2), sigma_run 0.00035 val / 0.00015 holdout (the frozen abl SCALE-UP form).
Slices: M<50, M>=50, Tue, Thu (own SE_feed). pmix = src/pmix.py private-mix delta (veto only).
STAGE-S RULE (pre-declared; STRENGTH lever): CONTINUE to stage E iff (s1) pooled val (last) > +1 SE_cal(val) AND (s2) pooled holdout (last) >= 0
AND (s3) no val slice < -2 SE_feed AND (s4) pmix >= -1 SE. Otherwise STOP. Diversity margin (rho_bb - rho_x) = information only.
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

ARM = sys.argv[1] if len(sys.argv) > 1 else "r2"
SEEDS = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "23,24").split(",")]
D = Path("<repo>/nn_x/w10/r2")
W = D / "preds"
REFS = {"abl": Path("<repo>/nn_x/abl/preds"), "w9": Path("<repo>/nn_x/w9/dw3/preds")}
REFV = {"abl": lambda s: f"abl_base_s{s}", "w9": lambda s: f"w9_base_s{s}"}
OUT = D / "logs" / f"s_readout_{ARM}.txt"
SIG_V, SIG_H = 0.00035, 0.00015
LINES = []


def P(s):
    print(s, flush=True)
    LINES.append(s)


fv, Mv, ov = feed_meta("val")
yv = np.load(FLAT / "val/cand_target.npy")
meta5 = np.load("<repo>/nn_x/w5/preds/holdout/holdout_meta.npy")
for nm, pth in [("r2", W / "holdout/holdout_meta.npy")] + [(k, v / "holdout/holdout_meta.npy") for k, v in REFS.items()]:
    assert (np.load(pth) == meta5).all(), f"holdout meta of {nm} != nn_x/w5"
Mh, oh = meta5
Mh = Mh.astype(np.int64)
oh = oh.astype(np.int64)
yh = np.load(W / "holdout/holdout_y.npy")
ev = lambda s: ndcg_per_feed(np.asarray(s, np.float32), yv, ov, Mv)  # noqa: E731
eh = lambda s: ndcg_per_feed(np.asarray(s, np.float32), yh, oh, Mh)  # noqa: E731
se = lambda d: d.std() / np.sqrt(len(d))  # noqa: E731
zv = lambda x: feed_rank_z(ov, Mv, np.asarray(x, np.float32))[:, 1]  # noqa: E731
zh = lambda x: feed_rank_z(oh, Mh, np.asarray(x, np.float32))[:, 1]  # noqa: E731
corr = lambda a, b: float(np.corrcoef(a, b)[0, 1])  # noqa: E731
tue = (fv["timestamp"].to_numpy() + 10800) // 86400 < 20720
SL = [("M<50", Mv < 50), ("M>=50", Mv >= 50), ("Tue", tue), ("Thu", ~tue)]
A = lambda s: f"w10_{ARM}_s{s}"  # noqa: E731
P(f"== STAGE S read-out {ARM} ({time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}): {[A(s) for s in SEEDS]} (nn_x/w10/r2, 2-epoch A_BASE + HYG + --xoff 0.5) vs "
  f"PRIMARY abl_base_s{SEEDS} (nn_x/abl) and REPLICATION w9_base_s{SEEDS} (nn_x/w9/dw3); LAST checkpoint primary, selected reported; rule nn_x/w5/logs/E_r2_predeclared.txt stage S")
PV, PH = {}, {}
for s in SEEDS:
    for tag, suf in (("last", "@last"), ("sel", "")):
        PV[(A(s), tag)] = np.load(W / f"nn_{A(s)}{suf}_trainonly_val.npy")
        PH[(A(s), tag)] = np.load(W / f"holdout/nn_{A(s)}{suf}_holdout.npy")
        for k, root in REFS.items():
            v = REFV[k](s)
            PV[(v, tag)] = np.load(root / f"nn_{v}{suf}_trainonly_val.npy")
            PH[(v, tag)] = np.load(root / f"holdout/nn_{v}{suf}_holdout.npy")
RES = {}
for ref in ("abl", "w9"):
    B = REFV[ref]
    RES[ref] = {}
    for tag in ("last", "sel"):
        dv, dh, per = [], [], {}
        for s in SEEDS:
            eb, et, hb, ht = ev(PV[(B(s), tag)]), ev(PV[(A(s), tag)]), eh(PH[(B(s), tag)]), eh(PH[(A(s), tag)])
            dv.append(et - eb)
            dh.append(ht - hb)
            per[s] = dict(base_val=float(eb.mean()), arm_val=float(et.mean()), base_hold=float(hb.mean()), arm_hold=float(ht.mean()),
                          d_val=float((et - eb).mean()), d_hold=float((ht - hb).mean()))
            P(f"[{ref} {tag}] seed {s}: base val {eb.mean():.5f} hold {hb.mean():.5f} | {ARM} val {et.mean():.5f} hold {ht.mean():.5f} | d val {(et - eb).mean():+.5f} d hold {(ht - hb).mean():+.5f}")
        dv, dh = np.mean(dv, 0), np.mean(dh, 0)
        zs = {n: float(dv[m].mean() / max(se(dv[m]), 1e-9)) for n, m in SL}
        ds = {n: float(dv[m].mean()) for n, m in SL}
        pm, pse = pmix.wmean_se(dv, pmix.weights()["private"])
        sec_v, sec_h = float(np.sqrt(se(dv) ** 2 + SIG_V ** 2 / 2)), float(np.sqrt(se(dh) ** 2 + SIG_H ** 2 / 2))
        P(f"[{ref} {tag}] POOLED member: val {dv.mean():+.5f}+-{se(dv):.5f} (z_feed {dv.mean() / se(dv):+.2f}; SE_cal {sec_v:.5f}, z_cal {dv.mean() / sec_v:+.2f}) | "
          f"holdout {dh.mean():+.5f}+-{se(dh):.5f} (z {dh.mean() / se(dh):+.2f}; SE_cal {sec_h:.5f}) | pmix {pm:+.5f}+-{pse:.5f} (z {pm / pse:+.2f}) | slices "
          + " ".join(f"{n} {ds[n]:+.5f} (z {zs[n]:+.1f})" for n in zs))
        ZV = {v: zv(PV[(v, tag)]) for v in [A(s) for s in SEEDS] + [B(s) for s in SEEDS]}
        rho_x = float(np.mean([corr(ZV[A(SEEDS[0])], ZV[B(SEEDS[1])]), corr(ZV[A(SEEDS[1])], ZV[B(SEEDS[0])])]))
        rho_bb = corr(ZV[B(SEEDS[0])], ZV[B(SEEDS[1])])
        rho_aa = corr(ZV[A(SEEDS[0])], ZV[A(SEEDS[1])])
        rho_same = float(np.mean([corr(ZV[A(s)], ZV[B(s)]) for s in SEEDS]))
        pA, pB = np.mean([ZV[A(s)] for s in SEEDS], 0), np.mean([ZV[B(s)] for s in SEEDS], 0)
        pv = ev(zv(pA)) - ev(zv(pB))
        hA = np.mean([zh(PH[(A(s), tag)]) for s in SEEDS], 0)
        hB = np.mean([zh(PH[(B(s), tag)]) for s in SEEDS], 0)
        ph = eh(zh(hA)) - eh(zh(hB))
        P(f"[{ref} {tag}] info: diversity margin rho_bb - rho_x {rho_bb - rho_x:+.5f} (rho_x {rho_x:.5f}, rho_bb {rho_bb:.5f}, arm-arm {rho_aa:.5f}, same-seed {rho_same:.5f}) | "
          f"2-member pool {ARM} vs base: val {pv.mean():+.5f}+-{se(pv):.5f} hold {ph.mean():+.5f}+-{se(ph):.5f}")
        RES[ref][tag] = dict(val=float(dv.mean()), val_se=float(se(dv)), se_cal_val=sec_v, hold=float(dh.mean()), hold_se=float(se(dh)), se_cal_hold=sec_h,
                             pmix=float(pm), pmix_se=float(pse), slices_d=ds, slices_z=zs, per_seed=per, rho_x=rho_x, rho_bb=rho_bb, margin=rho_bb - rho_x,
                             rho_aa=rho_aa, rho_same=rho_same, pool_val=float(pv.mean()), pool_hold=float(ph.mean()))
# run-noise read: the two references against each other (same args, different code copies / GPUs)
for tag in ("last", "sel"):
    dv = np.mean([ev(PV[(REFV["w9"](s), tag)]) - ev(PV[(REFV["abl"](s), tag)]) for s in SEEDS], 0)
    dh = np.mean([eh(PH[(REFV["w9"](s), tag)]) - eh(PH[(REFV["abl"](s), tag)]) for s in SEEDS], 0)
    P(f"[refs {tag}] w9_base - abl_base (same args, run noise): val {dv.mean():+.5f}+-{se(dv):.5f} | holdout {dh.mean():+.5f}+-{se(dh):.5f}")
    RES[f"refnoise_{tag}"] = dict(val=float(dv.mean()), val_se=float(se(dv)), hold=float(dh.mean()), hold_se=float(se(dh)))
r = RES["abl"]["last"]
gates = {"(s1) val > +1 SE_cal": r["val"] > r["se_cal_val"], "(s2) holdout >= 0": r["hold"] >= 0,
         "(s3) no val slice < -2 SE": min(r["slices_z"].values()) >= -2, "(s4) pmix >= -1 SE": r["pmix"] >= -r["pmix_se"]}
ok = all(gates.values())
P("gates (PRIMARY abl reference, LAST ckpt): " + "; ".join(f"{k} {'OK' if v else 'NO'}" for k, v in gates.items()))
r9 = RES["w9"]["last"]
P(f"replication reference w9 (information): val {r9['val']:+.5f} (z_cal {r9['val'] / r9['se_cal_val']:+.2f}), holdout {r9['hold']:+.5f}, pmix {r9['pmix']:+.5f}, "
  f"min slice z {min(r9['slices_z'].values()):+.1f} -> would be {'CONTINUE' if (r9['val'] > r9['se_cal_val'] and r9['hold'] >= 0 and min(r9['slices_z'].values()) >= -2 and r9['pmix'] >= -r9['pmix_se']) else 'STOP'}")
P(f"VERDICT stage S {ARM}: " + ("CONTINUE to stage E (10-epoch pair s951/952)" if ok else "STOP (futility)")
  + f" (val {r['val']:+.5f} +- {r['val_se']:.5f} [SE_cal {r['se_cal_val']:.5f}], holdout {r['hold']:+.5f} +- {r['hold_se']:.5f}, pmix {r['pmix']:+.5f}, margin {r['margin']:+.5f})")
P(f"registry: w10_{ARM}_s{'/'.join(map(str, SEEDS))} (A_BASE + HYG + --xoff 0.5, nn_x/w10/r2) vs abl_base (2-epoch pair, LAST ckpt, pre-declared stage-S rule 16:57:46Z): "
  f"val {r['val']:+.5f} +- {r['val_se']:.5f} (z_cal {r['val'] / r['se_cal_val']:+.1f}), holdout {r['hold']:+.5f} +- {r['hold_se']:.5f} (z {r['hold'] / r['hold_se']:+.1f}), "
  f"pmix {r['pmix']:+.5f}, " + " ".join(f"{n} {r['slices_d'][n]:+.5f}" for n in r["slices_d"]) + f", diversity margin {r['margin']:+.5f}, "
  f"pool val {r['pool_val']:+.5f} / hold {r['pool_hold']:+.5f}; vs w9_base val {r9['val']:+.5f} / hold {r9['hold']:+.5f}; selected ckpt val {RES['abl']['sel']['val']:+.5f} / hold {RES['abl']['sel']['hold']:+.5f} -> {'CONTINUE' if ok else 'STOP'} ({OUT})")
open(OUT, "w").write("\n".join(LINES) + "\n")
json.dump(dict(arm=ARM, seeds=SEEDS, pass_=bool(ok), gates={k: bool(v) for k, v in gates.items()}, res=RES), open(str(OUT)[:-4] + ".json", "w"), indent=1)
print(f"wrote {OUT}")
