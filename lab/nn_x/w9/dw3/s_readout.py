"""STAGE-S (2-epoch futility screen) read-out of the WAVE-9 arm DW3 (nn_x/w9/dw3; rule nn_x/w5/logs/E_dw3_predeclared.txt).
w9_<arm>_s23 / s24 paired per feed vs w9_base_s23 / s24 (same code copy, A_BASE + HYG). CPU only, read-only on the preds.
  cd <repo> && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 .venv/bin/python nn_x/w9/dw3/s_readout.py [arm=dw3] [seeds=23,24]
      -> nn_x/w9/dw3/logs/s_readout_<arm>.{txt,json}
Primary read = the LAST checkpoint (step 19,670; the 2-epoch screen convention of nn_x/w5 stage A and nn_x/abl: the holdout is
then an unselected judge); the selected checkpoint (holdout-best at step >= 12k) is reported. Pooled = per-feed mean of the
paired deltas over the seeds; SE_feed = sd / sqrt(n_feeds); SE_cal = sqrt(SE_feed^2 + sigma_run^2 / 2), sigma_run 0.00035 val /
0.00015 holdout (information). Slices: M<50, M>=50, Tue, Thu (own SE_feed). pmix = src/pmix.py private-mix delta.
Diversity at the member level (val, within-feed rank-z, Pearson): rho_x = mean(corr(arm_s23, base_s24), corr(arm_s24, base_s23)),
rho_bb = corr(base_s23, base_s24); margin = rho_bb - rho_x; also arm-arm, same-seed arm-base, the holdout margin, and the 2-member
pool arm vs base (val / holdout).
STAGE-S RULE (pre-declared): CONTINUE to stage E iff (s1) pooled val (last) >= -0.0010 AND (s2) pooled holdout (last) >= -0.0005 AND
(s3) no val slice < -2 SE_feed AND (s4) pmix >= -1 SE AND (s5) member margin rho_bb - rho_x >= +0.003 on val. Otherwise STOP.
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

ARM = sys.argv[1] if len(sys.argv) > 1 else "dw3"
SEEDS = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "23,24").split(",")]
D = Path("<repo>/nn_x/w9/dw3")
W = D / "preds"
OUT = D / "logs" / f"s_readout_{ARM}.txt"
SIG_V, SIG_H = 0.00035, 0.00015
LINES = []


def P(s):
    print(s, flush=True)
    LINES.append(s)


fv, Mv, ov = feed_meta("val")
yv = np.load(FLAT / "val/cand_target.npy")
Mh, oh = np.load(W / "holdout/holdout_meta.npy")
assert (np.load(W / "holdout/holdout_meta.npy") == np.load("<repo>/nn_x/w5/preds/holdout/holdout_meta.npy")).all(), "holdout meta != nn_x/w5"
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
A = lambda s: f"w9_{ARM}_s{s}"  # noqa: E731
B = lambda s: f"w9_base_s{s}"  # noqa: E731
P(f"== STAGE S read-out {ARM} ({time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}): {[A(s) for s in SEEDS]} vs {[B(s) for s in SEEDS]} "
  f"(nn_x/w9/dw3, 2-epoch A_BASE + HYG; LAST checkpoint primary, selected reported; rule nn_x/w5/logs/E_dw3_predeclared.txt stage S)")
PV, PH = {}, {}
for s in SEEDS:
    for v in (A(s), B(s)):
        for tag, suf in (("last", "@last"), ("sel", "")):
            PV[(v, tag)] = np.load(W / f"nn_{v}{suf}_trainonly_val.npy")
            PH[(v, tag)] = np.load(W / f"holdout/nn_{v}{suf}_holdout.npy")
RES = {}
for tag in ("last", "sel"):
    dv, dh, per = [], [], {}
    for s in SEEDS:
        eb, et, hb, ht = ev(PV[(B(s), tag)]), ev(PV[(A(s), tag)]), eh(PH[(B(s), tag)]), eh(PH[(A(s), tag)])
        dv.append(et - eb)
        dh.append(ht - hb)
        per[s] = dict(base_val=float(eb.mean()), arm_val=float(et.mean()), base_hold=float(hb.mean()), arm_hold=float(ht.mean()),
                      d_val=float((et - eb).mean()), d_hold=float((ht - hb).mean()))
        P(f"[{tag}] seed {s}: base val {eb.mean():.5f} hold {hb.mean():.5f} | {ARM} val {et.mean():.5f} hold {ht.mean():.5f} | d val {(et - eb).mean():+.5f} d hold {(ht - hb).mean():+.5f}")
    dv, dh = np.mean(dv, 0), np.mean(dh, 0)
    zs = {n: float(dv[m].mean() / max(se(dv[m]), 1e-9)) for n, m in SL}
    ds = {n: float(dv[m].mean()) for n, m in SL}
    pm, pse = pmix.wmean_se(dv, pmix.weights()["private"])
    sec_v, sec_h = float(np.sqrt(se(dv) ** 2 + SIG_V ** 2 / 2)), float(np.sqrt(se(dh) ** 2 + SIG_H ** 2 / 2))
    P(f"[{tag}] POOLED member: val {dv.mean():+.5f}+-{se(dv):.5f} (z {dv.mean() / se(dv):+.2f}; SE_cal {sec_v:.5f}) | holdout {dh.mean():+.5f}+-{se(dh):.5f} "
      f"(z {dh.mean() / se(dh):+.2f}; SE_cal {sec_h:.5f}) | pmix {pm:+.5f}+-{pse:.5f} (z {pm / pse:+.2f}) | slices "
      + " ".join(f"{n} {ds[n]:+.5f} (z {zs[n]:+.1f})" for n in zs))
    RES[tag] = dict(val=float(dv.mean()), val_se=float(se(dv)), hold=float(dh.mean()), hold_se=float(se(dh)), pmix=float(pm), pmix_se=float(pse),
                    slices_d=ds, slices_z=zs, per_seed=per, se_cal_val=sec_v, se_cal_hold=sec_h)
    # diversity (member level) + 2-member pool
    ZV = {v: zv(PV[(v, tag)]) for v in [A(s) for s in SEEDS] + [B(s) for s in SEEDS]}
    ZH = {v: zh(PH[(v, tag)]) for v in ZV}
    rho_x = float(np.mean([corr(ZV[A(SEEDS[0])], ZV[B(SEEDS[1])]), corr(ZV[A(SEEDS[1])], ZV[B(SEEDS[0])])]))
    rho_bb = corr(ZV[B(SEEDS[0])], ZV[B(SEEDS[1])])
    rho_aa = corr(ZV[A(SEEDS[0])], ZV[A(SEEDS[1])])
    rho_same = float(np.mean([corr(ZV[A(s)], ZV[B(s)]) for s in SEEDS]))
    rho_xh = float(np.mean([corr(ZH[A(SEEDS[0])], ZH[B(SEEDS[1])]), corr(ZH[A(SEEDS[1])], ZH[B(SEEDS[0])])]))
    rho_bbh = corr(ZH[B(SEEDS[0])], ZH[B(SEEDS[1])])
    pA, pB = np.mean([ZV[A(s)] for s in SEEDS], 0), np.mean([ZV[B(s)] for s in SEEDS], 0)
    hA, hB = np.mean([ZH[A(s)] for s in SEEDS], 0), np.mean([ZH[B(s)] for s in SEEDS], 0)
    pv, ph = ev(zv(pA)) - ev(zv(pB)), eh(zh(hA)) - eh(zh(hB))
    P(f"[{tag}] DIVERSITY (member, val rank-z): rho_x (arm-base cross-seed) {rho_x:.5f} | rho_bb (base-base) {rho_bb:.5f} | margin {rho_bb - rho_x:+.5f} | "
      f"arm-arm {rho_aa:.5f} | same-seed arm-base {rho_same:.5f} | holdout: rho_x {rho_xh:.5f} rho_bb {rho_bbh:.5f} margin {rho_bbh - rho_xh:+.5f}")
    P(f"[{tag}] 2-member POOL {ARM} vs base: val {pv.mean():+.5f}+-{se(pv):.5f} | holdout {ph.mean():+.5f}+-{se(ph):.5f}")
    RES[tag].update(rho_x=rho_x, rho_bb=rho_bb, margin=rho_bb - rho_x, rho_aa=rho_aa, rho_same=rho_same, rho_xh=rho_xh, rho_bbh=rho_bbh,
                    margin_h=rho_bbh - rho_xh, pool_val=float(pv.mean()), pool_hold=float(ph.mean()))
r = RES["last"]
gates = {"(s1) val >= -0.0010": r["val"] >= -0.0010, "(s2) holdout >= -0.0005": r["hold"] >= -0.0005,
         "(s3) no val slice < -2 SE": min(r["slices_z"].values()) >= -2, "(s4) pmix >= -1 SE": r["pmix"] >= -r["pmix_se"],
         "(s5) member margin >= +0.003": r["margin"] >= 0.003}
ok = all(gates.values())
P("gates (LAST ckpt): " + "; ".join(f"{k} {'OK' if v else 'NO'}" for k, v in gates.items()))
P(f"VERDICT stage S {ARM}: " + ("CONTINUE to stage E (10-epoch pair s951/952)" if ok else "STOP (futility)")
  + f" (val {r['val']:+.5f} +- {r['val_se']:.5f}, holdout {r['hold']:+.5f} +- {r['hold_se']:.5f}, margin {r['margin']:+.5f}, pmix {r['pmix']:+.5f})")
P(f"registry: w9_{ARM}_s{'/'.join(map(str, SEEDS))} vs w9_base (2-epoch A_BASE + HYG pair in nn_x/w9/dw3, LAST ckpt, pre-declared stage-S rule): "
  f"val {r['val']:+.5f} +- {r['val_se']:.5f} (z {r['val'] / r['val_se']:+.1f}), holdout {r['hold']:+.5f} +- {r['hold_se']:.5f} (z {r['hold'] / r['hold_se']:+.1f}), "
  f"pmix {r['pmix']:+.5f}, " + " ".join(f"{n} {r['slices_d'][n]:+.5f}" for n in r["slices_d"]) + f", member margin {r['margin']:+.5f} (rho_x {r['rho_x']:.4f} rho_bb {r['rho_bb']:.4f}), "
  f"pool val {r['pool_val']:+.5f} / hold {r['pool_hold']:+.5f}; selected ckpt val {RES['sel']['val']:+.5f} / hold {RES['sel']['hold']:+.5f} -> {'CONTINUE' if ok else 'STOP'} ({OUT})")
open(OUT, "w").write("\n".join(LINES) + "\n")
json.dump(dict(arm=ARM, seeds=SEEDS, pass_=bool(ok), gates={k: bool(v) for k, v in gates.items()}, res=RES), open(str(OUT)[:-4] + ".json", "w"), indent=1)
print(f"wrote {OUT}")
