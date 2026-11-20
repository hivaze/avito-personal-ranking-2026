"""STAGE-S read-out of the WAVE-9 arm CATGRAD (rule: nn_x/w5/logs/E_catgrad_predeclared.txt, frozen 2026-11-13T13:38:39Z).
2-epoch screen w9_catgrad_s23 / s24 vs w9_base_s23 / s24 (nn_x/w9/catgrad/preds), paired per seed, LAST checkpoint PRIMARY (@last preds),
selected checkpoint reported as information. CONTINUE iff all of (s1) strength, (s2) 2-epoch diversity margin, (s3) within-category
decorrelation, (s4) category-head strength (graft + category NDCG), (s5) within-part strength (graft), (s6) M<50 slice veto.
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 .venv/bin/python nn_x/w9/catgrad/w9_eval_s_catgrad.py [arm=catgrad] [base=base] [seeds=23,24]
      -> nn_x/w9/catgrad/logs/w9_eval_S_<arm>.{txt,json}
"""
import json
import os
import sys
import time

for v_ in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ.setdefault(v_, "12")
assert os.environ.get("CUDA_VISIBLE_DEVICES", "x") == "", "CPU only (read-out of saved predictions)"
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

sys.path.insert(0, "<repo>/src")
from common import FLAT, feed_meta  # noqa: E402
from metrics import ndcg_per_feed  # noqa: E402

ROOT = Path("<repo>")
ARM = sys.argv[1] if len(sys.argv) > 1 else "catgrad"
BASE = sys.argv[2] if len(sys.argv) > 2 else "base"
SEEDS = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "23,24").split(",")]
W9 = ROOT / "nn_x/w9/catgrad"
P9 = W9 / "preds"
OUT = W9 / "logs" / f"w9_eval_S_{ARM}.txt"
GATES = dict(s1_val=-0.0010, s1_hold=-0.0005, s2_margin=0.003, s3_within_drop=0.002, s4_graft_val=-0.0006, s4_catndcg=-0.002,
             s5_graft_val=-0.0008, s5_graft_hold=-0.0005, s6_m50_z=-2.0)
LINES = []


def P(s=""):
    print(s, flush=True)
    LINES.append(s)


class Split:
    """Per-split machinery: labels, (feed, category) groups, within-feed z, between/within split, chain-rule parts, category NDCG."""

    def __init__(self, name, M, off, y, cat):
        self.name = name
        self.M, self.off, self.y, self.cat = M.astype(np.int64), off.astype(np.int64), y.astype(np.float64), cat.astype(np.int64)
        self.n = len(M)
        self.fid = np.repeat(np.arange(self.n), self.M)
        key = self.fid * 64 + self.cat
        self.uk, self.gi = np.unique(key, return_inverse=True)
        self.gcnt = np.bincount(self.gi)
        self.gfeed = self.uk // 64
        self.G = len(self.uk)
        self.gy = (np.bincount(self.gi, self.y, self.G) > 0).astype(np.float64)  # group holds >= 1 positive
        self.Gn = np.bincount(self.gfeed, minlength=self.n)  # categories per feed
        self.goff = np.concatenate([[0], np.cumsum(self.Gn)[:-1]])
        self.mask2 = self.M[self.fid] >= 2
        self.single = (self.gcnt == 1)[self.gi]  # candidate sits alone in its (feed, category)

    def nd(self, s):
        return ndcg_per_feed(np.asarray(s, np.float64), self.y, self.off, self.M)

    def zs(self, s):
        s = np.asarray(s, np.float64)
        mu = np.repeat(np.bincount(self.fid, s, self.n) / self.M, self.M)
        d = s - mu
        sd = np.sqrt(np.repeat(np.bincount(self.fid, d * d, self.n) / self.M, self.M))
        return np.where(sd > 0, d / np.maximum(sd, 1e-12), 0.0)

    def split(self, z):
        b = (np.bincount(self.gi, z, self.G) / self.gcnt)[self.gi]
        return b, z - b

    def _lse(self, s, g, Gn):
        mx = np.full(Gn, -np.inf)
        np.maximum.at(mx, g, s)
        return np.log(np.bincount(g, np.exp(s - mx[g]), Gn)) + mx

    def parts(self, s):
        """(log P(cat(i)), log P(i | cat)) of the feed softmax of raw logits s (exact decomposition)."""
        s = np.asarray(s, np.float64)
        Lc = self._lse(s, self.gi, self.G)
        Lf = self._lse(Lc, self.gfeed, self.n)
        return (Lc - Lf[self.gfeed])[self.gi], s - Lc[self.gi], Lc - Lf[self.gfeed]

    def cat_ndcg(self, gscore):
        """Category-level NDCG per feed (catlib.cat_ndcg): categories ranked by gscore [G], relevance = holds >= 1 positive."""
        return ndcg_per_feed(np.asarray(gscore, np.float64), self.gy, self.goff, self.Gn)

    def corr(self, a, b, m=None):
        m = self.mask2 if m is None else m
        return float(np.corrcoef(a[m], b[m])[0, 1])


def holdout_split():
    """The 30K fixed holdout (train.py: rng 12345 over p_random > 0.9 & M >= 50 train feeds) with its category ids; checked against
    nn_x/w5/preds/holdout/holdout_{meta,y}.npy."""
    f = pl.read_parquet(FLAT / "train/feeds.parquet", columns=["M", "cand_off"])
    M = f["M"].to_numpy().astype(np.int64)
    coff = f["cand_off"].to_numpy().astype(np.int64)
    pr = np.load(ROOT / "eda/labels_position/out/train_prandom.npz")
    p = np.nan_to_num(pr["p_random"].astype(np.float32), nan=0.0)
    p[M < 50] = 0.0
    cand = np.where((p > 0.9) & (M >= 50))[0]
    hold = np.sort(np.random.default_rng(12345).choice(cand, 30000, replace=False))
    rows = np.concatenate([np.arange(o, o + m) for o, m in zip(coff[hold], M[hold])])
    Mh, oh = np.load(ROOT / "nn_x/w5/preds/holdout/holdout_meta.npy")
    yh = np.load(ROOT / "nn_x/w5/preds/holdout/holdout_y.npy")
    assert np.array_equal(Mh.astype(np.int64), M[hold]), "holdout meta mismatch (rng 12345 reconstruction)"
    cat = np.asarray(np.load(FLAT / "train/cand_category_id.npy", mmap_mode="r")[rows]).astype(np.int64)
    yt = np.asarray(np.load(FLAT / "train/cand_target.npy", mmap_mode="r")[rows]).astype(np.float32)
    assert np.array_equal(yt, yh.astype(np.float32)), "holdout labels mismatch"
    return Split("holdout", Mh.astype(np.int64), oh.astype(np.int64), yh, cat), hold


def main():
    t0 = time.time()
    fv, Mv, ov = feed_meta("val")
    yv = np.load(FLAT / "val/cand_target.npy")
    cv = np.load(FLAT / "val/cand_category_id.npy")
    V = Split("val", Mv, ov, yv, cv)
    H, hold = holdout_split()
    tue = (fv["timestamp"].to_numpy() + 10800) // 86400 < 20720
    SL = [("M<50", Mv < 50), ("M>=50", Mv >= 50), ("Tue", tue), ("Thu", ~tue)]
    se = lambda d: float(d.std() / np.sqrt(len(d)))  # noqa: E731
    P(f"== STAGE S read-out WAVE-9 {ARM} ({time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}): w9_{ARM}_s{SEEDS} vs w9_{BASE}_s{SEEDS}, "
      f"paired per seed, LAST checkpoint primary (selected ckpt = information). Rule: nn_x/w5/logs/E_catgrad_predeclared.txt (13:38:39Z): "
      f"gates {GATES}")
    P(f"val {V.n} feeds / {len(V.y)} cands / {V.G} (feed, category) groups; holdout {H.n} feeds / {len(H.y)} cands / {H.G} groups; "
      f"singleton-category positives: val {float(V.single[V.y > 0].mean()):.4f}, holdout {float(H.single[H.y > 0].mean()):.4f}; "
      f"val M<50 positives in singleton categories {float(V.single[(V.y > 0) & (Mv[V.fid] < 50)].mean()):.4f}")
    res = {}
    for view, suf in (("LAST", "@last"), ("SELECTED", "")):
        P(f"\n---- {view} checkpoint view ({'PRIMARY' if view == 'LAST' else 'information'})")
        pv, ph = {}, {}
        for s in SEEDS:
            for a in (BASE, ARM):
                pv[a, s] = np.load(P9 / f"nn_w9_{a}_s{s}{suf}_trainonly_val.npy").astype(np.float64)
                ph[a, s] = np.load(P9 / f"holdout/nn_w9_{a}_s{s}{suf}_holdout.npy").astype(np.float64)
        r = {}
        # (s1) strength + slices
        dv = [V.nd(pv[ARM, s]) - V.nd(pv[BASE, s]) for s in SEEDS]
        dh = [H.nd(ph[ARM, s]) - H.nd(ph[BASE, s]) for s in SEEDS]
        for i, s in enumerate(SEEDS):
            P(f"seed {s}: base val {V.nd(pv[BASE, s]).mean():.5f} hold {H.nd(ph[BASE, s]).mean():.5f} | {ARM} val {V.nd(pv[ARM, s]).mean():.5f} "
              f"hold {H.nd(ph[ARM, s]).mean():.5f} | d val {dv[i].mean():+.5f} d hold {dh[i].mean():+.5f}")
        dv, dh = np.mean(dv, 0), np.mean(dh, 0)
        r["d_val"], r["d_val_se"], r["d_hold"], r["d_hold_se"] = float(dv.mean()), se(dv), float(dh.mean()), se(dh)
        r["slices"] = {n: dict(d=float(dv[m].mean()), z=float(dv[m].mean() / max(se(dv[m]), 1e-12))) for n, m in SL}
        P(f"(s1) POOLED member: val {dv.mean():+.5f}+-{se(dv):.5f} (z {dv.mean() / se(dv):+.2f}) | holdout {dh.mean():+.5f}+-{se(dh):.5f} "
          f"(z {dh.mean() / se(dh):+.2f}) | slices " + " ".join(f"{n} {r['slices'][n]['d']:+.5f} (z {r['slices'][n]['z']:+.1f})" for n, _ in SL))
        # z-scores, between / within split
        Z = {k: V.zs(v) for k, v in pv.items()}
        S = {k: V.split(z) for k, z in Z.items()}
        s_, t_ = SEEDS
        rho_bb = V.corr(Z[BASE, s_], Z[BASE, t_])
        rho_x = 0.5 * (V.corr(Z[ARM, s_], Z[BASE, t_]) + V.corr(Z[ARM, t_], Z[BASE, s_]))
        w_bb = V.corr(S[BASE, s_][1], S[BASE, t_][1])
        w_x = 0.5 * (V.corr(S[ARM, s_][1], S[BASE, t_][1]) + V.corr(S[ARM, t_][1], S[BASE, s_][1]))
        b_bb = V.corr(S[BASE, s_][0], S[BASE, t_][0])
        b_x = 0.5 * (V.corr(S[ARM, s_][0], S[BASE, t_][0]) + V.corr(S[ARM, t_][0], S[BASE, s_][0]))
        r.update(rho_bb=rho_bb, rho_x=rho_x, margin=rho_bb - rho_x, within_bb=w_bb, within_x=w_x, within_drop=w_bb - w_x, between_bb=b_bb,
                 between_x=b_x, rho_aa=V.corr(Z[ARM, s_], Z[ARM, t_]), rho_same=float(np.mean([V.corr(Z[ARM, s], Z[BASE, s]) for s in SEEDS])))
        P(f"(s2) diversity margin: rho_bb {rho_bb:.5f} - rho_x {rho_x:.5f} = {rho_bb - rho_x:+.5f} (need >= +{GATES['s2_margin']}) | arm-arm {r['rho_aa']:.5f}, "
          f"same-seed arm-base {r['rho_same']:.5f}")
        P(f"(s3) within-category cross-seed corr: base-base {w_bb:.5f}, arm-base {w_x:.5f}, drop {w_bb - w_x:+.5f} (need >= +{GATES['s3_within_drop']}) | "
          f"between-category (info): base-base {b_bb:.5f}, arm-base {b_x:.5f}, drop {b_bb - b_x:+.5f}")
        # chain-rule grafts + category NDCG
        g1 = []  # catgrad category + base within   (s4)
        g2 = []  # base category + catgrad within   (s5)
        g2h = []
        cn = []
        for s in SEEDS:
            bc, bw, bg = V.parts(pv[BASE, s])
            ac, aw, ag = V.parts(pv[ARM, s])
            b0 = V.nd(pv[BASE, s])
            g1.append(V.nd(ac + bw) - b0)
            g2.append(V.nd(bc + aw) - b0)
            cn.append(V.cat_ndcg(ag) - V.cat_ndcg(bg))
            hbc, hbw, _ = H.parts(ph[BASE, s])
            hac, haw, _ = H.parts(ph[ARM, s])
            g2h.append(H.nd(hbc + haw) - H.nd(ph[BASE, s]))
        g1, g2, g2h, cn = np.mean(g1, 0), np.mean(g2, 0), np.mean(g2h, 0), np.mean(cn, 0)
        base_cat = float(np.mean([V.cat_ndcg(V.parts(pv[BASE, s])[2]).mean() for s in SEEDS]))
        r.update(graft_cat_val=float(g1.mean()), graft_cat_val_se=se(g1), cat_ndcg_d=float(cn.mean()), cat_ndcg_se=se(cn), base_cat_ndcg=base_cat,
                 graft_within_val=float(g2.mean()), graft_within_val_se=se(g2), graft_within_hold=float(g2h.mean()), graft_within_hold_se=se(g2h))
        P(f"(s4) category head: graft [{ARM} category + base within] - base: val {g1.mean():+.5f}+-{se(g1):.5f} (need >= {GATES['s4_graft_val']}) | "
          f"category NDCG {ARM} softmax(g) - base LSE: {cn.mean():+.5f}+-{se(cn):.5f} (need >= {GATES['s4_catndcg']}; base LSE category NDCG {base_cat:.4f})")
        P(f"(s5) within part: graft [base category + {ARM} within] - base: val {g2.mean():+.5f}+-{se(g2):.5f} (need >= {GATES['s5_graft_val']}) | "
          f"holdout {g2h.mean():+.5f}+-{se(g2h):.5f} (need >= {GATES['s5_graft_hold']})")
        P(f"(s6) M<50 slice: d {r['slices']['M<50']['d']:+.5f} z {r['slices']['M<50']['z']:+.2f} (need z >= {GATES['s6_m50_z']})")
        # information: 2-member pools, singleton positives
        pool_bb = V.nd(Z[BASE, s_] + Z[BASE, t_])
        pg = 0.5 * ((V.nd(Z[BASE, s_] + Z[ARM, t_]) - pool_bb).mean() + (V.nd(Z[BASE, t_] + Z[ARM, s_]) - pool_bb).mean())
        pool_aa = (V.nd(Z[ARM, s_] + Z[ARM, t_]) - pool_bb).mean()
        r.update(pool_mixed=float(pg), pool_arm=float(pool_aa))
        P(f"(info) 2-member pools vs base23+base24 ({pool_bb.mean():.5f}): mixed base_s + {ARM}_s' {pg:+.5f}; {ARM}23+{ARM}24 {pool_aa:+.5f}")
        gates = {"s1 val >= -0.0010": r["d_val"] >= GATES["s1_val"], "s1 holdout >= -0.0005": r["d_hold"] >= GATES["s1_hold"],
                 "s2 margin >= +0.003": r["margin"] >= GATES["s2_margin"], "s3 within drop >= +0.002": r["within_drop"] >= GATES["s3_within_drop"],
                 "s4 graft(cat) val >= -0.0006": r["graft_cat_val"] >= GATES["s4_graft_val"], "s4 catNDCG >= -0.002": r["cat_ndcg_d"] >= GATES["s4_catndcg"],
                 "s5 graft(within) val >= -0.0008": r["graft_within_val"] >= GATES["s5_graft_val"],
                 "s5 graft(within) holdout >= -0.0005": r["graft_within_hold"] >= GATES["s5_graft_hold"], "s6 M<50 z >= -2": r["slices"]["M<50"]["z"] >= GATES["s6_m50_z"]}
        r["gates"] = {k: bool(v) for k, v in gates.items()}
        r["continue"] = bool(all(gates.values()))
        P("gates: " + "; ".join(f"{k} {'OK' if v else 'NO'}" for k, v in gates.items()))
        P(f"{view} verdict: {'CONTINUE' if r['continue'] else 'CLOSED'}")
        res[view] = r
    # secondary reference: abl_base_s23/24 (information; different code lineage, selected ckpt)
    try:
        A = ROOT / "nn_x/abl/preds"
        dva = np.mean([V.nd(np.load(P9 / f"nn_w9_{ARM}_s{s}_trainonly_val.npy")) - V.nd(np.load(A / f"nn_abl_base_s{s}_trainonly_val.npy")) for s in SEEDS], 0)
        dvb = np.mean([V.nd(np.load(P9 / f"nn_w9_{BASE}_s{s}_trainonly_val.npy")) - V.nd(np.load(A / f"nn_abl_base_s{s}_trainonly_val.npy")) for s in SEEDS], 0)
        za = {s: V.zs(np.load(A / f"nn_abl_base_s{s}_trainonly_val.npy")) for s in SEEDS}
        zb = {s: V.zs(np.load(P9 / f"nn_w9_{BASE}_s{s}_trainonly_val.npy")) for s in SEEDS}
        P(f"\n(info) secondary reference abl_base_s23/24 (selected ckpt, nn_x/abl code): {ARM}(selected) - abl_base val {dva.mean():+.5f}+-{se(dva):.5f}; "
          f"w9_base(selected) - abl_base val {dvb.mean():+.5f}+-{se(dvb):.5f}; corr(w9_base_s, abl_base_s) same seed "
          f"{np.mean([V.corr(zb[s], za[s]) for s in SEEDS]):.5f}, abl base-base {V.corr(za[SEEDS[0]], za[SEEDS[1]]):.5f}")
        res["abl_ref"] = dict(arm_minus_abl=float(dva.mean()), w9base_minus_abl=float(dvb.mean()))
    except FileNotFoundError as e:
        P(f"(info) abl_base reference not available: {e}")
    # train minutes / learned head
    import torch
    for s in SEEDS:
        for a in (BASE, ARM):
            j = json.load(open(W9 / "logs" / f"nn_w9_{a}_s{s}.json".replace("nn_", "", 1)))
            msg = f"w9_{a}_s{s}: best_step {j['best_step']} val(selected) {j['val_ndcg']:.5f} train {j['train_minutes']:.1f} min"
            if a == ARM:
                ck = torch.load(W9 / "ckpt" / f"w9_{a}_s{s}@last.pt", map_location="cpu", weights_only=False)["state_dict"]
                msg += (f"; head: cg_cnt w {float(ck['cg_cnt.weight']):+.3f} b {float(ck['cg_cnt.bias']):+.3f}, |cg_cat| max {float(ck['cg_cat.weight'].abs().max()):.3f}, "
                        f"|cg_wide| max {float(ck['cg_wide.weight'].abs().max()):.3f}, |cg_q.3| max {float(ck['cg_q.3.weight'].abs().max()):.3f}")
            P("(info) " + msg)
    r = res["LAST"]
    P(f"\nVERDICT {ARM} (LAST ckpt, pre-declared stage-S rule): {'CONTINUE to stage E' if r['continue'] else 'CLOSED at stage S'} | selected-ckpt view: "
      f"{'CONTINUE' if res['SELECTED']['continue'] else 'CLOSED'}")
    P(f"registry: w9_{ARM}_s{'/'.join(map(str, SEEDS))} vs w9_{BASE} (2-epoch screen A_BASE + HYG, @last ckpt, rule E_catgrad_predeclared.txt 13:38:39Z): "
      f"val {r['d_val']:+.5f} +- {r['d_val_se']:.5f}, holdout {r['d_hold']:+.5f} +- {r['d_hold_se']:.5f}, margin {r['margin']:+.4f} (bb {r['rho_bb']:.4f}), "
      f"within drop {r['within_drop']:+.4f} (bb {r['within_bb']:.4f}), graft cat {r['graft_cat_val']:+.5f}, catNDCG {r['cat_ndcg_d']:+.5f}, graft within "
      f"{r['graft_within_val']:+.5f} / {r['graft_within_hold']:+.5f}, M<50 z {r['slices']['M<50']['z']:+.1f}, mixed 2-pool {r['pool_mixed']:+.5f} -> "
      f"{'CONTINUE' if r['continue'] else 'CLOSED'} ({OUT}); {time.time() - t0:.0f}s")
    open(OUT, "w").write("\n".join(LINES) + "\n")
    json.dump(dict(arm=ARM, base=BASE, seeds=SEEDS, gates=GATES, views=res, hold_feeds=int(len(hold))), open(str(OUT)[:-4] + ".json", "w"), indent=1)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
