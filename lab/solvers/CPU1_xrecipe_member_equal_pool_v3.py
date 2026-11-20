"""CPU1_xrecipe_member_equal_pool_v3: cross-recipe member-equal pools on the n2cpv3 fine-tune basis (CPU only, no training).

Pre-declared (task spec, sol_ensemble section 4 / CURRENT_STATUS 3k), nothing below is tuned on val:
  Families (TO = preds/nn_<m>_trainonly_{val,test}; FT = preds/nn_<m>_n2cpv3_{val_oof,test}, n2cpf only where n2cpv3 does not
  exist yet -> the pool is labelled 'info, not final'; holdout = preds/holdout/nn_<m>_holdout):
    F5   = v12_all_{e10,s81,s82}, v12_allprew_{s83,s84,s85}
    F6   = v13_lr1_mix1, v14_s92..s96
    F5cd = v15_cd_all_{e10,s81,s82}, v15_cd_allprew_{s83,s84,s85}
    H    = v13_lr1_mix0, v13_lr2_mix1, v14_mix1_s91   (v13_lr1_mix1 counted once, in F6; v13_lr1_mix3 collapsed, excluded)
    F7   = v17_pre2s_s101..s106 (phase B only; override with --f7)
  Strength gate: a family whose mean single-member TO NDCG is > 0.002 below the best family's on val OR on the holdout is
    excluded (dropped from every pool; pools that become identical collapse).
  Combiner: per-feed z of each member; member-equal mean inside each branch; final = 0.5 z(FT mean) + 0.5 z(TO mean).
  Pools: phase A Q1 = F5+F6, Q2 = Q1+F5cd, Q3 = Q2+H;  phase B Q1B = F5+F6+F7, Q3B = Q3+F7.
  Choice: inside a phase, the pool with the highest holdout TO-half ensemble NDCG (<= 3 choices, holdout only, never val).
  Compare the chosen pool by paired val vs R = 0.5 z(nn_final5v3_val_oof) + 0.5 z(nn_final5_trainonly_val) (sub_v12) and vs
    v11 = 0.5 z(nn_final5_val_oof) + 0.5 z(nn_final5_trainonly_val).
  Acceptance (frozen final-pick rule, CURRENT_STATUS 3k), vs both R and v11: paired val > 2 SE; no slice (M<50, M>=50, Tue,
    Thu) below -2 SE; holdout TO-half >= max(0.32162, R's holdout); presub_gate A-F PASS; file checks; FT halves all n2cpv3.

Nothing is fitted on val, so there is no nested CV to run; per-fold deltas over nn/finetune.py's GroupKFold(5)-by-user folds
are reported as a stability check. Writes only under solvers/out/CPU1_xrecipe_member_equal_pool_v3/ (never preds/ or subs/).

Run:  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=16 .venv/bin/python -u solvers/CPU1_xrecipe_member_equal_pool_v3.py --phase A
"""
import argparse
import contextlib
import io
import json
import os
import sys
import time

for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"]:
    os.environ[k] = "16"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np  # noqa: E402

sys.path.insert(0, "<repo>/solvers/ensemble")
from ens_common import BOUND, EDGES, HOLD, PREDS, ROOT, Holdout, Split, se  # noqa: E402

OUT = f"{ROOT}/solvers/out/CPU1_xrecipe_member_equal_pool_v3"
N_TEST = 2904806
FAM = {
    "F5": ["v12_all_e10", "v12_all_s81", "v12_all_s82", "v12_allprew_s83", "v12_allprew_s84", "v12_allprew_s85"],
    "F6": ["v13_lr1_mix1", "v14_s92", "v14_s93", "v14_s94", "v14_s95", "v14_s96"],
    "F5cd": ["v15_cd_all_e10", "v15_cd_all_s81", "v15_cd_all_s82", "v15_cd_allprew_s83", "v15_cd_allprew_s84",
             "v15_cd_allprew_s85"],
    "H": ["v13_lr1_mix0", "v13_lr2_mix1", "v14_mix1_s91"],
    "F7": [f"v17_pre2s_s{s}" for s in range(101, 107)],
}
POOLS = {"A": {"Q1": ["F5", "F6"], "Q2": ["F5", "F6", "F5cd"], "Q3": ["F5", "F6", "F5cd", "H"]},
         "B": {"Q1B": ["F5", "F6", "F7"], "Q3B": ["F5", "F6", "F5cd", "H", "F7"]}}
GATE_GAP = 0.002
HOLD_FLOOR = 0.32162


def log(*a):
    print(*a, flush=True)


def pfile(stem):
    return f"{PREDS}/{stem}.npy"


def ft_suffix(m):
    for suf in ("n2cpv3", "n2cpf"):
        if all(os.path.exists(pfile(f"nn_{m}_{suf}_{sp}")) for sp in ("val_oof", "test")):
            return suf
    return None


def member_ok(m):
    return (ft_suffix(m) is not None and all(os.path.exists(pfile(f"nn_{m}_trainonly_{sp}")) for sp in ("val", "test"))
            and os.path.exists(f"{HOLD}/nn_{m}_holdout.npy"))


class Bank:
    """Per-feed z of every member file, cached."""

    def __init__(self, Sv, St, Ho):
        self.S = {"val": Sv, "test": St}
        self.Ho = Ho
        self.c = {}

    def _get(self, key, path, S):
        if key not in self.c:
            x = np.load(path).astype(np.float64)
            assert len(x) == S.N and np.isfinite(x).all(), path
            self.c[key] = S.z(x)
        return self.c[key]

    def to(self, m, split):
        stem = f"nn_{m}_trainonly_{'val' if split == 'val' else 'test'}"
        return self._get(("to", m, split), pfile(stem), self.S[split])

    def ft(self, m, split):
        suf = ft_suffix(m)
        stem = f"nn_{m}_{suf}_{'val_oof' if split == 'val' else 'test'}"
        return self._get(("ft", m, split), pfile(stem), self.S[split])

    def ho(self, m):
        return self._get(("ho", m), f"{HOLD}/nn_{m}_holdout.npy", self.Ho)


def combine(S, ft_list, to_list):
    return S.z(0.5 * S.z(np.mean(ft_list, 0)) + 0.5 * S.z(np.mean(to_list, 0)))


def mh4_weights(Sv, St, kind):
    """M bucket (calib.py EDGES) x MSK hour bin {0-5, 6-11, 12-17, 18-23} cell weights val -> public/private (review_shift s06)."""
    hb = lambda ts: np.digitize(((ts + 3 * 3600) % 86400) // 3600, [6, 12, 18])  # noqa: E731
    mb = lambda M: np.searchsorted(EDGES, M, side="right")  # noqa: E731
    kv = mb(Sv.M) * 10 + hb(Sv.ts)
    mask = St.ts < BOUND if kind == "public" else St.ts >= BOUND
    kt = mb(St.M[mask]) * 10 + hb(St.ts[mask])
    pv = np.bincount(kv, minlength=110) / len(kv)
    pt = np.bincount(kt, minlength=110) / len(kt)
    w = (pt / np.maximum(pv, 1e-12))[kv]
    return w / w.mean(), float(pt[pv == 0].sum())


def pf_corr(S, a, b, rank=False):
    if rank:
        a, b = S.z(S.rank01(a)), S.z(S.rank01(b))
    else:
        a, b = S.z(a), S.z(b)
    return float((np.bincount(S.fid, a * b, S.n) / S.M).mean())


def ms(d, m=None):
    x = d if m is None else d[m]
    return [float(x.mean()), float(x.std() / np.sqrt(len(x)))]


def run_gate(tag, val_vec, test_vec, ref, members_by_suf):
    """presub_gate A-F on our own vectors: P is pointed at a link directory (no file under preds/ or subs/ is written)."""
    sys.path.insert(0, f"{ROOT}/src")
    from pathlib import Path

    import presub_gate as pg
    from make_sub import make_sub
    ld = f"{OUT}/gate_links"
    os.makedirs(ld, exist_ok=True)
    for stem in ["nn_final5v3_val_oof", "nn_final5_trainonly_val", "nn_final5v3_test", "nn_final5_trainonly_test",
                 "nn_final5_val_oof", "nn_final5_test"]:
        dst = f"{ld}/{stem}.npy"
        if not os.path.lexists(dst):
            os.symlink(pfile(stem), dst)
    np.save(f"{ld}/cand_{tag}_val.npy", val_vec.astype(np.float32))
    np.save(f"{ld}/cand_{tag}_test.npy", test_vec.astype(np.float32))
    sub = f"{OUT}/gate_{tag}.parquet"
    if not os.path.exists(sub):
        make_sub(test_vec.astype(np.float64), sub)
    refs = {"R": ("0.5*nn_final5v3_val_oof+0.5*nn_final5_trainonly_val", "0.5*nn_final5v3_test+0.5*nn_final5_trainonly_test"),
            "v11": ("0.5*nn_final5_val_oof+0.5*nn_final5_trainonly_val", "0.5*nn_final5_test+0.5*nn_final5_trainonly_test")}
    rv, rt = refs[ref]
    pg.P = Path(ld)
    pg.ST = {"FAIL": 0, "WARN": 0}
    sufs = sorted(members_by_suf)
    main_suf = "n2cpv3" if "n2cpv3" in members_by_suf else sufs[0]
    argv = ["presub_gate.py", "--sub", sub, "--val", f"1*cand_{tag}_val", "--test", f"1*cand_{tag}_test",
            "--ref_val", rv, "--ref_test", rt, "--members", ",".join(members_by_suf[main_suf]), "--ft_suffix", main_suf]
    buf = io.StringIO()
    old = sys.argv
    sys.argv = argv
    code = None
    with contextlib.redirect_stdout(buf):
        try:
            pg.main()
        except SystemExit as e:
            code = e.code
        for suf in sufs:
            if suf != main_suf:
                print(f"--- extra audit: members with fine-tune suffix {suf}")
                pg.audit_members(members_by_suf[suf], suf)
    sys.argv = old
    txt = buf.getvalue()
    if len(sufs) > 1:  # main() printed its verdict before the extra audit: recompute from the final counters
        verdict = "FAIL" if pg.ST["FAIL"] else ("PASS with WARN" if pg.ST["WARN"] else "PASS")
        txt += f"\nGATE (incl. extra audit): {verdict}  (FAIL {pg.ST['FAIL']}, WARN {pg.ST['WARN']})\n"
    else:
        verdict = "FAIL" if pg.ST["FAIL"] else ("PASS with WARN" if pg.ST["WARN"] else "PASS")
    open(f"{OUT}/gate_{tag}_vs_{ref}.log", "w").write(" ".join(argv) + "\n\n" + txt)
    warns = [ln for ln in txt.splitlines() if ln.startswith("[WARN]") or ln.startswith("[FAIL]")]
    return {"verdict": verdict, "fail": pg.ST["FAIL"], "warn": pg.ST["WARN"], "exit": code,
            "warn_lines": warns, "log": f"{OUT}/gate_{tag}_vs_{ref}.log"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="A", choices=["A", "B"])
    ap.add_argument("--f7", default=",".join(FAM["F7"]), help="F7 member list (phase B)")
    ap.add_argument("--no_gate", action="store_true")
    a = ap.parse_args()
    FAM["F7"] = [m for m in a.f7.split(",") if m]
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    rep = {"phase": a.phase, "families": {}, "members": {}, "pools": {}}
    fams = sorted({f for p in POOLS[a.phase].values() for f in p}, key=list(FAM).index)
    missing = {f: [m for m in FAM[f] if not member_ok(m)] for f in fams}
    if any(missing.values()):
        log("MISSING member files:", {k: v for k, v in missing.items() if v})
        if a.phase == "B":
            log("phase B needs every F7 member (TO, FT, holdout). Stop.")
            json.dump({"phase": "B", "missing": missing}, open(f"{OUT}/phase{a.phase}_missing.json", "w"), indent=1)
            return
        raise SystemExit(1)

    Sv, St, Ho = Split("val"), Split("test"), Holdout()
    B = Bank(Sv, St, Ho)
    cold = np.load(f"{ROOT}/eda/errors/synthesis/derived/val/cold_item.npy").astype(np.float64)
    npos = np.bincount(Sv.fid, Sv.y, Sv.n)
    ncp = np.bincount(Sv.fid, Sv.y * cold, Sv.n)
    assert (npos > 0).all()
    SL = {"all": np.ones(Sv.n, bool), "M<50": Sv.M < 50, "M>=50": Sv.M >= 50, "Tue": Sv.day == 0, "Thu": Sv.day == 1,
          "all-cold": ncp == npos, "all-warm": ncp == 0}
    W = {"priv_M": Sv.private_w("private"), "pub_M": Sv.private_w("public")}
    W["priv_MxH4"], miss_pr = mh4_weights(Sv, St, "private")
    W["pub_MxH4"], miss_pu = mh4_weights(Sv, St, "public")
    vw = np.load(f"{ROOT}/solvers/out/advval/val_weights.npz")
    W["priv_adv_intr"] = vw["adv_private_intrinsic"].astype(np.float64)
    W["pub_adv_intr"] = vw["adv_public_intrinsic"].astype(np.float64)
    assert np.allclose(vw["M_private"], W["priv_M"]), "advval M weights differ from ens_common's"
    rep["weights_note"] = {"MxH4_target_mass_without_val_cell": {"private": miss_pr, "public": miss_pu},
                           "ess": {k: float(w.sum() ** 2 / (w ** 2).sum()) for k, w in W.items()}}
    log(f"slices n: " + ", ".join(f"{k} {int(m.sum())}" for k, m in SL.items()), " | MxH4 missing mass", miss_pr, miss_pu)

    # ---------------- references
    zp = lambda S, stem: S.z(np.load(pfile(stem)).astype(np.float64))  # noqa: E731
    ref_v = {"R": Sv.z(0.5 * zp(Sv, "nn_final5v3_val_oof") + 0.5 * zp(Sv, "nn_final5_trainonly_val")),
             "v11": Sv.z(0.5 * zp(Sv, "nn_final5_val_oof") + 0.5 * zp(Sv, "nn_final5_trainonly_val"))}
    ref_t = {"R": St.z(0.5 * zp(St, "nn_final5v3_test") + 0.5 * zp(St, "nn_final5_trainonly_test")),
             "v11": St.z(0.5 * zp(St, "nn_final5_test") + 0.5 * zp(St, "nn_final5_trainonly_test"))}
    ref_pf = {k: Sv.per_feed(v) for k, v in ref_v.items()}
    # R reproduction from member files with our combiner (F5 only, n2cpv3)
    f5_chk = combine(Sv, [B.ft(m, "val") for m in FAM["F5"]], [B.to(m, "val") for m in FAM["F5"]])
    rep["check_R_from_members"] = [float(Sv.per_feed(f5_chk).mean()), float(ref_pf["R"].mean())]
    log(f"R val {ref_pf['R'].mean():.5f} (F5 n2cpv3 members via combiner {rep['check_R_from_members'][0]:.5f}); "
        f"v11 val {ref_pf['v11'].mean():.5f}")
    ho_ref = Ho.per_feed(Ho.z(np.mean([B.ho(m) for m in FAM["F5"]], 0)))  # R and v11 share the F5 train-only half
    rep["ref_holdout"] = float(ho_ref.mean())
    log(f"R/v11 holdout TO-half {ho_ref.mean():.5f}")

    # ---------------- members and strength gate
    for f in fams:
        rows = []
        for m in FAM[f]:
            r = {"family": f, "ft_suffix": ft_suffix(m), "to_val": float(Sv.per_feed(B.to(m, "val")).mean()),
                 "ft_val": float(Sv.per_feed(B.ft(m, "val")).mean()), "to_holdout": float(Ho.per_feed(B.ho(m)).mean())}
            rep["members"][m] = r
            rows.append(r)
            log(f"  {f:5s} {m:22s} FT={r['ft_suffix']:6s} TO val {r['to_val']:.5f}  FT val {r['ft_val']:.5f}  "
                f"TO holdout {r['to_holdout']:.5f}")
        rep["families"][f] = {"n": len(rows), "mean_to_val": float(np.mean([r["to_val"] for r in rows])),
                              "mean_to_holdout": float(np.mean([r["to_holdout"] for r in rows])),
                              "mean_ft_val": float(np.mean([r["ft_val"] for r in rows])),
                              "ft_all_n2cpv3": all(r["ft_suffix"] == "n2cpv3" for r in rows)}
    best_v = max(v["mean_to_val"] for v in rep["families"].values())
    best_h = max(v["mean_to_holdout"] for v in rep["families"].values())
    for f, v in rep["families"].items():
        v["gap_val"], v["gap_holdout"] = best_v - v["mean_to_val"], best_h - v["mean_to_holdout"]
        v["gate_pass"] = bool(v["gap_val"] <= GATE_GAP and v["gap_holdout"] <= GATE_GAP)
        log(f"gate {f:5s} n {v['n']}  mean TO val {v['mean_to_val']:.5f} (gap {v['gap_val']:.5f})  holdout "
            f"{v['mean_to_holdout']:.5f} (gap {v['gap_holdout']:.5f})  FT n2cpv3 {v['ft_all_n2cpv3']}  -> "
            f"{'PASS' if v['gate_pass'] else 'EXCLUDED'}")

    # ---------------- pools (gated), holdout choice
    gated, seen = {}, {}
    for p, fl in POOLS[a.phase].items():
        g = tuple(f for f in fl if rep["families"][f]["gate_pass"])
        if g in seen:
            log(f"pool {p} collapses onto {seen[g]} after the gate")
            rep["pools"][p] = {"families": list(fl), "gated_families": list(g), "collapsed_onto": seen[g]}
            continue
        seen[g] = p
        gated[p] = g
    for p, g in gated.items():
        mem = [m for f in g for m in FAM[f]]
        assert len(mem) == len(set(mem)), "duplicate member"
        sufs = {}
        for m in mem:
            sufs.setdefault(ft_suffix(m), []).append(m)
        final_ok = set(sufs) == {"n2cpv3"}
        hz = Ho.per_feed(Ho.z(np.mean([B.ho(m) for m in mem], 0)))
        hd = hz - ho_ref
        rep["pools"][p] = {"families": list(POOLS[a.phase][p]), "gated_families": list(g), "members": mem,
                           "n_members": len(mem), "ft_suffixes": {k: len(v) for k, v in sufs.items()},
                           "label": "final-eligible (all FT n2cpv3)" if final_ok else "info, not final (FT uses n2cpf)",
                           "holdout_to": float(hz.mean()), "holdout_delta_vs_R": ms(hd), "holdout_late20": ms(hd, Ho.late),
                           "_sufs": sufs}
        log(f"pool {p:4s} {'+'.join(g):18s} {len(mem):2d} members  FT {rep['pools'][p]['ft_suffixes']}  holdout TO-half "
            f"{hz.mean():.5f}  d {hd.mean():+.5f}+-{ms(hd)[1]:.5f}  late20 {ms(hd, Ho.late)[0]:+.5f}+-{ms(hd, Ho.late)[1]:.5f}")
    chosen = max(gated, key=lambda p: rep["pools"][p]["holdout_to"])
    rep["chosen"] = chosen
    log(f"CHOSEN on the holdout (phase {a.phase}): {chosen}")

    # ---------------- val / test for every pool (chosen = decision, others = info) + hybrid info row
    cand_v, cand_t = {}, {}
    for p in gated:
        mem = rep["pools"][p]["members"]
        cand_v[p] = combine(Sv, [B.ft(m, "val") for m in mem], [B.to(m, "val") for m in mem])
        cand_t[p] = combine(St, [B.ft(m, "test") for m in mem], [B.to(m, "test") for m in mem])
    hyb = {}
    for p in gated:
        mem = rep["pools"][p]["members"]
        v3 = [m for m in mem if ft_suffix(m) == "n2cpv3"]
        if 0 < len(v3) < len(mem):
            h = f"{p}h"
            hyb[h] = p
            cand_v[h] = combine(Sv, [B.ft(m, "val") for m in v3], [B.to(m, "val") for m in mem])
            cand_t[h] = combine(St, [B.ft(m, "test") for m in v3], [B.to(m, "test") for m in mem])
            rep["pools"][h] = {"members_ft": v3, "members_to": mem, "label": "info only, not pre-declared (FT branch = "
                               "n2cpv3 members only, TO branch = full pool)"}
    # TO-half-only val comparison (independent of the FT preset)
    to_ref = Sv.per_feed(Sv.z(np.mean([B.to(m, "val") for m in FAM["F5"]], 0)))
    for p in gated:
        tv = Sv.per_feed(Sv.z(np.mean([B.to(m, "val") for m in rep["pools"][p]["members"]], 0)))
        rep["pools"][p]["val_to_half"] = float(tv.mean())
        rep["pools"][p]["val_to_half_delta_vs_F5"] = ms(tv - to_ref)
    rep["val_to_half_F5"] = float(to_ref.mean())

    for p in cand_v:
        assert np.isfinite(cand_t[p]).all() and len(cand_t[p]) == N_TEST == St.N, p
        assert np.isfinite(cand_v[p]).all() and len(cand_v[p]) == Sv.N, p
        np.save(f"{OUT}/cand_{a.phase}_{p}_test.npy", cand_t[p].astype(np.float32))
        np.save(f"{OUT}/cand_{a.phase}_{p}_val.npy", cand_v[p].astype(np.float32))
        pv = Sv.per_feed(cand_v[p])
        r = rep["pools"][p]
        r["val"] = float(pv.mean())
        r["file_checks"] = {"finite": True, "len_test": int(len(cand_t[p])), "len_val": int(len(cand_v[p]))}
        r["calib"] = {k: float((W[k] * pv).mean()) for k in ["pub_M", "priv_M", "pub_MxH4", "priv_MxH4"]}
        r["vs"] = {}
        for ref in ["R", "v11"]:
            d = pv - ref_pf[ref]
            o = {s: ms(d, m) for s, m in SL.items()}
            for wk in ["priv_M", "priv_MxH4", "priv_adv_intr", "pub_adv_intr"]:
                o[wk] = [float(x) for x in se(d, W[wk])]
            o["folds"] = [float(d[Sv.fold == k].mean()) for k in range(5)]
            o["z"] = o["all"][0] / o["all"][1]
            o["delta_sd_per_feed"] = float(d.std())
            r["vs"][ref] = o
        r["test_corr"] = {ref: {"pearson_z": pf_corr(St, cand_t[p], ref_t[ref]),
                                "spearman": pf_corr(St, cand_t[p], ref_t[ref], rank=True)} for ref in ["R", "v11"]}
        r["val_spearman_vs_R"] = pf_corr(Sv, cand_v[p], ref_v["R"], rank=True)
        o = r["vs"]["R"]
        log(f"\n{p:5s} val {r['val']:.5f} [{r['label']}]")
        for ref in ["R", "v11"]:
            o = r["vs"][ref]
            log(f"   vs {ref:3s} " + "  ".join(f"{s} {o[s][0]:+.5f}+-{o[s][1]:.5f}" for s in SL) +
                f"  | privM {o['priv_M'][0]:+.5f}  privMxH4 {o['priv_MxH4'][0]:+.5f}  adv_intr priv {o['priv_adv_intr'][0]:+.5f}"
                f"+-{o['priv_adv_intr'][1]:.5f} pub {o['pub_adv_intr'][0]:+.5f}  folds " + " ".join(f"{x:+.4f}" for x in o["folds"]))
        log(f"   calib pub/priv M {r['calib']['pub_M']:.5f}/{r['calib']['priv_M']:.5f}  MxH4 {r['calib']['pub_MxH4']:.5f}/"
            f"{r['calib']['priv_MxH4']:.5f}  test corr vs v11 {r['test_corr']['v11']['pearson_z']:.4f}/"
            f"{r['test_corr']['v11']['spearman']:.4f}  vs R(v12) {r['test_corr']['R']['pearson_z']:.4f}/"
            f"{r['test_corr']['R']['spearman']:.4f} (Pearson z / Spearman); val Spearman vs R {r['val_spearman_vs_R']:.4f}")
    rep["refs"] = {k: {"val": float(v.mean()), "calib": {wk: float((W[wk] * v).mean()) for wk in ["pub_M", "priv_M", "pub_MxH4", "priv_MxH4"]}}
                   for k, v in ref_pf.items()}
    rep["refs_test_corr_R_v11"] = {"pearson_z": pf_corr(St, ref_t["R"], ref_t["v11"]),
                                   "spearman": pf_corr(St, ref_t["R"], ref_t["v11"], rank=True)}
    # pairwise test Spearman among pools (pick-2 rule)
    keys = list(cand_t)
    rep["test_spearman_pools"] = {f"{x}|{y}": pf_corr(St, cand_t[x], cand_t[y], rank=True)
                                  for i, x in enumerate(keys) for y in keys[i + 1:]}
    # vs nn_final5cd12 mix (existing frozen hedge), information
    cd12 = Sv.z(0.5 * zp(Sv, "nn_final5cd12_val_oof") + 0.5 * zp(Sv, "nn_final5cd12_trainonly_val"))
    rep["cd12_val"] = float(Sv.per_feed(cd12).mean())
    # phase B: also paired vs the phase-A holdout choice (information)
    pa = f"{OUT}/phaseA.json"
    if a.phase == "B" and os.path.exists(pa):
        ca = json.load(open(pa))["chosen"]
        va = Sv.per_feed(np.load(f"{OUT}/cand_A_{ca}_val.npy").astype(np.float64))
        rep["vs_phaseA_choice"] = {"phaseA_choice": ca}
        for p in cand_v:
            d = Sv.per_feed(cand_v[p]) - va
            rep["vs_phaseA_choice"][p] = {s: ms(d, m) for s, m in SL.items()}
            log(f"{p} vs phase-A choice {ca}: " + "  ".join(f"{s} {ms(d, m)[0]:+.5f}+-{ms(d, m)[1]:.5f}" for s, m in SL.items()))

    # ---------------- presub gate + frozen acceptance
    acc = {}
    for p in gated:
        r = rep["pools"][p]
        do_gate = (not a.no_gate) and (p == chosen)
        r["gate"] = {}
        for ref in ["R", "v11"]:
            if do_gate or ((not a.no_gate) and ref == "R"):
                r["gate"][ref] = run_gate(f"{a.phase}_{p}", cand_v[p], cand_t[p], ref, r["_sufs"])
                log(f"presub_gate {p} vs {ref}: {r['gate'][ref]['verdict']} (FAIL {r['gate'][ref]['fail']}, WARN "
                    f"{r['gate'][ref]['warn']}) -> {r['gate'][ref]['log']}")
        for ref in ["R", "v11"]:
            o = r["vs"][ref]
            crit = {"val_gt_2SE": o["all"][0] > 2 * o["all"][1],
                    "no_slice_below_-2SE": all(o[s][0] >= -2 * o[s][1] for s in ["M<50", "M>=50", "Tue", "Thu"]),
                    "holdout_ge_floor": r["holdout_to"] >= max(HOLD_FLOOR, rep["ref_holdout"]),
                    "presub_gate_pass": (r["gate"][ref]["verdict"].startswith("PASS") if ref in r["gate"] else "not run"),
                    "file_checks": True,
                    "ft_all_n2cpv3": set(r["_sufs"]) == {"n2cpv3"}}
            ok = all(v is True for v in crit.values())
            acc[f"{p}_vs_{ref}"] = {"criteria": crit, "pass": ok}
            log(f"ACCEPT {p} vs {ref}: " + "  ".join(f"{k}={v}" for k, v in crit.items()) + f"  => {'PASS' if ok else 'no'}")
    rep["acceptance"] = acc
    for p in gated:
        rep["pools"][p].pop("_sufs", None)
    rep["runtime_s"] = time.time() - t0
    json.dump(rep, open(f"{OUT}/phase{a.phase}.json", "w"), indent=1, default=float)
    log(f"\nwrote {OUT}/phase{a.phase}.json  ({rep['runtime_s']:.0f} s)")


if __name__ == "__main__":
    main()
