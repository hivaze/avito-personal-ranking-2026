"""CPU2: stale-val replay of pre2s (Phase A: 3-epoch pairs, Phase B: F7 vs F6 families, Phase C: sx_* GPU arms).

CPU only (CUDA_VISIBLE_DEVICES=""), 16 threads. Read-only on nn/ckpt, nn_x/solvers/ckpt, nn/feat_cache, data/feats,
preds/. Writes only to solvers/out/stale/{cf,replay}/ and solvers/out/CPU2_stale_replay_pre2s_and_arms/.

  python solvers/CPU2_stale_replay_pre2s_and_arms.py prep     # pre2both counterfactual files + pre-checks (a)(b)(c)
  python solvers/CPU2_stale_replay_pre2s_and_arms.py A        # Phase A replays + readout
  python solvers/CPU2_stale_replay_pre2s_and_arms.py B        # Phase B (needs v17_pre2s_s101..106)
  python solvers/CPU2_stale_replay_pre2s_and_arms.py C        # Phase C (needs nn_x/solvers/ckpt/sx_*.pt)

Scenarios (val inputs):
  base           live val inputs.
  both           review_shift 'both' popularity / co-vis at private-like coverage (it_hist_*_luo, seq_cv_{invd20,norm,pop}).
  both_pre2both  both + pre2both: newly_cold = (both_it_hist_all_luo == 0) & (val it_hist_all_luo > 0);
                 pre2_fe' = where(newly_cold, pre2_fe_noid, pre2_fe); pre2_fe_z' = within-feed z built exactly like the
                 live pre2_fe_z (per-config mean of z, see build_pre2both); pre2_fe_s' = (pre2_fe' + 20) / 6.
                 pre2_new*, pre2_new_s, pre2_hmax32 are NOT emulated (left at live values).

Readouts (paired per feed, A = baseline, B = treatment, pairs averaged per feed before the SE):
  base delta = B - A on base;  stale delta = B - A on the scenario;  diff-in-diff = stale delta - base delta.
  Slices: all, Tue, Thu, M<50, M>=50, all-cold-positive / all-warm-positive feeds (cold = it_hist_all_luo == 0 under
  the scenario's own popularity; base-defined cold also reported).

Pre-declared decision rules (from the task, applied exactly):
  (i)  pre2s 'staleness-fragile' if diff-in-diff < -2 SE AND stale delta < 0 (slice 'all').
  (ii) a GPU-arm winner is vetoed if its stale delta < -2 SE pooled over seeds 23/24.
  Valid only if pre-checks (a), (b), (d) pass; if (c) fails, both-only is primary.
"""
import glob
import json
import os
import sys
import time

for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"]:
    os.environ[k] = "16"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import polars as pl

ROOT = "<repo>"
sys.path.insert(0, f"{ROOT}/solvers/stale")
sys.path.insert(0, f"{ROOT}/src")
sys.path.insert(0, f"{ROOT}/pretrain")

CF_BOTH = "/tmp/scratchpad/review/shift/out/cf"
STALE = f"{ROOT}/solvers/out/stale"
CF = f"{STALE}/cf"
REP = f"{STALE}/replay"
OUT = f"{ROOT}/solvers/out/CPU2_stale_replay_pre2s_and_arms"
for d in (CF, REP, OUT):
    os.makedirs(d, exist_ok=True)
FEAT = f"{ROOT}/nn/feat_cache/val"
PRE_TAGS = [("v3At2", "v3t2"), ("v4At2", "v4t2")]  # pre2 val assembly: assemble.py --val v3At2,v4At2 --calib v3t2,v4t2

PAIRS_A = [("v8_nd_e3", "v18_pre2s_e3_s23"), ("v8_nd_e3_s42", "v18_pre2s_e3_s42")]
F6 = ["v13_lr1_mix1", "v14_s92", "v14_s93", "v14_s94", "v14_s95", "v14_s96"]
F7 = [f"v17_pre2s_s{s}" for s in range(101, 107)]
F7_EXTRA = ["v17_pre2s_s107"]  # pre-declared F7 in CURRENT_STATUS 3m 19:50 adds s107; reported as a sensitivity row


# ----------------------------------------------------------------------------------------------- meta
def val_meta():
    f = pl.read_parquet(f"{ROOT}/data/flat/val/feeds.parquet")
    M = f["M"].to_numpy().astype(np.int64)
    off = f["cand_off"].to_numpy().astype(np.int64)
    y = np.load(f"{ROOT}/data/flat/val/cand_target.npy").astype(np.float64)
    day = (f["timestamp"].to_numpy() + 3 * 3600) // 86400
    day = np.where(day <= day.min() + 1, 0, 1)  # same Tue/Thu split as nn/pcmp.py
    fid = np.repeat(np.arange(len(M)), M)
    return M, off, y, day, fid


def fz(s, fid, M):
    s = s.astype(np.float64)
    mu = np.bincount(fid, s, len(M)) / M
    sd = np.sqrt(np.maximum(np.bincount(fid, s * s, len(M)) / M - mu ** 2, 0)) + 1e-6
    return (s - mu[fid]) / sd[fid]


# ----------------------------------------------------------------------------------------------- scenarios
def scenarios():
    both = {n[len("both_"):-4]: f"{CF_BOTH}/{n}" for n in sorted(os.listdir(CF_BOTH)) if n.startswith("both_")}
    assert set(both) == {"it_hist_all_luo", "it_hist_et1_luo", "it_hist_et2_luo", "it_hist_et3_luo", "it_hist_et4_luo",
                         "seq_cv_invd20", "seq_cv_norm", "seq_cv_pop"}, both
    p2 = {k: f"{CF}/pre2both_{k}.npy" for k in ("pre2_fe", "pre2_fe_z", "pre2_fe_s")}
    # sensitivity only (NOT the pre-declared scenario): also emulate the 'new' head (pre2_new / _z / _s) with new_noid
    p3 = {k: f"{CF}/pre2full_{k}.npy" for k in ("pre2_new", "pre2_new_z", "pre2_new_s")}
    return {"base": {}, "both": both, "both_pre2both": {**both, **p2}, "both_pre2full": {**both, **p2, **p3}}


# ----------------------------------------------------------------------------------------------- prep
def build_pre2both():
    """pre2both counterfactual files + pre-checks (a), (b), (c). Returns the check dict."""
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    from stack_val import feed_rank_z
    from assemble import affine
    M, off, y, day, fid = val_meta()
    L = lambda n: np.load(f"{FEAT}/{n}.npy")  # noqa: E731  live files the models read (hard links of data/feats*)
    fe, fs, fzl = L("pre2_fe"), L("pre2_fe_s"), L("pre2_fe_z")
    idc = L("pre2_id_contrib")
    noid = np.load(f"{ROOT}/data/feats/val/pre2_fe_noid.npy")
    luo = L("it_hist_all_luo")
    bluo = np.load(f"{CF_BOTH}/both_it_hist_all_luo.npy")
    chk = {}
    # (a)
    chk["a_max_abs_fe_s_minus_(fe+20)/6"] = float(np.nanmax(np.abs(fs - (fe + 20) / 6)))
    chk["a_pass"] = bool(chk["a_max_abs_fe_s_minus_(fe+20)/6"] <= 1e-6)
    # (b) naive z of the config-mean pre2_fe vs the faithful builder (mean over configs of z of each aligned config)
    zn = feed_rank_z(off, M, fe.astype(np.float32))[:, 1]
    chk["b_naive_z(pre2_fe)_max_abs_diff"] = float(np.abs(zn - fzl).max())
    fes, noids, zs = [], [], []
    for t, c in PRE_TAGS:
        L0 = lambda k: np.load(f"{ROOT}/preds/pre_{t}_{k}_val.npy").astype(np.float32)  # noqa: E731
        R = lambda k: np.load(f"{ROOT}/preds/pre_{c}_{k}_val.npy").astype(np.float32)  # noqa: E731
        ffe, _ = affine(L0("fe"), R("fe"))
        fno, _ = affine(L0("fe_noid"), R("fe_noid"))
        fes.append(ffe(L0("fe")))
        noids.append(fno(L0("fe_noid")))
        zs.append(feed_rank_z(off, M, fes[-1])[:, 1])
    chk["b_builder_repro_pre2_fe_max_abs"] = float(np.abs(np.mean(fes, 0) - fe).max())
    chk["b_builder_repro_pre2_fe_noid_max_abs"] = float(np.abs(np.mean(noids, 0) - noid).max())
    chk["b_builder_repro_pre2_fe_z_max_abs"] = float(np.abs(np.mean(zs, 0) - fzl).max())
    chk["b_pass"] = bool(chk["b_builder_repro_pre2_fe_z_max_abs"] < 1e-4 and chk["b_builder_repro_pre2_fe_max_abs"] < 1e-5)
    # (c)
    c0 = idc[luo == 0] == 0
    c1 = idc[luo > 0] == 0
    has_id, warm = idc != 0, luo > 0
    chk["c_P(idc==0|luo==0)"] = float(c0.mean())
    chk["c_P(idc==0|luo>0)"] = float(c1.mean())
    chk["c_P(luo>0|idc!=0)"] = float(warm[has_id].mean())
    chk["c_jaccard(ID, LUO)"] = float((has_id & warm).sum() / (has_id | warm).sum())
    # pre-declared reading: the emulation needs "LUO-cold => ID-less"; overlap = share of ID-covered rows inside LUO
    # coverage = P(luo>0 | idc!=0) (equivalently P(idc==0 | luo==0) must be ~1). Jaccard reported as the symmetric read.
    chk["c_overlap_directional"] = chk["c_P(luo>0|idc!=0)"]
    chk["c_pass_directional"] = bool(chk["c_overlap_directional"] >= 0.80)
    chk["c_pass_symmetric"] = bool(chk["c_jaccard(ID, LUO)"] >= 0.80)
    # counterfactual
    nc = (bluo == 0) & (luo > 0)
    fe2 = np.where(nc, noid, fe).astype(np.float32)
    fes2 = [np.where(nc, n_, f_).astype(np.float32) for f_, n_ in zip(fes, noids)]
    z2 = np.mean([feed_rank_z(off, M, x)[:, 1] for x in fes2], 0).astype(np.float32)
    chk["cf_pre2_fe'_spec_vs_per_config_max_abs"] = float(np.abs(np.mean(fes2, 0) - fe2).max())
    chk["cf_naive_z(pre2_fe')_vs_builder_max_abs"] = float(np.abs(feed_rank_z(off, M, fe2)[:, 1] - z2).max())
    fs2 = ((fe2 + 20) / 6).astype(np.float32)
    chk["cf_newly_cold_share"] = float(nc.mean())
    chk["cf_newly_cold_with_id_share"] = float((idc[nc] != 0).mean())
    chk["cf_rows_changed"] = int((fe2 != fe).sum())
    chk["cf_mean_shift_changed_rows_fe"] = float((fe2 - fe)[fe2 != fe].mean())
    chk["cf_mean_shift_changed_rows_fe_z"] = float((z2 - fzl)[fe2 != fe].mean())
    chk["cf_abs_shift_unchanged_rows_fe_z_mean"] = float(np.abs(z2 - fzl)[fe2 == fe].mean())
    chk["cf_P5_noid_vs_fe_on_no_id_rows_abs_max"] = float(np.abs(fe - noid)[idc == 0].max())
    chk["cold_share_val"] = float((luo == 0).mean())
    chk["cold_share_both"] = float((bluo == 0).mean())
    np.save(f"{CF}/newly_cold.npy", nc)
    np.save(f"{CF}/pre2both_pre2_fe.npy", fe2)
    np.save(f"{CF}/pre2both_pre2_fe_z.npy", z2)
    np.save(f"{CF}/pre2both_pre2_fe_s.npy", fs2)
    json.dump(chk, open(f"{OUT}/prechecks_abc.json", "w"), indent=1)
    for k, v in chk.items():
        print(f"  {k}: {v}")
    return chk


def build_pre2full():
    """Sensitivity scenario (not pre-declared): emulate the un-emulated 'new' head too. Per config, the new-head
    score of a newly cold item is replaced by its no-ID score, aligned with the new head's own val->test map."""
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    from stack_val import feed_rank_z
    from assemble import affine
    M, off, y, day, fid = val_meta()
    L = lambda n: np.load(f"{FEAT}/{n}.npy")  # noqa: E731
    new, nz, ns, idc = L("pre2_new"), L("pre2_new_z"), L("pre2_new_s"), L("pre2_id_contrib")
    nc = np.load(f"{CF}/newly_cold.npy")
    chk = {"new_s_vs_(new+20)/6_max_abs": float(np.abs(ns - (new + 20) / 6).max())}
    news, news2, zs, zs2, raw_eq = [], [], [], [], []
    for t, c in PRE_TAGS:
        L0 = lambda k: np.load(f"{ROOT}/preds/pre_{t}_{k}_val.npy").astype(np.float32)  # noqa: E731
        R = lambda k: np.load(f"{ROOT}/preds/pre_{c}_{k}_val.npy").astype(np.float32)  # noqa: E731
        fn, _ = affine(L0("new"), R("new"))
        n1, n0 = fn(L0("new")), fn(L0("new_noid"))
        raw_eq.append(float(np.abs(L0("new") - L0("new_noid"))[idc == 0].max()))
        news.append(n1); zs.append(feed_rank_z(off, M, n1)[:, 1])
        n2 = np.where(nc, n0, n1).astype(np.float32)
        news2.append(n2); zs2.append(feed_rank_z(off, M, n2)[:, 1])
    chk["repro_new_max_abs"] = float(np.abs(np.mean(news, 0) - new).max())
    chk["repro_new_z_max_abs"] = float(np.abs(np.mean(zs, 0) - nz).max())
    chk["raw_new_minus_new_noid_on_no_id_rows_max_abs"] = raw_eq
    new2 = np.mean(news2, 0).astype(np.float32)
    z2 = np.mean(zs2, 0).astype(np.float32)
    ch = new2 != new
    chk["rows_changed"] = int(ch.sum())
    chk["mean_shift_changed_rows_new"] = float((new2 - new)[ch].mean())
    chk["mean_shift_changed_rows_new_z"] = float((z2 - nz)[ch].mean())
    np.save(f"{CF}/pre2full_pre2_new.npy", new2)
    np.save(f"{CF}/pre2full_pre2_new_z.npy", z2)
    np.save(f"{CF}/pre2full_pre2_new_s.npy", ((new2 + 20) / 6).astype(np.float32))
    json.dump(chk, open(f"{OUT}/prechecks_pre2full.json", "w"), indent=1)
    for k, v in chk.items():
        print(f"  {k}: {v}")


# ----------------------------------------------------------------------------------------------- replay
_DATA = {}


def ref_path(m):
    for p in (f"{ROOT}/preds/nn_{m}_trainonly_val.npy", f"{ROOT}/nn_x/solvers/preds/nn_{m}_trainonly_val.npy",
              f"{ROOT}/nn_x/solvers/preds/{m}_trainonly_val.npy"):
        if os.path.exists(p):
            return p
    return None


def replay(m, scen, log):
    """Cached CPU replay of member m on val under scenario scen. Returns float32 scores (cand-aligned)."""
    p = f"{REP}/cf_{m}_{scen}.npy"
    if os.path.exists(p):
        return np.load(p)
    import torch
    import run as R
    sc = scenarios()[scen]
    if "data" not in _DATA:
        import data as D
        _DATA["data"] = D.SplitData("val", torch.device("cpu"), max_hist=256)
    t0 = time.time()
    m_, ck = R.load(m)
    xt = ck["cfg"].get("xt_names") or []
    if scen.startswith("both_pre2") and not any("pre2" in n for n in xt):
        # the model reads no pre2 input -> identical inputs to 'both' (verified in Phase A: v8 replays identical)
        pr = replay(m, "both", log)
        np.save(p, pr)
        log.write(f"[alias] {m} {scen} := both (model reads no pre2 input)\n"); log.flush()
        return pr
    if scen != "base" and any(n.startswith(("prew", "has:prew")) for n in xt):
        # prew_* is NaN where it_hist_all_luo == 0 (pretrain/make_prew.py): mask it consistently with the scenario's
        # popularity, exactly as solvers/errors/nncpu/cf_pool.py did (review_shift prewboth_* files)
        sc = {**sc, **{n[len("prewboth_"):-4]: f"{CF_BOTH}/{n}" for n in os.listdir(CF_BOTH) if n.startswith("prewboth_")}}
    used = [k for k in sc if k in xt or f"has:{k}" in xt]
    pr, _ = R.run(m, "val", sc, data=_DATA["data"])
    pr = pr.astype(np.float32)
    np.save(p, pr)
    msg = f"[replay] {m} {scen}: {time.time() - t0:.0f}s, overridden inputs used by the model: {used}"
    if scen == "base":
        rp = ref_path(m)
        if rp:
            r = np.load(rp)
            msg += f"; corr with stored {os.path.basename(rp)} = {np.corrcoef(r, pr)[0, 1]:.7f}, max|diff| {np.abs(r - pr).max():.2e}"
    print(msg, flush=True)
    log.write(msg + "\n"); log.flush()
    return pr


def check_d(models, log):
    out = {}
    for m in models:
        pr = replay(m, "base", log)
        rp = ref_path(m)
        if rp is None:
            out[m] = {"corr": None, "pass": False, "ref": None}
            continue
        r = np.load(rp)
        c = float(np.corrcoef(r, pr)[0, 1])
        out[m] = {"corr": c, "max_abs_diff": float(np.abs(r - pr).max()), "pass": bool(c >= 0.99999), "ref": rp}
    return out


# ----------------------------------------------------------------------------------------------- readout
def ens(members, scen, fid, M, log):
    return np.mean([fz(replay(m, scen, log), fid, M) for m in members], 0)


def stats(d, mask):
    x = d[mask]
    return float(x.mean()), float(x.std() / np.sqrt(mask.sum())), int(mask.sum())


def coldpos_slices(y, cold, fid, M):
    pos = y > 0
    npos = np.bincount(fid, pos.astype(float), len(M))
    ncold = np.bincount(fid, (pos & cold).astype(float), len(M))
    return (npos > 0) & (ncold == npos), (npos > 0) & (ncold == 0)


def readout(pairs, scen_list, log, label):
    """pairs: list of (A_members, B_members). Per-feed deltas averaged over pairs, SE over feeds."""
    from metrics import ndcg_per_feed
    M, off, y, day, fid = val_meta()
    luo = np.load(f"{FEAT}/it_hist_all_luo.npy")
    bluo = np.load(f"{CF_BOTH}/both_it_hist_all_luo.npy")
    base_slices = {"all": np.ones(len(M), bool), "Tue": day == 0, "Thu": day == 1, "M<50": M < 50, "M>=50": M >= 50}
    cb, wb = coldpos_slices(y, luo == 0, fid, M)
    cs, ws = coldpos_slices(y, bluo == 0, fid, M)
    nd = {}
    for scen in ["base"] + scen_list:
        for A, B in pairs:
            for side in (A, B):
                key = ("+".join(side), scen)
                if key not in nd:
                    nd[key] = ndcg_per_feed(ens(side, scen, fid, M, log), y, off, M)
    res = {"label": label, "pairs": [["+".join(A), "+".join(B)] for A, B in pairs], "n_feeds": int(len(M)),
           "slice_sizes": {"allcold_pos(base)": int(cb.sum()), "allwarm_pos(base)": int(wb.sum()),
                           "allcold_pos(both)": int(cs.sum()), "allwarm_pos(both)": int(ws.sum())}}
    dbase = np.mean([nd[("+".join(B), "base")] - nd[("+".join(A), "base")] for A, B in pairs], 0)
    res["per_side_mean"] = {f"{k[0]}|{k[1]}": float(v.mean()) for k, v in nd.items()}
    res["per_pair"] = {}
    for A, B in pairs:
        a, b = "+".join(A), "+".join(B)
        res["per_pair"][f"{a}->{b}"] = {s: float((nd[(b, s)] - nd[(a, s)]).mean()) for s in ["base"] + scen_list}
        res["per_pair"][f"{a}->{b}"].update({f"stale_cost_{x}_{s}": float((nd[(x, s)] - nd[(x, 'base')]).mean())
                                             for s in scen_list for x in (a, b)})
    res["scen"] = {}
    for scen in scen_list:
        dst = np.mean([nd[("+".join(B), scen)] - nd[("+".join(A), scen)] for A, B in pairs], 0)
        did = dst - dbase
        r = {}
        sl_base = {**base_slices, "allcold_pos": cb, "allwarm_pos": wb}
        sl_stale = {**base_slices, "allcold_pos": cs, "allwarm_pos": ws}
        r["base_delta"] = {k: stats(dbase, m) for k, m in sl_base.items()}
        r["stale_delta"] = {k: stats(dst, m) for k, m in sl_stale.items()}
        r["did(stale-defined cold)"] = {k: stats(did, m) for k, m in sl_stale.items()}
        r["did(base-defined cold)"] = {k: stats(did, m) for k, m in {"allcold_pos": cb, "allwarm_pos": wb}.items()}
        r["base_delta(stale-defined cold)"] = {k: stats(dbase, m) for k, m in {"allcold_pos": cs, "allwarm_pos": ws}.items()}
        mu, se, _ = r["did(stale-defined cold)"]["all"]
        smu, sse, _ = r["stale_delta"]["all"]
        r["rule_i_flag"] = bool(mu < -2 * se and smu < 0)
        r["rule_ii_veto"] = bool(smu < -2 * sse)
        res["scen"][scen] = r
    return res


def fmt(res):
    lines = [f"== {res['label']}: pairs {res['pairs']}"]
    for k, v in res["per_pair"].items():
        lines.append(f"  pair {k}: " + ", ".join(f"{s} {x:+.5f}" for s, x in v.items()))
    for scen, r in res["scen"].items():
        lines.append(f"  -- scenario {scen}: rule(i) flag={r['rule_i_flag']}  rule(ii) veto={r['rule_ii_veto']}")
        for name in ("base_delta", "stale_delta", "did(stale-defined cold)", "did(base-defined cold)",
                     "base_delta(stale-defined cold)"):
            lines.append(f"     {name:32s} " + "  ".join(f"{k} {m:+.5f}+-{s:.5f}(n={n})" for k, (m, s, n) in r[name].items()))
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------------- phases
def phase(label, pairs, models, scen_list, extra=None):
    log = open(f"{OUT}/{label}.log", "a")
    log.write(f"\n### {time.strftime('%Y-%m-%d %H:%M:%S')} {label}\n")
    chk = json.load(open(f"{OUT}/prechecks_abc.json"))
    d = check_d(models, log)
    for m in models:
        for s in scen_list:
            replay(m, s, log)
    res = readout(pairs, scen_list, log, label)
    res["check_d"] = d
    res["check_d_pass"] = bool(all(v["pass"] for v in d.values()))
    res["checks_abd_pass"] = bool(chk["a_pass"] and chk["b_pass"] and res["check_d_pass"])
    res["check_c_directional"] = chk["c_pass_directional"]
    res["check_c_symmetric"] = chk["c_pass_symmetric"]
    if extra:
        res["extra"] = extra(log)
    json.dump(res, open(f"{OUT}/{label}.json", "w"), indent=1)
    txt = fmt(res) + f"\n  check (d): {json.dumps(d)}\n  checks a,b,d pass: {res['checks_abd_pass']}"
    print(txt, flush=True)
    log.write(txt + "\n")
    log.close()
    return res


F5 = ["v12_all_e10", "v12_all_s81", "v12_all_s82", "v12_allprew_s83", "v12_allprew_s84", "v12_allprew_s85"]
CFPOOL = "/tmp/scratchpad/solvers/errors/out/cf"


def pools(f7, log):
    """Supplementary (NOT pre-declared, information only): TO halves of the final-ensemble candidate sets,
    pool = equal-weight mean over families of the per-feed z of each family's member z-mean."""
    from metrics import ndcg_per_feed
    M, off, y, day, fid = val_meta()
    fam = {"F5": F5, "F6": F6, "F7": f7}
    out = {}
    for scen in ("base", "both_pre2both"):
        fz_fam = {k: fz(ens(v, scen, fid, M, log), fid, M) for k, v in fam.items()}
        P = {"F5": fz_fam["F5"], "F5+F6": (fz_fam["F5"] + fz_fam["F6"]) / 2,
             "F5+F6+F7": (fz_fam["F5"] + fz_fam["F6"] + fz_fam["F7"]) / 3, "F7": fz_fam["F7"], "F6": fz_fam["F6"]}
        out[scen] = {k: ndcg_per_feed(v, y, off, M) for k, v in P.items()}
    res = {}
    for a, b in [("F5", "F5+F6"), ("F5+F6", "F5+F6+F7"), ("F5", "F5+F6+F7"), ("F5", "F7")]:
        db = out["base"][b] - out["base"][a]
        ds = out["both_pre2both"][b] - out["both_pre2both"][a]
        r = {"base": stats(db, np.ones(len(M), bool)), "stale": stats(ds, np.ones(len(M), bool)),
             "did": stats(ds - db, np.ones(len(M), bool)),
             "level_base": [float(out["base"][a].mean()), float(out["base"][b].mean())],
             "level_stale": [float(out["both_pre2both"][a].mean()), float(out["both_pre2both"][b].mean())]}
        for nm, msk in (("M<50", M < 50), ("M>=50", M >= 50), ("Tue", day == 0), ("Thu", day == 1)):
            r[f"stale_{nm}"] = stats(ds, msk)
        res[f"{a} -> {b}"] = r
    # cross-check of this replay pipeline against the earlier cf_pool replays (same checkpoints, same 'both' inputs)
    xc = {}
    for m in F5 + ["v13_lr1_mix1"]:
        for s_ in ("base", "both"):
            q = f"{CFPOOL}/cf_{m}_{s_}.npy"
            if os.path.exists(q):
                xc[f"{m}|{s_}"] = float(np.corrcoef(np.load(q), replay(m, s_, log))[0, 1])
    res["crosscheck_corr_vs_cf_pool"] = xc
    return res


def avail(models):
    """Members whose final checkpoint AND stored train-only val preds exist (train.py writes the selected ckpt at
    the end of training, then the preds; requiring both, with ckpt not newer than preds, avoids replaying a
    checkpoint that is still being written)."""
    import run as R
    ok = []
    for m in models:
        try:
            cp = R.ckpt_path(m)
        except FileNotFoundError:
            continue
        rp = ref_path(m)
        if rp and os.path.getmtime(cp) <= os.path.getmtime(rp) + 1:
            ok.append(m)
    return ok


def main():
    cmd = sys.argv[1]
    S3 = ["both", "both_pre2both"]
    if cmd == "prep":
        build_pre2both()
    elif cmd == "prep_full":
        build_pre2full()
    elif cmd == "replay":  # replay <m1,m2,...> <scen1,scen2,...>   (cache warm-up)
        log = open(f"{OUT}/replay_warm.log", "a")
        for m in sys.argv[2].split(","):
            for s_ in sys.argv[3].split(","):
                replay(m, s_, log)
    elif cmd == "warm_f5":
        log = open(f"{OUT}/replay_warm.log", "a")
        for m in F5:
            for s_ in ("base", "both", "both_pre2both"):
                replay(m, s_, log)
        for m in F5 + ["v13_lr1_mix1"]:
            for s_ in ("base", "both"):
                q = f"{CFPOOL}/cf_{m}_{s_}.npy"
                print(m, s_, "corr vs cf_pool replay", np.corrcoef(np.load(q), replay(m, s_, log))[0, 1], flush=True)
    elif cmd == "A_full":  # sensitivity: Phase A pairs under the harsher (non-pre-declared) pre2full emulation
        models = [m for p in PAIRS_A for m in p]
        phase("phaseA_sens_pre2full", [([a], [b]) for a, b in PAIRS_A], models, ["both_pre2full"])
    elif cmd == "A":
        models = [m for p in PAIRS_A for m in p]
        phase("phaseA", [([a], [b]) for a, b in PAIRS_A], models, S3)
    elif cmd == "B":
        f7 = avail(F7)
        f7x = avail(F7_EXTRA)
        if len(f7) < len(F7):
            print("Phase B: missing F7 members:", sorted(set(F7) - set(f7)))
            if "--partial" not in sys.argv:
                return
        f6 = avail(F6)
        assert len(f6) == len(F6), f6
        # F6 members have no pre2 inputs -> both_pre2both == both for them (replayed anyway; the cache makes it cheap
        # only if identical inputs were used; we alias explicitly to save CPU and assert the input lists).
        import run as R
        for m in f6:
            xt = R.load(m)[1]["cfg"].get("xt_names") or []
            assert not any("pre2" in n for n in xt), (m, xt)
            src, dst = f"{REP}/cf_{m}_both.npy", f"{REP}/cf_{m}_both_pre2both.npy"
            if not os.path.exists(dst):
                log = open(f"{OUT}/phaseB.log", "a")
                replay(m, "both", log)
                np.save(dst, np.load(src))
                log.write(f"[alias] {m} both_pre2both := both (model reads no pre2 input)\n")
                log.close()
        tag = "phaseB" if len(f7) == len(F7) else f"phaseB_partial{len(f7)}"

        def extra(log):
            e = {"pools_supplementary": pools(f7, log)}
            if f7x:
                e["F7(+s107) vs F6"] = readout([(f6, f7 + f7x)], S3, log, "F7(+s107) vs F6")
            return e
        for m in f7x:
            for s in ["base"] + S3:
                replay(m, s, open(f"{OUT}/{tag}.log", "a"))
        phase(tag, [(f6, f7)], f6 + f7 + f7x, S3, extra)
    elif cmd == "C":
        import run as R
        cks = avail(sorted(os.path.basename(p)[:-3] for p in glob.glob(f"{ROOT}/nn_x/solvers/ckpt/sx_*.pt")
                           if "@" not in os.path.basename(p)))
        print("Phase C checkpoints:", cks)
        if not cks:
            return
        log = open(f"{OUT}/phaseC.log", "a")
        d = check_d(cks, log)
        for m in cks:
            replay(m, "both_pre2both", log)
        json.dump({"check_d": d, "models": cks}, open(f"{OUT}/phaseC_replays.json", "w"), indent=1)
        print(json.dumps(d, indent=1))
        # pairing (control vs arm per seed) is read from argv: C ctrl_s23:arm_s23,ctrl_s24:arm_s24 [label]
        if len(sys.argv) > 2:
            pairs = [tuple(p.split(":")) for p in sys.argv[2].split(",")]
            label = sys.argv[3] if len(sys.argv) > 3 else "phaseC_" + pairs[0][1]
            phase(label, [([a], [b]) for a, b in pairs], [m for p in pairs for m in p], ["both_pre2both"])


if __name__ == "__main__":
    main()
