"""Read-out of the nn_x/w5 WAVE-5 arm VALCD (val-in-cooldown, idea D1 of reports/EXTERNAL_TOP1_1113.md and
nn_x/audit/critique2/critique2.txt item 3; helpers cd_arms7.py; lane.sh B:valcd:<root>:<fold>). CPU only, read-only on every
prediction / log file; its only outputs are logs/w5_eval_valcd.{txt,json} and the assembled member files below.

  cd <repo> && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 .venv/bin/python nn_x/w5/w5_eval_valcd.py
        -> nn_x/w5/logs/w5_eval_valcd.{txt,json} (OUT=<path> overrides); member files nn_x/w5/preds/nn_w5_cd_valcd_r<r>[@last]_val_oof.npy
           (OOF val scores: the fold-k feeds scored by run k) and nn_w5_cd_valcd_r<r>[@last]_test.npy (mean over the 5 fold runs'
           per-feed z-scores of test), written only when all 5 fold runs of the root exist.
  ... w5_eval_valcd.py selftest   plumbing check with the m02ref@last files standing in for every fold run (no member files written;
           every delta vs m02ref and every in-sample gap must be exactly 0) -> logs/w5_eval_valcd_selftest.{txt,json}; verdict meaningless

RUNS: w5_cd_valcd_r<r>_f<k>, r in 101-104, k in 0-4 (20 runs; lane.sh B:valcd:<r>:<k> = B:m02ref:<r> + --val_mix 0.01 --val_fold
<k>; CDSEED 900 + r; the 2-epoch m02 re-cool of nn/ckpt/v17_pre2s_s<r>@77500.pt with the wave-5 hygiene flags). Run k trains
(in the cooldown, additive share 0.01 of the non-pool steps, ~193 val batches = ~3.8 passes) on the labelled val feeds of the
folds != k (nn/finetune.py's user-grouped GroupKFold(5), user-0 singletons, sizes 2885/2885/2885/2884/2884; cd_arms7.val_folds)
and NEVER on fold k. Reads: the LAST checkpoint (@last; primary, as every wave-5 read: the holdout judge is then unselected)
and the selected checkpoint (holdout-best at step >= 12k; secondary, info only).

PRE-DECLARED READ-OUT (frozen 2026-11-12 23:41 UTC, `date -u`, BEFORE any valcd run existed: only this code and its tests
existed; no valcd prediction had been produced or read):
  OOF member of root r = per feed, the val scores of the ONE run that never trained on that feed (fold-k feeds from run k);
    in-sample scores are never read for any gate. Test member of root r = mean over k of the per-feed z-scores of run k's test
    preds (the 5 fold models averaged, as nn/finetune.py does).
  References (same roots 101-104):
    (a) m02ref = w5_cd_m02ref_r<r>@last train-only val preds (same CDSEED 900+r, same re-cool, no val labels);
    (b) FT = the fine-tune branch of the r6 m02 member of the same root (a CDSEED 811 re-cool): per feed
        0.5 z(preds/nn_r6_cd_m02_r<r>_n2cpv3_val_oof.npy) + 0.5 z(preds/nn_r6_cd_m02_r<r>_trainonly_val.npy), i.e. the
        pipeline's "0.5 fine-tuned + 0.5 train-only" member; test likewise from the _n2cpv3_test / _trainonly_test files.
        Also reported: the r6 m02 train-only member alone (seed 811), so that (b) - (a) separates the CDSEED and FT effects.
  Pools (4 roots, member-equal per-feed z-mean): valcd pool = mean_r z(OOF_r); m02ref pool = mean_r z(m02ref_r);
    FT pool = mean_r [0.5 z(ft_r) + 0.5 z(to_r)]; r6 train-only pool = mean_r z(to_r). Every delta = valcd - reference,
    per-feed paired over the 14,423 val feeds, SE = sd / sqrt(n_feeds), z = delta / SE; slices M<50 / M>=50 / Tue / Thu /
    coldpos / warmpos (nn_x/r6/r6_eval.py definitions); pmix = src/pmix.py private-mix reweighting of the per-feed deltas
    (report + veto only, as everywhere in the project).
  Member level (reported): mean over roots of the per-feed paired deltas OOF_r - reference_r.
  Holdout (the fixed 30K random-like train feeds; the holdout never contains val feeds): member level = per root, the mean over
    the 5 fold runs of ndcg_feed(valcd_{r,k}@last) - ndcg_feed(m02ref_r@last), pooled over the 4 roots (per-feed SE); also the
    20-run z-mean vs the 4-run m02ref z-mean (ensemble, info).
  Forward check: the OOF pool delta vs the FT pool on the Tue feeds and on the Thu feeds separately (val = two exploration
    days; test is later: a gain present on both days is the forward-consistent one; reported, not gated beyond rule 4).
  In-sample diagnostic (reported): per run, NDCG on its 4 training folds vs m02ref on the same feeds (the memorisation gap);
    a large gap with a null OOF delta reads as over-fitting of the val labels, a null gap with a null OOF delta as a dose too
    small; the arm is NOT re-run on this evidence.
  PASS iff ALL of (last checkpoints, 4 roots complete):
    (1) 4-root OOF pool delta vs the FT pool >= +0.0005 AND its z > 2;
    (2) member-level holdout delta vs m02ref >= -1 SE;
    (3) pmix (private-mix) pool delta vs the FT pool >= -1 SE;
    (4) no val slice (M<50, M>=50, Tue, Thu, coldpos, warmpos) of the pool delta vs the FT pool < -2 SE.
  Otherwise KILL: the arm is dropped. The m02ref comparisons are reported but not gated (they mix the CDSEED-811-vs-900+r
  and the val-label effects). INCOMPLETE if any of the 20 runs or any reference file is missing (nothing is gated).
  Secondary (selected checkpoints): the same tables, info only.
"""
import json
import os
import re
import sys

assert os.environ.get("CUDA_VISIBLE_DEVICES", "x") == "", "CPU only (read-out of saved predictions)"
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

ROOT = "<repo>"
W5 = f"{ROOT}/nn_x/w5"
sys.path.insert(0, f"{ROOT}/nn_x/r6")
sys.path.insert(0, f"{ROOT}/src")
sys.path.insert(0, W5)
import r6_eval as R  # noqa: E402  (read-only import: val / holdout meta + slices, per-feed z, stats)
from metrics import ndcg_per_feed  # noqa: E402
from cd_arms7 import FOLD_SIZES, N_FOLDS, val_folds  # noqa: E402  (the very fold construction train.py uses)

R.DIRS["w5"] = (f"{W5}/preds", f"{W5}/logs")
P = R.P
ROOTS = (101, 102, 103, 104)
READS = (("last", "@last"), ("sel", ""))
GATE_MIN_DELTA = 0.0005
SELFTEST = len(sys.argv) > 1 and sys.argv[1] == "selftest"


def fz(s, ctx):
    return R.fz(np.asarray(s, np.float64), ctx["fid"], ctx["M"])


def pf(s, ctx):
    return ndcg_per_feed(np.asarray(s, np.float32), ctx["y"], ctx["off"], ctx["M"])


def st_line(name, d, slices):
    st = R.stats(d, slices)
    P(f"  {name:44s} {R.fmt(st)}")
    return st


def pmix_private(d):
    try:
        import pmix
        Wt = pmix.weights()
        m, se = pmix.wmean_se(d, Wt["private"])
        return dict(delta=float(m), se=float(se), z=float(m / se))
    except Exception as e:  # noqa: BLE001
        P(f"  PMIX unavailable ({type(e).__name__}: {e})")
        return None


def rows_of(feeds, off, M):
    return np.concatenate([np.arange(off[f], off[f] + M[f]) for f in feeds])


def valcd_paths(r, k, suf):
    if SELFTEST:  # stand-in: the m02ref@last files for every fold run (plumbing check)
        return dict(val=f"{W5}/preds/nn_w5_cd_m02ref_r{r}@last_trainonly_val.npy",
                    test=f"{W5}/preds/nn_w5_cd_m02ref_r{r}@last_trainonly_test.npy",
                    hold=f"{W5}/preds/holdout/nn_w5_cd_m02ref_r{r}@last_holdout.npy")
    return dict(val=f"{W5}/preds/nn_w5_cd_valcd_r{r}_f{k}{suf}_trainonly_val.npy",
                test=f"{W5}/preds/nn_w5_cd_valcd_r{r}_f{k}{suf}_trainonly_test.npy",
                hold=f"{W5}/preds/holdout/nn_w5_cd_valcd_r{r}_f{k}{suf}_holdout.npy")


def ref_paths(r):
    return dict(m02ref_val=f"{W5}/preds/nn_w5_cd_m02ref_r{r}@last_trainonly_val.npy",
                m02ref_test=f"{W5}/preds/nn_w5_cd_m02ref_r{r}@last_trainonly_test.npy",
                m02ref_hold=f"{W5}/preds/holdout/nn_w5_cd_m02ref_r{r}@last_holdout.npy",
                m02ref_sel_val=f"{W5}/preds/nn_w5_cd_m02ref_r{r}_trainonly_val.npy",
                m02ref_sel_hold=f"{W5}/preds/holdout/nn_w5_cd_m02ref_r{r}_holdout.npy",
                ft_oof=f"{ROOT}/preds/nn_r6_cd_m02_r{r}_n2cpv3_val_oof.npy", ft_test=f"{ROOT}/preds/nn_r6_cd_m02_r{r}_n2cpv3_test.npy",
                to_val=f"{ROOT}/preds/nn_r6_cd_m02_r{r}_trainonly_val.npy", to_test=f"{ROOT}/preds/nn_r6_cd_m02_r{r}_trainonly_test.npy")


def runs_block(suf):
    P(f"== runs ({'last' if suf else 'selected'} read): selected step | selected val (full, in-sample inflated) / holdout | last val / holdout | OOF, "
      f"in-sample NDCG at the last eval (log) | minutes")
    out = {}
    for r in ROOTS:
        for k in range(N_FOLDS):
            v = f"w5_cd_valcd_r{r}_f{k}"
            jf = f"{W5}/logs/{v}.json"
            if not os.path.exists(jf):
                P(f"  {v:24s} (missing)")
                continue
            j = json.load(open(jf))
            hh = j["history"]
            sel = [x for x in hh if x["step"] == j["best_step"]][0]
            lg = open(f"{W5}/logs/{v}.log").read().splitlines() if os.path.exists(f"{W5}/logs/{v}.log") else []
            vl = [ln for ln in lg if ln.startswith("wave-5 valcd [step ")]
            mo = re.search(r"OOF fold \d \(\d+ feeds\) ([\d.]+) \| in-sample [^ ]+ \(\d+ feeds\) ([\d.]+)", vl[-1]) if vl else None
            nb = re.search(r"val batches (\d+)", vl[-1]) if vl else None
            x = dict(best_step=j["best_step"], total=hh[-1]["step"], sel_val=sel["val_ndcg"], sel_hold=sel["holdout_ndcg"],
                     last_val=hh[-1]["val_ndcg"], last_hold=hh[-1]["holdout_ndcg"], minutes=j["train_minutes"],
                     oof_last=float(mo.group(1)) if mo else None, ins_last=float(mo.group(2)) if mo else None,
                     val_batches=int(nb.group(1)) if nb else None)
            out[v] = x
            P(f"  {v:24s} best {x['best_step']:>6}/{x['total']} | sel {x['sel_val']:.5f} / {x['sel_hold']:.5f} | last {x['last_val']:.5f} / "
              f"{x['last_hold']:.5f} | OOF {x['oof_last'] if x['oof_last'] is not None else float('nan'):.5f} in-sample "
              f"{x['ins_last'] if x['ins_last'] is not None else float('nan'):.5f} ({x['val_batches']} val batches) | {x['minutes']:.0f} min")
    return out


def read(suf, v, h, fold, te_meta):
    """One read (suf '@last' primary / '' selected). Returns the result dict (None if incomplete)."""
    tag = "LAST" if suf else "SELECTED"
    P(f"\n#### {tag} checkpoints ({'primary, gated' if suf else 'secondary, info only'})")
    M, off = v["M"], v["off"]
    fold_rows = [rows_of(np.flatnonzero(fold == k), off, M) for k in range(N_FOLDS)]
    members, miss = {}, []
    for r in ROOTS:
        ps = [valcd_paths(r, k, suf) for k in range(N_FOLDS)]
        rp = ref_paths(r)
        need = [p[x] for p in ps for x in ("val", "test", "hold")] + [rp["m02ref_val" if suf else "m02ref_sel_val"],
                                                                     rp["m02ref_hold" if suf else "m02ref_sel_hold"],
                                                                     rp["ft_oof"], rp["to_val"], rp["m02ref_test"], rp["ft_test"], rp["to_test"]]
        mm = [p for p in need if not os.path.exists(p)]
        if mm:
            miss += mm
            continue
        V = np.zeros(len(v["y"]), np.float32)
        ins, hold_d, Tz = [], [], None
        a_val = np.load(rp["m02ref_val" if suf else "m02ref_sel_val"])
        pa = pf(a_val, v)
        ha = pf(np.load(rp["m02ref_hold" if suf else "m02ref_sel_hold"]), h)
        for k in range(N_FOLDS):
            pk = np.load(ps[k]["val"])
            V[fold_rows[k]] = pk[fold_rows[k]]
            pk_f = pf(pk, v)
            ins.append(float((pk_f - pa)[fold != k].mean()))  # in-sample gap vs m02ref on the 4 training folds
            hold_d.append(pf(np.load(ps[k]["hold"]), h) - ha)
            tz = fz(np.load(ps[k]["test"]), te_meta)
            Tz = tz if Tz is None else Tz + tz
        Tz = (Tz / N_FOLDS).astype(np.float32)
        if not SELFTEST:
            np.save(f"{W5}/preds/nn_w5_cd_valcd_r{r}{suf}_val_oof.npy", V)
            np.save(f"{W5}/preds/nn_w5_cd_valcd_r{r}{suf}_test.npy", Tz)
        ft = 0.5 * fz(np.load(rp["ft_oof"]), v) + 0.5 * fz(np.load(rp["to_val"]), v)
        members[r] = dict(V=V, zV=fz(V, v), pV=pf(V, v), pA=pa, zA=fz(a_val, v), ft=ft, pFT=pf(ft, v),
                          to=fz(np.load(rp["to_val"]), v), ins=ins, hold_d=np.mean(hold_d, 0),
                          hold_zV=np.mean([fz(np.load(p["hold"]), h) for p in ps], 0), hold_zA=fz(np.load(rp["m02ref_hold" if suf else "m02ref_sel_hold"]), h))
        members[r]["pTO"] = pf(members[r]["to"], v)
    if miss:
        P(f"  INCOMPLETE: {len(miss)} files missing, e.g. {miss[:3]}")
    if not members:
        return None
    rs = sorted(members)
    P(f"  roots read: {rs}; member files written: preds/nn_w5_cd_valcd_r<r>{suf}_val_oof.npy / _test.npy")
    res = dict(roots=rs, missing=miss, member={}, pool={}, holdout={}, insample={})
    P("== member level (per root, OOF valcd - reference; val NDCG):")
    for r in rs:
        m = members[r]
        P(f"  r{r}: OOF {m['pV'].mean():.5f} | m02ref {m['pA'].mean():.5f} ({(m['pV'] - m['pA']).mean():+.5f}) | FT blend {m['pFT'].mean():.5f} "
          f"({(m['pV'] - m['pFT']).mean():+.5f}) | r6 train-only {m['pTO'].mean():.5f} ({(m['pV'] - m['pTO']).mean():+.5f}) | in-sample gap vs m02ref "
          f"by fold {np.round(m['ins'], 5).tolist()} | holdout vs m02ref {m['hold_d'].mean():+.5f}")
        res["member"][r] = dict(oof=float(m["pV"].mean()), m02ref=float(m["pA"].mean()), ft=float(m["pFT"].mean()), to=float(m["pTO"].mean()),
                                insample_gap=m["ins"], hold_delta=float(m["hold_d"].mean()))
    P("== member level pooled over roots (mean of the per-feed paired deltas):")
    dA = np.mean([members[r]["pV"] - members[r]["pA"] for r in rs], 0)
    dF = np.mean([members[r]["pV"] - members[r]["pFT"] for r in rs], 0)
    dT = np.mean([members[r]["pV"] - members[r]["pTO"] for r in rs], 0)
    res["member"]["pooled_vs_m02ref"] = st_line("valcd OOF - m02ref (member)", dA, v["slices"])
    res["member"]["pooled_vs_ft"] = st_line("valcd OOF - FT blend (member)", dF, v["slices"])
    res["member"]["pooled_vs_r6to"] = st_line("valcd OOF - r6 m02 train-only (member)", dT, v["slices"])
    P(f"== {len(rs)}-root pools (member-equal per-feed z-mean):")
    pV = pf(np.mean([members[r]["zV"] for r in rs], 0), v)
    pA = pf(np.mean([members[r]["zA"] for r in rs], 0), v)
    pF = pf(np.mean([members[r]["ft"] for r in rs], 0), v)
    pT = pf(np.mean([members[r]["to"] for r in rs], 0), v)
    P(f"  pool NDCG: valcd OOF {pV.mean():.5f} | m02ref {pA.mean():.5f} | FT {pF.mean():.5f} | r6 train-only {pT.mean():.5f}")
    res["pool"]["ndcg"] = dict(valcd=float(pV.mean()), m02ref=float(pA.mean()), ft=float(pF.mean()), r6to=float(pT.mean()))
    res["pool"]["vs_ft"] = st_line("POOL valcd - FT pool (GATED)", pV - pF, v["slices"])
    res["pool"]["vs_m02ref"] = st_line("POOL valcd - m02ref pool (info)", pV - pA, v["slices"])
    res["pool"]["vs_r6to"] = st_line("POOL valcd - r6 train-only pool (info)", pV - pT, v["slices"])
    res["pool"]["ft_vs_m02ref"] = st_line("POOL FT - m02ref (reference gap, info)", pF - pA, v["slices"])
    pm = pmix_private(pV - pF)
    pm2 = pmix_private(pV - pA)
    if pm:
        P(f"  PMIX private-mix: valcd - FT pool {pm['delta']:+.5f} +- {pm['se']:.5f} (z {pm['z']:+.2f}); valcd - m02ref pool "
          f"{pm2['delta']:+.5f} +- {pm2['se']:.5f} (z {pm2['z']:+.2f})")
    res["pool"]["pmix_vs_ft"], res["pool"]["pmix_vs_m02ref"] = pm, pm2
    fw = res["pool"]["vs_ft"]
    P(f"  FORWARD check (valcd - FT pool): Tue {fw['Tue']['delta']:+.5f} (z {fw['Tue']['z']:+.2f})  Thu {fw['Thu']['delta']:+.5f} "
      f"(z {fw['Thu']['z']:+.2f})  M<50 {fw['M<50']['delta']:+.5f} (z {fw['M<50']['z']:+.2f})  M>=50 {fw['M>=50']['delta']:+.5f} "
      f"(z {fw['M>=50']['z']:+.2f})")
    P("== holdout (30K random-like train feeds; never contains val):")
    hd = np.mean([members[r]["hold_d"] for r in rs], 0)
    res["holdout"]["member_vs_m02ref"] = st_line("HOLD valcd (mean of 5 folds) - m02ref (member, GATED)", hd, h["slices"])
    he = pf(np.mean([members[r]["hold_zV"] for r in rs], 0), h) - pf(np.mean([members[r]["hold_zA"] for r in rs], 0), h)
    res["holdout"]["ens_vs_m02ref"] = st_line(f"HOLD {5 * len(rs)}-run z-mean - {len(rs)}-run m02ref z-mean (info)", he, h["slices"])
    gaps = [g for r in rs for g in members[r]["ins"]]
    res["insample"] = dict(mean_gap_vs_m02ref=float(np.mean(gaps)), per_run=gaps)
    P(f"== in-sample diagnostic: mean NDCG gap vs m02ref on the 4 training folds of each run {np.mean(gaps):+.5f} (per run min "
      f"{min(gaps):+.5f} max {max(gaps):+.5f}); OOF member delta vs m02ref {dA.mean():+.5f}")
    if suf:
        complete = len(rs) == len(ROOTS) and not miss
        g = fw["all"]
        hm = res["holdout"]["member_vs_m02ref"]["all"]
        gates = {f"pool OOF - FT >= +{GATE_MIN_DELTA}": g["delta"] >= GATE_MIN_DELTA, "pool OOF - FT z > 2": g["z"] > 2,
                 "holdout member - m02ref >= -1 SE": hm["delta"] >= -hm["se"],
                 "pmix (private-mix) pool - FT >= -1 SE": pm is not None and pm["delta"] >= -pm["se"],
                 "no val slice (pool - FT) < -2 SE": all(s_["z"] >= -2 for kk, s_ in fw.items() if kk != "all")}
        verdict = "INCOMPLETE" if not complete else ("PASS" if all(gates.values()) else "KILL")
        P("  gates: " + "; ".join(f"{k} {'OK' if x else 'NO'}" for k, x in gates.items()))
        P(f"  ==> VALCD verdict ({len(rs)} roots, last checkpoints): {verdict}  [pool OOF - FT {g['delta']:+.5f} +- {g['se']:.5f} "
          f"(z {g['z']:+.2f}); holdout member {hm['delta']:+.5f} +- {hm['se']:.5f}; pmix {pm['delta'] if pm else float('nan'):+.5f}]")
        res.update(gates=gates, verdict=verdict, complete=complete)
    return res


def main():
    P(f"#### nn_x/w5 VALCD read-out (val-in-cooldown; w5_eval_valcd.py; rules frozen 2026-11-12 23:41 UTC)"
      + (" #### SELFTEST: m02ref@last stands in for every fold run; verdict meaningless" if SELFTEST else ""))
    v, h = R.setup("val"), R.setup("hold")
    fv = pl.read_parquet(f"{ROOT}/data/flat/val/feeds.parquet")
    fold = val_folds(fv["user_id"].to_numpy())
    sizes = tuple(int(x) for x in np.bincount(fold, minlength=N_FOLDS))
    assert sizes == FOLD_SIZES, sizes
    ft = pl.read_parquet(f"{ROOT}/data/flat/test/feeds.parquet", columns=["M"])
    Mt = ft["M"].to_numpy().astype(np.int64)
    te_meta = dict(M=Mt, fid=np.repeat(np.arange(len(Mt)), Mt))
    P(f"  val folds {sizes} (nn/finetune.py construction); roots {ROOTS}; reads: @last (primary), selected (info)")
    out = {"runs": runs_block("@last")}  # the run table reads the real valcd logs (empty in the selftest)
    for rd, suf in (READS[:1] if SELFTEST else READS):
        out[rd] = read(suf, v, h, fold, te_meta)
    if SELFTEST:
        r_ = out["last"]
        ok = (r_ is not None and abs(r_["pool"]["vs_m02ref"]["all"]["delta"]) == 0.0 and abs(r_["member"]["pooled_vs_m02ref"]["all"]["delta"]) == 0.0
              and abs(r_["holdout"]["member_vs_m02ref"]["all"]["delta"]) == 0.0 and r_["insample"]["mean_gap_vs_m02ref"] == 0.0
              and abs(r_["holdout"]["ens_vs_m02ref"]["all"]["delta"]) < 1e-9 and r_["pool"]["pmix_vs_m02ref"]["delta"] == 0.0)
        P(f"#### SELFTEST {'OK' if ok else 'FAILED'}: every delta vs m02ref and the in-sample gap are 0 with the stand-in files; "
          f"the FT-reference deltas and slices are the m02ref-vs-FT numbers")
        out["selftest_ok"] = ok
    P("\n== summary")
    for rd, suf in READS:
        r_ = out.get(rd)
        if r_ is None:
            P(f"  {rd}: no complete root")
            continue
        g = r_["pool"]["vs_ft"]["all"]
        a = r_["pool"]["vs_m02ref"]["all"]
        hm = r_["holdout"]["member_vs_m02ref"]["all"]
        pm = r_["pool"]["pmix_vs_ft"]
        P(f"  {rd:5s} roots {r_['roots']}: pool OOF - FT {g['delta']:+.5f} (z {g['z']:+.2f}) | pool OOF - m02ref {a['delta']:+.5f} (z {a['z']:+.2f}) | "
          f"holdout member {hm['delta']:+.5f} (z {hm['z']:+.2f}) | pmix - FT {pm['delta'] if pm else float('nan'):+.5f} | Tue "
          f"{r_['pool']['vs_ft']['Tue']['delta']:+.5f} Thu {r_['pool']['vs_ft']['Thu']['delta']:+.5f} | in-sample gap "
          f"{r_['insample']['mean_gap_vs_m02ref']:+.5f}" + (f" -> {r_['verdict']}" if "verdict" in r_ else ""))
    outp = os.environ.get("OUT") or f"{W5}/logs/w5_eval_valcd{'_selftest' if SELFTEST else ''}.txt"
    open(outp, "w").write("\n".join(R.OUT) + "\n")
    json.dump(out, open(re.sub(r"\.txt$", "", outp) + ".json", "w"), indent=1, default=float)
    print(f"wrote {outp}")


if __name__ == "__main__":
    main()
