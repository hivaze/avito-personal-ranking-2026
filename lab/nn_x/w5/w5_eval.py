"""Read-out of the nn_x/w5 WAVE-5 arms TRUNC-PROD and AUXF (reports/INSIGHTS_1112.md section 3.1 items 1, 3, 8).
CPU only, read-only on every prediction / log file (nn_x/r6/r6_eval.py is imported read-only for the paired per-feed
machinery: R.compare = per-feed mean of the paired NDCG deltas, SE = sd / sqrt(n_feeds)).

  cd <repo> && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 .venv/bin/python nn_x/w5/w5_eval.py A
        stage A: 2-epoch from-scratch screen (r7 BASE2 + --cooldown_mix 0.02), seeds 23/24, w5_{trunc,auxf}_s<s> vs
        w5_base_s<s>                                               -> nn_x/w5/logs/w5_eval_A.{txt,json}
  ... w5_eval.py B          stage B: CD re-cool of v17_pre2s_s<r>@77500 (2 epochs, m02, CDSEED 900+r), roots 101-104,
        w5_cd_{truncp,auxf}_r<r> vs w5_cd_m02ref_r<r>              -> nn_x/w5/logs/w5_eval_B.{txt,json}
  ... w5_eval.py B2 <arm>   stage 2 (roots 101-107; m02ref + arm on 105-107 must exist) -> logs/w5_eval_B2.{txt,json}
  ... w5_eval.py selftest   plumbing check on EXISTING runs (stage A: r7_dwa vs r7_base; stage B: r6_cd_rec33 vs
        r6_cd_m02, seed 811); verdicts printed there are meaningless       -> logs/w5_eval_selftest.txt

Runs (lane.sh): every run writes the SELECTED checkpoint (holdout-best among evals at step >= 12000, --select_min_step)
as <v> and the LAST checkpoint (step 19,670 in both stages) as <v>@last, each with train-only val preds and holdout preds
(preds/holdout/nn_<v>[@last]_holdout.npy, the fixed 30K random-like train holdout).

PRE-DECLARED RULES (frozen 2026-11-12 19:07 UTC, `date -u`, BEFORE any wave-5 run; no w5 training run existed, only this
code; CDSEED rule = 900 + root, INSIGHTS item 8):
  Primary read = the LAST checkpoint (k = 19,670 >= 12k for every run; INSIGHTS item 8: the holdout is then an unselected
      judge; the m02 families are end-rising). Secondary read (reported, never a gate) = the selected checkpoint
      (holdout-best at k >= 12k).
  All deltas are arm - reference on the same seed / root (B = arm), pooled = per-feed mean of the paired deltas.
  STAGE A (TRUNC-A: w5_trunc vs w5_base; AUXF-ii: w5_auxf vs w5_base; seeds 23/24):
      STOP iff pooled val delta < 0 OR pooled holdout delta < 0. Otherwise PASS-A (eligible for L1 per INSIGHTS:
      "If A or B passes, go to L1"; AUXF: L1 needs both (i) and (ii) to pass).
  STAGE B (TRUNC-B: w5_cd_truncp vs w5_cd_m02ref; AUXF-i: w5_cd_auxf vs w5_cd_m02ref; roots 101-104):
      STOP iff member val delta < 0 OR member holdout delta < 0 OR val M<50 slice delta < -1 SE OR private-mix (src/pmix.py)
      member delta < -1 SE OR stale-val member delta < -1 SE (stale-val = val re-predicted under the project's
      private-like 'both_pre2both' inputs, stale_pred.py, last checkpoints of both arms). If stale-val preds are missing
      and nothing else fires: PENDING. Otherwise CONTINUE -> stage 2 (roots 105-107, arm + m02ref, same CDSEED rule).
  STAGE 2 (B2, roots 101-107, last checkpoints): PASS-CD iff member val delta > 2 SE AND member holdout delta >= 0 AND
      7-member arm ensemble vs 7-member m02ref ensemble: holdout >= 0 and val >= -1 SE AND no val slice < -2 SE AND
      pmix >= -1 SE AND stale-val >= -1 SE (the wave-4 PASS-CD rule + the wave-5 vetoes).
  Diagnostics (reported, never gates): per-seed / per-root deltas; slices M<50 / M>=50 / Tue / Thu / coldpos / warmpos;
      SE_cal = sqrt(SE_feed^2 + sigma_run^2 / 2) (r7: sigma_run 0.00035 val, 0.00015 holdout) for stage A; the selected
      read; ensemble deltas; the per-positive-class lever table (nn_x/insights/errors 08_levers.py classes warm|cold x
      recent|mid|novel, + = arm reduces the loss) and the cat#1|w3+ loss block (05_mass.py: positive in G3's top
      category but at within-category rank >= 3); CLIPLOG (share of clipped plateau steps; close clipping iff < 20%;
      main / aux gradient-norm ratio by Mpad, the AUXF profile); m02ref (CDSEED 900+r) vs r6 m02 (seed 811), info.
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
import r6_eval as R  # noqa: E402  (read-only import of the r6 paired-evaluation helpers)

R.DIRS["w5"] = (f"{W5}/preds", f"{W5}/logs")
R.DIRS["r7"] = (f"{ROOT}/nn_x/r7/preds", f"{ROOT}/nn_x/r7/logs")
P = R.P
SIG_RUN = {"val": 0.00035, "hold": 0.00015}
SEEDS = (23, 24)
ROOTS1 = (101, 102, 103, 104)
ROOTS2 = (101, 102, 103, 104, 105, 106, 107)
# name maps (overridden by selftest): stage A base / arms, stage B reference / arms
CFG = dict(
    A_base=lambda s: f"w5:w5_base_s{s}", A_arm=lambda arm, s: f"w5:w5_{arm}_s{s}", A_arms=("trunc", "auxf"),
    B_ref=lambda r: f"w5:w5_cd_m02ref_r{r}", B_arm=lambda arm, r: f"w5:w5_cd_{arm}_r{r}", B_arms=("truncp", "auxf"),
    last="@last", tag="")


def L(n, read):
    """name of the read: primary 'last' -> <v>@last, secondary 'sel' -> <v>."""
    return n + CFG["last"] if read == "last" else n


def have(ctxs, names):
    return all(os.path.exists(c["path"](n)) for c in ctxs for n in names)


# ------------------------------------------------------------------------------------------------ helpers
def pooled(ctx, pairs):
    return R.compare(ctx, pairs)


def pmix_block(v, pairs):
    try:
        sys.path.insert(0, f"{ROOT}/src")
        import pmix  # noqa: E402
        Wt = pmix.weights()
    except Exception as e:  # noqa: BLE001
        P(f"  PMIX: unavailable ({type(e).__name__}: {e})")
        return None
    d = np.mean([R.per_feed(v, b) - R.per_feed(v, a) for a, b in pairs], 0)
    m, se = pmix.wmean_se(d, Wt["private"])
    P(f"  PMIX private-mix pooled delta {m:+.5f} +- {se:.5f} (z {m / se:+.2f}); < -1 SE: {'YES' if m < -se else 'no'}")
    return dict(delta=float(m), se=float(se), below_1se=bool(m < -se))


def stale_ctx():
    s = R.setup("val")
    s["path"] = lambda n: f"{R.split_name(n)[0][0]}/stale/nn_{R.split_name(n)[1]}_staleval.npy"  # stale_pred.py output
    s["cache"] = {}
    return s


def stale_block(sctx, pairs):
    miss = [n for p in pairs for n in p if not os.path.exists(sctx["path"](n))]
    if miss:
        P(f"  STALE-VAL: pending (missing {len(miss)} stale preds: {' '.join(R.split_name(n)[1] for n in miss)})")
        return None
    res = R.compare(sctx, pairs)["pooled"]
    a = res["all"]
    P(f"  STALE-VAL pooled delta (private-like inputs): {a['delta']:+.5f} +- {a['se']:.5f} (z {a['z']:+.2f}); "
      f"< -1 SE: {'YES' if a['delta'] < -a['se'] else 'no'}")
    return res


_E = {}


def errors_art(v):
    """Artifacts of nn_x/insights/errors (read-only): keep (non-leak feeds), cold / novelty per candidate, G3 classes."""
    if _E:
        return _E
    E = f"{ROOT}/nn_x/insights/errors/out"
    T = pl.read_parquet(f"{E}/feed_table.parquet")
    keep = ~T["leak"].to_numpy()
    y, fid, M, off = v["y"], v["fid"], v["M"], v["off"]
    assert len(keep) == len(M)
    minh = np.load(f"{E}/cand_cat_minh.npy")
    cold = np.load(f"{E}/cold_luo.npy")
    cat = np.load(f"{ROOT}/data/flat/val/cand_category_id.npy").astype(np.int64)
    z = np.asarray(np.load(f"{E}/cand_scores.npz")["G3"], dtype=np.float64)
    pi = np.flatnonzero(y > 0)
    npos = np.bincount(fid, y, len(M))
    idt = np.array([np.sum(1 / np.log2(np.arange(int(k)) + 2)) for k in range(0, 400)])
    nov = np.select([minh[pi] < 8, minh[pi] < 32], ["recent", "mid"], "novel")
    cls = np.char.add(np.char.add(np.where(cold[pi], "cold", "warm"), "|"), nov.astype(str))
    ck = fid * 64 + cat
    uk, inv = np.unique(ck, return_inverse=True)
    lse = np.log(np.bincount(inv, np.exp(z)))
    kf = uk // 64
    o = np.lexsort((-lse, kf))
    cr = np.empty(len(uk), np.int64)
    cr[o] = np.arange(len(uk)) - np.searchsorted(kf[o], kf[o])
    o2 = np.lexsort((y, -z, ck))
    wr = np.empty(len(z), np.int64)
    wr[o2] = np.arange(len(z)) - np.searchsorted(ck[o2], ck[o2])
    blk = (cr[inv][pi] == 0) & (wr[pi] >= 3)
    _E.update(keep=keep, pi=pi, fp=fid[pi], ID=idt[npos.astype(int)], cls=cls, blk=blk)
    return _E


def ppl(v, s, E):
    """exact per-positive loss (05_mass / 08_levers): positive with the j-th best pessimistic rank r owns
    (1/log2(j+2) - 1/log2(r+2)) / IDCG_f; aligned to E['pi']."""
    y, fid, M, off = v["y"], v["fid"], v["M"], v["off"]
    order = np.lexsort((y, -np.asarray(s, np.float64), fid))
    r = np.empty(len(order), np.int64)
    r[order] = np.arange(len(order)) - np.repeat(off, M)
    pi = E["pi"]
    o = np.lexsort((r[pi], fid[pi]))
    pis = pi[o]
    fp = fid[pis]
    j = np.arange(len(pis)) - np.searchsorted(fp, fp)
    out = np.empty(len(pi))
    out[o] = (1 / np.log2(j + 2) - 1 / np.log2(r[pis] + 2)) / E["ID"][fp]
    return out


def lever_block(v, pairs):
    """pooled loss reduction (ref loss - arm loss, mean over pairs) per positive class, per (non-leak) val feed."""
    try:
        E = errors_art(v)
    except Exception as e:  # noqa: BLE001
        P(f"  LEVERS: unavailable ({type(e).__name__}: {e})")
        return None
    d = np.mean([ppl(v, np.load(v["path"](a)), E) - ppl(v, np.load(v["path"](b)), E) for a, b in pairs], 0)
    keep, fp = E["keep"], E["fp"]
    kp = keep[fp]
    nf = keep.sum()
    n = len(keep)
    out = {}
    classes = ["warm|recent", "warm|mid", "warm|novel", "cold|recent", "cold|mid", "cold|novel"]
    for c in classes + ["cat#1|w3+", "total"]:
        m = kp & (E["cls"] == c if c in classes else (E["blk"] if c == "cat#1|w3+" else np.ones_like(kp)))
        fs = np.bincount(fp[m], d[m], n)[keep]
        out[c] = dict(delta=float(fs.sum() / nf), se=float(fs.std(ddof=1) / np.sqrt(nf) if nf > 1 else np.nan),
                      pos_share=float(m.sum() / kp.sum()))
    P("  LEVERS (pooled loss reduction per val feed, + = arm helps; feed-clustered SE; 08_levers classes / 05_mass block):")
    P("    " + "  ".join(f"{c} {o['delta']:+.5f}+-{o['se']:.5f} (pos {o['pos_share']:.3f})" for c, o in out.items()))
    return out


def cliplog_block(names):
    rows = {}
    for n in names:
        (pdir, ldir), v = R.split_name(n)
        f = f"{ldir}/{v}_cliplog.npz"
        if not os.path.exists(f):
            continue
        s = json.loads(str(np.load(f)["summary"]))
        rows[v] = s
        ph = s["phases"]
        bm = s["by_mpad"]
        prof = " ".join(f"{k}:{x['aux_over_main']:.2f}" for k, x in sorted(bm.items(), key=lambda t: int(t[0]))[::3])
        P(f"  CLIPLOG {v}: " + "; ".join(f"{k} clipped {x['clipped_share']:.3f} (median {x['median']:.3f})"
                                         for k, x in ph.items()) + f" | aux/main grad norm by Mpad {prof}")
    if rows:
        pl_ = [s["phases"].get("plateau", {}).get("clipped_share") for s in rows.values()]
        pl_ = [x for x in pl_ if x is not None]
        if pl_:
            P(f"  CLIPLOG decision (INSIGHTS item 1): max plateau clipped share {max(pl_):.3f} -> "
              + ("clipping CLOSED as an issue (< 20%)" if max(pl_) < 0.2 else ">= 20%: one CD arm with clip 5.0"))
    return rows


def runs_block(names):
    P("== runs: selected step / total | selected val, holdout (in-training eval) | last val, holdout | minutes")
    out = {}
    for n in names:
        (pdir, ldir), vn = R.split_name(n)
        if not os.path.exists(f"{ldir}/{vn}.json"):
            P(f"  {n:30s} (missing)")
            continue
        j = json.load(open(f"{ldir}/{vn}.json"))
        hh = j["history"]
        sel = [r for r in hh if r["step"] == j["best_step"]][0]
        x = dict(best_step=j["best_step"], total=hh[-1]["step"], sel_val=sel["val_ndcg"], sel_hold=sel["holdout_ndcg"],
                 final_val=hh[-1]["val_ndcg"], final_hold=hh[-1]["holdout_ndcg"], minutes=j["train_minutes"])
        out[n] = x
        P(f"  {n:30s} best {x['best_step']:>6}/{x['total']} | sel val {x['sel_val']:.5f} hold {x['sel_hold']:.5f} | "
          f"last val {x['final_val']:.5f} hold {x['final_hold']:.5f} | {x['minutes']:.0f} min")
    return out


def read_pairs(v, h, pairs, title, se_cal=False):
    rv, rh = pooled(v, pairs), pooled(h, pairs)
    for p in rv["pairs"]:
        P(f"    VAL  {R.split_name(p['B'])[1]} {p['B_mean']:.5f} vs {R.split_name(p['A'])[1]} {p['A_mean']:.5f}: "
          f"{p['slices']['all']['delta']:+.5f}")
    for p in rh["pairs"]:
        P(f"    HOLD {R.split_name(p['B'])[1]} {p['B_mean']:.5f} vs {R.split_name(p['A'])[1]} {p['A_mean']:.5f}: "
          f"{p['slices']['all']['delta']:+.5f}")
    P(f"  [{title}] VAL  pooled: {R.fmt(rv['pooled'])}")
    P(f"  [{title}] HOLD pooled: {R.fmt(rh['pooled'])}")
    if se_cal:
        for nm, r in (("val", rv), ("hold", rh)):
            a = r["pooled"]["all"]
            sc = np.sqrt(a["se"] ** 2 + SIG_RUN[nm] ** 2 / 2)
            a["se_cal"], a["z_cal"] = float(sc), float(a["delta"] / sc)
            P(f"  [{title}] {nm} SE_cal {sc:.5f} (z_cal {a['delta'] / sc:+.2f})")
    return rv, rh


def ens_block(v, h, refs, arms):
    out = {}
    for nm, ctx in (("VAL", v), ("HOLD", h)):
        e = R.compare(ctx, [("+".join(refs), "+".join(arms))])
        out[nm] = e["pooled"]["all"]
        P(f"  ENS {nm} {len(arms)}-member arm vs {len(refs)}-member ref: {R.fmt({'all': out[nm]})} "
          f"(ref {e['pairs'][0]['A_mean']:.5f}, arm {e['pairs'][0]['B_mean']:.5f})")
    return out


# ------------------------------------------------------------------------------------------------ stages
def stage_a(arms=None):
    arms = arms or CFG["A_arms"]
    v, h = R.setup("val"), R.setup("hold")
    P(f"== WAVE-5 STAGE A (2-epoch from-scratch, seeds {SEEDS}): STOP iff pooled val < 0 or pooled holdout < 0 vs base "
      f"(primary = last checkpoint)")
    base = CFG["A_base"]
    runs_block([base(s) for s in SEEDS] + [CFG["A_arm"](a, s) for a in arms for s in SEEDS])
    out = {}
    for arm in arms:
        pairs = {rd: [(L(base(s), rd), L(CFG["A_arm"](arm, s), rd)) for s in SEEDS] for rd in ("last", "sel")}
        if not have((v, h), [n for p in pairs["last"] for n in p]):
            P(f"\n== {arm}: incomplete (needs base + arm, seeds {SEEDS}, last-checkpoint val + holdout preds): not read")
            continue
        P(f"\n== {arm} vs base (B = arm)")
        rv, rh = read_pairs(v, h, pairs["last"], "last", se_cal=True)
        dv, dh = rv["pooled"]["all"]["delta"], rh["pooled"]["all"]["delta"]
        res = dict(val=rv["pooled"], hold=rh["pooled"], per_seed_val=[p["slices"]["all"]["delta"] for p in rv["pairs"]],
                   per_seed_hold=[p["slices"]["all"]["delta"] for p in rh["pairs"]])
        if have((v, h), [n for p in pairs["sel"] for n in p]):
            sv, sh = read_pairs(v, h, pairs["sel"], "selected (info)")
            res["sel"] = dict(val=sv["pooled"]["all"], hold=sh["pooled"]["all"])
        res["pmix"] = pmix_block(v, pairs["last"])
        res["levers"] = lever_block(v, pairs["last"])
        res["cliplog"] = cliplog_block([base(s) for s in SEEDS] + [CFG["A_arm"](arm, s) for s in SEEDS])
        stop = dv < 0 or dh < 0
        res["verdict"] = (f"STOP (val {dv:+.5f}, holdout {dh:+.5f})" if stop else
                          f"PASS-A (val {dv:+.5f}, holdout {dh:+.5f}) -> L1-eligible per INSIGHTS")
        res["stop"] = stop
        P(f"  ==> {arm}: val {dv:+.5f} (z {rv['pooled']['all']['z']:+.2f}, z_cal {rv['pooled']['all']['z_cal']:+.2f}), "
          f"holdout {dh:+.5f} (z {rh['pooled']['all']['z']:+.2f}) -> {res['verdict']}")
        out[arm] = res
    P("\n== summary (stage A, last checkpoints)")
    for arm, r in out.items():
        s = r["val"]
        P(f"  {arm:6s} val {s['all']['delta']:+.5f} hold {r['hold']['all']['delta']:+.5f} | M<50 {s['M<50']['delta']:+.5f} "
          f"M>=50 {s['M>=50']['delta']:+.5f} Tue {s['Tue']['delta']:+.5f} Thu {s['Thu']['delta']:+.5f} coldpos "
          f"{s['coldpos']['delta']:+.5f} | pmix {r['pmix']['delta'] if r['pmix'] else float('nan'):+.5f} -> {r['verdict']}")
    return out


def stage_b(arms=None, roots=ROOTS1, final=False):
    arms = arms or CFG["B_arms"]
    v, h = R.setup("val"), R.setup("hold")
    sctx = stale_ctx()
    ref = CFG["B_ref"]
    P(f"== WAVE-5 STAGE {'2 (PASS-CD)' if final else 'B'} (CD re-cool, roots {roots}, reference m02ref, primary = last "
      f"checkpoint): " + ("PASS-CD iff val > 2 SE, hold >= 0, ENS hold >= 0 & val >= -1 SE, no val slice < -2 SE, "
                          "pmix >= -1 SE, stale-val >= -1 SE" if final else
                          "STOP iff val < 0 or hold < 0 or M<50 < -1 SE or pmix < -1 SE or stale-val < -1 SE"))
    runs_block([ref(r) for r in roots] + [CFG["B_arm"](a, r) for a in arms for r in roots])
    out = {}
    for arm in arms:
        pairs = {rd: [(L(ref(r), rd), L(CFG["B_arm"](arm, r), rd)) for r in roots] for rd in ("last", "sel")}
        if not have((v, h), [n for p in pairs["last"] for n in p]):
            P(f"\n== {arm}: incomplete on roots {roots} (needs m02ref + arm last-checkpoint val + holdout preds): not read")
            continue
        P(f"\n== {arm} vs m02ref (B = arm)")
        rv, rh = read_pairs(v, h, pairs["last"], "last")
        sv, sh = rv["pooled"], rh["pooled"]
        res = dict(val=sv, hold=sh, per_root_val=[p["slices"]["all"]["delta"] for p in rv["pairs"]],
                   per_root_hold=[p["slices"]["all"]["delta"] for p in rh["pairs"]])
        if have((v, h), [n for p in pairs["sel"] for n in p]):
            xv, xh = read_pairs(v, h, pairs["sel"], "selected (info)")
            res["sel"] = dict(val=xv["pooled"]["all"], hold=xh["pooled"]["all"])
        res["ens"] = ens_block(v, h, [a for a, _ in pairs["last"]], [b for _, b in pairs["last"]])
        res["corr_same_root"] = [R.corr(v, a, b) for a, b in pairs["last"]]
        P("  same-root corr arm vs m02ref (val, within-feed z): " + " ".join(f"{x:.4f}" for x in res["corr_same_root"]))
        pm = pmix_block(v, pairs["last"])
        st = stale_block(sctx, pairs["last"])
        res["pmix"], res["stale"] = pm, st
        res["levers"] = lever_block(v, pairs["last"])
        res["cliplog"] = cliplog_block([ref(r) for r in roots[:1]] + [CFG["B_arm"](arm, r) for r in roots[:1]])
        dv, dh, m50 = sv["all"]["delta"], sh["all"]["delta"], sv["M<50"]
        pm_bad = pm is not None and pm["below_1se"]
        st_bad = st is not None and st["all"]["delta"] < -st["all"]["se"]
        if not final:
            fired = {"val < 0": dv < 0, "holdout < 0": dh < 0, "M<50 < -1 SE": m50["delta"] < -m50["se"],
                     "pmix < -1 SE": pm_bad, "stale-val < -1 SE": st_bad}
            stop = any(fired.values())
            pending = not stop and st is None
            verdict = ("STOP (" + ", ".join(k for k, x in fired.items() if x) + ")" if stop else
                       "PENDING (stale-val preds missing; nothing else fires)" if pending else
                       "CONTINUE -> stage 2 (roots 105-107: m02ref + arm, CDSEED 900+root)")
            res.update(fired=fired, stop=stop, pending=pending)
        else:
            ev, eh = res["ens"]["VAL"], res["ens"]["HOLD"]
            gates = {"member val > 2 SE": sv["all"]["z"] > 2, "member holdout >= 0": dh >= 0,
                     "ensemble holdout >= 0": eh["delta"] >= 0, "ensemble val >= -1 SE": ev["delta"] >= -ev["se"],
                     "no val slice < -2 SE": all(s_["z"] >= -2 for s_ in sv.values()),
                     "pmix >= -1 SE": not pm_bad, "stale-val >= -1 SE (available)": st is not None and not st_bad}
            verdict = "PASS-CD" if all(gates.values()) else "FAIL"
            P("  gates: " + "; ".join(f"{k} {'OK' if x else 'NO'}" for k, x in gates.items()))
            res["gates"] = gates
        res["verdict"] = verdict
        P(f"  Tue {sv['Tue']['delta']:+.5f}  Thu {sv['Thu']['delta']:+.5f}  M<50 {m50['delta']:+.5f}+-{m50['se']:.5f}  "
          f"M>=50 {sv['M>=50']['delta']:+.5f}  coldpos {sv['coldpos']['delta']:+.5f}")
        P(f"  ==> {arm}: member val {dv:+.5f} (z {sv['all']['z']:+.2f}), holdout {dh:+.5f} (z {sh['all']['z']:+.2f}) "
          f"-> {verdict}")
        out[arm] = res
    if not final and CFG["tag"] == "":  # info: the new m02ref (CDSEED 900+r) vs the r6 m02 reference (seed 811)
        old = [(f"r6:r6_cd_m02_r{r}", ref(r)) for r in roots]
        if have((v, h), [n for p in old for n in p]):
            P("\n== info: m02ref (CDSEED 900+root, selected) vs r6 m02 (seed 811, selected), member level")
            read_pairs(v, h, old, "m02ref - r6 m02 (info)")
    P(f"\n== summary (stage {'2' if final else 'B'}, last checkpoints)")
    for arm, r in out.items():
        P(f"  {arm:7s} val {r['val']['all']['delta']:+.5f} (z {r['val']['all']['z']:+.2f}) hold "
          f"{r['hold']['all']['delta']:+.5f} (z {r['hold']['all']['z']:+.2f}) M<50 {r['val']['M<50']['delta']:+.5f} "
          f"pmix {r['pmix']['delta'] if r['pmix'] else float('nan'):+.5f} stale "
          + (f"{r['stale']['all']['delta']:+.5f}" if r["stale"] else "pending") + f" -> {r['verdict']}")
    cont = [a for a, r in out.items() if not final and not r["stop"] and not r["pending"]]
    if cont:
        J2 = " ".join(f"B:{x}:{r}" for r in (105, 106, 107) for x in ["m02ref"] + cont)
        P(f"\n  stage-2 jobs: J2=\"{J2}\"; then w5_eval.py B2 {' '.join(cont)}")
    return out


def selftest():
    """Plumbing check on existing runs (no w5 run needed): stage A r7_dwa vs r7_base, stage B r6_cd_rec33 vs r6_cd_m02."""
    CFG.update(A_base=lambda s: f"r7:r7_base_s{s}", A_arm=lambda arm, s: f"r7:r7_{arm}_s{s}", A_arms=("dwa",),
               B_ref=lambda r: f"r6:r6_cd_m02_r{r}", B_arm=lambda arm, r: f"r6:r6_cd_{arm}_r{r}", B_arms=("rec33",),
               last="", tag="selftest")
    P("#### SELFTEST (stand-in runs; 'last' read = the stored selected checkpoint; verdicts meaningless)")
    a = stage_a()
    b = stage_b()
    ok = "dwa" in a and "rec33" in b and b["rec33"]["levers"] is not None and b["rec33"]["stale"] is not None
    P(f"#### SELFTEST {'OK' if ok else 'INCOMPLETE'}: stage A read {list(a)}, stage B read {list(b)}")
    return dict(A=a, B=b, ok=ok)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "A"
    assert mode in ("A", "B", "B2", "selftest"), "usage: w5_eval.py A | B | B2 <arm ...> | selftest"
    if mode == "A":
        res = stage_a(sys.argv[2:] or None)
    elif mode == "B":
        res = stage_b(sys.argv[2:] or None)
    elif mode == "B2":
        assert sys.argv[2:], "B2 needs the continuing arm(s)"
        res = stage_b(sys.argv[2:], ROOTS2, final=True)
    else:
        res = selftest()
    out = os.environ.get("OUT") or f"{W5}/logs/w5_eval_{mode}.txt"
    open(out, "w").write("\n".join(R.OUT) + "\n")
    json.dump(res, open(re.sub(r"\.txt$", "", out) + ".json", "w"), indent=1, default=float)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
