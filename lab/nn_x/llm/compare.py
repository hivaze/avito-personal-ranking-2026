"""Pre-declared A1 / A2 evaluation of the matched-data test (LLM arm L vs ranker arm B, control R).

  ../../.venv_llm/bin/python compare.py --L llm06_L1_s23 llm06_L1_s24 --B llmcmp_base_s23 llmcmp_base_s24 \
      --R llm06rand_L1_s23 --tag stageA

Inputs (train-only predictions, never val-fine-tuned):
  L / R : nn_x/llm/preds/<name>_trainonly_val.npy, nn_x/llm/preds/holdout/<name>_holdout.npy
  B     : nn_x/llm/baseline/preds/nn_<name>_trainonly_val.npy, nn_x/llm/baseline/preds/holdout/nn_<name>_holdout.npy
Seeds are paired by position (L[i] <-> B[i], same subset). Output: results/compare_<tag>.{json,txt}

Pre-launch amendments (reports/LLM_FEASIBILITY.md 5.2, 2026-11-12) encoded here:
  E1 (under-training guard), decided BEFORE compare and at most once:
      compare.py --e1_decide --L llm06_L1_s23 llm06_L1_s24      -> results/e1_decision.json (+ last line E1_EXTEND=0/1)
    reads ONLY the step + holdout_ndcg lines of logs/<L>.json. If for either seed the holdout-best evaluation is the
    final one AND the final evaluation beats the previous one, BOTH seeds are re-run once with --epochs 4 (variants
    <name with _e4 before _s<seed>>, arm E1 of run_matched.sh) and A1/A2 are judged on those runs. The decision file is
    written once and never re-decided. compare (default --e1 require) refuses to run without it and substitutes the
    4-epoch runs when it says extend.
    R23 gets the same guard only if the P4B exception would otherwise fire: then compare stops with decision
    PENDING and writes results/e1_R_decision.json (needed = true); after arm R23_E1 (4 epochs) compare is re-run and
    judges the exception with the 4-epoch control.
  V0: the L/R trainers never score val (train_llm.py --no_val_log); compare.py is the first reader of val labels.
  I3: compare.py --probe <probe dir> prints the probe gate (>= 25K padded tok/s, <= 30 GB) and the I3 decision;
      compare.py --i3_flag <probe dir> prints "--final_fp32" or "" for run_matched.sh (exit 2 if the probe is missing,
      exit 3 if the probe gate FAILED: every training arm of run_matched.sh calls it, so no arm runs after a failed gate).
  6 GPU-h hard cap: compare.py --gpu_hours <logs/gpu_ledger.tsv> --cap 6 sums the lane wall-clock hours of every GPU
      arm run_matched.sh recorded and exits 3 at/over the cap (checked before each GPU arm). With --project_e1 (arm E1,
      after the decision says extend) it also exits 3 unless BOTH 4-epoch runs fit (2 x the L23 + L24 lane-hours):
      one 4-epoch run without its pair is useless to compare.

Large-model arm sets (reports/llm_large.md, 2026-11-12; Qwen3.5-4B / 9B through qwen35_branch, same flags as L):
  compare.py --L llmq4_L1_s23 llmq4_L1_s24 --B llmcmp_base_s23 llmcmp_base_s24 --R llm06rand_L1_s23 --tag q4 \
      --e1 info --size qwen3.5-4b
    --e1 info: no E1 extension exists for these arms (pre-declared); each arm's holdout trace (still improving at
    the last evaluation?) is reported as information only. A1/A2 are computed by exactly the same code as Stage A.
    The P4B exception does not apply (it was the escalation to a 4B model); R (the 0.6B random-init control) is
    reported as information. Decision text names the size.
  compare.py --probe_large probe_q4: size-probe gate (peak reserved memory <= [n] GB AND estimated arm time <= 10 h),
    exit 3 on FAIL, exit 2 if missing; stdout = GB of free memory an arm of that size needs (peak reserved + 10).
  compare.py --curve 0.6B:stageA 4B:q4 9B:q9 --tag size_curve: size curve from the compare_<tag>.json files.

Rules (reports/llm_research.md 4.6, copied verbatim into run_matched.sh):
  A1: pooled paired train-only val L-B > 2 SE, both seeds > 0, holdout >= 0, no slice below -2 SE.
  A2: (i) L >= B - 0.002 on val AND on holdout; (ii) mean_s eq(L_s,B_s) - eq(B23,B24) > 0 on val with z >= 1.5 and
      holdout >= 0. Also report rho and the law-predicted gain 0.0186(1-rho) - 0.378|gap|.
  Decision: both fail -> STOP (exception: L >= B - 1 SE and L - R > 2 SE -> one Qwen3-4B-Instruct-2507 probe);
            A1 passes -> 600K slope check (continue only if the advantage keeps >= 50%);
            A2 passes (or slope check passes) -> full-data 0.6B members (1 epoch), then gates (a)+(b), frozen pool.
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

import data_llm as D

HERE = Path(__file__).resolve().parent
E1_RULE = ("E1: after L23 and L24 finish, read only their holdout lines; if for either seed the holdout-best evaluation "
           "is the final one and it beats the previous evaluation, re-run both seeds once with --epochs 4 and judge "
           "A1/A2 on those runs (at most one extension; R23 gets the same guard only if the P4B exception would "
           "otherwise fire)")
PROBE_GATE = dict(min_padded_tok_s=25000.0, max_peak_mem_gb=30.0)


def e4_name(name):
    """Variant name of the single pre-declared 4-epoch re-run: llm06_L1_s23 -> llm06_L1_e4_s23."""
    out = re.sub(r"_s(\d+)$", r"_e4_s\1", name)
    assert out != name, name
    return out


def holdout_trace(root, name):
    """E1 input: ONLY (step, holdout_ndcg) of every evaluation in logs/<name>.json (val is never read)."""
    j = json.loads((Path(root) / "logs" / f"{name}.json").read_text())
    h = [(int(e["step"]), float(e["holdout_ndcg"])) for e in j["history"]]
    assert h, f"{name}: no evaluations logged"
    vals = np.array([v for _, v in h])
    bi = int(np.argmax(vals))  # first maximum == the trainer's strict '>' selection
    final_is_best = bi == len(h) - 1
    beats_prev = len(h) < 2 or vals[-1] > vals[-2]  # one evaluation only: no evidence of a plateau -> improving
    return dict(n_evals=len(h), best_step=h[bi][0], final_step=h[-1][0], final=float(vals[-1]),
                previous=float(vals[-2]) if len(h) > 1 else None, final_is_best=bool(final_is_best),
                final_beats_previous=bool(beats_prev), still_improving=bool(final_is_best and beats_prev),
                trace=[[s_, round(v, 6)] for s_, v in h])


def e1_decide(root, L):
    """Decide E1 once from the L runs' holdout lines; the decision file is never overwritten."""
    root = Path(root)
    f = root / "results" / "e1_decision.json"
    if f.exists():
        dec = json.loads(f.read_text())
        assert dec["L"] == list(L), f"existing E1 decision is for {dec['L']}, not {list(L)}"
        print(f"E1 already decided at {dec['decided_utc']} (never re-decided): extend = {dec['extend']}")
        return dec
    for n in L:
        assert (root / "preds" / f"{n}_trainonly_val.npy").exists(), f"{n} has not finished; E1 runs after L23 and L24"
    seeds = {n: holdout_trace(root, n) for n in L}
    extend = any(v["still_improving"] for v in seeds.values())
    dec = dict(rule=E1_RULE, L=list(L), seeds=seeds, extend=bool(extend),
               L_judged=[e4_name(n) for n in L] if extend else list(L),
               decided_utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
    f.parent.mkdir(exist_ok=True)
    f.write_text(json.dumps(dec, indent=2))
    for n, v in seeds.items():
        print(f"E1 {n}: {v['n_evals']} holdout evals, best step {v['best_step']} / final {v['final_step']}, final "
              f"{v['final']:.5f} vs previous {v['previous']}, still improving = {v['still_improving']}")
    print(f"E1 decision: extend = {extend} -> judge A1/A2 on {dec['L_judged']}")
    return dec


def probe_report(probe_dir, quiet=False):
    """Probe gate (throughput, memory) + I3 decision from a finished probe run (run_matched.sh probe)."""
    d = Path(probe_dir)
    i3f, lf = d / "i3_precision.json", d / "logs" / "llm06_probe.json"
    if not (i3f.exists() and lf.exists()):
        return None
    i3, lj = json.loads(i3f.read_text()), json.loads(lf.read_text())
    tok, mem = float(lj["padded_tokens_per_s"]), lj.get("peak_mem_gb")
    gate = tok >= PROBE_GATE["min_padded_tok_s"] and mem is not None and mem == mem and mem <= PROBE_GATE["max_peak_mem_gb"]
    if not quiet:
        print(f"PROBE: {tok:.0f} padded tok/s (gate >= {PROBE_GATE['min_padded_tok_s']:.0f}), peak mem {mem} GB "
              f"(gate <= {PROBE_GATE['max_peak_mem_gb']:.0f}) -> PROBE GATE {'PASS' if gate else 'FAIL (run no arms)'}")
        print(f"I3: holdout NDCG {i3['ndcg_inference_path']:.6f} ({i3['inference_path']}) vs {i3['ndcg_fp32']:.6f} "
              f"(fp32 trunk), delta {i3['delta']:+.6f} (threshold {i3['threshold']:g}) -> final predictions in "
              f"{'FP32 (--final_fp32)' if i3['final_fp32'] else 'bf16 (as trained)'}")
    return dict(gate=gate, padded_tok_s=tok, peak_mem_gb=mem, i3=i3)


def pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def gpu_hours(ledger, now=None):
    """6 GPU-h hard cap (LLM_FEASIBILITY 5.3): lane wall-clock hours of every GPU arm recorded by run_matched.sh in
    logs/gpu_ledger.tsv (tab-separated lines: start|end, unix time, gpu, arm, lane pid). A finished arm counts
    end - start; an arm still running in a live lane counts now - start. An arm whose lane died without an 'end'
    line (SIGKILL) cannot be costed: it is returned in `stale` and the caller refuses until an 'end' line is added
    by hand. Wall-clock on a shared lane >= exclusive GPU time, so the cap binds early rather than late."""
    now = time.time() if now is None else now
    p = Path(ledger)
    open_, rows, stale, used = {}, [], [], 0.0
    if not p.exists():
        return used, rows, stale
    for ln in p.read_text().splitlines():
        if not ln.strip():
            continue
        kind, t, gpu, arm, pid = ln.split("\t")
        key = (gpu, arm, pid)
        if kind == "start":
            open_[key] = float(t)
        elif kind == "end" and key in open_:
            dt = (float(t) - open_.pop(key)) / 3600
            used += dt
            rows.append((arm, gpu, round(dt, 3)))
    for (gpu, arm, pid), t0 in open_.items():
        if pid_alive(int(pid)):
            dt = (now - t0) / 3600
            used += dt
            rows.append((f"{arm} (running)", gpu, round(dt, 3)))
        else:
            stale.append((arm, gpu, pid, time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(t0))))
    return used, rows, stale


def pf(s, y, off, M):
    return D.ndcg_per_feed(s, y, off, M)


def ms(d, m=None):
    d = d if m is None else d[m]
    return float(d.mean()), float(d.std() / np.sqrt(len(d)))


def rho(za, zb, M):
    fid = np.repeat(np.arange(len(M)), M)
    return float((np.bincount(fid, za * zb, len(M)) / M).mean())


def load_ctx(root):
    f = pl.read_parquet(D.FLAT / "val" / "feeds.parquet")
    M = f["M"].to_numpy().astype(np.int64)
    off = f["cand_off"].to_numpy().astype(np.int64)
    y = np.load(D.FLAT / "val" / "cand_target.npy").astype(np.float64)
    day = (f["timestamp"].to_numpy() + 3 * 3600) // 86400
    day = np.where(day <= day.min() + 1, 0, 1)  # 0 = Tue, 1 = Thu (as nn/pcmp.py)
    fid = np.repeat(np.arange(len(M)), M)
    luo = np.load(D.feat_file("val", "it_hist_all_luo"), mmap_mode="r")
    cold = (np.asarray(luo) == 0).astype(np.float64)
    npos = np.bincount(fid, y, len(M))
    allcold = (npos > 0) & (np.bincount(fid, cold * y, len(M)) == npos)
    val = dict(M=M, off=off, y=y, slices={"all": np.ones(len(M), bool), "M<50": M < 50, "M>=50": M >= 50,
                                          "Tue": day == 0, "Thu": day == 1, "allcold": allcold})
    M_all, ts_all = D.train_meta()
    pr = D.load_prandom(M_all)
    hold = D.holdout_feeds(pr, M_all)
    Mh = M_all[hold]
    offh = np.concatenate([[0], np.cumsum(Mh)[:-1]])
    yh = None
    for p in (root / "preds" / "holdout" / "holdout_y.npy", root / "baseline" / "preds" / "holdout" / "holdout_y.npy"):
        if p.exists():
            yh = np.load(p).astype(np.float64)
            break
    if yh is None:
        st = D.FeedStore("train", hold, "cpu", verbose=False)
        yh = st.target_np.astype(np.float64)
    ts = ts_all[hold]
    ho = dict(M=Mh, off=offh, y=yh, slices={"all": np.ones(len(Mh), bool), "late20%": ts >= np.quantile(ts, 0.8)})
    return val, ho


def paths(root, name, kind):
    if kind == "B":
        return (root / "baseline" / "preds" / f"nn_{name}_trainonly_val.npy",
                root / "baseline" / "preds" / "holdout" / f"nn_{name}_holdout.npy")
    return root / "preds" / f"{name}_trainonly_val.npy", root / "preds" / "holdout" / f"{name}_holdout.npy"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", nargs="+")
    ap.add_argument("--B", nargs="+")
    ap.add_argument("--R", default="")
    ap.add_argument("--tag", default="stageA")
    ap.add_argument("--root", default=str(HERE), help="directory holding preds/ and baseline/ (tests use a scratch dir)")
    ap.add_argument("--e1", default="require", choices=["require", "skip", "info"],
                    help="require: compare runs only after the E1 decision (Stage A, 0.6B); info: arm sets without an "
                         "E1 extension (large-model arms, reports/llm_large.md): holdout traces reported only; "
                         "skip: stand-in tests only")
    ap.add_argument("--size", default="Qwen3-0.6B", help="label of the L arm set (decision text and size curve)")
    ap.add_argument("--probe_large", default="", help="size-probe directory: print its gate; exit 3 on FAIL; stdout "
                                                      "= GB of free GPU memory an arm of that size needs")
    ap.add_argument("--curve", nargs="+", default=None, help="size curve: <label>:<compare tag> ...")
    ap.add_argument("--e1_decide", action="store_true", help="decide E1 from --L's holdout lines, then exit")
    ap.add_argument("--e1_R_needed", action="store_true", help="print 1 if the R23 E1 re-run was demanded, else 0")
    ap.add_argument("--probe", default="", help="print the probe gate + I3 decision of this probe directory, then exit")
    ap.add_argument("--i3_flag", default="", help="print '--final_fp32' or '' from this probe directory's I3 result "
                                                  "(exit 2 without a probe, exit 3 if the probe gate failed)")
    ap.add_argument("--gpu_hours", default="", help="ledger of run_matched.sh: print the lane-hours used; exit 3 if "
                                                    ">= --cap or if an arm cannot be costed")
    ap.add_argument("--cap", type=float, default=6.0, help="hard cap in lane wall-clock hours (LLM_FEASIBILITY 5.3)")
    ap.add_argument("--project_e1", action="store_true",
                    help="with --gpu_hours: also require room for both E1 4-epoch runs (2 x the L23 + L24 lane-hours)")
    a = ap.parse_args()
    root = Path(a.root)
    if a.probe_large:
        f = Path(a.probe_large) / "probe.json"
        if not f.exists():
            print(f"no size probe in {a.probe_large}: run its probe arm (P4 / P9) first", file=sys.stderr)
            sys.exit(2)
        r = json.loads(f.read_text())
        e, g = r["est_hours"], r["gate"]
        mem = max(r["peak_reserved_train_gb"], r["peak_reserved_fp32_gb"])
        print(f"SIZE PROBE {r['backbone']} (gpu {r['gpu']}, grad_accum {r['grad_accum']}): {r['s_per_step']:.2f} s/step, "
              f"{r['train_padded_tok_s']:.0f} padded tok/s; est. arm {e['total']:.2f} h (train {e['train']:.2f}, "
              f"{r['n_evals']} evals {e['evals']:.2f}, fp32 final {e['fp32_final']:.2f}, setup {e['setup']:.2f}) vs "
              f"<= {g['hours']:g} h; peak reserved {mem:.1f} GB vs <= {g['mem_gb']:g} -> SIZE PROBE GATE "
              f"{'PASS' if g['PASS'] else 'FAIL'}", file=sys.stderr)
        if not g["PASS"]:
            sys.exit(3)
        print(int(np.ceil(mem)) + 10)
        return
    if a.curve:
        size_curve(root, a.curve, a.tag)
        return
    if a.gpu_hours:
        used, rows, stale = gpu_hours(a.gpu_hours)
        need, note = 0.0, ""
        if a.project_e1:  # E1 = two 4-epoch L runs, useful only as a pair: both must fit under the cap or neither starts
            fin = {}
            for arm, _, h in rows:
                if arm in ("L23", "L24"):
                    fin[arm] = max(fin.get(arm, 0.0), h)
            if len(fin) == 2:
                need = 2 * (fin["L23"] + fin["L24"])
                note = f"; E1 pair projected at {need:.2f} h (2 x the L23 + L24 lane-hours)"
            else:
                note = "; E1 projection unavailable (L23/L24 not both in the ledger): plain cap check only"
        print(f"GPU budget: {used:.2f} of {a.cap:g} lane-hours used so far ("
              + (", ".join(f"{r[0]} gpu{r[1]} {r[2]:.2f} h" for r in rows) or "nothing yet") + ")" + note)
        if stale:
            print(f"GPU budget: cannot cost {stale} (lane gone without an 'end' line): add 'end<TAB><unix time>"
                  f"<TAB><gpu><TAB><arm><TAB><pid>' to {a.gpu_hours} by hand, then retry; not starting", file=sys.stderr)
            sys.exit(3)
        if used >= a.cap or (need and used + need > a.cap):
            print(f"GPU budget: HARD CAP ({used:.2f} used + {need:.2f} needed vs {a.cap:g} lane-hours); not starting. "
                  f"Only the lead may raise CAP_GPU_H in run_matched.sh (or decide what to do without this arm).",
                  file=sys.stderr)
            sys.exit(3)
        return
    if a.probe or a.i3_flag:
        r = probe_report(a.probe or a.i3_flag, quiet=bool(a.i3_flag))
        if r is None:
            print(f"no finished probe in {a.probe or a.i3_flag} (I3 undecided): run 'run_matched.sh <gpu> probe' first",
                  file=sys.stderr)
            sys.exit(2)
        if a.i3_flag:
            if not r["gate"]:  # pre-declared: run no arms unless the probe passed (LLM_FEASIBILITY 5.3)
                print(f"PROBE GATE FAILED ({r['padded_tok_s']:.0f} padded tok/s vs >= "
                      f"{PROBE_GATE['min_padded_tok_s']:.0f}; peak mem {r['peak_mem_gb']} GB vs <= "
                      f"{PROBE_GATE['max_peak_mem_gb']:.0f}): run no arms; wait for a quieter lane and re-run the probe",
                      file=sys.stderr)
                sys.exit(3)
            print("--final_fp32" if r["i3"]["final_fp32"] else "")
        return
    if a.e1_R_needed:
        f = root / "results" / "e1_R_decision.json"
        print(int(f.exists() and json.loads(f.read_text())["needed"]))
        return
    if a.e1_decide:
        dec = e1_decide(root, a.L)
        print(f"E1_EXTEND={int(dec['extend'])}")
        return
    assert a.L and a.B and len(a.L) == len(a.B)
    e1 = None
    if a.e1 == "require":  # E1: compare runs only after the (single) extension decision
        f = root / "results" / "e1_decision.json"
        if not f.exists():
            sys.exit("E1 not decided: run arm E1 (compare.py --e1_decide) first; compare runs only after the E1 decision")
        e1 = json.loads(f.read_text())
        assert e1["L"] == list(a.L), (e1["L"], a.L)
        if e1["extend"]:
            missing = [n for n in e1["L_judged"] if not (root / "preds" / f"{n}_trainonly_val.npy").exists()]
            if missing:
                sys.exit(f"E1 extension decided but the 4-epoch runs are not finished: {missing} (if arm E1 refused "
                         f"for the GPU cap, the lead decides; the 2-epoch runs are NOT judged instead)")
            print(f"E1: extension applied -> judging {e1['L_judged']} instead of {a.L}")
            a.L = list(e1["L_judged"])
    traces = {}
    if a.e1 == "info":  # large-model arms: no extension exists; holdout traces are information only
        for n in a.L:
            try:
                traces[n] = holdout_trace(root, n)
            except (FileNotFoundError, AssertionError) as ex:
                traces[n] = {"error": str(ex)}
    R_base = a.R
    if a.R and a.e1 == "require":  # R23 4-epoch run exists only if its E1 guard was demanded earlier
        fr = root / "results" / "e1_R_decision.json"
        if fr.exists() and json.loads(fr.read_text())["needed"]:
            if not (root / "preds" / f"{e4_name(a.R)}_trainonly_val.npy").exists():
                sys.exit(f"E1 guard on {a.R} demanded: run arm R23_E1 ({e4_name(a.R)}) before compare")
            a.R = e4_name(a.R)
            print(f"E1(R): judging the control with {a.R}")
    val, ho = load_ctx(root)
    S = {"val": val, "hold": ho}
    sc = {}  # (name, split) -> raw scores
    for kind, names in (("L", a.L), ("B", a.B), ("R", [a.R] if a.R else [])):
        for n in names:
            pv, ph = paths(root, n, kind)
            sc[(n, "val")], sc[(n, "hold")] = np.load(pv), np.load(ph)
    nd = {k: pf(v, S[k[1]]["y"], S[k[1]]["off"], S[k[1]]["M"]) for k, v in sc.items()}
    z = {k: D.feed_z(v, S[k[1]]["M"]) for k, v in sc.items()}

    def eqnd(names, split):
        s = np.mean([z[(n, split)] for n in names], 0)
        return pf(s, S[split]["y"], S[split]["off"], S[split]["M"])

    R = {"L": a.L, "B": a.B, "R": a.R, "single": {}, "paired": {}, "per_seed": {}, "A2": {}, "control": {}}
    for k, v in nd.items():
        R["single"][f"{k[0]}|{k[1]}"] = float(v.mean())
    lines = []
    # ---------------- A1: pooled paired L - B
    for split in ("val", "hold"):
        d = np.mean([nd[(l, split)] - nd[(b, split)] for l, b in zip(a.L, a.B)], 0)
        R["paired"][split] = {k: ms(d, m) for k, m in S[split]["slices"].items()}
        R["per_seed"][split] = [ms(nd[(l, split)] - nd[(b, split)]) for l, b in zip(a.L, a.B)]
    pv_m, pv_se = R["paired"]["val"]["all"]
    hold_d = R["paired"]["hold"]["all"][0]
    bad = [f"{sp}:{k}" for sp in ("val", "hold") for k, (m, se) in R["paired"][sp].items() if m < -2 * se]
    slices_ok = not bad  # gating slices: val all/M<50/M>=50/Tue/Thu/allcold, holdout all/late20% (research 4.5-4.6)
    A1 = pv_m > 2 * pv_se and all(m > 0 for m, _ in R["per_seed"]["val"]) and hold_d >= 0 and slices_ok
    # ---------------- A2
    gap_val = float(np.mean([nd[(l, "val")].mean() - nd[(b, "val")].mean() for l, b in zip(a.L, a.B)]))
    gap_hold = float(np.mean([nd[(l, "hold")].mean() - nd[(b, "hold")].mean() for l, b in zip(a.L, a.B)]))
    A2i = gap_val >= -0.002 and gap_hold >= -0.002
    a2 = {}
    if len(a.B) >= 2:
        for split in ("val", "hold"):
            eq_lb = np.mean([eqnd([l, b], split) for l, b in zip(a.L, a.B)], 0)
            eq_bb = eqnd(a.B[:2], split)
            a2[split] = ms(eq_lb - eq_bb)
        m2, se2 = a2["val"]
        A2ii = m2 > 0 and m2 / max(se2, 1e-12) >= 1.5 and a2["hold"][0] >= 0
    else:
        A2ii = False
    rhos = [rho(z[(l, "val")], z[(b, "val")], val["M"]) for l, b in zip(a.L, a.B)]
    rho_bb = rho(z[(a.B[0], "val")], z[(a.B[1], "val")], val["M"]) if len(a.B) >= 2 else float("nan")
    r_mean = float(np.mean(rhos))
    law = 0.0186 * (1 - r_mean) - 0.378 * abs(gap_val)
    A2 = A2i and A2ii
    R["A2"] = dict(gap_val=gap_val, gap_hold=gap_hold, strength_ok=A2i, eqLB_minus_eqBB=a2, diversity_ok=A2ii,
                   rho_LB=rhos, rho_BB=rho_bb, law_predicted_gain_val=law)
    R["A2_detail"] = R["A2"]  # R["A2"] is overwritten by the boolean below; keep the details in the JSON
    # ---------------- control
    exception = False
    if a.R:
        l0, b0 = a.L[0], a.B[0]
        for split in ("val", "hold"):
            R["control"][f"L-R|{split}"] = ms(nd[(l0, split)] - nd[(a.R, split)])
            R["control"][f"R-B|{split}"] = ms(nd[(a.R, split)] - nd[(b0, split)])
        lr_m, lr_se = R["control"]["L-R|val"]
        exception = pv_m >= -pv_se and lr_m > 2 * lr_se and a.e1 != "info"  # P4B exception: Stage A (0.6B) only
        if exception and a.R == R_base and a.e1 == "require" and not A1 and not A2:
            # the P4B exception would fire: apply the E1 guard to R23 first (pre-declared)
            tr_R = holdout_trace(root, R_base)
            (root / "results").mkdir(exist_ok=True)
            (root / "results" / "e1_R_decision.json").write_text(json.dumps(dict(
                rule=E1_RULE, R=R_base, trace=tr_R, needed=tr_R["still_improving"], R_rerun=e4_name(R_base),
                decided_utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())), indent=2))
            R["control"]["e1_R"] = tr_R
            if tr_R["still_improving"]:
                R["pending"] = "R23_E1"
        # information, not a gate (research 4.6): L - R <= 1 SE -> any A1/A2 gain is capacity/architecture
        R["control"]["attribution"] = (
            "pretraining helps (L-R > 2 SE)" if lr_m > 2 * lr_se else
            "no evidence that language pretraining helps (L-R <= 1 SE): an A1/A2 gain would be capacity or "
            "architecture, so a small from-scratch family is the cheaper way to get it" if lr_m <= lr_se else
            "inconclusive (1 SE < L-R <= 2 SE)")
    # ---------------- decision
    if a.e1 == "info":  # large-model arm sets (reports/llm_large.md): same A1/A2, size-named decisions, no exception
        imp = [n for n, t in traces.items() if t.get("still_improving")]
        R["control"]["note"] = ("R is the 0.6B random-init control (information only for this size; the P4B "
                                "exception does not apply)") if a.R else ""
        R["e1_info"] = dict(traces=traces, still_improving=imp)
        tail = (f" NOTE (information, pre-declared): {imp} still improving at the last holdout evaluation; no "
                f"extension exists for this arm set.") if imp else ""
        if A1:
            nxt = (f"A1 PASS ({a.size}) -> 600K-feed slope check with {a.size} (seed 23); continue only if the L-B "
                   f"advantage keeps >= 50%.") + tail
        elif A2:
            nxt = (f"A2 PASS ({a.size}) -> full-data {a.size} members (1 epoch, 2 seeds), then gates (a)+(b), "
                   f"frozen pool.") + tail
        else:
            nxt = f"A1+A2 FAIL ({a.size}) -> STOP for this size." + tail
    elif R.get("pending"):
        nxt = (f"PENDING (E1 guard on the control): the P4B exception would fire, but {R_base} was still improving at "
               f"its last holdout evaluation -> run 'run_matched.sh <gpu> R23_E1' ({e4_name(R_base)}, 4 epochs), then "
               f"re-run compare; the exception is judged with the 4-epoch control.")
    elif A1:
        nxt = "A1 PASS -> Stage B: 600K-feed slope check (seed 23); continue only if the L-B advantage keeps >= 50%."
    elif A2:
        nxt = "A2 PASS -> Stage C: full-data Qwen3-0.6B members (1 epoch, 2 seeds), then gates (a)+(b), frozen pool."
    elif exception:
        nxt = "A1+A2 FAIL, exception met (L >= B - 1 SE and L - R > 2 SE) -> ONE Qwen3-4B-Instruct-2507 L1 probe."
    else:
        nxt = "A1+A2 FAIL -> STOP. Add 'LLM backbones (soft-token Qwen3-0.6B, matched 150K)' to the do-not list."
    R.update(A1=A1, A2=A2, exception=exception, decision=nxt, slices_below_2se=bad, L_judged=list(a.L), R_judged=a.R,
             e1=e1 and {k: e1[k] for k in ("extend", "L", "L_judged", "decided_utc")}, size=a.size, e1_mode=a.e1)
    lines.append(f"single models (val | holdout): " + ", ".join(
        f"{n} {nd[(n, 'val')].mean():.5f} | {nd[(n, 'hold')].mean():.5f}" for n in a.L + a.B + ([a.R] if a.R else [])))
    for sp in ("val", "hold"):
        lines.append(f"paired L-B {sp}: " + "  ".join(f"{k} {m:+.5f}+-{se:.5f}" for k, (m, se) in R["paired"][sp].items()))
        lines.append(f"  per seed: " + "  ".join(f"{m:+.5f}+-{se:.5f}" for m, se in R["per_seed"][sp]))
    lines.append(f"A1 = {A1} (val z {pv_m / max(pv_se, 1e-12):+.2f}; slices below -2 SE: {bad or 'none'})")
    lines.append(f"A2(i) gap val {gap_val:+.5f} hold {gap_hold:+.5f} -> {A2i}; A2(ii) eq(L,B)-eq(B,B) "
                 + "  ".join(f"{k} {m:+.5f}+-{se:.5f}" for k, (m, se) in a2.items()) + f" -> {A2ii}; A2 = {A2}")
    lines.append(f"rho(L,B) {['%.4f' % r for r in rhos]} rho(B23,B24) {rho_bb:.4f}; law-predicted eq gain {law:+.5f}")
    if a.R:
        lines.append("control: " + "  ".join(f"{k} {v[0]:+.5f}+-{v[1]:.5f}" for k, v in R["control"].items()
                                             if isinstance(v, tuple)) + f"; {R['control']['attribution']}")
    if e1 is not None:
        lines.append(f"E1: extend = {e1['extend']} (decided {e1['decided_utc']}); judged L = {a.L}; R = {a.R or '-'}")
    if traces:
        lines.append("holdout traces (information; no E1 for this arm set): " + "; ".join(
            f"{n}: best step {t['best_step']}/{t['final_step']}, still improving = {t['still_improving']}"
            if "error" not in t else f"{n}: {t['error']}" for n, t in traces.items()))
    lines.append(f"DECISION: {nxt}")
    out = root / "results"
    out.mkdir(exist_ok=True)
    (out / f"compare_{a.tag}.json").write_text(json.dumps(R, indent=2, default=float))
    (out / f"compare_{a.tag}.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def size_curve(root, specs, tag):
    """Size curve over finished compare_<tag>.json files (pre-declared in reports/llm_large.md): per size the mean
    single val / holdout NDCG of L and of B, the paired L-B (pooled over seeds), eq(L,B)-eq(B,B), rho(L,B), rho(B,B),
    the law-predicted eq gain, A1 / A2, and the 0.6B random-init control R (val / holdout)."""
    rows, lines = [], []
    hdr = (f"{'size':<7} {'L val':>8} {'L hold':>8} {'B val':>8} {'B hold':>8} {'L-B val':>17} {'L-B hold':>17} "
           f"{'eqLB-eqBB val':>17} {'rho LB':>7} {'rho BB':>7} {'law':>8}  A1/A2")
    lines.append(hdr)
    Rrow = None
    for sp in specs:
        lab, t = sp.split(":", 1)
        f = Path(root) / "results" / f"compare_{t}.json"
        if not f.exists():
            lines.append(f"{lab:<7} (missing {f.name})")
            continue
        j = json.loads(f.read_text())
        sg = j["single"]
        Lv = float(np.mean([sg[f"{n}|val"] for n in j["L_judged"]]))
        Lh = float(np.mean([sg[f"{n}|hold"] for n in j["L_judged"]]))
        Bv = float(np.mean([sg[f"{n}|val"] for n in j["B"]]))
        Bh = float(np.mean([sg[f"{n}|hold"] for n in j["B"]]))
        pv, ph = j["paired"]["val"]["all"], j["paired"]["hold"]["all"]
        a2 = j["A2_detail"]
        eqv = a2["eqLB_minus_eqBB"].get("val", [float("nan"), float("nan")])
        r = dict(size=lab, tag=t, L=j["L_judged"], L_val=Lv, L_hold=Lh, B_val=Bv, B_hold=Bh, LB_val=pv, LB_hold=ph,
                 eqLB_minus_eqBB_val=eqv, eqLB_minus_eqBB_hold=a2["eqLB_minus_eqBB"].get("hold"),
                 rho_LB=float(np.mean(a2["rho_LB"])), rho_BB=a2["rho_BB"], law=a2["law_predicted_gain_val"],
                 A1=j["A1"], A2=bool(j["A2"]), decision=j["decision"])
        rows.append(r)
        lines.append(f"{lab:<7} {Lv:8.5f} {Lh:8.5f} {Bv:8.5f} {Bh:8.5f} {pv[0]:+9.5f}+-{pv[1]:.5f} "
                     f"{ph[0]:+9.5f}+-{ph[1]:.5f} {eqv[0]:+9.5f}+-{eqv[1]:.5f} {r['rho_LB']:7.4f} {r['rho_BB']:7.4f} "
                     f"{r['law']:+8.5f}  {r['A1']}/{r['A2']}")
        if j.get("R_judged") and Rrow is None:
            Rn = j["R_judged"]
            Rrow = dict(R=Rn, val=sg.get(f"{Rn}|val"), hold=sg.get(f"{Rn}|hold"))
    if Rrow:
        lines.append(f"random-init control {Rrow['R']} (0.6B architecture, all weights trained): val {Rrow['val']:.5f} "
                     f"| holdout {Rrow['hold']:.5f}")
    out = Path(root) / "results"
    out.mkdir(exist_ok=True)
    (out / f"{tag}.json").write_text(json.dumps(dict(rows=rows, R=Rrow), indent=2, default=float))
    (out / f"{tag}.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
