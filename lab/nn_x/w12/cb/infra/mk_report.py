"""Assemble reports/CATBOOST_1115/infra.md from the measured JSON files (no heavy compute, numbers are never typed by hand
in the tables). usage: CUDA_VISIBLE_DEVICES="" CBX_THREADS=2 python mk_report.py <verdict.txt> [<notes.txt>]"""
import glob, json, os, re, subprocess, sys
import numpy as np
R = "<repo>"; H = f"{R}/nn_x/w12/cb/infra"; O = f"{H}/out"
sys.path.insert(0, H)
from tables import gpu_curve, curve  # noqa: E402

VERDICT = open(sys.argv[1]).read().strip()
NOTES = open(sys.argv[2]).read().strip() if len(sys.argv) > 2 else ""


def J(tag):
    p = f"{O}/{tag}.eval.json"
    return json.load(open(p)) if os.path.exists(p) else None


def model_rows(tags):
    out = ["| model | selection | fit feeds | trees used (built) | val NDCG ± SE (n = 14,423) | holdout NDCG ± SE (n = 30,000) | val half A | val half B | rho val / hold | law record / refit | member rule 0.9 z(W16) + 0.1 z(X): val, holdout, pooled z |",
           "|---|---|---:|---:|---|---|---:|---:|---|---|---|"]
    for tag, label, sel in tags:
        r = J(tag)
        if r is None:
            out.append(f"| {label} | {sel} | not finished | | | | | | | | |"); continue
        e = r.get("extra", {}); m = r["member_rule"]
        trees = f"{e['best_iter'] + 1} ({e['trees_built']})" if "best_iter" in e else e.get("trees", "-")
        hA = f"{e['val_halfA']:.5f}" if "val_halfA" in e else e.get("halfA", "-"); hB = f"{e['val_halfB']:.5f}" if "val_halfB" in e else e.get("halfB", "-")
        out.append(f"| {label} | {sel} | {e.get('n_fit_feeds', e.get('feeds', '-'))} | {trees} | {r['val']:.5f} ± {r['val_se']:.5f} | {r['hold']:.5f} ± {r['hold_se']:.5f} | "
                   f"{hA} | {hB} | {r['rho_val']:.3f} / {r['rho_hold']:.3f} | {r['law_record']:+.4f} / {r['law_refit']:+.4f} | "
                   f"{m['val']['delta']:+.5f} ± {m['val']['se']:.5f}, {m['hold']['delta']:+.5f} ± {m['hold']['se']:.5f}, {m['pooled_z']:+.2f} {'PASS' if m['pass'] else 'FAIL'} |")
    return "\n".join(out)


def curve_table(tags):
    g = gpu_curve(); cs = {t: curve(t) for t in tags}
    cs = {t: c for t, c in cs.items() if c is not None}
    out = ["| trees | GPU record, val half A | " + " | ".join(f"`{t}`: selection-set NDCG (minutes of fit)" for t in cs) + " |", "|---:|---:|" + "---:|" * len(cs)]
    mx = max(int(c[0][-1]) for c in cs.values())
    for k in sorted(set([1, 26, 51] + list(range(101, mx + 1, 50)) + [int(c[0][-1]) for c in cs.values()] + list(range(101, 2000, 100)) + [1936])):
        if k > max(mx, 1936): continue
        row = [f"{c[2][k - 1]:.5f} ({c[1][k - 1] / 60:.0f})" if k <= c[0][-1] else "" for c in cs.values()]
        if k in g or any(row):
            out.append(f"| {k} | {('%.5f' % g[k]) if k in g else ''} | " + " | ".join(row) + " |")
    return "\n".join(out)


def matched_level(tag):
    g = gpu_curve(); c = curve(tag)
    if c is None: return ""
    it, t, v = c; rm = np.maximum.accumulate(v)
    out = ["| GPU trees | GPU half-A NDCG | CPU trees to reach the same half-A NDCG | CPU minutes |", "|---:|---:|---:|---:|"]
    for k in (101, 301, 501, 1001, 1501, 1936):
        if k not in g: continue
        hit = np.where(rm >= g[k])[0]
        out.append(f"| {k} | {g[k]:.5f} | {int(it[hit[0]]) if len(hit) else 'not reached in ' + str(int(it[-1])) + ' trees (best ' + ('%.5f' % v.max()) + ')'} | {('%.0f' % (t[hit[0]] / 60)) if len(hit) else ''} |")
    return "\n".join(out)


def staged(tags):
    out = []
    for t in tags:
        r = J(t)
        if not r or "staged" not in r.get("extra", {}): continue
        st = r["extra"]["staged"]; e = r["extra"]
        step = max(1, len(st) // 14)
        keep = [s for i, s in enumerate(st) if i % step == step - 1 or i == len(st) - 1]
        out.append(f"Staged evaluation of `{t}` (evaluation only; the model used above has {e['best_iter'] + 1} trees):\n")
        out.append("| trees | val | val half A | val half B | holdout |\n|---:|---:|---:|---:|---:|")
        out += [f"| {s['trees']} | {s['val']:.5f} | {s['val_halfA']:.5f} | {s['val_halfB']:.5f} | {s['hold']:.5f} |" for s in keep]
        bA = max(st, key=lambda s: s["val_halfA"]); bH = max(st, key=lambda s: s["hold"]); bV = max(st, key=lambda s: s["val"])
        out.append(f"\nBest staged point by val half A: {bA['trees']} trees (val {bA['val']:.5f}, holdout {bA['hold']:.5f}); by full val: {bV['trees']} trees "
                   f"({bV['val']:.5f}); by holdout: {bH['trees']} trees ({bH['hold']:.5f}).\n")
    return "\n".join(out)


def timing():
    rows = {}
    pool = None
    for p in sorted(glob.glob(f"{O}/timing*.jsonl")):
        for line in open(p):
            d = json.loads(line)
            if d["name"] == "_pool": pool = pool or d
            elif "error" not in d: rows[d["name"]] = d
    t33 = []
    for loss in ("YetiRank", "QuerySoftMax", "RMSE"):
        cells = []
        for dp in (6, 8, 10):
            d = rows.get(f"{loss}_d{dp}")
            cells.append(f"{d['s_per_iter']:.2f} ({d['rows_per_s'] / 1e6:.2f})" if d else "-")
        t33.append(f"| {loss} | " + " | ".join(cells) + " |")
    lev = []
    base = {"YetiRank": rows["YetiRank_d8"]["s_per_iter"], "RMSE": rows["RMSE_d8"]["s_per_iter"]}
    names = {"YetiRank_d8": "YetiRank default", "YetiRank_d8_rsm0.3": "YetiRank, rsm 0.3", "YetiRank_d8_rsm0.1": "YetiRank, rsm 0.1",
             "YetiRank_d8_bern0.5": "YetiRank, Bernoulli subsample 0.5 (objects)", "YetiRank_d8_bern0.5_group": "YetiRank, Bernoulli subsample 0.5, sampling_unit Group",
             "YetiRank_d8_perm3": "YetiRank:permutations=3", "YetiRank_d8_rsm0.3_bern0.5": "YetiRank, rsm 0.3 + Bernoulli 0.5",
             "RMSE_d8": "RMSE default", "RMSE_d8_rsm0.3": "RMSE, rsm 0.3", "RMSE_d8_mvs": "RMSE, MVS subsample 0.5",
             "RMSE_d8_catnum": "RMSE, category_id as a number (no categorical feature)", "YetiRank_d8_catnum": "YetiRank, category_id as a number",
             "RMSE_d8+item_id": "RMSE, + item_id as a native categorical feature (CTR)", "RMSE_d8+item_id+user_id": "RMSE, + item_id and user_id as native categorical features (CTR)"}
    for k, lab in names.items():
        if k in rows:
            b = base["YetiRank" if k.startswith("Yeti") else "RMSE"]
            lev.append(f"| {lab} | {rows[k]['s_per_iter']:.2f} | {rows[k]['s_per_iter'] / b:.2f}x |")
    feas = []
    for loss in ("YetiRank", "QuerySoftMax", "RMSE"):
        d = rows[f"{loss}_d8"]; s = d["s_per_iter"]; n0 = d["rows"]
        f = lambda n: int(7000 / (s * n / n0))
        feas.append(f"| {loss} | {f(n0)} | {f(25103552)} | {f(19741607 + 16622487)} | {s * 83.0e6 / n0:.0f} s, {f(83.0e6)} iterations, about {31.8 * 83.0e6 / n0:.0f} GB |")
    return "\n".join(t33), "\n".join(lev), "\n".join(feas), pool


def smoke():
    p = f"{H}/logs/smoke_big.log"
    L = [x.rstrip("\n") for x in open(p)]
    npass = sum(x.startswith("PASS") for x in L); nfail = sum(x.startswith("FAIL") for x in L)
    last = [x for x in L if x.startswith(("ALL PASS", "FAILED"))]
    groups = {"6 ": "training path opens no val / test / holdout label file; `ds.y` raises for val, hold, test",
              "1 ": "177 features == cb26 `member_features()` == record JSON",
              "2 ": "shapes of X / gid / item_id / cat for xm, hold, val, test, xs, prod200k",
              "3a": "f1, f5, category_id, M, H, emb_missing, item_id, fidx recomputed from raw for 1,000 random rows per set",
              "3b": "the same 1,000 rows x 177 columns equal `cbdata.load_tab` of cb26",
              "3c": "dot_u, cos_u, item_norm recomputed from the cbx raw embeddings (max abs error 0 / 6e-8 / 0)",
              "3d": "hmean / h5 / h20 and their counts recomputed in numpy for 100 feeds per set",
              "3e": "history order: hist_hpos == 0..H-1",
              "3f": "user embedding rows == user_emb.npy",
              "4a": "teacher value of a row == the stored file (1,000 rows per train set)",
              "4b": "teacher_oof == teacher_xf<k> / 3 with k = cbx.xfold_of(user)",
              "4c": "soft labels sum to 1 per feed; centred forms have zero feed mean",
              "5 ": "no holdout feed in xm, xs, prod200k, concat(xm, xs); holdout == cb26 `hold_index()`; fit / selection share no known user",
              "7a": "NDCG kernel is a verbatim copy of `pc_common._ndcg`",
              "7b": "fast NDCG == `src/metrics.ndcg_per_feed` on val (W16 pool, heavy ties, constant, random): max abs diff 0",
              "7c": "the same on the holdout", "7d": "feed z == `stack_val.feed_rank_z[:, 1]`", "7e": "W16 pool = 0.40144 / 0.32460",
              "9a": "evaluator reproduces the record's 0.38201 / 0.30424 from its saved predictions (row-order check for val and holdout)",
              "11a": "tiny YetiRank fit on `make_pool`, prediction on val", "11b": "embedding-feature pools train with YetiRank, RMSE (teacher logit), QuerySoftMax (soft teacher)",
              "11c": "baseline pool (YetiRank on top of the centred OOF teacher logit)", "11d": "item_id / user_id as native categorical features + OOF NN logit as a column",
              "12a": "holdout: mean of the 11 members' prediction files == teacher_mean11 on the holdout rows",
              "12b": "NN inputs on val / holdout score as expected", "12c": "NN input on test has the right shape", "12d": "OOF teacher agrees with the xf15 input on the holdout"}
    out = [f"{npass} PASS, {nfail} FAIL ({last[-1] if last else 'no final line'}); log `nn_x/w12/cb/infra/logs/smoke_big.log` [M].\n", "| check | what | result |", "|---|---|---|"]
    for k, lab in groups.items():
        hit = [x for x in L if x.startswith(("PASS", "FAIL")) and x[6:].startswith(k)]
        res = "not run" if not hit else ("PASS" if all(x.startswith("PASS") for x in hit) else "FAIL") + f" ({len(hit)})"
        out.append(f"| {k.strip()} | {lab} | {res} |")
    return "\n".join(out)


T33, TLEV, TFEAS, POOL = timing()
BODY = open(f"{H}/report_template.txt").read()
rep = BODY
rep = rep.replace("@@VERDICT@@", VERDICT).replace("@@NOTES@@", NOTES).replace("@@LIMITS@@", open(f"{H}/limits.txt").read().strip())
rep = rep.replace("@@T_MODELS@@", model_rows([("record_e3_xm_yeti_GPU", "record: GPU YetiRank `e3_xm_yeti`, lr 0.08 (predictions on record, re-scored here)", "val half A (record)"),
                                             ("repro_valA", "CPU `repro_valA`: record configuration, lr 0.08", "val half A (record rule)"),
                                             ("ts_lr020_s0", "CPU `ts_lr020_s0`: lr 0.2, od_wait 100, otherwise the record configuration", "train-side split (no val label)")]))
rep = rep.replace("@@T_CURVE@@", curve_table(["repro_valA", "ts_lr020_s0"]))
rep = rep.replace("@@T_STAGED@@", staged(["repro_valA", "ts_lr020_s0"])).replace("@@T_MATCH@@", matched_level("repro_valA"))
rep = rep.replace("@@T_TIMING33@@", T33).replace("@@T_LEVERS@@", TLEV).replace("@@T_FEAS@@", TFEAS)
rep = rep.replace("@@T_RECORD@@", model_rows([("record_e3_xm_yeti_GPU", "GPU CatBoost YetiRank of record (hard label)", "val half A"),
                                             ("record_gbdt_H_xm_134118", "LightGBM lambdarank, hard label, round 600 (long_bets)", "fixed rounds"),
                                             ("record_gbdt_K_xm_134118", "LightGBM L2 on teacher_mean11, 134K feeds, round 600 (long_bets)", "fixed rounds"),
                                             ("record_gbdt_X_xm_60000", "LightGBM L2 on teacher_oof, 60K feeds, round 600 (long_bets)", "fixed rounds"),
                                             ("record_bundle_cb_yeti+lgb_kd", "bundle: z-mean of the GPU CatBoost and the LightGBM KD student", "-")]))
rep = rep.replace("@@T_SMOKE@@", smoke())
left = re.findall(r"@@[A-Z_0-9]+@@", rep)
assert not left, left
os.makedirs(f"{R}/reports/CATBOOST_1115", exist_ok=True)
open(f"{R}/reports/CATBOOST_1115/infra.md", "w").write(rep)
print("written", len(rep), "chars;", rep.count("\n"), "lines")
