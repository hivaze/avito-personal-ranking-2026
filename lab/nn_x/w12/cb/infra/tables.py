"""Markdown tables for reports/CATBOOST_1115/infra.md from out/*.eval.json, out/*.curve.json and out/timing*.jsonl (no heavy compute).
usage: CUDA_VISIBLE_DEVICES="" CBX_THREADS=4 python tables.py"""
import glob, json, os, re, sys
import numpy as np
O = "<repo>/nn_x/w12/cb/infra/out"


def ev_rows(tags):
    print("| model | fit feeds | trees used (built) | val NDCG ± SE (n=14,423) | holdout NDCG ± SE (n=30,000) | half A | half B | rho val / hold | law (record / refit) | member rule: val / holdout / pooled z |")
    print("|---|---:|---:|---|---|---:|---:|---|---|---|")
    for tag, label in tags:
        p = f"{O}/{tag}.eval.json"
        if not os.path.exists(p):
            print(f"| {label} | not finished | | | | | | | | |"); continue
        r = json.load(open(p)); e = r.get("extra", {}); m = r["member_rule"]
        trees = f"{e['best_iter'] + 1} ({e['trees_built']})" if e else "-"
        print(f"| {label} | {e.get('n_fit_feeds', '-')} | {trees} | {r['val']:.5f} ± {r['val_se']:.5f} | {r['hold']:.5f} ± {r['hold_se']:.5f} | "
              f"{e.get('val_halfA', float('nan')):.5f} | {e.get('val_halfB', float('nan')):.5f} | {r['rho_val']:.3f} / {r['rho_hold']:.3f} | "
              f"{r['law_record']:+.4f} / {r['law_refit']:+.4f} | {m['val']['delta']:+.5f} ± {m['val']['se']:.5f} / {m['hold']['delta']:+.5f} ± {m['hold']['se']:.5f} / "
              f"{m['pooled_z']:+.2f} {'PASS' if m['pass'] else 'FAIL'} |")


def gpu_curve():
    g = {}
    for line in open("<repo>/nn_x/cb26/logs/e3_xm_yeti.log"):
        m = re.match(r"^(\d+):\s+test: ([0-9.]+)", line)
        if m: g[int(m.group(1)) + 1] = float(m.group(2))  # log index is 0-based: index 100 = 101 trees
        m = re.match(r"^bestTest = ([0-9.]+)", line)
        if m: best = float(m.group(1))
        m = re.match(r"^bestIteration = (\d+)", line)
        if m: g[int(m.group(1)) + 1] = best
    return g


def curve(tag):
    p = f"{O}/{tag}.curve.json"
    if not os.path.exists(p): return None
    r = json.load(open(p))
    return np.array([x[0] for x in r]), np.array([x[1] for x in r]), np.array([x[2]["validation:NDCG:type=Base"] for x in r])


if __name__ == "__main__":
    print("## models\n")
    ev_rows([(t, l) for t, l in [("record_e3_xm_yeti_GPU", "record: GPU YetiRank e3_xm_yeti (val-A early stop, 1,936 trees)")] +
             [(os.path.basename(p)[:-10], os.path.basename(p)[:-10]) for p in sorted(glob.glob(f"{O}/*.eval.json")) if "record_e3" not in p]])
    print("\n## selection-set curve vs the GPU record (val half A, 7,212 feeds)\n")
    g = gpu_curve(); cs = {t: curve(t) for t in sys.argv[1:] or ["repro_valA"]}
    print("| trees | GPU record half A | " + " | ".join(f"{t} (its own selection set)" for t in cs) + " |")
    print("|---:|---:|" + "---:|" * len(cs))
    mx = max(int(c[0][-1]) for c in cs.values() if c is not None)
    for k in sorted(set([1, 25, 50] + list(range(101, mx + 1, 100)) + [mx])):
        row = [f"{c[2][k - 1]:.5f} ({c[1][k - 1] / 60:.0f} min)" if c is not None and k <= c[0][-1] else "" for c in cs.values()]
        print(f"| {k} | {('%.5f' % g[k]) if k in g else ''} | " + " | ".join(row) + " |")
    print("\n## staged curves (evaluation only)\n")
    for p in sorted(glob.glob(f"{O}/*.eval.json")):
        r = json.load(open(p)); st = r.get("extra", {}).get("staged")
        if not st: continue
        print(f"\n{r['tag']}:\n\n| trees | val | val half A | val half B | holdout |\n|---:|---:|---:|---:|---:|")
        for s in st:
            print(f"| {s['trees']} | {s['val']:.5f} | {s['val_halfA']:.5f} | {s['val_halfB']:.5f} | {s['hold']:.5f} |")
    print("\n## timing\n")
    rows = {}
    for p in sorted(glob.glob(f"{O}/timing*.jsonl")):
        for line in open(p):
            d = json.loads(line)
            if d["name"] == "_pool":
                print(f"pool: {d['rows']} rows, Pool build {d['t_pool_s']:.0f}s, quantize {d['t_quantize_s']:.0f}s, RSS GB {d['mem_gb']}, load {d['load']}, {os.path.basename(p)} started {d['started']}\n")
            elif "error" in d:
                rows[d["name"] + " [" + os.path.basename(p) + "]"] = d
            else:
                rows[d["name"] + " [" + os.path.basename(p) + "]"] = d
    print("| config | s / iteration | M rows / s | iterations in 2 h (7,000 s net) | est. s / iteration at 83M rows (1M production feeds, 83 rows / feed) | RSS GB (peak) |")
    print("|---|---:|---:|---:|---:|---:|")
    for k, d in rows.items():
        if "error" in d:
            print(f"| {k} | FAILED: {d['error'][:120]} | | | | |"); continue
        print(f"| {k} | {d['s_per_iter']:.2f} | {d['rows_per_s'] / 1e6:.2f} | {int(7000 / d['s_per_iter'])} | {d['s_per_iter'] * 83.0e6 / d['rows']:.1f} | {d['rss_gb']:.1f} ({d['peak_rss_gb']:.1f}) |")
