"""INVENTORY B, step 8: merge the sweeps into tables.
Outputs (out/): all_columns.tsv, negctl_existing_inputs.txt, noise_summary.txt, support_table.txt.  Read-only, CPU, seconds."""
import json, math
import numpy as np
from scipy.stats import norm
O = "<repo>/reports/INPUTS_1116/residual_rows/out/"
L = lambda f: {j["key"]: j for j in map(json.loads, open(O + f))}
SV, SF, SH = L("sweep_val_TO16.jsonl"), L("sweep_val_FILE.jsonl"), L("sweep_hold_TO16.jsonl")
CV, CF, CH = L("cond_val_TO16.jsonl"), L("cond_val_FILE.jsonl"), L("cond_hold_TO16.jsonl")
meta = {}
hdr = None
for l in open(O + "store_meta.tsv"):
    c = l.rstrip("\n").split("\t")
    if hdr is None: hdr = c; continue
    meta[c[0]] = dict(zip(hdr, c))
XT34 = {n for n, m in meta.items() if m["in_xt34"] == "1"}
INMODEL = set(("f1 f2 f3 f4 f5 emb_missing item_norm cos_u dot_u seq_max_all seq_max_new8 maxsim32 meansim32 attn32 seq_top3 cat_cnt_all cat_share_all cat_min_hpos feed_cat_share "
               + " ".join(f"cos_het{t}" for t in range(1, 11)) + " " + " ".join(f"cat_cnt_et{t}" for t in range(1, 11))).split())


def status(key):
    g, n = key.split(":", 1)
    if g != "store": return "gate array"
    base = n
    for suf in ("_z", "_rk", "_gap"):
        if n.endswith(suf) and n[:-len(suf)] in (XT34 | INMODEL): base = n[:-len(suf)]
    if n in XT34: return "TRAINED INPUT (one of the 34)"
    if base in XT34: return "re-encoding of a trained input"
    if n in INMODEL or base in INMODEL: return "in-model column (copy)"
    m = meta.get(n, {})
    lk = m.get("leak_status_1111", "")
    if "BANNED" in lk: return "BANNED (leak)"
    if "in-sample" in lk or "own label" in lk or "count themselves" in lk: return "train rows in-sample (holdout read invalid)"
    if m.get("in_ft42") == "1": return "fine-tune extra only"
    return "not read by the ranker"


g = lambda d, k, f: d.get(k, {}).get(f, float("nan"))
keys = sorted(set(SV) | set(SH))
rows = []
for k in keys:
    rows.append(dict(key=k, status=status(k), nan_val=g(SV, k, "nanshare"),
                     LR_valTO=g(SV, k, "LR_z"), LR_valFILE=g(SF, k, "LR_z"), LR_hold=g(SH, k, "LR_z"),
                     nest_valTO=g(SV, k, "nest"), nest_valTO_se=g(SV, k, "nest_se"), nest_valTO_pmix=g(SV, k, "nest_pmix"), nest_valFILE=g(SF, k, "nest"), nest_valFILE_se=g(SF, k, "nest_se"),
                     nest_hold=g(SH, k, "nest"), nest_hold_se=g(SH, k, "nest_se"),
                     cz_valTO=g(CV, k, "cz_z"), czrk_valTO=g(CV, k, "cz_rk"), cz_valFILE=g(CF, k, "cz_z"), czrk_valFILE=g(CF, k, "cz_rk"), cz_hold=g(CH, k, "cz_z"), czrk_hold=g(CH, k, "cz_rk"),
                     cznan_valTO=g(CV, k, "cz_nan"), cznan_hold=g(CH, k, "cz_nan"), r2_val=g(CV, k, "r2_z"), dnats_val=g(CV, k, "dnats_z"), dnats_hold=g(CH, k, "dnats_z"),
                     skip=SV.get(k, {}).get("skip", SV.get(k, {}).get("error", ""))))
for r in rows:  # a column that pool + inputs reproduce (R2 >= 0.98 on val) is a copy: its score z rests on a vanishing residual variance and is not read
    if r["status"] in ("gate array", "not read by the ranker", "fine-tune extra only") and r["r2_val"] == r["r2_val"] and r["r2_val"] >= 0.98:
        r["status"] = "copy of pool + inputs (R2 >= 0.98)"
cols = list(rows[0].keys())
with open(O + "all_columns.tsv", "w") as f:
    f.write("\t".join(cols) + "\n")
    for r in rows:
        f.write("\t".join((f"{r[c]:.6g}" if isinstance(r[c], float) else str(r[c])) for c in cols) + "\n")

# ---- negative control: inputs the ranker already reads, plain LR lens
with open(O + "negctl_existing_inputs.txt", "w") as f:
    f.write("Plain likelihood ratio (1 df, z form) of columns the ranker ALREADY reads, against the pool. LR_valTO = 16 held-out wide students train-only; LR_valFILE = FD16P20 file; LR_hold = holdout.\n")
    f.write(f"{'column':24s} {'status':34s} LR_valTO LR_valFILE LR_hold | cond cz valTO / valFILE / hold (given pool + control block; ~0 by construction when the column is in the block)\n")
    ex = [r for r in rows if r["status"] in ("TRAINED INPUT (one of the 34)", "in-model column (copy)") and r["LR_valTO"] == r["LR_valTO"]]
    for r in ex:
        f.write(f"{r['key'][6:]:24s} {r['status']:34s} {r['LR_valTO']:8.1f} {r['LR_valFILE']:10.1f} {r['LR_hold']:7.1f} | {r['cz_valTO']:+6.2f} {r['cz_valFILE']:+6.2f} {r['cz_hold']:+6.2f}\n")
    for nm in ("LR_valTO", "LR_valFILE", "LR_hold"):
        v = np.array([r[nm] for r in ex if r[nm] == r[nm]])
        f.write(f"{nm}: n {len(v)} median {np.median(v):.1f} p75 {np.quantile(v, .75):.1f} p90 {np.quantile(v, .9):.1f} max {v.max():.1f}; share above 3.84: {(v > 3.84).mean():.2f}; share above 10.83: {(v > 10.83).mean():.2f}\n")

# ---- noise summary
with open(O + "noise_summary.txt", "w") as f:
    for sp in ("val", "hold"):
        c = json.load(open(O + f"calib_{sp}.json")); P = [p for p in c["placebo"] if "LR_z" in p]
        lr = np.array([p["LR_z"] for p in P]); ne = np.array([p["nest"] for p in P]); se = np.array([p["nest_se"] for p in P]); z = ne / se
        f.write(f"[{sp}] {len(P)} label-free placebo columns on the pool of 16 held-out wide students (150 N(0,1), 150 within-feed permutations of real columns)\n")
        f.write(f"   plain LR (1 df): mean {lr.mean():.2f}, p95 {np.quantile(lr, .95):.2f}, p99 {np.quantile(lr, .99):.2f}, max {lr.max():.2f}\n")
        f.write(f"   nested NDCG read: mean {ne.mean():+.5f}, sd {ne.std():.5f}, own SE (mean) {se.mean():.5f}; quantiles 5/50/95/99: {np.quantile(ne, .05):+.5f} {np.quantile(ne, .5):+.5f} {np.quantile(ne, .95):+.5f} {np.quantile(ne, .99):+.5f}; max {ne.max():+.5f}; max z {z.max():+.2f}; share z >= 2: {(z >= 2).mean():.3f}\n")
        for N in (16, 39, 94, 254, 300, 508, 1000, 11617):
            e = norm.ppf(1 - 1 / (N + 1)) if N > 1 else 0  # typical maximum of N independent N(0,1)
            f.write(f"   best of N = {N:5d} pure-noise reads: typical max z {e:.2f} -> nested NDCG {ne.mean() + e * ne.std():+.5f} for a 1-column noise read (sd {ne.std():.5f}); a read with SE 0.0003 would print {e * 0.0003:+.5f}\n")
    for nm in ("val_TO16", "val_FILE", "hold_TO16"):
        c = json.load(open(O + f"cond_{nm}.json")); z = np.array(c["placebo_cz"])
        f.write(f"[conditional score test, {nm}] placebo cz: n {len(z)}, mean {z.mean():+.2f}, sd {z.std():.2f}, max |cz| {np.abs(z).max():.2f}, share |cz| >= 2: {(np.abs(z) >= 2).mean():.3f}, >= 3: {(np.abs(z) >= 3).mean():.3f};"
                f" control block {c['K']} columns, LR(block | pool) {c['LR_block']:.1f}, nested NDCG of re-weighting the ranker's own inputs {c['nest_block']:+.5f} +- {c['nest_block_se']:.5f}\n")

# ---- support table: candidates that are NOT trained inputs / copies / banned
cand = [r for r in rows if r["status"] in ("gate array", "not read by the ranker", "fine-tune extra only") and r["cz_valTO"] == r["cz_valTO"]]
both = [r for r in cand if r["cz_hold"] == r["cz_hold"]]
def best(r, a, b):
    x, y = r[a], r[b]
    return x if (y != y or abs(x) >= abs(y)) else y
for r in cand:
    r["bv"] = best(r, "cz_valTO", "czrk_valTO"); r["bf"] = best(r, "cz_valFILE", "czrk_valFILE"); r["bh"] = best(r, "cz_hold", "czrk_hold") if r["cz_hold"] == r["cz_hold"] else float("nan")
with open(O + "support_table.txt", "w") as f:
    n = len(cand); nb = len(both)
    f.write(f"candidates (not a trained input, not a copy of one, not banned, not in-sample on train rows): {n} columns with a val read, {nb} with val AND holdout arrays\n")
    for thr in (2.0, 2.5, 3.0, 4.0):
        pv = 2 * norm.sf(thr)
        a = sum(abs(r["cz_valTO"]) >= thr for r in cand); b = sum(abs(r["cz_hold"]) >= thr for r in both)
        c = sum(abs(r["cz_valTO"]) >= thr and abs(r["cz_hold"]) >= thr and np.sign(r["cz_valTO"]) == np.sign(r["cz_hold"]) for r in both)
        f.write(f"   |cz| >= {thr} (z form): val {a} of {n} (noise expects {n * pv:.1f}); holdout {b} of {nb} ({nb * pv:.1f}); BOTH with the same sign {c} of {nb} (noise expects {nb * pv * pv / 2:.2f})\n")
    f.write("\nTwo-witness list: same sign on val (train-only pool) and holdout, |cz| >= 2.5 on both in the z or the rank form. cz = conditional score z given pool + the ranker's own inputs.\n")
    f.write(f"{'column':30s} {'status':26s} cz val TO (z / rk)  val FILE (z / rk)   holdout (z / rk)   R2   dnats val / hold   plain nested NDCG val TO / holdout\n")
    sel = [r for r in both if abs(r["bv"]) >= 2.5 and abs(r["bh"]) >= 2.5 and np.sign(r["bv"]) == np.sign(r["bh"])]
    sel.sort(key=lambda r: -min(abs(r["bv"]), abs(r["bh"])))
    for r in sel:
        f.write(f"{r['key']:30s} {r['status']:26s} {r['cz_valTO']:+6.2f} / {r['czrk_valTO']:+6.2f}   {r['cz_valFILE']:+6.2f} / {r['czrk_valFILE']:+6.2f}   {r['cz_hold']:+6.2f} / {r['czrk_hold']:+6.2f}   {r['r2_val']:.2f}  {r['dnats_val']:.5f} / {r['dnats_hold']:.5f}"
                f"   {r['nest_valTO']:+.5f} +- {r['nest_valTO_se']:.5f} / {r['nest_hold']:+.5f} +- {r['nest_hold_se']:.5f}\n")
    f.write(f"\nVal-only candidates (no holdout array) with |cz| >= 3 on the train-only pool:\n")
    for r in sorted([r for r in cand if r["cz_hold"] != r["cz_hold"] and abs(r["bv"]) >= 3], key=lambda r: -abs(r["bv"])):
        f.write(f"{r['key']:30s} {r['status']:26s} {r['cz_valTO']:+6.2f} / {r['czrk_valTO']:+6.2f}   {r['cz_valFILE']:+6.2f} / {r['czrk_valFILE']:+6.2f}   R2 {r['r2_val']:.2f}  plain nested {r['nest_valTO']:+.5f} +- {r['nest_valTO_se']:.5f}\n")
    f.write(f"\nOne witness only (|cz| >= 3.5 on one, below 2 or opposite sign on the other):\n")
    for r in sorted([r for r in both if (abs(r["bv"]) >= 3.5 and (abs(r["bh"]) < 2 or np.sign(r["bv"]) != np.sign(r["bh"]))) or (abs(r["bh"]) >= 3.5 and (abs(r["bv"]) < 2 or np.sign(r["bv"]) != np.sign(r["bh"])))], key=lambda r: -max(abs(r["bv"]), abs(r["bh"]))):
        f.write(f"{r['key']:30s} {r['status']:26s} val {r['cz_valTO']:+6.2f} / {r['czrk_valTO']:+6.2f}   holdout {r['cz_hold']:+6.2f} / {r['czrk_hold']:+6.2f}\n")
    f.write("\nBanned / in-sample columns (the lens sees the known leaks; not candidates):\n")
    for r in sorted([r for r in rows if r["status"] in ("BANNED (leak)", "train rows in-sample (holdout read invalid)") and r["cz_hold"] == r["cz_hold"]], key=lambda r: -abs(r["cz_hold"]))[:14]:
        f.write(f"{r['key']:30s} {r['status']:44s} val {r['cz_valTO']:+6.2f} / {r['czrk_valTO']:+6.2f}   holdout {r['cz_hold']:+7.2f} / {r['czrk_hold']:+7.2f}\n")
    st = {}
    for r in rows: st[r["status"]] = st.get(r["status"], 0) + 1
    f.write("\nstatus counts over all columns read: " + json.dumps(st) + "\n")
print(open(O + "noise_summary.txt").read()); print(open(O + "support_table.txt").read())
