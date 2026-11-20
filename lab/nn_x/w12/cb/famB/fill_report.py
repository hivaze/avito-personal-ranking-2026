"""Fill the arm table and the nats lines of reports/CATBOOST_1115/famB.md from the prediction files (run once, at the end).
usage: python -u fill_report.py"""
import json, sys, io, contextlib
sys.path.insert(0, "<repo>/nn_x/w12/cb/famB"); sys.path.insert(0, "<repo>/nn_x/w12/cb")
from fb_common import *
from nats import screen

WHAT = {
    "b0": "B0 reference (infra `ts_lr020_s0`): the 177 features, `category_id` native with default CTR",
    "b0_s1": "B0 recipe, seed 1 (62-minute budget)",
    "b3_abc_s0": "B3: B0 + 70 hand-made columns (blocks a, b, c: user x cluster / category statistics, transition and user-cluster x item-cluster lifts, nearest-train-item statistics)",
    "b3_abc_s1": "B3, seed 1 (80-minute budget)",
    "b1_mcc4_s0": "B1: B0 + cl256 / cl1024 / cl4096 native, `has_time`, rows shuffled inside feeds, 255 CTR borders, max_ctr_complexity 4",
    "b1_mcc2_s0": "B1 with max_ctr_complexity 2 (45-minute budget)",
    "b2_item_s0": "B2: B0 + `item_id` native (`has_time`, 255 CTR borders, complexity 1, model_size_reg 0, ctr_leaf_count_limit 2M), pool xm + xs (436,320 fit feeds)",
}
ARMS = ["b0", "b0_s1", "b3_abc_s0", "b3_abc_s1", "b1_mcc4_s0", "b1_mcc2_s0", "b2_item_s0"]
BUNDLES = ["b0+b3_abc_s0", "b0+b0_s1+b3_abc_s0+b3_abc_s1", "b3_abc_s0+b3_abc_s1", "b0+b0_s1", "b0+b1_mcc4_s0", "b0+b1_mcc4_s0+b3_abc_s0"]
P = lambda tag, sp: np.load((B0V if sp == "val" else B0H) if tag == "b0" else PREDS / f"{tag}_{sp}.npy")
rows, lines, res = [], [], {}
hdr = "| tag | what changed against B0 | trees used (built) | fit min | level val / holdout | rho | NATS val / holdout (bar 0.0016 on both) | PROMISING | member rule 0.9 / 0.1: val / holdout, pooled z |\n|---|---|---|---|---|---|---|---|---|"


def fmt(tag, what, r, trees="", mins=""):
    v, h = r["val"], r["hold"]
    return (f"| `{tag}` | {what} | {trees} | {mins} | {v['level']:.5f} / {h['level']:.5f} | {v['rho']:.3f} / {h['rho']:.3f} | {v['nats']:+.5f} ± {v['nats_se']:.5f} / {h['nats']:+.5f} ± {h['nats_se']:.5f} | "
            f"{'YES' if r['promising'] else 'no'} | {v['mix01']:+.5f} ± {v['mix01_se']:.5f} / {h['mix01']:+.5f} ± {h['mix01_se']:.5f}, {r['pooled_z']:+.2f} ({'PASS' if r['member_pass'] else 'no pass'}) |")


def run(tag, pv, ph):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = screen(pv, ph, tag=tag)
    lines.append(buf.getvalue().strip()); return r


for tag in ARMS:
    if tag != "b0" and not (PREDS / f"{tag}_val.npy").exists():
        continue
    r = run(tag, P(tag, "val"), P(tag, "hold")); res[tag] = r
    ej = cbx.ROOT / "nn_x/w12/cb/infra/out/ts_lr020_s0.eval.json" if tag == "b0" else PREDS / f"{tag}.eval.json"
    e = json.load(open(ej)).get("extra", {})
    rows.append(fmt(tag, WHAT[tag], r, f"{e['best_iter'] + 1} ({e.get('trees_built', '')})", f"{e.get('fit_seconds', 0) / 60:.0f}"))
for b in BUNDLES:
    tags = b.split("+")
    if not all(t == "b0" or (PREDS / f"{t}_val.npy").exists() for t in tags):
        continue
    r = run("BUNDLE(" + b + ")", [P(t, "val") for t in tags], [P(t, "hold") for t in tags]); res["bundle:" + b] = r
    rows.append(fmt("bundle(" + " + ".join(tags) + ")", "member-equal mean of per-feed z (information only: no arm is PROMISING)", r))
json.dump(res, open(HERE / "out/final_table.json", "w"), indent=1, default=float)
(HERE / "out/final_table.md").write_text(hdr + "\n" + "\n".join(rows) + "\n")
(HERE / "out/final_nats_lines.txt").write_text("\n".join(lines) + "\n")
p = cbx.ROOT / "reports/CATBOOST_1115/famB.md"; s = p.read_text()
if "ARM_TABLE_PENDING" in s:
    s = s.replace("ARM_TABLE_PENDING", hdr + "\n" + "\n".join(rows))
if "NATS_LINES_PENDING" in s:
    s = s.replace("NATS_LINES_PENDING", "\n".join(lines))
p.write_text(s)
print(hdr + "\n" + "\n".join(rows))
