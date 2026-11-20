"""Markdown rows for the report: one row per arm (tag, trees, fit minutes, level, rho, nats, member-rule statistic) from preds/<tag>.eval.json and nats.screen.
usage: python -u mk_table.py tag1 tag2 ... [--bundle a+b ...]      ("b0" = infra ts_lr020_s0)"""
import json, sys
sys.path.insert(0, "<repo>/nn_x/w12/cb/famB"); sys.path.insert(0, "<repo>/nn_x/w12/cb")
from fb_common import *
from nats import screen

args = sys.argv[1:]; bundles = []
if "--bundle" in args:
    i = args.index("--bundle"); bundles = args[i + 1:]; args = args[:i]


def P(tag, sp):
    return np.load((B0V if sp == "val" else B0H) if tag == "b0" else PREDS / f"{tag}_{sp}.npy")


def row(tag, r, trees="", mins=""):
    v, h = r["val"], r["hold"]
    return (f"| `{tag}` | {trees} | {mins} | {v['level']:.5f} / {h['level']:.5f} | {v['rho']:.3f} / {h['rho']:.3f} | {v['nats']:+.5f} ± {v['nats_se']:.5f} / {h['nats']:+.5f} ± {h['nats_se']:.5f} | "
            f"{'PROMISING' if r['promising'] else 'no'} | {v['mix01']:+.5f} ± {v['mix01_se']:.5f} / {h['mix01']:+.5f} ± {h['mix01_se']:.5f}, pooled z {r['pooled_z']:+.2f} {'PASS' if r['member_pass'] else 'no pass'} |")


print("| tag | trees used (built) | fit min | level val / holdout | rho | NATS val / holdout | screen | member rule 0.9 / 0.1 |\n|---|---|---|---|---|---|---|---|")
for tag in args:
    r = screen(P(tag, "val"), P(tag, "hold"), verbose=False, tag=tag)
    ej = cbx.ROOT / "nn_x/w12/cb/infra/out/ts_lr020_s0.eval.json" if tag == "b0" else PREDS / f"{tag}.eval.json"
    trees = mins = ""
    if ej.exists():
        e = json.load(open(ej)).get("extra", {})
        if "best_iter" in e:
            trees = f"{e['best_iter'] + 1} ({e.get('trees_built', '')})"; mins = f"{e.get('fit_seconds', 0) / 60:.0f}"
    print(row(tag, r, trees, mins))
for b in bundles:
    tags = b.split("+")
    r = screen([P(t, "val") for t in tags], [P(t, "hold") for t in tags], verbose=False, tag=b)
    print(row("bundle(" + " + ".join(tags) + ")", r))
