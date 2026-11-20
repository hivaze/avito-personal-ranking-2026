"""Markdown table of famS arms from preds/<tag>.eval.json + reads/nats_<tag>.json + curve (train-side selection NDCG).
usage: python fs_table.py <tag> [<tag> ...]"""
import sys, json
from pathlib import Path
H = Path("<repo>/nn_x/w12/cb/famS")
print("| tag | trees used / built | fit min | sel NDCG (train-side) | val | holdout | rho val / hold | NATS val | NATS holdout | PROMISING | member rule w 0.1: val, holdout, pooled z |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for t in sys.argv[1:]:
    try:
        n = json.load(open(H / f"reads/nats_{t}.json"))
    except FileNotFoundError:
        print(f"| {t} | no nats read |"); continue
    v, h = n["val"], n["hold"]
    tr = fm = sel = "n/a"
    p = H / f"preds/{t}.eval.json"
    if p.exists():
        e = json.load(open(p))["extra"]
        tr = f"{e['best_iter'] + 1} / {e['trees_built']}"; fm = f"{e['fit_seconds'] / 60:.0f}"
        c = json.load(open(H / f"preds/{t}.curve.json")); vv = [m.get("validation:NDCG:type=Base") for _, _, m in c if m]
        sel = f"{max(vv):.5f}"
    print(f"| {t} | {tr} | {fm} | {sel} | {v['level']:.5f} | {h['level']:.5f} | {v['rho']:.3f} / {h['rho']:.3f} | {v['nats']:+.5f} ± {v['nats_se']:.5f} | {h['nats']:+.5f} ± {h['nats_se']:.5f} | "
          f"{'YES' if n['promising'] else 'no'} | {v['mix01']:+.5f} ± {v['mix01_se']:.5f}, {h['mix01']:+.5f} ± {h['mix01_se']:.5f}, z {n['pooled_z']:+.2f} {'PASS' if n['member_pass'] else 'no pass'} |")
