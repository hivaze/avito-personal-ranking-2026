"""Lead's notice (2), second half: LABEL-FREE distribution of every new column on train rows (xm, holdout) against val and test rows.
Per column and set: share of NaN rows, quantiles 10 / 50 / 90 of the finite values, mean. Writes out/col_dist.md and out/col_dist.json; prints the columns
whose median or NaN share differs most between xm and val / test.   usage: python -u col_dist.py [lag]   ("lag": train-row files of the test-lag variant)"""
import json, sys
sys.path.insert(0, "<repo>/nn_x/w12/cb/famB")
from fb_common import *
lag = len(sys.argv) > 1 and sys.argv[1] == "lag"
res = {}; lines = ["| column | set | NaN share | p10 | p50 | p90 | mean |", "|---|---|---|---|---|---|---|"]
for blk in ("a", "b", "c"):
    names = json.load(open(FEAT / f"{blk}_names.json"))
    A = {s: np.load(FEAT / f"{'lag_' if lag and s in ('xm', 'hold') else ''}{blk}_{s}.npy", mmap_mode="r") for s in ("xm", "hold", "val", "test")}
    for j, n in enumerate(names):
        res[n] = {}
        for s in ("xm", "hold", "val", "test"):
            x = np.asarray(A[s][:, j]) if s != "xm" else np.asarray(A[s][::4, j])       # xm: every 4th row (4.9M rows) is enough for quantiles
            f = np.isfinite(x); v = x[f]
            q = np.quantile(v, [0.1, 0.5, 0.9]) if len(v) else [np.nan] * 3
            res[n][s] = dict(nan=float(1 - f.mean()), p10=float(q[0]), p50=float(q[1]), p90=float(q[2]), mean=float(v.mean()) if len(v) else float("nan"))
            lines.append(f"| `{n}` | {s} | {1 - f.mean():.4f} | {q[0]:.4g} | {q[1]:.4g} | {q[2]:.4g} | {res[n][s]['mean']:.4g} |")
tag = "_lag" if lag else ""
(HERE / f"out/col_dist{tag}.md").write_text("\n".join(lines) + "\n"); json.dump(res, open(HERE / f"out/col_dist{tag}.json", "w"), indent=1)
# shift score: |p50 difference| / (p90 - p10 on xm), and NaN-share difference
print(f"columns: {len(res)}; shift of val / test against xm{' (lagged train rows)' if lag else ''}: (median shift in units of the xm p10-p90 range) / (NaN-share difference)")
rows = []
for n, r in res.items():
    rng = max(r["xm"]["p90"] - r["xm"]["p10"], 1e-9)
    rows.append((max(abs(r["val"]["p50"] - r["xm"]["p50"]), abs(r["test"]["p50"] - r["xm"]["p50"])) / rng, n,
                 (r["val"]["p50"] - r["xm"]["p50"]) / rng, (r["test"]["p50"] - r["xm"]["p50"]) / rng, r["val"]["nan"] - r["xm"]["nan"], r["test"]["nan"] - r["xm"]["nan"], (r["hold"]["p50"] - r["xm"]["p50"]) / rng))
for sc, n, dv, dt, nv, nt, dh in sorted(rows, reverse=True)[:16]:
    print(f"  {n:18s} median shift val {dv:+.2f} test {dt:+.2f} holdout {dh:+.2f} | NaN share val {nv:+.3f} test {nt:+.3f}")
