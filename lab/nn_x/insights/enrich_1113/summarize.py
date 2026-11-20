import json
rows = [json.loads(l) for l in open("out/results.jsonl")]
print(f"{'item':4s} {'columns':52s} {'AUC(first col)':14s} {'val delta':>10s} {'SE':>8s} {'z':>6s} {'holdout':>9s} {'late':>9s} {'pmix':>9s} verdict")
for r in rows:
    a = list(r["auc"].items())
    print(f"{r['id']:4s} {','.join(k for k, _ in a)[:52]:52s} {a[0][1]:.4f}         {r['val_delta']:+.5f} {r['val_se']:.5f} {r['val_z']:+6.2f} {r['hold_delta']:+.5f} {r['late_delta']:+.5f} {r['pmix_delta']:+.5f} {r['verdict']}")
