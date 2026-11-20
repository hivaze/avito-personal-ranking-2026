"""Print ctx_ harness results from reports/feature_evals.jsonl."""
import json
for l in open("<repo>/reports/feature_evals.jsonl"):
    d = json.loads(l)
    if "ctx" in d["tag"] or any(n.startswith("ctx_") for n in d["family"]):
        print(f"{d['tag'][:44]:44s} n={d['n_family']:3d} base {d['base_ndcg']:.5f} cand {d['cand_ndcg']:.5f} "
              f"delta {d['delta']:+.5f} se {d['delta_se']:.5f} seeds {[round(x, 5) for x in d['per_seed_delta']]} {d['seconds']}s")
