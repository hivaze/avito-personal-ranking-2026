"""Pilot on validation: a 5% eval slice by user (blind cards), a disjoint labelled study set,
and calibration numbers from all non-eval feeds. Labels of the eval slice go to private/ only."""
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from cards import Split  # noqa: E402
from avito_ranker.metrics import ndcg_per_feed, ndcg  # noqa: E402

OUT = Path(__file__).parent / "val"
EVAL_PER_BATCH = 15
STUDY_BATCHES, STUDY_PER_BATCH = 5, 20

s = Split("val")
F = s.feeds
nd = ndcg_per_feed(s.score, s.target, F.offsets, F.sizes)
print("val NDCG of FD16P20 final score:", round(float(nd.mean()), 5), "(record 0.40271)")

rng = np.random.default_rng(20261117)
g = s.group_key()
groups = np.unique(g)
rng.shuffle(groups)
target_n = int(np.ceil(0.05 * F.n_feeds))
cnt = {k: v for k, v in zip(*np.unique(g, return_counts=True))}
eval_groups, n = [], 0
for k in groups:
    if n >= target_n:
        break
    eval_groups.append(k); n += cnt[k]
is_eval = np.isin(g, np.array(eval_groups))
eval_idx = np.nonzero(is_eval)[0]
rest = np.nonzero(~is_eval)[0]
print("eval feeds", len(eval_idx), "users", len(eval_groups), "rest", len(rest))

# study set: disjoint users; half random, half feeds where the model's best positive is below rank 5
best_rank = np.empty(F.n_feeds, int)
for i in range(F.n_feeds):
    rows = s.order(i)
    best_rank[i] = int(np.nonzero(s.target[rows])[0][0]) + 1
rest_users = {}
for i in rng.permutation(rest):
    rest_users.setdefault(g[i], i)          # one feed per user
pool = np.array(list(rest_users.values()))
n_half = STUDY_BATCHES * STUDY_PER_BATCH // 2
rand_part = pool[:n_half]
bad_pool = pool[n_half:][best_rank[pool[n_half:]] > 5]
bad_part = bad_pool[:n_half]
study = rng.permutation(np.concatenate([rand_part, bad_part]))
print("study feeds", len(study), "of which model-miss", np.isin(study, bad_part).sum())

# calibration on all non-eval feeds
calib = {}
r_rest = best_rank[rest]
calib["ndcg_mean"] = float(nd[rest].mean())
calib["share_best_positive_at_rank"] = {k: float(v) for k, v in {
    "1": (r_rest == 1).mean(), "2-3": ((r_rest >= 2) & (r_rest <= 3)).mean(), "4-10": ((r_rest >= 4) & (r_rest <= 10)).mean(),
    "11-30": ((r_rest >= 11) & (r_rest <= 30)).mean(), "31+": (r_rest >= 31).mean()}.items()}
buckets = [(1, 1), (2, 2), (3, 3), (4, 5), (6, 10), (11, 20), (21, 50), (51, 100), (101, 300)]
pr = {f"{a}-{b}": [0, 0] for a, b in buckets}
size_bins = [(5, 30), (31, 100), (101, 300)]
for i in rest:
    rows = s.order(i)
    t = s.target[rows]
    for a, b in buckets:
        seg = t[a - 1:b]
        pr[f"{a}-{b}"][0] += int(seg.sum()); pr[f"{a}-{b}"][1] += len(seg)
calib["positive_rate_by_model_rank"] = {k: round(v[0] / v[1], 4) for k, v in pr.items() if v[1]}
calib["mean_positives_per_feed"] = float(np.mean([s.target[F.offsets[i]:F.offsets[i] + F.sizes[i]].sum() for i in rest]))
calib["share_feeds_one_positive"] = float(np.mean([s.target[F.offsets[i]:F.offsets[i] + F.sizes[i]].sum() == 1 for i in rest]))
print(json.dumps(calib, indent=1))

# write cards
for sub in ["eval", "study", "private"]:
    (OUT / sub).mkdir(parents=True, exist_ok=True)
eval_order = rng.permutation(eval_idx)
n_batches = int(np.ceil(len(eval_order) / EVAL_PER_BATCH))
manifest = {"eval": [], "study": []}
key = {}
for b in range(n_batches):
    idx = eval_order[b::n_batches]               # round-robin keeps batch sizes similar
    text, labels = [], []
    for j, i in enumerate(idx):
        lab = f"E{b:02d}-{j:02d}"
        key[lab] = int(i)
        text.append(s.card(int(i), lab, with_labels=False))
    p = OUT / "eval" / f"batch_{b:02d}.txt"
    p.write_text("\n".join(text))
    manifest["eval"].append({"file": str(p), "feeds": [f"E{b:02d}-{j:02d}" for j in range(len(idx))],
                             "lines": sum(1 for _ in open(p)), "chars": p.stat().st_size})
for b in range(STUDY_BATCHES):
    idx = study[b * STUDY_PER_BATCH:(b + 1) * STUDY_PER_BATCH]
    text = [s.card(int(i), f"S{b}-{j:02d}", with_labels=True) for j, i in enumerate(idx)]
    p = OUT / "study" / f"study_{b}.txt"
    p.write_text("\n".join(text))
    manifest["study"].append({"file": str(p), "lines": sum(1 for _ in open(p)), "chars": p.stat().st_size})
json.dump(key, open(OUT / "private" / "eval_key.json", "w"))
json.dump(calib, open(OUT / "calibration.json", "w"), indent=1)
json.dump(manifest, open(OUT / "manifest.json", "w"), indent=1)
print("eval batches", n_batches, "chars/batch mean", np.mean([m["chars"] for m in manifest["eval"]]),
      "max", max(m["chars"] for m in manifest["eval"]), "lines max", max(m["lines"] for m in manifest["eval"]))
print("study chars", [m["chars"] for m in manifest["study"]])
