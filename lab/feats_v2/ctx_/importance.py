"""Gain importance of family features in one LightGBM lambdarank fit on all of val (base + family), harness params.
Usage: python importance.py <prefixes comma-separated | =name1,name2,...> [threads]
"""
import sys, json
sys.path.insert(0, "<repo>/src")
from common import *
import lightgbm as lgb
from eval_family import PARAMS, ROUNDS, base_features

pref = tuple(sys.argv[1].split(","))
threads = int(sys.argv[2]) if len(sys.argv) > 2 else 16
f, M, off = feed_meta("val")
y = np.load(FLAT / "val/cand_target.npy").astype(np.float32)
base = base_features()
fam = [n for n in all_feature_names("val") if n.startswith(pref) and n not in base] if not sys.argv[1].startswith("=") \
    else sys.argv[1][1:].split(",")
names = base + fam
X = load_feats("val", names)
b = lgb.train(dict(PARAMS, num_threads=threads, seed=0), lgb.Dataset(X, y, group=M, feature_name=names,
              categorical_feature=["category_id"]), ROUNDS)
g = b.feature_importance("gain")
g = g / g.sum()
order = np.argsort(-g)
print("rank name gain_share")
for r, i in enumerate(order):
    if names[i] in fam or r < 15:
        print(r, names[i], f"{g[i]:.4f}", "*" if names[i] in fam else "")
