"""Cold-item check: mean within-feed percentile rank of positives (0.5 = random), split by whether the candidate item was
seen in train (as candidate or in history). usage: python cold.py name1,name2,..."""
import sys
sys.path.insert(0, "<repo>/feats_v2/clu_")
from clu_common import *
from build import rel_rank

nb.set_num_threads(8)
p = WORK / "val_seen.npy"
if not p.exists():
    NI = 111_400_000
    seen = np.zeros(NI, bool)
    seen[np.load(FLAT / "train/cand_item_id.npy")] = True
    seen[np.load(FLAT / "train/hist_item_id.npy")] = True
    np.save(p, seen[np.load(FLAT / "val/cand_item_id.npy")])
seen = np.load(p)
fe = feeds("val")
y = np.load(FLAT / "val/cand_target.npy").astype(bool)
print(f"val unseen rows {1-seen.mean():.3f}, positives unseen {1-seen[y].mean():.3f}")
print("| feature | AUC-like seen pos | AUC-like unseen pos |\n|---|---|---|")
for n in sys.argv[1].split(","):
    x = np.load(FEATS / "val" / f"{n}.npy").astype(np.float32)
    r = rel_rank(x + np.float32(1e-6) * np.random.default_rng(0).random(len(x)).astype(np.float32), fe["co"], fe["M"])
    r = np.where(np.isnan(r), 0.0, r)
    print(f"| {n} | {r[y & seen].mean():.4f} | {r[y & ~seen].mean():.4f} |", flush=True)
