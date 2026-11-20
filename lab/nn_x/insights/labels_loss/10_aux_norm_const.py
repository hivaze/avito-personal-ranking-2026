"""Constant c for a feed-normalised K9 aux that keeps the epoch-mean aux weight of the live recipe:
aux_feed = aux_w * c * sum_f (1-p_f) CE_aux_f / max_feeds  vs live aux = aux_w * sum_f (1-p_f) CE_aux_f / sum_f (1-p_f).
Also the epoch-mean aux:main ratio before/after K1 (main batch-mean -> sum/512)."""
import sys, numpy as np
sys.path.insert(0, "<repo>/src")
from common import feed_meta
f, M, _ = feed_meta("train")
pr = np.load("<repo>/eda/labels_position/out/train_prandom.npz")
p = np.nan_to_num(pr["p_random"].astype(np.float32), nan=0.0); p[M < 50] = 0
cand = np.where((p > 0.9) & (M >= 50))[0]
hold = np.sort(np.random.default_rng(12345).choice(cand, 30000, replace=False))
tr = np.ones(len(M), bool); tr[hold] = False
Mp = (M + 15) // 16 * 16
nb, sq, sB = 0, 0.0, 0.0
per = []
for mp in np.unique(Mp[tr]):
    m = tr & (Mp == mp); B = min(512, 32768 // mp); n = m.sum(); k = int(np.ceil(n / B))
    nb += k; sq += (1 - p[m]).sum() / 512; sB += n / 512
    per.append((mp, B, (1 - p[m]).mean() * B / 512))
c = nb / sq
print(f"regular batches/epoch {nb}; mean sum(1-p)/512 per batch {sq/nb:.4f} -> c = {c:.3f}")
print(f"epoch-mean main weight per batch: batch-mean 1.0 vs feed-norm {sB/nb:.3f}; aux:main ratio 0.5 -> {0.5/(sB/nb):.3f}")
for mp, B, r in per[::3]:
    print(f"  Mpad {mp:3d} B {B:3d}: live aux/main per batch {0.5/(B/512):.2f}; feed-norm aux/main {0.5*c*r/(B/512):.2f}")
