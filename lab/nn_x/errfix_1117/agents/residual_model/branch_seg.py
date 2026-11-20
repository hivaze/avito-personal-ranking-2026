"""Per-feed NDCG under branch weights w*s_ft+(1-w)*s_to, aggregated by feed segments."""
import numpy as np, polars as pl
import prep
from prep import kit
df = kit.load()
off, sz = kit.feed_layout(df); y = df['label'].to_numpy()
sft = df['s_ft'].to_numpy().astype(float); sto = df['s_to'].to_numpy().astype(float)
W = [0.0, 0.25, 0.5, 0.75, 1.0]
nd = np.column_stack([kit.ndcg_per_feed(w*sft+(1-w)*sto, y, off, sz) for w in W])
d = nd - nd[:, [2]]
fl = df[off]
seg = {
 'M': np.digitize(fl['M'].to_numpy(), [20, 50, 100, 200]),
 'H': np.digitize(fl['H'].to_numpy(), [1, 10, 50, 150, 256]),
 'guest': fl['guest'].to_numpy().astype(int),
 'uvec_missing': fl['uvec_missing'].to_numpy().astype(int),
 'newest_src': fl['newest_src'].to_numpy(),
 'newest_type': fl['newest_type'].to_numpy(),
 'hour': np.digitize(fl['hour'].to_numpy(), [6, 12, 18]),
 'n_feed_cats': np.digitize(fl['n_feed_cats'].to_numpy(), [3, 6, 10, 15]),
 'n_hist_cats32': np.digitize(fl['n_hist_cats32'].to_numpy(), [1, 2, 4, 8]),
}
for name, g in seg.items():
    print('==', name)
    for v in np.unique(g):
        m = g == v
        if m.sum() < 100: continue
        se = d[m].std(0, ddof=1) / np.sqrt(m.sum())
        print(f"  {v:>4} n={m.sum():5d} " + " ".join(f"w{W[i]}:{d[m,i].mean():+.4f}({se[i]:.4f})" for i in (0, 1, 3, 4)))
