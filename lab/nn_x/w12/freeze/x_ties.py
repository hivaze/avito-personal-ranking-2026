"""Side check for audit items (b) / (f): exact within-feed ties of the final float32 test score of each candidate (stored branches).
Reason: the gate's line "B reproduction: min per-feed Spearman" reads 0.999987 .. 1.000000, while b_branches.py finds every file row-exact
against src/make_sub.py's rank of the stored branches. An exact float tie is ordered by make_sub's fixed random key; the gate's Spearman
ranks a tie differently, which is the only way two orders built from the same scores can differ. CPU, read-only. Output: out/x_ties.json."""
import sys, json
sys.path.insert(0, '<repo>/nn_x/w12/freeze'); sys.path.insert(0, '<repo>/src')
import numpy as np
from cands import OUT, PREDS, CANDS
from common import feed_meta
from stack_val import feed_rank_z
ft_, Mt, ot = feed_meta('test'); f = np.repeat(np.arange(len(Mt)), Mt)
Z = lambda x: feed_rank_z(ot, Mt, np.asarray(x, np.float32))[:, 1]
res = {'test_feeds': int(len(Mt)), 'test_feeds_M_lt_50': int((Mt < 50).sum()), 'test_feeds_M_ge_50': int((Mt >= 50).sum())}
for c in CANDS:
    s = (0.5 * Z(np.load(PREDS / f'ens_{c["tag"]}_ft_test.npy')) + 0.5 * Z(np.load(PREDS / f'ens_{c["tag"]}_to_test.npy'))).astype(np.float32)
    o = np.lexsort((s, f)); same = (np.diff(s[o]) == 0) & (np.diff(f[o]) == 0)
    res[c['id']] = dict(tied_adjacent_pairs=int(same.sum()), feeds_with_a_tie=int(len(np.unique(f[o][1:][same]))))
    print(f"{c['id']:18s} exact within-feed ties in the final float32 test score: {res[c['id']]['tied_adjacent_pairs']} adjacent pairs in {res[c['id']]['feeds_with_a_tie']} feeds")
print(f"test feeds {res['test_feeds']}: M < 50 {res['test_feeds_M_lt_50']}, M >= 50 {res['test_feeds_M_ge_50']}")
json.dump(res, open(OUT / 'x_ties.json', 'w'), indent=1)
