"""Check of the odd f2_lvlrank quintile act/exp in 09: bin sizes and act/exp by explicit level-rank bins."""
import sys; sys.path.insert(0, "<repo>/nn_x/codes/features")
exec(open("<repo>/nn_x/codes/features/09_residual_g3.py").read().split("# (b) nested")[0].replace('print(f"\\n(a)', 'pass #'))
x = X["f2_lvlrank"]
q = np.quantile(x, [0.2, 0.4, 0.6, 0.8]); b = np.searchsorted(q, x, side="right")
print("quintile cut points", q, "bin sizes", np.bincount(b, minlength=5) / len(x))
bins = [(-0.01, 0.0001), (0.0001, 0.25), (0.25, 0.4999), (0.4999, 0.5001), (0.5001, 0.75), (0.75, 0.9999), (0.9999, 1.01)]
for lo, hi in bins:
    m = (x >= lo) & (x < hi)
    print(f"lvlrank in [{lo},{hi}): rows {m.mean():.3f}, f2_zero share {X['f2_zero'][m].mean():.3f}, f2_mode share {X['f2_mode'][m].mean():.3f}, act/exp {y[m].sum()/exp_[m].sum():.3f}, rate {y[m].mean()*100:.2f}%")
