import hk, numpy as np
c = hk.Ctx(); s = c.score; r = c.col("rank")
for nm in ["h10", "h5", "h32", "hmax", "hmean5"]:
    x = c.col(nm).astype(float); xz = np.nan_to_num((x - np.nanmean(x)) / np.nanstd(x), nan=0.0)
    line = []
    for w in [-0.01, -0.02, -0.03, -0.05, 0.02]:
        m, se, d = c.delta(s + w * xz)
        line.append(f"w={w}:{m*1e4:+.1f}({m/se:+.1f}) f[{','.join(f'{d[c.fold_feed==f].mean()*1e4:+.0f}' for f in range(5))}]")
    print(nm, " | ".join(line))
