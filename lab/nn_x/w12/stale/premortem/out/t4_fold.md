## T4b. Fold-to-fold agreement of the fine-tune branch on TEST rows (label-free; rule R5)
| unit | feed-mean Pearson r between two fold predictions (10 pairs) | v1 = 1 - r | r(fold k, 5-fold mean) | public feeds r / private feeds r |
|---|---|---|---|---|
| single member (mean of the 16 wide students) | 0.9969 | 0.0031 | 0.9989 | 0.9969 / 0.9969 |
| W8 FT pool (8 members) | 0.9973 | 0.0027 | 0.9989 (same top-1 item 0.927) | 0.9973 / 0.9974 |
| W16 FT pool (16 members) | 0.9974 | 0.0026 | 0.9990 (same top-1 item 0.928) | 0.9974 / 0.9974 |
| Pd FT pool (4 pop1 d128 students) | 0.9979 | 0.0021 | 0.9991 (same top-1 item 0.931) | 0.9978 / 0.9979 |

Fold noise of a single member v1 = 0.0031; of the 8-pool 0.0027; of the 16-pool 0.0026. If the fold noise were independent across members the 8-pool would have about 0.0004 and the 16-pool 0.0002: the part common to all members (same GroupKFold split) is what a val row keeps and a test row averages over 5 folds.

## T4c. SIMEX on val: what removing 4/5 of the fold noise of the out-of-fold FT pool would be worth (rule R5 white noise, R5b structured noise)
| pool | noise | FINAL val at lambda 0 | lambda 0.5 | lambda 1 | lambda 2 | extrapolated to lambda -0.8 (quadratic) | same, linear fit | gain FINAL at -0.8 (quadratic / linear) | FT branch alone: gain at -0.8 (quadratic / linear) |
|---|---|---|---|---|---|---|---|---|---|
| W8 | white | 0.40165 | 0.40156 | 0.40159 | 0.40159 | 0.40177 | 0.40163 | **+0.00012** / -0.00002 | +0.00044 / +0.00014 |
| W16 | white | 0.40158 | 0.40156 | 0.40144 | 0.40139 | 0.40172 | 0.40166 | **+0.00015** / +0.00008 | -0.00030 / +0.00048 |
| W16 | struct | 0.40158 | 0.40152 | 0.40152 | 0.40148 | 0.40166 | 0.40160 | **+0.00008** / +0.00002 | -0.00005 / +0.00015 |

**R5:** structured-noise SIMEX gain at FINAL level for W16 = +0.00008 (white noise: W8 +0.00012, W16 +0.00015). Bars: SUPPORTED iff >= 0.0002; NOT SUPPORTED AS THE MAIN CAUSE iff < 0.0001. -> NOT SUPPORTED AS THE MAIN CAUSE. For comparison: public reads the FT half of W8 at +0.00048 ± 0.00055, plain val at +0.00009 ± 0.00043, pmix-public at +0.00066 ± 0.00054.
