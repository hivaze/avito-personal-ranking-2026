### 2.4 YetiRank on the teacher logit (S4y; "if time allows", information only) [M]

`YetiRank` with the real-valued centred teacher logit as the label (pair weight proportional to the label gap [D, docs_rank 2c]), lr 0.3, `l2_leaf_reg` 5, 33-minute cap: 117 trees at 17.1 s per tree (4.5x an RMSE tree, 2.6x a QuerySoftMax tree), best selection iteration 114. Level 0.38193 / 0.30461.

| comparison | val | holdout | pooled z | reading |
|---|---|---|---|---|
| S4y - S1f, equal rounds 50 | +0.00249 ± 0.00075 | +0.00294 ± 0.00048 | +6.65 | per tree YetiRank learns as fast as the listwise soft label |
| S4y - S1f, equal rounds 100 | +0.00220 ± 0.00077 | +0.00217 ± 0.00047 | +5.25 | |
| S4y - S4b, equal rounds 50 | +0.00108 ± 0.00073 | -0.00010 ± 0.00045 | +0.89 | |
| S4y - S4b, equal rounds 100 | +0.00055 ± 0.00072 | -0.00057 ± 0.00044 | -0.38 | |
| S4y - S1f, equal wall time (33 min: 114 vs 500 trees) | -0.00130 ± 0.00086 | -0.00190 ± 0.00049 | -3.82 | per CPU minute it loses to RMSE |
| S4y - S4b, equal wall time (33 min: 114 vs 300 trees) | -0.00175 ± 0.00079 | -0.00313 ± 0.00046 | -6.33 | and to QuerySoftMax on the soft label |
| S4y - `ts_lr020_s0` (hard labels, 509 trees) | +0.00043 ± 0.00088 | -0.00001 ± 0.00050 | +0.33 | at its cap it is at the hard-label level |

- S4y has the LOWEST rho to W16 of the family (0.837 / 0.840, below the hard-label tree's 0.847) and no more NATS than the other students (+0.00058 ± 0.00027 / +0.00038 ± 0.00014). Its low rho is an unfinished fit, not independent information: given [W16, hard-label tree] it adds +0.00014 ± 0.00013 / +0.00000 ± 0.00000. **rho alone does not screen a tree arm; the NATS does.**
- On CPU the pairwise loss on a real-valued label is not worth its cost: 500 trees would take 2.4 hours.
