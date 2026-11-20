p = '<repo>/reports/NEXT_1115/u20wit.md'
s = open(p).read()
R = [
 ("on H0 both B - A (-0.00086 ± 0.00076) and the out-of-sample C - A (+0.00038 ± 0.00082, against +0.00143 ± 0.00040 on the cross-user holdout) read low, and base - xf0 does the same (+0.00152 ± 0.00081 against +0.00251 ± 0.00048); I have no mechanism for it and each gap is within 2 SE.",
  "on the fold-0 holdout feeds both B - A and the out-of-sample C - A read lower than on the cross-user holdout (GPU files: B - A -0.00111 ± 0.00075 against +0.00061 ± 0.00038; C - A +0.00055 ± 0.00083 against +0.00143 ± 0.00040), and base - xf0 does the same (+0.00127 ± 0.00097 against +0.00251 ± 0.00048, seeds 951, 952). A is common to the first two gaps, so a fluctuation of the two A seeds on these 5,974 feeds would produce both [I]; I have no mechanism, and each gap is within 2 SE (B - A: 2.0 SE)."),
 ("- **Short feeds** (39% of test):", "- **Short feeds** (39% of test feeds by SYNTHESIS bet 3; not re-measured):"),
 ("positives per feed match val M < 50 [M].", "positives per feed within 0.04 of val M < 50 in every size bin [M]."),
]
for a, b in R:
    assert s.count(a) == 1, a[:60]
    s = s.replace(a, b)
open(p, 'w').write(s); print('patched', len(R))
