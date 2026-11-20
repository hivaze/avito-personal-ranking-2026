## T5a. More fine-tune seeds (3 seeds per fold against 1), scored pairs on record
| pair | val FINAL | pmix-public | pmix-private | PUBLIC ± SE |
|---|---|---|---|---|
| C2 - G3 (G3 members, FT 3 seeds - 1 seed) | +0.00018 ± 0.00012 | +0.00023 ± 0.00015 | +0.00021 ± 0.00019 | -0.000066 ± 0.00016 |
| C3 - C1 (G3 + pop1 x8, FT 3 seeds - 1 seed) | +0.00012 ± 0.00011 | +0.00018 ± 0.00013 | +0.00022 ± 0.00016 | +0.000060 ± 0.00014 |
| C1 - G3 (pop1 x8 added, FT 1 seed) | +0.00057 ± 0.00040 | +0.00090 ± 0.00050 | +0.00107 ± 0.00062 | +0.000574 ± 0.00051 |
| C3 - C2 (pop1 x8 added, FT 3 seeds) | +0.00051 ± 0.00040 | +0.00085 ± 0.00051 | +0.00108 ± 0.00063 | +0.000700 ± 0.00052 |

## T5b. Level: the 7 scored files against the line on record (public = 0.0074 + 0.963 val; no refit)
| file | val FINAL | PUBLIC | line | residual | pmix-public level | PUBLIC - pmix-public level |
|---|---|---|---|---|---|---|
| W8P20 | 0.40214 | 0.395009 | 0.39466 | +0.00035 | 0.39923 | -0.00422 |
| W8 | 0.40165 | 0.394486 | 0.39419 | +0.00030 | 0.39860 | -0.00411 |
| W16 | 0.40158 | 0.394483 | 0.39412 | +0.00036 | 0.39874 | -0.00426 |
| S16 | 0.40173 | 0.394078 | 0.39426 | -0.00019 | 0.39838 | -0.00430 |
| W8TO | 0.40157 | 0.394009 | 0.39411 | -0.00010 | 0.39794 | -0.00393 |
| C3 | 0.40101 | 0.393454 | 0.39357 | -0.00012 | 0.39812 | -0.00467 |
| G3 | 0.40032 | 0.392820 | 0.39291 | -0.00009 | 0.39705 | -0.00423 |
Residual sd over the 7 files 0.00025 (on record: 0.0006); mean +0.00007.

## T5c. Slope of the public delta on each val column, 6 independent scored pairs (through the origin; SE from the public SEs only)
| column | slope ± SE | chi2 of public against slope = 1 (6 df) |
|---|---|---|
| val FINAL | +1.08 ± 0.49 | 2.1 |
| pmix-public FINAL | +0.78 ± 0.34 | 1.8 |
| pmix-private FINAL | +0.59 ± 0.26 | 4.1 |
| SVF | +0.56 ± 0.27 | 5.1 |
