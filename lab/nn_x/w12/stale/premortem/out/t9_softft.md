## T9. Soft-target fine-tune pilot re-read (8655 feeds done by all arms of both members; final EMA; mean of the two members; paired per feed)
| quantity | plain val (pilot: see right) | pmix-private weights | cold share >= 0.30 | cold share < 0.30 | M >= 50 | M < 50 |
|---|---|---|---|---|---|---|
| file proxy, B soft targets - A baseline | -0.00036 ± 0.00031 (pilot -0.00036 ± 0.00031) | -0.00044 ± 0.00046 | -0.00034 ± 0.00048 | -0.00037 ± 0.00040 | -0.00002 ± 0.00033 | -0.00084 ± 0.00058 |
| member - TO, B - A | +0.00053 ± 0.00038 (pilot +0.00053 ± 0.00038) | +0.00012 ± 0.00056 | +0.00002 ± 0.00058 | +0.00093 ± 0.00051 | +0.00047 ± 0.00047 | +0.00062 ± 0.00065 |
| file proxy of the baseline A - TO pool | +0.00054 ± 0.00053 (pilot +0.00054 ± 0.00053) | +0.00143 ± 0.00086 | +0.00116 ± 0.00085 | +0.00007 ± 0.00068 | -0.00018 ± 0.00063 | +0.00157 ± 0.00094 |
CONTROL (plain val within 0.00003 of -0.00036): VALID. RULE (block 6): pmix-private -0.96 SE, cold >= 0.30 -0.71 SE -> the closure also holds at the private mix (same low power).
