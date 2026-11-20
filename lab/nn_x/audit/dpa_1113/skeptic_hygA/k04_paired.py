"""Paired per-feed difference uncapped - capped (same columns, same GBM / LIN harness) from k01 npz outputs."""
import numpy as np, sys
sys.path.insert(0, "<repo>/src")
import pmix
W = pmix.weights()["private"]
d = "/tmp/scratchpad/dpa/skeptic_hygA/out/"
for base in ["G3", "r6_cd_m02_r101"]:
    for a, c in [("T1c_mass_uncapped_gbm", "K1_mass_capped_gbm"), ("T1a_mass_log_z_lin", "K2_mass_capped_log_z_lin"),
                 ("K1_mass_capped_gbm", "noise2_gbm"), ("T1c_mass_uncapped_gbm", "noise2_gbm")]:
        try:
            A = np.load(f"{d}k01_{a}_{base}.npz"); C = np.load(f"{d}k01_{c}_{base}.npz")
        except FileNotFoundError:
            continue
        dv = A["dv"] - C["dv"]; dh = A["dh"] - C["dh"]
        sv = dv.std() / np.sqrt(len(dv)); sh = dh.std() / np.sqrt(len(dh)); pm, pse = pmix.wmean_se(dv, W)
        print(f"{base:16s} {a} - {c}: val {dv.mean():+.5f} ({sv:.5f}) z {dv.mean()/sv:+.2f} | pmix {pm:+.5f} ({pse:.5f}) | hold {dh.mean():+.5f} ({sh:.5f}) z {dh.mean()/sh:+.2f}")
