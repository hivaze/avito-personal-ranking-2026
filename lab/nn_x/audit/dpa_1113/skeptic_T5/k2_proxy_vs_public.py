# numbers copied from dpa/validation/out/s7_m50.log (contrast vs G3): public | val all | val M>=50 | val-TO M>=50 | holdout
import numpy as np
rows = {"Cd": (-0.00074, -0.00131, -0.00139, -0.00126, -0.00060),
        "Pall": (-0.00038, -0.00136, -0.00133, -0.00109, -0.00045),
        "PF7x": (-0.00010, -0.00058, -0.00046, -0.00050, -0.00020),
        "M2": (-0.00039, +0.00062, +0.00038, +0.00036, -0.00018),
        "Q1a": (+0.00007, +0.00036, +0.00051, +0.00097, -0.00052),
        "Q2a": (+0.00039, +0.00029, +0.00088, +0.00116, -0.00002)}
A = np.array(list(rows.values()))
pub = A[:, 0]
names = ["val_all", "val_M50", "valTO_M50", "holdout"]
for j, n in enumerate(names, 1):
    x = A[:, j]
    k = (x @ pub) / (x @ x)
    r = np.corrcoef(x, pub)[0, 1]
    rmse1 = np.sqrt(np.mean((pub - x) ** 2)); rmsek = np.sqrt(np.mean((pub - k * x) ** 2))
    sign = int((np.sign(x) == np.sign(pub)).sum())
    print(f"{n:10s} corr {r:+.3f}  k {k:+.2f}  rmse(k=1) {rmse1:.5f}  rmse(k fit) {rmsek:.5f}  sign agree {sign}/6")
print(f"rmse(public - 0) {np.sqrt(np.mean(pub**2)):.5f}  (paired public SE per contrast ~0.0004-0.0006, s4_power / CURRENT_STATUS 14:17)")
# bootstrap-free sanity: with 6 points and public noise SE 0.00045, simulate corr of a PERFECT proxy (truth = proxy) with noisy public
rng = np.random.default_rng(0)
for j, n in enumerate(names, 1):
    x = A[:, j]
    cs = [np.corrcoef(x, x + rng.normal(0, 0.00045, 6))[0, 1] for _ in range(20000)]
    cs_k = [np.corrcoef(x, 0.5 * x + rng.normal(0, 0.00045, 6))[0, 1] for _ in range(20000)]
    print(f"{n:10s} if public = proxy + noise(0.00045): corr 5-50-95% {np.percentile(cs,[5,50,95]).round(3)} ; if public = 0.5*proxy + noise: {np.percentile(cs_k,[5,50,95]).round(3)}")
