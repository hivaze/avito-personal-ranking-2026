"""C03 test: one LIN with the 6 primary columns loc16, mode6, mode96, g_shift, g_f2res, g_cmatch on G3."""
import harness as Hn
from harness import np, log
v = np.load(Hn.CACHE / "c03_val.npz")
h = np.load(Hn.CACHE / "c03_hold.npz")
N = ["loc16", "mode6", "mode96", "g_shift", "g_f2res", "g_cmatch"]
info = {}
for n in N:
    info[n] = Hn.lin_info(n, [v[n]], f"single column {n} (information)")
Hn.run_item("C03", N, [v[n] for n in N], [h[n] for n in N], extra=info)
