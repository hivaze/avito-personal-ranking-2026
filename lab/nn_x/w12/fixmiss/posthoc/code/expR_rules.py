"""Read-out of a few declared pop-light / FT-weight rules on the current members vs the current best file (no fitting;
these are grid points of the already-declared rule family; in-sample by construction, so information only)."""
import ph, numpy as np
D = ph.Data("val"); C = np.load("../out/comp_val.npz"); z = {k: ph.zf(C[k], D.off, D.M) for k in C.files}
Mr = np.repeat(D.M, D.M)
def fin(ws, wl, wft=0.5, wto_pl=None):
    w = np.where(Mr >= 50, wl, ws)
    ftm = ph.zf((1 - w) * z["ft_base"] + w * z["ft_pl"], D.off, D.M); tom = ph.zf((1 - w) * z["to_base"] + w * z["to_pl"], D.off, D.M)
    return wft * ftm + (1 - wft) * tom
for args, nm in (((0.4, 0.2), "current m0.4_0.2 (check = 0)"), ((0.4, 0.4), "m0.4_0.4"), ((0.4, 0.3), "m0.4_0.3"), ((0.5, 0.2), "m0.5_0.2"),
                 ((0.5, 0.3), "m0.5_0.3"), ((0.3, 0.2), "m0.3_0.2"), ((0.2, 0.2), "FD16P20 m0.2_0.2"), ((0.4, 0.2, 0.6), "m0.4_0.2 FT 0.6"), ((0.4, 0.2, 0.4), "m0.4_0.2 FT 0.4")):
    D.report(fin(*args), nm)
