"""M6 (CPU): robustness of the M-split pop-light FT-branch form (pop-light part of the FT branch from train-only predictions
on feeds with M >= thr): thresholds 30/50/75/100/150, leave-one-pop-member-out, and the label-free test shift of the file
(FD16P20@m0.4_0.2, the current best). Information for the lead; the threshold 50 was fixed before this scan."""
import sys
sys.path.insert(0, "<repo>/lab/nn_x/w12/fixmiss/training")
import numpy as np
from pathlib import Path
from m3_branch_views import pool, Z, ev, line, Mv, SETS
from common import feed_meta
from stack_val import feed_rank_z
MROW = np.repeat(Mv, Mv)
base, pop = SETS["FD16P20"]
Fb, Fp, Tb, Tp = pool(base, "ft"), pool(pop, "ft"), pool(base, "to"), pool(pop, "to")
w = np.where(MROW >= 50, 0.2, 0.4).astype(np.float32)
to = (1 - w) * Z(Tb) + w * Z(Tp)
cur = ev(0.5 * Z((1 - w) * Z(Fb) + w * Z(Fp)) + 0.5 * Z(to))
stored = np.load("<repo>/preds/ens_FD16P20_m04_02kd_ft_val.npy") * 0.5 + np.load("<repo>/preds/ens_FD16P20_m04_02kd_to_val.npy") * 0.5
print(f"rebuilt m0.4_0.2 {cur.mean():.5f} vs stored file {ev(stored).mean():.5f}")
for thr in (30, 50, 75, 100, 150):
    Fpm = np.where(MROW >= thr, Z(Tp), Z(Fp))
    v = ev(0.5 * Z((1 - w) * Z(Fb) + w * Fpm) + 0.5 * Z(to))
    print(f"thr {thr:3d}: {line(v - cur)}")
for k in range(4):
    keep = [m for i, m in enumerate(pop) if i != k]
    Fp3, Tp3 = pool(keep, "ft"), pool(keep, "to")
    to3 = (1 - w) * Z(Tb) + w * Z(Tp3); c3 = ev(0.5 * Z((1 - w) * Z(Fb) + w * Z(Fp3)) + 0.5 * Z(to3))
    v3 = ev(0.5 * Z((1 - w) * Z(Fb) + w * np.where(MROW >= 50, Z(Tp3), Z(Fp3))) + 0.5 * Z(to3))
    print(f"leave out {pop[k]}: {line(v3 - c3)}")
# label-free test shift
P = Path("<repo>/preds"); ft_, Mt, ot = feed_meta("test"); MT = np.repeat(Mt, Mt)
Zt = lambda x: feed_rank_z(ot, Mt, np.asarray(x, np.float32))[:, 1]
pt = lambda ms, suf: np.mean([Zt(np.load(P / f"nn_{m}_{suf}.npy")) for m in ms], 0)
tFb, tFp, tTb, tTp = pt(base, "n2cpv3s3_test"), pt(pop, "n2cpv3s3_test"), pt(base, "trainonly_test"), pt(pop, "trainonly_test")
wt = np.where(MT >= 50, 0.2, 0.4).astype(np.float32)
tto = (1 - wt) * Zt(tTb) + wt * Zt(tTp)
tcur = 0.5 * Zt((1 - wt) * Zt(tFb) + wt * Zt(tFp)) + 0.5 * Zt(tto)
tnew = 0.5 * Zt((1 - wt) * Zt(tFb) + wt * np.where(MT >= 50, Zt(tTp), Zt(tFp))) + 0.5 * Zt(tto)
st = 0.5 * np.load(P / "ens_FD16P20_m04_02kd_ft_test.npy") + 0.5 * np.load(P / "ens_FD16P20_m04_02kd_to_test.npy")
print(f"test: corr(rebuilt, stored) {np.corrcoef(tcur, st)[0, 1]:.6f} | corr(new, stored) {np.corrcoef(tnew, st)[0, 1]:.6f}")
top = lambda s, o, M: np.array([np.argmax(s[a:a + m]) for a, m in zip(o, M)])
print(f"test top-1 changed in {(top(tnew, ot, Mt) != top(st, ot, Mt)).mean():.4f} of feeds (M>=50 only by construction: {(Mt >= 50).mean():.3f} of test feeds)")
