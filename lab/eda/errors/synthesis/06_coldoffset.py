"""Nested test of a cold-candidate offset on top of nn_final3: score = z(base) + delta * flag.

delta is chosen on 4/5 of the leak-free feeds (5-fold GroupKFold by user, 2 seeds) from a coarse grid and applied to the other 1/5.
Reports the paired per-feed delta vs delta = 0 on raw val and on val reweighted to the test (M x cold-share) mix, and per
positive-coldness segment. Flags come from 05_coldflag.py.
"""
import sys
sys.path.insert(0, "<repo>/src"); sys.path.insert(0, "<repo>/eda/errors/segments")
sys.path.insert(0, "<repo>/eda/errors/synthesis")
from common import *
from stack_val import feed_rank_z
from segdefs import seg_defs
from sklearn.model_selection import GroupKFold

SNAP = __import__("pathlib").Path("<repo>/eda/errors/synthesis/snap")
def pp(fn):
    import pathlib
    return SNAP / fn if (SNAP / fn).exists() else pathlib.Path("<repo>/preds") / fn


D = Path("<repo>/eda/errors/synthesis")
P = Path("<repo>/preds")
d_all = pl.read_parquet("<repo>/eda/errors/segments/feed_table.parquet")
te = pl.read_parquet("<repo>/eda/errors/segments/test_segments.parquet")
kf = ~d_all["leak"].to_numpy()
S = seg_defs(d_all); St = seg_defs(te, test=True)
key_v = np.char.add(S["M"].astype(str), S["cold_share"].astype(str)); key_t = np.char.add(St["M"].astype(str), St["cold_share"].astype(str))
w = np.zeros(len(d_all))
for k in np.unique(key_v[kf]):
    w[(key_v == k) & kf] = (key_t == k).mean() / (key_v[kf] == k).mean()
w = w[kf] / w[kf].mean()
cold_all = d_all["pos_cold_all"].to_numpy().astype(bool)[kf]; warm = ~d_all["pos_cold_any"].to_numpy().astype(bool)[kf]
f, M, off = feed_meta("val"); y = np.load(FLAT / "val/cand_target.npy").astype(np.float32)
users = f["user_id"].to_numpy().copy(); users[users == 0] = -np.arange(1, (users == 0).sum() + 1)
GRID = [-0.25, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0]


def z_of(fn):
    return feed_rank_z(off, M, np.load(fn).astype(np.float32))[:, 1].astype(np.float64)


bases = {"nn_final3_tr": z_of(pp("nn_final3_trainonly_val.npy")), "nn_final3_oof": z_of(pp("nn_final3_val_oof.npy")),
         "nn_final2_tr": z_of(pp("nn_final2_trainonly_val.npy"))}
bases["blend_f3oof_f3tr"] = 0.5 * bases["nn_final3_oof"] + 0.5 * bases["nn_final3_tr"]


def wse(x, ww):
    m = np.average(x, weights=ww)
    return np.sqrt(np.average((x - m) ** 2, weights=ww) / (ww.sum() ** 2 / (ww ** 2).sum()))


for flag in sys.argv[1].split(","):
    c = np.load(D / "derived" / "val" / f"{flag}.npy").astype(np.float64)
    for bn, zb in bases.items():
        pers = {g: ndcg_per_feed(zb + g * c, y, off, M) for g in GRID}
        base = pers[0]
        outs, chosen = [], []
        for seed in range(2):
            o = np.zeros(len(M))
            for tr, va in GroupKFold(5, shuffle=True, random_state=seed).split(np.arange(len(M)), groups=users):
                trk = tr[kf[tr]]
                g = max(GRID, key=lambda g: pers[g][trk].mean())
                chosen.append(g)
                o[va] = pers[g][va] - base[va]
            outs.append(o)
        dd = np.mean(outs, 0)[kf]
        full = {g: (pers[g] - base)[kf].mean() for g in GRID}
        print(f"{flag:12s} {bn:17s} nested {dd.mean():+.5f} ± {dd.std() / np.sqrt(len(dd)):.5f} | test-mix {np.average(dd, weights=w):+.5f} ± {wse(dd, w):.5f}"
              f" | all-cold-pos {dd[cold_all].mean():+.5f} warm-pos {dd[warm].mean():+.5f} | chosen {sorted(set(chosen))} | "
              f"full-val curve " + " ".join(f"{g}:{v:+.4f}" for g, v in full.items()), flush=True)
