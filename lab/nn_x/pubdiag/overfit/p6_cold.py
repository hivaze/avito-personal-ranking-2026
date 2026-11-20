"""Part 6 (supports the synthesis): is the cold-candidate promotion that separates M2 from Cd paid on the holdout too,
and is its val payoff stable across val days? Writes p6_cold.txt."""
from lib import *

C = OUT / "cache"
idx = {m: i for i, m in enumerate(MEMBERS)}
lines = []
P_ = lambda s="": (print(s, flush=True), lines.append(s))
V, Hs = Split("val"), Split("holdout")
Zv = np.load(C / "Z_val_to.npy", mmap_mode="r"); Zh = np.load(C / "Z_holdout_to.npy", mmap_mode="r")

# holdout rows -> train rows (same selection as train.py --holdout)
tf = pl.read_parquet(FLAT / "train/feeds.parquet", columns=["M", "cand_off"])
Mt, ot = tf["M"].to_numpy(), tf["cand_off"].to_numpy()
pr = np.nan_to_num(np.load(ROOT / "eda/labels_position/out/train_prandom.npz")["p_random"].astype(np.float32), nan=0.0)
pr[Mt < 50] = 0.0
hold = np.sort(np.random.default_rng(12345).choice(np.where((pr > 0.9) & (Mt >= 50))[0], 30000, replace=False))
rows = np.concatenate([np.arange(o, o + m) for o, m in zip(ot[hold], Mt[hold])])
luo_tr = np.load(ROOT / "nn/feat_cache/train/it_hist_all_luo.npy", mmap_mode="r")
coldh = np.nan_to_num(np.asarray(luo_tr[rows], np.float64), nan=0.0) == 0
coldv = np.nan_to_num(np.load(ROOT / "nn/feat_cache/val/it_hist_all_luo.npy").astype(np.float64), nan=0.0) == 0
assert len(coldh) == len(Hs.y)


def mix(S, Z, ms):
    return S.z(np.asarray(Z[[idx[m] for m in ms]], np.float64).mean(0))


def segs(S, cold):
    npos = np.bincount(S.fid, S.y, S.n); ncp = np.bincount(S.fid, S.y * cold, S.n)
    return (npos > 0) & (ncp == npos), (npos > 0) & (ncp == 0), npos > 0


def promo(S, d, cold):
    nc = np.bincount(S.fid, cold, S.n); nw = S.M - nc
    mc = np.bincount(S.fid, d * cold, S.n) / np.maximum(nc, 1); mw = np.bincount(S.fid, d * ~cold, S.n) / np.maximum(nw, 1)
    ok = (nc > 0) & (nw > 0)
    return (mc - mw)[ok].mean()


P_(f"cold-candidate share: val {coldv.mean():.3f}, holdout {coldh.mean():.3f}; positive rate cold/warm: val "
   f"{V.y[coldv].mean():.4f}/{V.y[~coldv].mean():.4f}, holdout {Hs.y[coldh].mean():.4f}/{Hs.y[~coldh].mean():.4f}")
for S, cold, nm in ((V, coldv, "val"), (Hs, coldh, "holdout")):
    cp, wp, hp = segs(S, cold)
    P_(f"{nm}: coldpos feeds {cp.mean():.3f}, warmpos {wp.mean():.3f}, mixed {(hp & ~cp & ~wp).mean():.3f}")
tue = (V.ts + 10800) // 86400 < 20720
pairs = [("M2 - Cd", pool_members("M2"), pool_members("Cd")), ("G3 - Cd", pool_members("G3"), pool_members("Cd")),
         ("PF7x - Cd", pool_members("PF7x"), pool_members("Cd")), ("re-cool F7b - F7", FAM["F7b"], FAM["F7"]),
         ("m02 F7x - F7b", FAM["F7x"], FAM["F7b"])]
P_()
P_("TO mixes; per split: cold promotion (z units) | gain all | coldpos | warmpos | mixed  (mean +- SE)")
for nm, A, B in pairs:
    for S, Z, cold, lab in ((V, Zv, coldv, "val"), (Hs, Zh, coldh, "hold")):
        a, b = mix(S, Z, A), mix(S, Z, B)
        d = S.nd(a) - S.nd(b)
        cp, wp, hp = segs(S, cold)
        mx = hp & ~cp & ~wp
        P_(f"  {nm:17s} {lab:4s} promo {promo(S, a - b, cold):+.4f} | all {d.mean():+.5f}+-{se(d):.5f} | coldpos {d[cp].mean():+.5f}+-{se(d[cp]):.5f}"
           f" | warmpos {d[wp].mean():+.5f}+-{se(d[wp]):.5f} | mixed {d[mx].mean():+.5f}+-{se(d[mx]):.5f}")
    a, b = mix(V, Zv, A), mix(V, Zv, B); d = V.nd(a) - V.nd(b); cp, wp, hp = segs(V, coldv)
    P_(f"  {'':17s} val coldpos by day: Tue {d[cp & tue].mean():+.5f}+-{se(d[cp & tue]):.5f}  Thu {d[cp & ~tue].mean():+.5f}+-{se(d[cp & ~tue]):.5f}"
       f" | warmpos Tue {d[wp & tue].mean():+.5f} Thu {d[wp & ~tue].mean():+.5f}")
(OUT / "p6_cold.txt").write_text("\n".join(lines) + "\n")
