"""Per-candidate coldness flags + calibration of nn_final3 on cold vs warm candidates (leak-free val)."""
import os; os.environ["POLARS_MAX_THREADS"] = "16"
import sys; sys.path.insert(0, "<repo>/src"); sys.path.insert(0, "<repo>/eda/errors/synthesis")
from common import *
from stack_val import feed_rank_z
from qt import leak_mask
OUT = Path("<repo>/eda/errors/synthesis/derived")
tr_c = pl.Series(np.load(FLAT / "train/cand_item_id.npy")).unique()
tr_h = pl.Series(np.load(FLAT / "train/hist_item_id.npy")).unique()
print("unique done", len(tr_c), len(tr_h), flush=True)
for split in ("val", "test"):
    it = np.load(FLAT / split / "cand_item_id.npy")
    sit = pl.Series(it); inc = sit.is_in(tr_c).to_numpy(); inh = sit.is_in(tr_h).to_numpy()
    cold = (~inc & ~inh).astype(np.float32)
    np.save(OUT / split / "cold_item.npy", cold)
    np.save(OUT / split / "cold_noimpr.npy", (~inc).astype(np.float32))
    cvz = (np.load(FEATS / split / "seq_cv_pop.npy") == 0).astype(np.float32)
    np.save(OUT / split / "cvpop_zero.npy", cvz)
    luoz = (np.load(FEATS / split / "it_hist_all_luo.npy") == 0).astype(np.float32)
    np.save(OUT / split / "luo_zero.npy", luoz)
    print(split, f"cold(never in train cand/hist) {cold.mean():.3f}  no train impr {(~inc).mean():.3f}  seq_cv_pop==0 {cvz.mean():.3f}  luo==0 {luoz.mean():.3f}")
# calibration: actual / expected positives under the within-feed softmax of the model (temperature fit on val by 1-param LR)
f, M, off = feed_meta("val"); y = np.load(FLAT / "val/cand_target.npy").astype(np.float64); kf = ~leak_mask()
fidx = np.load(FLAT / "val/cand_fidx.npy"); rk = kf[fidx]
P = Path("<repo>/preds")
for name in ["nn_final2_trainonly", "nn_final3_trainonly", "nn_final3_val_oof"]:
    fn = P / (f"{name}.npy" if name.endswith("oof") else f"{name}_val.npy")
    z = feed_rank_z(off, M, np.load(fn).astype(np.float32))[:, 1].astype(np.float64)
    # expected positives per feed = npos * softmax(a z); pick a by max listwise likelihood on a grid
    best = None
    for a in np.arange(0.5, 3.01, 0.1):
        e = np.exp(a * z); s = np.add.reduceat(e, off); p = e / np.repeat(s, M)
        ll = (y * np.log(p + 1e-12))[rk].sum()
        if best is None or ll > best[0]:
            best = (ll, a, p)
    _, a, p = best
    npos = np.add.reduceat(y, off); ex = p * np.repeat(npos, M)
    out = [f"{name} (a={a:.1f})"]
    for flag in ["cold_item", "cold_noimpr", "cvpop_zero", "luo_zero"]:
        c = np.load(OUT / "val" / f"{flag}.npy").astype(bool)
        for lab, m in (("1", c), ("0", ~c)):
            mm = m & rk
            act, exp_ = y[mm].sum(), ex[mm].sum()
            out.append(f"{flag}={lab}: act/exp {act / exp_:.3f} (±{np.sqrt(act) / exp_:.3f})")
    print(" | ".join(out))
