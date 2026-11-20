"""Information (label-free): which emulation of it_hist / seq_cv coverage is closest to PUBLIC and to PRIVATE rows?
fresh val, 'notest' (no test-split snapshots), 'trunc17' (train Sep 17-19 removed), 'both', and the 50/50 feed mix fresh+both
(what --stale_train 0.5 trains on, as a row mixture). Same statistics as gate.py (dZ, dM by listing-age bucket, val-row weights)."""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"]:
    os.environ[k] = "12"
import numpy as np, polars as pl
from pathlib import Path
sys.path.insert(0, "<repo>/nn_x/w12/stale/staleft")
import gate as G
ROOT = Path("<repo>"); FC = ROOT / "nn/feat_cache"; FLAT = ROOT / "data/flat"
SC = Path("/tmp/scratchpad/review/shift/out/cf")
ft = pl.read_parquet(FLAT / "test/feeds.parquet")
bkv = G.age_bucket(np.load(FLAT / "val/cand_f4.npy")); bkt = G.age_bucket(np.load(FLAT / "test/cand_f4.npy"))
fi = np.load(FLAT / "test/cand_fidx.npy"); pub = (ft["timestamp"].to_numpy().astype(np.int64) < G.PUB_TS)[fi]
wv = np.bincount(bkv, minlength=7) / len(bkv)
out = ["column | view | zero share | PUBLIC dZ dM | PRIVATE dZ dM"]
for n in ["it_hist_all_luo", "it_hist_et1_luo", "it_hist_et2_luo", "it_hist_et4_luo", "seq_cv_pop", "seq_cv_invd20"]:
    rv = np.load(FC / "val" / f"{n}.npy").astype(np.float32); rt = np.load(FC / "test" / f"{n}.npy").astype(np.float32)
    views = {"fresh": rv}
    for sc in ("notest", "trunc17", "both"):
        p = SC / f"{sc}_{n}.npy"
        if p.exists():
            views[sc] = np.load(p).astype(np.float32)
    Xt, Zt = G.xt_transform(n, rt), rt == 0
    st = {}
    for vn, r in views.items():
        st[vn] = G.stats(G.xt_transform(n, r), r == 0, bkv) + (float((r == 0).mean()),)
    st["mix50 (fresh+both)"] = (0.5 * (st["fresh"][0] + st["both"][0]), 0.5 * (st["fresh"][1] + st["both"][1]), 0.5 * (st["fresh"][2] + st["both"][2]))
    tp = {}
    for pn, pm in (("pub", pub), ("priv", ~pub)):
        tp[pn] = G.stats(Xt[pm], Zt[pm], bkt[pm]) + (float(Xt[pm].std()), float(Zt[pm].mean()))
    for vn, (m, z, zs) in st.items():
        s = f"{n} | {vn} | {zs:.4f} |"
        for pn in ("pub", "priv"):
            mt, zt_, sd, _ = tp[pn]
            s += f" {float((wv * np.abs(z - zt_)).sum()):.4f} {float((wv * np.abs(m - mt)).sum() / sd):.4f} |"
        out.append(s)
    out.append(f"{n} | TEST rows | pub {tp['pub'][3]:.4f} priv {tp['priv'][3]:.4f} | |")
open("<repo>/nn_x/w12/stale/staleft/logs/gate_scen.txt", "w").write("\n".join(out) + "\n")
print("\n".join(out))
