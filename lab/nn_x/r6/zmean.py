"""Equal-weight per-feed z-mean of existing prediction files, for val and test.

  python nn/zmean.py <out> <in1> <in2> ...   # reads preds/nn_<in>_{val_oof,test}.npy, writes preds/nn_<out>_{val_oof,test}.npy
(<in> may also name a *_mix family file such as final5_mix.)
"""
import sys

import numpy as np
import polars as pl

ROOT = "<repo>"


def fz(s, fid, M):
    s = s.astype(np.float64)
    mu = np.bincount(fid, s, len(M)) / M
    sd = np.sqrt(np.maximum(np.bincount(fid, s * s, len(M)) / M - mu ** 2, 0)) + 1e-6
    return (s - mu[fid]) / sd[fid]


def main(out, ins):
    sys.path.insert(0, f"{ROOT}/src")
    from metrics import ndcg
    for split, suf in (("val", "val_oof"), ("test", "test")):
        f = pl.read_parquet(f"{ROOT}/data/flat/{split}/feeds.parquet")
        M = f["M"].to_numpy().astype(np.int64)
        fid = np.repeat(np.arange(len(M)), M)
        z = np.mean([fz(np.load(f"{ROOT}/nn_x/r6/preds/nn_{n}_{suf}.npy"), fid, M) for n in ins], 0).astype(np.float32)
        np.save(f"{ROOT}/nn_x/r6/preds/nn_{out}_{suf}.npy", z)
        if split == "val":
            y = np.load(f"{ROOT}/data/flat/val/cand_target.npy")
            print(f"{out}: val NDCG {ndcg(z, y, f['cand_off'].to_numpy(), M):.5f} from {ins}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
