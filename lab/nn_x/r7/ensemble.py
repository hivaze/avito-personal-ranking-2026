"""Average several NN variants (per-feed z-scored scores) for both the fine-tuned (OOF/test) and train-only predictions.

  python nn/ensemble.py --variants nopos_aux_s0,nopos_aux_s1 --out ens
reads  preds/nn_<v>_{val_oof,test,trainonly_val,trainonly_test}.npy  (missing ones are skipped)
writes preds/nn_<out>_{val_oof,test,trainonly_val,trainonly_test}.npy
"""
import argparse
import sys

import numpy as np

from blend import FLAT, feed_meta, feed_z, ndcg  # noqa: E402

PREDS = "<repo>/nn_x/r7/preds"  # nn_x/r7 copy

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--variants", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--trainonly_variants", default="",
                   help="comma list of variants whose train-only preds form the train-only ensemble "
                        "(default: --variants with a '_mix' suffix stripped)")
    p.add_argument("--w_trainonly", type=float, default=0.2,
                   help="weight of the train-only (epoch-0) predictions mixed into each fine-tuned model; 0.2 == "
                        "snapshot mean over fine-tune epochs 0..4 instead of 1..4")
    a = p.parse_args()
    vs = a.variants.split(",")
    _, Mv, offv = feed_meta("val")
    _, Mt, _ = feed_meta("test")
    y = np.load(FLAT / "val/cand_target.npy")
    res = {}
    for val_suf, test_suf in [("val_oof", "test"), ("trainonly_val", "trainonly_test")]:
        zs_v, zs_t, used = [], [], []
        if val_suf == "val_oof":
            names = vs
        elif a.trainonly_variants:
            names = a.trainonly_variants.split(",")
        else:
            names = list(dict.fromkeys(v.removesuffix("_mix") for v in vs))
        for v in names:  # "<v>_mix" fine-tunes share the train-only predictions of <v>
            try:
                sv = np.load(f"{PREDS}/nn_{v}_{val_suf}.npy").astype(np.float64)
                st = np.load(f"{PREDS}/nn_{v}_{test_suf}.npy").astype(np.float64)
            except FileNotFoundError:
                continue
            print(f"  {val_suf:14s} {v:24s} {ndcg(sv, y, offv, Mv):.5f}")
            zs_v.append(feed_z(sv, Mv))
            zs_t.append(feed_z(st, Mt))
            used.append(v)
        if not used:
            continue
        ev, et = np.mean(zs_v, 0), np.mean(zs_t, 0)
        print(f"{val_suf}: ensemble of {len(used)} -> val NDCG {ndcg(ev, y, offv, Mv):.5f}")
        res[val_suf] = (ev, et, used)
    if a.w_trainonly > 0 and "val_oof" in res and "trainonly_val" in res:
        if not a.trainonly_variants:
            base = set(v.removesuffix("_mix") for v in res["val_oof"][2])
            assert base == set(res["trainonly_val"][2]), "need train-only preds for every fine-tuned variant"
        w = a.w_trainonly
        ev = (1 - w) * res["val_oof"][0] + w * res["trainonly_val"][0]
        et = (1 - w) * res["val_oof"][1] + w * res["trainonly_val"][1]
        print(f"fine-tuned mixed with train-only (w={w}): val NDCG {ndcg(ev, y, offv, Mv):.5f}")
        res["val_oof"] = (ev, et, res["val_oof"][2])
    for val_suf, test_suf in [("val_oof", "test"), ("trainonly_val", "trainonly_test")]:
        if val_suf in res:
            np.save(f"{PREDS}/nn_{a.out}_{val_suf}.npy", res[val_suf][0].astype(np.float32))
            np.save(f"{PREDS}/nn_{a.out}_{test_suf}.npy", res[val_suf][1].astype(np.float32))
