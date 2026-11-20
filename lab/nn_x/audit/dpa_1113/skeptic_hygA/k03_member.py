"""k02 nested residuals with a single member as the base (r6_cd_m02_r101, v23_gru_pre2s_s501; TO val / holdout)."""
import os, sys
for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"]:
    os.environ[k] = "12"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import k02_capexcess_nested as K
for mem in ["r6_cd_m02_r101", "v23_gru_pre2s_s501"]:
    K.b = K.feed_z(np.load(K.ROOT / f"preds/nn_{mem}_trainonly_val.npy").astype(np.float64), K.offv, K.Mv)
    K.bh = K.feed_z(np.load(K.ROOT / f"preds/holdout/nn_{mem}_holdout.npy").astype(np.float64), K.offh, K.Mh)
    print(f"#### member {mem}: base val {K.ndcg_per_feed(K.b, K.yv, K.offv, K.Mv).mean():.5f} hold {K.ndcg_per_feed(K.bh, K.yh, K.offh, K.Mh).mean():.5f}")
    exc = lambda R: sum(np.log(np.maximum(R[c], 10.0) / 10.0) for c in K.MASS)
    K.nested("capexcess_sum4", exc(K.Rv), exc(K.Rh))
    K.nested("capexcess_soft05", np.log(np.maximum(K.Rv["ctx_hn_soft05"], 10.0) / 10.0), np.log(np.maximum(K.Rh["ctx_hn_soft05"], 10.0) / 10.0))
    a = K.lz(K.Rv, K.offv, K.Mv) - K.lc(K.Rv, K.offv, K.Mv); ah = K.lz(K.Rh, K.offh, K.Mh) - K.lc(K.Rh, K.offh, K.Mh)
    K.nested("z_log1p minus z_capped (repr change)", a, ah)
    rng = np.random.default_rng(1)
    K.nested("noise control", rng.standard_normal(len(K.yv)), rng.standard_normal(len(K.yh)))
