"""FT-stage inputs (n2cpv3, 42 extras): replicate nn/data.py attach_extra (val stats reused for test: log1p if
min>=0 & max>20, standardise, clip +-6, NaN -> 0) and measure NaN rates, log rule, clip binding, PSI val->pub/priv
after the transform, and within-feed tie share of *_rk columns.  CPU only."""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""; os.environ["OMP_NUM_THREADS"] = "12"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s02_stats import psi, OUT, BOUND
import numpy as np
from pathlib import Path
FC = Path("<repo>/nn/feat_cache")
names = ['centrality', 'it_hist_all_luo', 'it_hist_all_luo_rk', 'it_hist_all_luo_zsl', 'it_hist_all_luo_gap', 'it_hist_et1_luo', 'it_hist_et2_luo', 'it_hist_et3_luo', 'it_hist_et4_luo', 'it_hist_et5_luo', 'it_hist_et6_luo', 'it_hist_et7_luo', 'it_hist_et8_luo', 'it_hist_et9_luo', 'it_hist_et10_luo', 'seq_cv_invd20', 'seq_cv_invd20_rk', 'seq_cv_pop', 'seq_max_all', 'seq_top3', 'seq_argmax_hpos', 'ctx_nd_hcent', 'ctx_nd_knn_f1', 'ctx_nd_knn_h', 'ctx_nd_knn_h_dev', 'ctx_nd_knn_u', 'ctx_nd_knn_u_dev', 'ctx_nd_max', 'ctx_nd_top5h', 'ctx_nd_top5u', 'ctx_nd_ucent', 'pre_fe', 'pre_fe_z', 'pre_new', 'pre_new_z', 'pre_hmax32_c', 'pre_hmax8_c', 'pre_fe_noid', 'pre_id_contrib_v2', 'pre_idmax32', 'pre_idmax32_has', 'pre_hmax32_sent']
tm = np.load(OUT / "test_meta.npz"); pub = np.repeat(tm["ts"] < BOUND, tm["M"])
print(f"{'extra':22s} log  nan_val nan_pub nan_priv  clip6_val clip6_priv  psi_v>pub psi_v>priv  p999/p50(val,transformed-abs)")
rows = []
for n in names:
    v = np.load(FC / "val" / f"{n}.npy").astype(np.float64); t = np.load(FC / "test" / f"{n}.npy").astype(np.float64)
    logm = (np.nanmin(v) >= 0) and (np.nanmax(v) > 20)
    f = (lambda x: np.log1p(np.maximum(x, 0))) if logm else (lambda x: x)
    vv, tt = f(v), f(t)
    mu, sd = np.nanmean(vv), np.nanstd(vv) + 1e-6
    zv, zt = (vv - mu) / sd, (tt - mu) / sd
    zp, zq = zt[pub], zt[~pub]
    c6v = np.nanmean(np.abs(zv) >= 6); c6q = np.nanmean(np.abs(zq) >= 6)
    pv, pp = psi(np.nan_to_num(np.clip(zv, -6, 6)), np.nan_to_num(np.clip(zp, -6, 6))), psi(np.nan_to_num(np.clip(zv, -6, 6)), np.nan_to_num(np.clip(zq, -6, 6)))
    a = np.abs(vv[np.isfinite(vv)]); r = np.quantile(a, 0.999) / max(np.median(a), 1e-9)
    print(f"{n:22s} {int(logm)}    {np.isnan(v).mean():.3f}   {np.isnan(t[pub]).mean():.3f}   {np.isnan(t[~pub]).mean():.3f}     {c6v:.4f}     {c6q:.4f}      {pv:.3f}     {pp:.3f}     {r:.1f}")
