"""C06 test: LIN residual with bt_theta, bt_seen (lam 3) on G3; INFO: lam 1 / 10 (val nested only) and the it_tr_coec contrast."""
import harness as Hn
from harness import np, log, ROOT
v = np.load(Hn.CACHE / "c06_val.npz")
h = np.load(Hn.CACHE / "c06_hold.npz")
log(f"val seen share {v['seen_l3'].mean():.4f}; holdout seen share {h['seen_l3'].mean():.4f}; corr(theta_l3, theta_l1) "
    f"{np.corrcoef(v['theta_l3'], v['theta_l1'])[0, 1]:.4f}")
coec = np.load(ROOT / "data/feats/val/it_tr_coec.npy").astype(np.float64)
info = {}
info["lam1"] = Hn.lin_info("C06_l1", [v["theta_l1"], v["seen_l1"]], "bt lam 1 [theta, seen]")
info["lam10"] = Hn.lin_info("C06_l10", [v["theta_l10"], v["seen_l10"]], "bt lam 10 [theta, seen]")
info["coec"] = Hn.lin_info("coec", [coec], "it_tr_coec alone")
info["coec_plus_bt"] = Hn.lin_info("coec_bt", [coec, v["theta_l3"], v["seen_l3"]], "it_tr_coec + bt lam 3")
cm = np.isfinite(coec)
log(f"corr(bt_theta, it_tr_coec) on seen & finite rows {np.corrcoef(v['theta_l3'][cm & (v['seen_l3'] > 0)], coec[cm & (v['seen_l3'] > 0)])[0, 1]:.4f}")
Hn.run_item("C06", ["bt_theta", "bt_seen"], [v["theta_l3"], v["seen_l3"]], [h["theta_l3"], h["seen_l3"]], extra=info)
