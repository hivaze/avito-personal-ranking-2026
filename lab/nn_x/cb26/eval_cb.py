"""cb26 evaluation: apply the pre-declared rule (cbrule.py) to E1/E2/E3 outputs. CPU only, light numpy.
Writes final score arrays to out/ (row order = preds/ for val/test, preds/holdout/ for trH)."""
import json, sys
import numpy as np
import polars as pl
sys.path.insert(0, "<repo>/nn_x/cb26"); sys.path.insert(0, "<repo>/src")
from cbdata import meta, ens_feats, zf, D, ROOT  # noqa: E402
from metrics import ndcg_per_feed  # noqa: E402
from cbrule import judge, pse  # noqa: E402
import pmix  # noqa: E402

OUT = ROOT / "nn_x/cb26/out"; OUT.mkdir(exist_ok=True)
POOL = sys.argv[1] if len(sys.argv) > 1 else "G3"
WHICH = sys.argv[2].split(",") if len(sys.argv) > 2 else ["e1_hr_qsm", "e2_vr_qsm", "e3_xm_yeti"]

d = pl.read_parquet(ROOT / "eda/errors/segments/feed_table.parquet")
np.save(D / "val_cold_share.npy", d["cold_share"].to_numpy())

fv, Mv, ov, yv = meta("val"); fh, Mh, oh, yh = meta("trH"); ft_, Mt, ot, _ = meta("test")
zv = lambda x: zf(ov, Mv, x); zh = lambda x: zf(oh, Mh, x); zt = lambda x: zf(ot, Mt, x)
ev, eh, et = ens_feats(POOL, "val"), ens_feats(POOL, "trH"), ens_feats(POOL, "test")
pool_v = zv(0.5 * zv(ev["ft"]) + 0.5 * zv(ev["to"])); pool_t = zt(0.5 * zt(et["ft"]) + 0.5 * zt(et["to"]))
to_h = zh(eh["to"])
nv = lambda s: ndcg_per_feed(s, yv, ov, Mv); nh = lambda s: ndcg_per_feed(s, yh, oh, Mh)
bv, bh = nv(pool_v), nh(to_h)
W = pmix.weights()
print(f"reference {POOL}: val {bv.mean():.5f}  trH(to) {bh.mean():.5f}")


def pm(d):
    return pmix.wmean_se(d, W["private"])


def corr_feed(a, b, M):
    fid = np.repeat(np.arange(len(M)), M)
    return float((np.bincount(fid, a * b, len(M)) / M).mean())  # a, b per-feed z


res = {}
if "e1_hr_qsm" in WHICH and (D / "e1_hr_qsm_beta.json").exists():
    j = json.loads((D / "e1_hr_qsm_beta.json").read_text()); b = j["beta_mean"]
    dv, dt = np.load(D / "e1_hr_qsm_delta_val.npy"), np.load(D / "e1_hr_qsm_delta_test.npy")
    oh_s = np.load(D / "e1_hr_qsm_oofscore_trH.npy")
    print(f"\nE1 HR (trH-trained residual, QuerySoftMax): beta {b:.3f}, best iters {j['iters']}, Delta sd val {dv.std():.4f}")
    s_prim = b * pool_v + dv
    s_sec = 0.5 * zv(ev["ft"]) + 0.5 * zv(b * zv(ev["to"]) + dv)
    s_to = b * zv(ev["to"]) + dv  # to-branch alone (diagnostic)
    print(f"   val to-branch alone: {nv(zv(ev['to'])).mean():.5f} -> +Delta {nv(s_to).mean():.5f} "
          f"(delta {pse(nv(s_to) - nv(zv(ev['to'])))[0]:+.5f} +- {pse(nv(s_to) - nv(zv(ev['to'])))[1]:.5f})")
    dh = nh(oh_s) - bh
    for name, s in [("E1 primary  beta*z(pool)+Delta", s_prim), ("E1 secondary to-branch repl.", s_sec)]:
        pv = nv(s); ok, txt = judge(name, pv - bv, dh, pm(pv - bv)); print(txt); res[name] = ok
    np.save(OUT / "e1_val_score.npy", s_prim.astype(np.float32))
    np.save(OUT / "e1_test_score.npy", (b * pool_t + dt).astype(np.float32))
    np.save(OUT / "e1_trH_oofscore.npy", oh_s.astype(np.float32))
if "e2_vr_qsm" in WHICH and (D / "e2_vr_qsm_beta.json").exists():
    j = json.loads((D / "e2_vr_qsm_beta.json").read_text()); b = j["beta_mean"]
    s_oof = np.load(D / "e2_vr_qsm_oofscore_val.npy"); dt = np.load(D / "e2_vr_qsm_delta_test.npy")
    base_v = np.load(D / f"base_vr_{POOL}_val.npy")
    assert np.abs(base_v - pool_v).max() < 1e-4
    print(f"\nE2 VR (val-nested residual, QuerySoftMax): beta {b:.3f}, best iters {j['iters']}")
    pv = nv(s_oof); ok, txt = judge("E2 nested residual", pv - bv, None, pm(pv - bv)); print(txt); res["E2"] = ok
    np.save(OUT / "e2_val_oofscore.npy", s_oof.astype(np.float32))
    np.save(OUT / "e2_test_score.npy", (b * pool_t + dt).astype(np.float32))
if "e3_xm_yeti" in WHICH and (D / "e3_xm_yeti.json").exists():
    j = json.loads((D / "e3_xm_yeti.json").read_text())
    cv, ch, ct = [np.load(D / f"e3_xm_yeti_{s}.npy") for s in ("val", "trH", "test")]
    hB = np.load(D / "e3_xm_yeti_halfB_feeds.npy")
    print(f"\nE3 XM (exploration-subset CatBoost member): train feeds {j['n_train_feeds']}, best iter {j['best_iter']}")
    print(f"   single: val {j['val_full']:.5f} (half A=ES {j['val_halfA_ES']:.5f}, half B {j['val_halfB']:.5f}); trH {j['trH']:.5f}")
    zc_v, zc_h, zc_t = zv(cv), zh(ch), zt(ct)
    rho_v, rho_h = corr_feed(zc_v, pool_v, Mv), corr_feed(zc_h, to_h, Mh)
    gap = bv.mean() - j["val_full"]
    print(f"   rho with pool: val {rho_v:.3f} trH {rho_h:.3f}; gap to pool {gap:.4f}; "
          f"law (eq-mix gain ~ 0.0186(1-rho) - 0.378 gap) = {0.0186 * (1 - rho_v) - 0.378 * gap:+.4f}")
    best = None
    for w in (0.05, 0.1, 0.2):
        pv = nv(pool_v + w * zc_v); ph = nh(to_h + w * zc_h)
        m, s = pse(pv - bv); mB, sB = pse((pv - bv)[hB])
        hm, hs = pse(ph - bh)
        print(f"   w={w:.2f}: val {m:+.5f} +- {s:.5f} (half B {mB:+.5f} +- {sB:.5f}) | trH {hm:+.5f} +- {hs:.5f}")
        if best is None or m > best[1]:
            best = (w, m, pv, ph)
    w0 = nv(pool_v + 0.05 * zc_v).mean() - bv.mean()
    ok, txt = judge(f"E3 member at selected w={best[0]}", best[2] - bv, best[3] - bh, pm(best[2] - bv), extra_cond=w0 > 0)
    print(txt); res["E3"] = ok
    for s, x in (("val", cv), ("test", ct), ("trH", ch)):
        np.save(OUT / f"e3_{s}_score.npy", x)
print("\nSUMMARY", json.dumps(res))
