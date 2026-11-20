"""C16 REGIME4 Step 0 read: regcorr = zf(s_regime) - zf(s_base) (seed-averaged scores, seeds 0/1/2) as a LIN residual on G3.
INFO: the spec's kill read = proxy val (and holdout) NDCG of regime / droppre / droppost vs base, paired per feed."""
from h import *
H = Harness()
O = Path(HERE) / "out"
def avg(arm, split, seeds=(0, 1, 2)):
    ps = [O / f"c16_{arm}_s{s}_{split}.npy" for s in seeds]
    ps = [p for p in ps if p.exists()]
    return np.mean([np.load(p).astype(np.float64) for p in ps], 0), len(ps)
sb, nb_ = avg("base", "val"); sr, nr = avg("regime", "val")
hb, _ = avg("base", "hold"); hr, _ = avg("regime", "hold")
assert nb_ == 3 and nr == 3
for arm in ["base", "regime", "droppre", "droppost"]:
    s, n = avg(arm, "val"); sh, _ = avg(arm, "hold")
    log(f"proxy {arm:9s} ({n} seeds): val {H.nd(s).mean():.5f} holdout {H.ndh(sh).mean():.5f}")
for arm in ["regime", "droppre", "droppost"]:
    s, n = avg(arm, "val"); sh, _ = avg(arm, "hold")
    d = H.nd(s) - H.nd(sb); dh = H.ndh(sh) - H.ndh(hb)
    m, se, z = H.stats(d); mh, seh, zh = H.stats(dh)
    log(f"INFO kill read {arm} - base (seed-avg, {n} seeds): proxy val {m:+.5f} ({se:.5f}) z {z:+.2f} {'> 1 SE' if abs(z) > 1 else '<= 1 SE'} | holdout {mh:+.5f} ({seh:.5f}) z {zh:+.2f}")
rc_v = H.zv(sr) - H.zv(sb); rc_h = H.zh(hr) - H.zh(hb)
coverage("regcorr val", rc_v); coverage("regcorr hold", rc_h)
log(f"corr(zf s_regime, zf s_base) val {np.corrcoef(H.zv(sr), H.zv(sb))[0,1]:.5f}")
H.run("C16_regcorr", [rc_v], [rc_h], out_jsonl=O / "results.jsonl")
