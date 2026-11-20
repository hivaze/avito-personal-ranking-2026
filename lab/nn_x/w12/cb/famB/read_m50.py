"""Lead's notice (3): NATS and the member-rule statistic on val feeds with M >= 50 separately (the holdout is M >= 50 by construction).
Same statistics as nats.py, restricted to the feed subset (the 1- and 2-parameter conditional logits are refitted on the subset).
usage: python -u read_m50.py tag [tag+tag (bundle) ...]     "b0" = infra ts_lr020_s0"""
import json, sys
sys.path.insert(0, "<repo>/nn_x/w12/cb/famB"); sys.path.insert(0, "<repo>/nn_x/w12/cb")
from fb_common import *
import nats
from nats import clogit
m = nats._meta()
P = lambda tag, sp: np.load((B0V if sp == "val" else B0H) if tag == "b0" else PREDS / f"{tag}_{sp}.npy")
y, off, M = m.y["val"].astype(np.float64), m.off["val"], m.M["val"]; zw = m.zw["val"]; gid = np.repeat(np.arange(len(M)), M)
res = {}
print(f"val feeds {len(M)}: M >= 50: {(M >= 50).sum()} ({(M >= 50).mean():.3f}), M < 50: {(M < 50).sum()}")
print("| model | val M >= 50: level | NATS | member rule 0.9 / 0.1 | val M < 50: NATS | member rule |\n|---|---|---|---|---|---|")
for b in sys.argv[1:]:
    zx = np.mean([m.z("val", P(t, "val")) for t in b.split("+")], 0); out = {}
    for nm, fm in (("m50", M >= 50), ("mlt50", M < 50)):
        rm = fm[gid]; Ms = M[fm]; os_ = np.r_[0, np.cumsum(Ms)[:-1]].astype(np.int64); ys = y[rm]; zws = zw[rm].astype(np.float64); zxs = zx[rm].astype(np.float64)
        _, _, lw = clogit(zws[:, None], ys, os_, Ms); bb, ss, l2 = clogit(np.stack([zws, zxs], 1), ys, os_, Ms); d = lw - l2; nf = int((lw != 0).sum())
        e0 = nats.ndcg_fast(zws.astype(np.float32), ys.astype(np.float32), os_, Ms); ex = nats.ndcg_fast(zxs.astype(np.float32), ys.astype(np.float32), os_, Ms)
        mix = (0.9 * zws + 0.1 * zxs).astype(np.float32)
        dm = nats.ndcg_fast(cbx.feed_z(os_, Ms, np.ascontiguousarray(mix)), ys.astype(np.float32), os_, Ms) - e0
        out[nm] = dict(n=int(fm.sum()), level=float(ex.mean()), w16=float(e0.mean()), nats=float(d.sum() / nf), nats_se=float(np.std(d) / np.sqrt(len(d)) * len(d) / nf),
                       mix=float(dm.mean()), mix_se=float(dm.std() / np.sqrt(len(dm))))
    res[b] = out; a, c = out["m50"], out["mlt50"]
    print(f"| `{b}` | {a['level']:.5f} (W16 {a['w16']:.5f}) | {a['nats']:+.5f} ± {a['nats_se']:.5f} | {a['mix']:+.5f} ± {a['mix_se']:.5f} | {c['nats']:+.5f} ± {c['nats_se']:.5f} | {c['mix']:+.5f} ± {c['mix_se']:.5f} |", flush=True)
json.dump(res, open(HERE / "out/read_m50.json", "w"), indent=1)
