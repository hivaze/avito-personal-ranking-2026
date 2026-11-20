"""Build (in this directory only) val/test final scores for the pop-light rule candidates on the FD16P20 members and run
the label-free parts of the gate that can run without writing to preds/: D (val delta, slices, leak probe on has-later
feeds), F (test/val parity of the change), A (finite, no within-feed ties on test). Not a submission."""
import ph, numpy as np, polars as pl, pmix
from scipy.stats import spearmanr
out = {}
for split in ("val", "test"):
    D = ph.Data(split); C = np.load(f"../out/comp_{split}.npz"); z = {k: ph.zf(C[k], D.off, D.M) for k in C.files}
    Mr = np.repeat(D.M, D.M)
    def fin(ws, wl):
        w = np.where(Mr >= 50, wl, ws)
        return 0.5 * ph.zf((1 - w) * z["ft_base"] + w * z["ft_pl"], D.off, D.M) + 0.5 * ph.zf((1 - w) * z["to_base"] + w * z["to_pl"], D.off, D.M)
    out[split] = {"ref": D.s, "m04_03": fin(0.4, 0.3), "m04_04": fin(0.4, 0.4), "D": D}
    print(split, "rebuild of current file: max |diff|", np.abs(fin(0.4, 0.2) - D.s).max())
Dv, Dt = out["val"]["D"], out["test"]["D"]
fv, ft = Dv.f, Dt.f
uv = fv["user_id"].to_numpy(); tv = fv["timestamp"].to_numpy()
allu = np.concatenate([uv, ft["user_id"].to_numpy()]); allt = np.concatenate([tv, ft["timestamp"].to_numpy()])
last = pl.DataFrame({"u": allu, "t": allt}).group_by("u").agg(pl.col("t").max()); lastd = dict(zip(last["u"].to_list(), last["t"].to_list()))
has_later = np.array([(u != 0) and (lastd[u] > t) for u, t in zip(uv, tv)])
def pf_corr(D, a, b):
    r = []
    for o, m in zip(D.off, D.M):
        if m > 2: r.append(spearmanr(a[o:o + m], b[o:o + m])[0])
    return np.nanmean(r)
for c in ("m04_03", "m04_04"):
    d = Dv.ev(out["val"][c]) - Dv.base
    se = lambda x: x.std() / np.sqrt(len(x))
    gap = d[has_later].mean() - d[~has_later].mean(); gse = np.sqrt(se(d[has_later]) ** 2 + se(d[~has_later]) ** 2)
    Dv.report(out["val"][c], f"candidate {c}")
    print(f"   leak probe has_later (n={has_later.sum()}) - no_later: {gap:+.5f} +- {gse:.5f} (z {gap / gse:+.2f})")
    print(f"   parity: per-feed Spearman new-vs-ref val {pf_corr(Dv, out['val'][c], out['val']['ref']):.5f} test {pf_corr(Dt, out['test'][c], out['test']['ref']):.5f}")
    t = out["test"][c].astype(np.float32); o = np.lexsort((t, Dt.fid)); ts_ = t[o]; ff = Dt.fid[o]
    print(f"   test finite {np.isfinite(t).all()} within-feed float32 ties {int(((ts_[1:] == ts_[:-1]) & (ff[1:] == ff[:-1])).sum())}; test corr to current {np.corrcoef(t, out['test']['ref'])[0, 1]:.5f}")
    np.save(f"../out/cand_{c}_test.npy", t); np.save(f"../out/cand_{c}_val.npy", out["val"][c].astype(np.float32))
