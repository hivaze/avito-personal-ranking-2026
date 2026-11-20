"""W25 column diagnostics: NaN share per split, PSI train(XM, M-matched? no: all XM rows) vs val and val vs test (10 val-quantile bins),
single-feature NDCG on trH (train labels) and val (descriptive), rank-corr with the strongest related base feature."""
import json, sys
import numpy as np
sys.path.insert(0, "<repo>/nn_x/cb26"); sys.path.insert(0, "<repo>/src")
sys.path.insert(0, "<repo>/nn_x/w25")
from cbdata import meta, trH_rows, load_tab, ROOT  # noqa: E402
from train_w25 import load_w25, w25_groups  # noqa: E402
from metrics import ndcg_per_feed  # noqa: E402

WD = ROOT / "nn_x/w25/data"
fv, Mv, ov, yv = meta("val"); fh, Mh, oh, yh = meta("trH"); ft_, Mt, ot, _ = meta("test")
rows_tr = np.load(WD / "train/rows.npy")
rng = np.random.default_rng(0)
sub = np.sort(rng.choice(len(rows_tr), 3_000_000, replace=False))


def psi(a, b, edges):
    pa = np.histogram(a, edges)[0] / max(len(a), 1); pb = np.histogram(b, edges)[0] / max(len(b), 1)
    pa = np.clip(pa, 1e-6, None); pb = np.clip(pb, 1e-6, None)
    return float(((pa - pb) * np.log(pa / pb)).sum())


groups = w25_groups()
names = [n for g in sys.argv[1:] or groups.keys() for n in groups[g]]
seen = set(); names = [n for n in names if not (n in seen or seen.add(n))]
print(f"{'column':26s} {'nanTr':>6s} {'nanV':>6s} {'nanT':>6s} {'psiTV':>7s} {'psiVT':>7s} {'ndcg_trH':>9s} {'ndcg_val':>9s}")
res = {}
for n in names:
    xt = np.load(WD / f"train/{n}.npy", mmap_mode="r")[sub].astype(np.float64)
    xv = np.load(WD / f"val/{n}.npy").astype(np.float64); xs = np.load(WD / f"test/{n}.npy").astype(np.float64)
    xh = load_w25("trH", [n])[:, 0].astype(np.float64)
    ok = ~np.isnan(xv)
    edges = np.unique(np.quantile(xv[ok], np.linspace(0, 1, 11))) if ok.sum() > 100 else np.array([0, 1])
    edges[0] = -np.inf; edges[-1] = np.inf
    p1 = psi(xt[~np.isnan(xt)], xv[ok], edges) if (~np.isnan(xt)).sum() > 100 else np.nan
    p2 = psi(xv[ok], xs[~np.isnan(xs)], edges) if (~np.isnan(xs)).sum() > 100 else np.nan
    fill = lambda x: np.where(np.isnan(x), np.nanmin(x) - 1 if np.isfinite(np.nanmin(x)) else 0, x)
    nh = ndcg_per_feed(fill(xh) + 1e-6 * rng.random(len(xh)), yh, oh, Mh).mean()
    nv_ = ndcg_per_feed(fill(xv) + 1e-6 * rng.random(len(xv)), yv, ov, Mv).mean()
    res[n] = dict(nan_tr=float(np.isnan(xt).mean()), nan_val=float(np.isnan(xv).mean()), nan_test=float(np.isnan(xs).mean()),
                  psi_tr_val=p1, psi_val_test=p2, ndcg_trH=float(nh), ndcg_val=float(nv_))
    r = res[n]
    print(f"{n:26s} {r['nan_tr']:6.3f} {r['nan_val']:6.3f} {r['nan_test']:6.3f} {r['psi_tr_val']:7.3f} {r['psi_val_test']:7.3f} {r['ndcg_trH']:9.4f} {r['ndcg_val']:9.4f}", flush=True)
pj = ROOT / "nn_x/w25/out/psi_w25.json"
old = json.loads(pj.read_text()) if pj.exists() else {}
old.update(res)
pj.write_text(json.dumps(old, indent=1))
