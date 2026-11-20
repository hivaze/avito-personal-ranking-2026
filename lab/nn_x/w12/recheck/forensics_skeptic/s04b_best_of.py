"""Honest version of the report's 'best of 248 columns = +0.00044 +- 0.00020': the column AND the weight are picked on the fit half."""
from hx import *
from concurrent.futures import ThreadPoolExecutor
D = load_val(); y, M, off = D["y"], D["M"], D["off"]; src = FEATS / "val"
BAN = lambda n: ("id_contrib" in n) or n.startswith("kv64") or n in ("M", "H", "hour", "category_id") or (n.startswith("it_hist_all") and "luo" not in n) or (n.startswith("it_hist_et") and not n.endswith("_luo"))
cols = {}
for n_ in sorted(p.stem for p in src.glob("*.npy")):
    if BAN(n_): continue
    x = np.load(src / f"{n_}.npy", mmap_mode="r")
    if x.ndim != 1 or x.shape[0] != len(y): continue
    x = np.nan_to_num(np.asarray(x, np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if x.std() > 0: cols[n_] = x
names = list(cols); p0 = ndcg_per_feed(D["base"], y, off, M); ws = (-0.05, -0.02, 0.02, 0.05)
def scr(n_):
    z = zfeed(cols[n_], off, M); return np.stack([ndcg_per_feed(D["base"] + w * z, y, off, M) - p0 for w in ws]).astype(np.float32)
with ThreadPoolExecutor(12) as ex: G = np.stack(list(ex.map(scr, names)))   # [col, w, feed]
print("columns", len(names))
for seed in (0, 1, 2, 3, 4):
    half = user_half(D["uid"], seed); d = np.zeros(len(M)); pk = []
    for a in (True, False):
        tr = half == a; m = G[:, :, tr].mean(2); ci, wi = np.unravel_index(np.argmax(m), m.shape); d[~tr] = G[ci, wi, ~tr]; pk.append((names[ci], ws[wi], round(float(m[ci, wi]), 5)))
    print(f"split seed {seed}: best column + weight picked on the fit half -> other half: {fmt(d)} (z {d.mean() / se(d):+.2f}); halves {fmt(d[half])} / {fmt(d[~half])}; picks {pk}", flush=True)
# null reference: distribution of the in-sample maximum over the columns
print("in-sample max over columns x weights:", float(G.mean(2).max()), "| share of columns with a positive best-w mean:", float((G.mean(2).max(1) > 0).mean()))
