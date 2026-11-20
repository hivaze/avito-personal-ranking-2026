"""Unit test of compact_hist: (a) keep=hmask is identity; (b) compaction == physically removing the events from the
flat history arrays (rebuilt hist_off/H) for 600 feeds; (c) crop: same check."""
import sys, time
sys.path.insert(0, "<repo>/solvers/r6")
import tta  # sets threads / CUDA_VISIBLE_DEVICES
import numpy as np, torch, copy
import data as D
m, ck = tta.load("v17_pre2s_s101")
d = D.SplitData("val", torch.device("cpu"), max_hist=256)
d.attach_xt(ck["cfg"]["xt_names"])
rng = np.random.default_rng(0)
fids = np.sort(rng.choice(d.n, 600, replace=False))
# include short-history feeds
fids = np.unique(np.concatenate([fids, np.where(d.H_np <= 3)[0][:40]]))
u = rng.random(d.n_hist)  # per flat history event
keep_flat = u >= 0.3
def logits(dd, fb, fn=None):
    b = dd.gather(fb)
    if fn: b = fn(b)
    with torch.no_grad():
        return m(b, use_pos=False)[b["cmask"]].numpy(), b
fb = fids[np.argsort(d.M_np[fids])]
t0=time.time()
base, b0 = logits(d, fb)
ident, _ = logits(d, fb, lambda b: tta.compact_hist(b, b["hmask"].clone()))
print("(a) identity max abs", np.abs(base - ident).max())
crop0, _ = logits(d, fb, tta.hist_crop_fn(256))
print("(a2) crop256 max abs", np.abs(base - crop0).max())
# (b) drop via global per-event mask
hist_off = d.hist_off.numpy()
def drop_fn(b):
    hi = (d.hist_off[b["fidx"]][:, None] + torch.arange(b["hmask"].shape[1])[None]).clamp_max(d.n_hist - 1)
    k = torch.from_numpy(keep_flat)[hi]
    return tta.compact_hist(b, b["hmask"] & k)
comp, bc = logits(d, fb, drop_fn)
# physical rebuild: new flat arrays for all feeds (events beyond max_hist kept as is)
d2 = copy.copy(d)
idx = []; newH = np.zeros(d.n, np.int64); newoff = np.zeros(d.n, np.int64)
for i in range(d.n):
    o, h = hist_off[i], int(d.H_np[i])
    ev = np.arange(o, o + h)
    kk = keep_flat[ev] | (np.arange(h) >= 256)
    newoff[i] = len(idx) and sum(map(len, idx))
    idx.append(ev[kk]); newH[i] = kk.sum()
idx = np.concatenate(idx)
newoff = np.concatenate([[0], np.cumsum(newH)[:-1]])
for nm in ["h_emb", "h_cat", "h_et", "h_es"]:
    setattr(d2, nm, getattr(d, nm)[torch.from_numpy(idx)])
d2.n_hist = len(idx); d2.hist_off = torch.from_numpy(newoff); d2.H = torch.from_numpy(newH); d2.H_np = newH
phys, bp = logits(d2, fb)
print("(b) drop: compaction vs physical removal: max abs", np.abs(comp - phys).max(), "corr", np.corrcoef(comp, phys)[0,1],
      "mean |base-comp|", np.abs(base-comp).mean(), "H equal", bool((bc["H"]==bp["H"]).all()))
# (c) crop 64 vs physical
d3 = copy.copy(d); d3.H = d.H.clamp_max(64); d3.H_np = np.minimum(d.H_np, 64)
c64, _ = logits(d, fb, tta.hist_crop_fn(64)); p64, _ = logits(d3, fb)
print("(c) crop64 vs physical max abs", np.abs(c64 - p64).max())
# (d) MC-dropout modules count, and randomness
n = tta.set_mc(m, True); torch.manual_seed(1); r1, _ = logits(d, fb); torch.manual_seed(1); r2, _ = logits(d, fb); torch.manual_seed(2); r3,_ = logits(d, fb)
tta.set_mc(m, False); back, _ = logits(d, fb)
print("(d) mc modules", n, "same-seed max abs", np.abs(r1-r2).max(), "diff-seed mean abs", np.abs(r1-r3).mean(), "vs det mean abs", np.abs(r1-base).mean(), "eval restored", np.abs(back-base).max())
print("ranker training flag under mc:", (tta.set_mc(m, True), m.training)[1]); tta.set_mc(m, False)
print(f"{time.time()-t0:.0f}s")
