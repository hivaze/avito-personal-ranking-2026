"""winner_model recheck, check 2 (CPU fp32 replay of saved train-only checkpoints on val; nothing written outside this dir).

 (a) exact permutation equivariance: permute the candidate axis (incl. padding) of gathered batches, compare logits.
 (b) sub-feed consistency: score every val feed on random candidate sub-sets, LABEL-FREE (unlike train.py --aug_cand,
     which keeps every positive): partition the feed's candidates at random into G groups; pass g scores the feed
     WITHOUT group g (kept share 1 - 1/G; G = 2 gives two complementary halves). Everything in-model is recomputed from
     the mask (within-feed z / percentile features, log M, feed-category share, set attention); the precomputed xt
     columns (ctx_*) still describe the full feed.
     Outputs per member: full-feed logits, mean sub-feed logits (raw and within-sub-feed z) per scheme, and the
     per-feed Spearman between full-feed and sub-feed logits on the kept candidates (first repetition).
 Usage: CUDA_VISIBLE_DEVICES="" python c02_perm_subfeed.py <variant> [--schemes 2x4,5x1] [--feeds N]
"""
import argparse
import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import torch

torch.set_num_threads(min(12, int(os.environ.get("OMP_NUM_THREADS", "12"))))
sys.path.insert(0, "<repo>/nn_x/w5")  # model.py / data.py only (read-only import; common.py is NOT imported)
from model import Ranker  # noqa: E402
from data import SplitData  # noqa: E402

OUT = "<repo>/nn_x/w12/recheck/winner_model/out"
os.makedirs(OUT, exist_ok=True)
CK = "<repo>/nn_x/w5/ckpt"

p = argparse.ArgumentParser()
p.add_argument("variant")
p.add_argument("--schemes", default="2x4,5x1", help="comma list GxR: G groups (leave-one-group-out), R repetitions")
p.add_argument("--feeds", type=int, default=0, help="subsample val feeds (0 = all)")
p.add_argument("--perm_batches", type=int, default=6)
a = p.parse_args()

dev = torch.device("cpu")
ck = torch.load(f"{CK}/{a.variant}.pt", map_location="cpu", weights_only=False)
model = Ranker(**ck["cfg"])
model.load_state_dict(ck["state_dict"])
model.eval()
va = SplitData("val", dev)
if ck["cfg"].get("xt_names"):
    va.attach_xt(ck["cfg"]["xt_names"])
feeds = np.arange(va.n)
if a.feeds:
    feeds = np.sort(np.random.default_rng(1).choice(va.n, a.feeds, replace=False))
batches = va.batches(feeds, 32768, 512, shuffle=False)
print(f"[{a.variant}] d {ck['cfg']['d']} params {sum(q.numel() for q in model.parameters()) / 1e6:.2f}M; "
      f"{len(feeds)} val feeds, {len(batches)} batches", flush=True)


@torch.no_grad()
def run(mask_fn=None):
    out = torch.full((va.n_cand,), float("nan"))
    for bi, fb in enumerate(batches):
        b = va.gather(fb)
        if mask_fn is not None:
            keep = mask_fn(bi, b)
            b["cmask"] = b["cmask"] & keep
            if "y" in b:
                b["y"] = b["y"] * b["cmask"]
        lg = model(b, use_pos=False)
        m = b["cmask"]
        out[b["ci"][m]] = lg[m].float()
    return out.numpy()


# ---------------------------------------------------------------- (a) permutation equivariance
CAND_KEYS = ["cmask", "ci", "ce_raw", "cmiss", "ccat", "cf", "y", "xt"]
g = torch.Generator().manual_seed(7)
mx, n_rows, flips = 0.0, 0, 0
with torch.no_grad():
    sel = np.linspace(0, len(batches) - 1, min(a.perm_batches, len(batches))).astype(int)
    for bi in sel:
        b = va.gather(batches[bi])
        lg0 = model(b, use_pos=False)
        b2 = va.gather(batches[bi])
        B, Mx = b2["cmask"].shape
        perm = torch.stack([torch.randperm(Mx, generator=g) for _ in range(B)])
        for k in CAND_KEYS:
            if k in b2:
                idx = perm if b2[k].dim() == 2 else perm[..., None].expand(-1, -1, b2[k].shape[-1])
                b2[k] = torch.gather(b2[k], 1, idx)
        lg1 = model(b2, use_pos=False)
        back = torch.gather(lg0, 1, perm)  # lg0 re-ordered like the permuted batch
        m = b2["cmask"]
        d = (lg1 - back)[m].abs()
        mx = max(mx, float(d.max()))
        n_rows += int(m.sum())
        # does the within-feed ORDER change anywhere?
        r0 = back.masked_fill(~m, -1e9).argsort(1, descending=True)
        r1 = lg1.masked_fill(~m, -1e9).argsort(1, descending=True)
        flips += int(((r0 != r1) & m.gather(1, r0)).any(1).sum())
print(f"(a) permutation of the candidate axis (incl. padding), {len(sel)} batches, {n_rows} rows: "
      f"max |dlogit| {mx:.2e}; feeds whose order changed {flips}", flush=True)

# ---------------------------------------------------------------- (b) sub-feed scoring
t0 = time.time()
full = run()
t_full = time.time() - t0
np.save(f"{OUT}/{a.variant}_full.npy", full.astype(np.float32))
print(f"full-feed fp32 pass {t_full:.0f}s", flush=True)

M_np, off_np = va.M_np, va.cand_off_np
for sch in a.schemes.split(","):
    G, R = (int(x) for x in sch.split("x"))
    s_raw = np.zeros(va.n_cand, np.float64)
    s_z = np.zeros(va.n_cand, np.float64)
    cnt = np.zeros(va.n_cand, np.int32)
    rho = None
    t0 = time.time()
    for r in range(R):
        gen = torch.Generator().manual_seed(1000 * G + r)
        # one random group id per candidate, balanced within the feed: rank of a random key modulo G
        grp = {}
        for bi, fb in enumerate(batches):
            Mx = int(-(-M_np[fb].max() // 16) * 16)
            key = torch.rand(len(fb), Mx, generator=gen)
            valid = torch.arange(Mx)[None, :] < torch.from_numpy(M_np[fb])[:, None]
            key = key.masked_fill(~valid, 2.0)
            rk = key.argsort(1).argsort(1)
            grp[bi] = rk % G
        for gi in range(G):
            sc = run(lambda bi, b: grp[bi] != gi)
            ok = ~np.isnan(sc)
            s_raw[ok] += sc[ok]
            cnt[ok] += 1
            # within-sub-feed z of this pass
            fid = np.repeat(np.arange(va.n), M_np)
            n_k = np.bincount(fid[ok], minlength=va.n).astype(np.float64)
            mu = np.bincount(fid[ok], weights=sc[ok], minlength=va.n) / np.maximum(n_k, 1)
            var = np.bincount(fid[ok], weights=sc[ok] ** 2, minlength=va.n) / np.maximum(n_k, 1) - mu ** 2
            sd = np.sqrt(np.maximum(var, 0)) + 1e-9
            s_z[ok] += (sc[ok] - mu[fid[ok]]) / sd[fid[ok]]
            if r == 0 and gi == 0:  # consistency: Spearman(full, sub) on the kept candidates, per feed
                rho = np.full(va.n, np.nan)
                for f in feeds:
                    o, m = off_np[f], M_np[f]
                    k = ok[o:o + m]
                    if k.sum() >= 5:
                        x, y_ = full[o:o + m][k], sc[o:o + m][k]
                        rx, ry = x.argsort().argsort().astype(np.float64), y_.argsort().argsort().astype(np.float64)
                        rho[f] = np.corrcoef(rx, ry)[0, 1]
        if R > 1:  # K-curve: cumulative mean after r + 1 repetitions
            np.save(f"{OUT}/{a.variant}_sub{sch}_z_r{r + 1}.npy", (s_z / np.maximum(cnt, 1)).astype(np.float32))
        print(f"  scheme {sch} rep {r} done, {time.time() - t0:.0f}s", flush=True)
    assert (cnt[np.isin(np.repeat(np.arange(va.n), M_np), feeds)] > 0).all()
    np.save(f"{OUT}/{a.variant}_sub{sch}_raw.npy", (s_raw / np.maximum(cnt, 1)).astype(np.float32))
    np.save(f"{OUT}/{a.variant}_sub{sch}_z.npy", (s_z / np.maximum(cnt, 1)).astype(np.float32))
    np.save(f"{OUT}/{a.variant}_sub{sch}_rho.npy", rho.astype(np.float32))
    print(f"(b) scheme G={G} R={R}: kept share {1 - 1 / G:.2f}, scores per candidate {cnt[cnt > 0].mean():.1f}, "
          f"per-feed Spearman(full, sub) mean {np.nanmean(rho):.4f} median {np.nanmedian(rho):.4f} p10 {np.nanpercentile(rho, 10):.4f}; "
          f"{time.time() - t0:.0f}s", flush=True)
