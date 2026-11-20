"""How ambiguous is HistRetrieval's top-16? Share of candidate rows (with an embedding, >16 valid history items) whose
16th and 17th largest candidate-history cosines are exactly equal, fp32 vs bf16-rounded cosine (GPU autocast path),
and how much of it is duplicate history items (same emb row). 1500 random val feeds."""
import os, sys
os.environ["OMP_NUM_THREADS"] = "4"
sys.path.insert(0, "<repo>/solvers/r6")
import tta
import numpy as np, torch
torch.set_num_threads(4)
import data as D
d = D.SplitData("val", torch.device("cpu"), max_hist=256)
rng = np.random.default_rng(0)
fids = np.sort(rng.choice(d.n, 1500, replace=False))
tot = dict(rows=0, tie32=0, tie16=0, anytie32=0, anytie16=0)
dup_share = []
with torch.no_grad():
    for fb in d.batches(fids, 32768, 512, shuffle=False):
        b = d.gather(fb)
        ce = b["ce_raw"] / b["ce_raw"].norm(dim=-1).clamp_min(1e-3)[..., None]
        he = b["he_raw"] / b["he_raw"].norm(dim=-1).clamp_min(1e-3)[..., None]
        hv = b["hemb_ok"]
        rows = b["cmask"] & ~b["cmiss"] & (hv.sum(1) > 16)[:, None]
        for prec in ("32", "16"):
            S = torch.einsum("bmd,bkd->bmk", ce, he) if prec == "32" else torch.einsum("bmd,bkd->bmk", ce.bfloat16(), he.bfloat16()).float()
            t, _ = S.masked_fill(~hv[:, None, :], -2.0).topk(17, -1)
            tot["tie" + prec] += ((t[..., 15] == t[..., 16]) & rows).sum().item()
            tot["anytie" + prec] += (((t[..., 1:16] == t[..., :15]).any(-1)) & rows).sum().item()
        tot["rows"] += rows.sum().item()
        # duplicate history items: share of valid history events whose emb row already occurs earlier in the history
        hi = (d.hist_off[b["fidx"]][:, None] + torch.arange(hv.shape[1])[None]).clamp_max(d.n_hist - 1)
        e = d.h_emb[hi].numpy(); hvn = hv.numpy()
        for i in range(len(e)):
            x = e[i][hvn[i]]
            if len(x): dup_share.append(1 - len(np.unique(x)) / len(x))
print({k: (v / tot["rows"] if k != "rows" else v) for k, v in tot.items()})
print("mean share of duplicate history events per feed", float(np.mean(dup_share)), "median", float(np.median(dup_share)))
