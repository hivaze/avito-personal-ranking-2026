import sys
sys.path.insert(0, "<repo>/solvers/r6")
import tta
import numpy as np, torch, copy
import data as D
m, ck = tta.load("v17_pre2s_s101")
d = D.SplitData("val", torch.device("cpu"), max_hist=256)
d.attach_xt(ck["cfg"]["xt_names"])
fb = np.where(d.H_np == 256)[0][:8]
d3 = copy.copy(d); d3.H = d.H.clamp_max(64); d3.H_np = np.minimum(d.H_np, 64)
b1 = tta.hist_crop_fn(64)(d.gather(fb)); b2 = d3.gather(fb)
for k in ["hmask", "hemb_ok", "he_raw", "hcat", "het", "hes", "H"]:
    x1, x2 = b1[k], b2[k]
    if x1.dim() > 1: x1 = x1[:, :x2.shape[1]]
    print(k, tuple(b1[k].shape), tuple(b2[k].shape), bool((x1 == x2).all()), b1[k][:, 64:].float().abs().sum().item() if b1[k].dim()>1 else '')
with torch.no_grad():
    X1 = m.dense(b1); X2 = m.dense(b2)
print("dense max abs", (X1 - X2).abs().max().item(), (X1-X2).abs().amax((0,1)).nonzero().flatten().tolist())
with torch.no_grad():
    l1 = m(b1); l2 = m(b2)
print("logit max abs", (l1 - l2).abs().max().item())
# width-only test: same data, pad compaction output to width 64 vs 256
# (1) garbage content: zero masked hcat/het/hes in b1
b1z = dict(b1); 
for k in ["hcat", "het", "hes"]:
    b1z[k] = b1[k] * b1["hmask"]
with torch.no_grad(): l1z = m(b1z)
print("b1 vs b1 zeroed-garbage", (l1 - l1z).abs().max().item())
# (2) width: b2 padded to 256 with copies of masked
def pad(b, K):
    bb = dict(b); k0 = b["hmask"].shape[1]
    for k in ["hmask", "hemb_ok", "hcat", "het", "hes"]:
        bb[k] = torch.cat([b[k], torch.zeros(b[k].shape[0], K - k0, dtype=b[k].dtype)], 1)
    bb["he_raw"] = torch.cat([b["he_raw"], torch.zeros(b["he_raw"].shape[0], K - k0, 128)], 1)
    return bb
with torch.no_grad(): l2p = m(pad(b2, 256)); l2p128 = m(pad(b2, 128))
print("b2 vs b2 padded to 256", (l2 - l2p).abs().max().item(), "to 128", (l2 - l2p128).abs().max().item(), "b1z vs b2pad256", (l1z - l2p).abs().max().item())
# per-module: hist_enc only
import torch.nn.functional as F
with torch.no_grad():
    for K in (64, 256):
        q = torch.randn(2, 4, K, 32); kk = torch.randn(2, 4, K, 32); v = torch.randn(2, 4, K, 32)
    q = torch.randn(1, 1, 5, 8); kk = torch.randn(1, 1, 64, 8); v = torch.randn(1, 1, 64, 8)
    mask = torch.zeros(1, 1, 1, 64, dtype=torch.bool); mask[..., :10] = True
    o1 = F.scaled_dot_product_attention(q, kk, v, attn_mask=mask)
    o2 = F.scaled_dot_product_attention(q, kk[..., :10, :], v[..., :10, :])
    print("sdpa bool mask exactness", (o1 - o2).abs().max().item())
    fm = torch.zeros(1,1,5,64).masked_fill(~mask, float('-inf'))
    o3 = F.scaled_dot_product_attention(q, kk, v, attn_mask=fm)
    print("sdpa float -inf mask", (o3 - o2).abs().max().item())
acts = {}
def hook(name):
    def f(mod, inp, out):
        acts.setdefault(name, []).append(out if not isinstance(out, tuple) else out[0])
    return f
hs = [m.hist_enc[0].register_forward_hook(hook("hist_enc0")), m.retr.register_forward_hook(hook("retr")),
      m.cross[0].register_forward_hook(hook("cross0")), m.sets[0].register_forward_hook(hook("sets0")),
      m.cross[0].attn.register_forward_hook(hook("cross0.attn")), m.hist_enc[0].attn.register_forward_hook(hook("hist0.attn"))]
with torch.no_grad():
    m(b2); m(pad(b2, 256))
for k, (a1, a2) in acts.items():
    if a1.shape[1] != a2.shape[1]:
        n = min(a1.shape[1], a2.shape[1]); a1, a2 = a1[:, :n], a2[:, :n]
    print(k, tuple(a1.shape), (a1 - a2).abs().max().item())
# tie hypothesis: topk indices differ only among equal S values?
with torch.no_grad():
    for bb in (b2, pad(b2, 256)):
        m.dense(bb)
    S1 = torch.einsum("bmd,bkd->bmk", b2["ce"], b2["he"]) * b2["hemb_ok"][:, None, :]
    bp = pad(b2, 256); m.dense(bp)
    S2 = torch.einsum("bmd,bkd->bmk", bp["ce"], bp["he"]) * bp["hemb_ok"][:, None, :]
    t1, i1 = S1.float().masked_fill(~b2["hemb_ok"][:, None, :], -2.0).topk(16, -1)
    t2, i2 = S2.float().masked_fill(~bp["hemb_ok"][:, None, :], -2.0).topk(16, -1)
    print("top values equal", bool((t1 == t2).all()), "idx sets differ (rows)", (np.sort(i1.numpy(), -1) != np.sort(i2.numpy(), -1)).any(-1).mean())
    # duplicates among top values
    dup = (t1[..., 1:] == t1[..., :-1]).any(-1).float().mean().item()
    print("rows with tied values inside top16", dup, "rows where 16th == 17th value", (S1.float().masked_fill(~b2['hemb_ok'][:, None, :], -2.0).topk(17, -1)[0][..., 15] == S1.float().masked_fill(~b2['hemb_ok'][:, None, :], -2.0).topk(17, -1)[0][..., 16]).float().mean().item())
