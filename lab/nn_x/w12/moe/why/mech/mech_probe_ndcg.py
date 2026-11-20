"""w12/moe/why/mech M4: how much of the ranking is a LINEAR function of the per-candidate inputs?
For each reference member (w5_e10_kd05w192_s951-954; X is identical across them: same normalisation stats):
  probe(layer -> logit): least squares of the member's within-feed-centred logit on the within-feed-centred layer, fit on the HOLDOUT
     (30,000 feeds), scored on VAL (time-forward) and on the holdout itself; layers X (104 normalised dense inputs), c0 (token entering
     block 1, 192), last block output (192), and the top-1 / top-8 principal directions of X.
  click-LS(X): least squares of the within-feed-centred CLICK label on X (a linear ranker on the 104 inputs, 105 numbers), fit on the
     holdout, scored on val.
NDCG deltas vs the member itself, paired per feed (val n=14,423; holdout n=30,000)."""
import sys, numpy as np, torch
sys.path.insert(0, '<repo>/lab/nn_x/w12/moe/why/mech')
from mech_common import *  # noqa
dev = torch.device('cuda:0')
m0, cfg = build('w5_e10_kd05w192_s951', dev)
va = load_val(dev, cfg['xt_names']); ho = load_hold(dev, cfg['xt_names'])
def capture(model, data, tf32):
    cap = {k: torch.zeros((data.n_cand, d), dtype=torch.float32, device=dev) for k, d in (('X', cfg['n_dense']), ('c0', 192), ('last', 192))}
    st = {}
    def pre(b): st['b'] = b
    hs = [model.c_mlp.register_forward_hook(lambda m_, a_, o_: st.__setitem__('X', a_[0])),
          model.cross[0].register_forward_hook(lambda m_, a_, o_: st.__setitem__('c0', a_[0])),
          model.sets[-1].register_forward_hook(lambda m_, a_, o_: st.__setitem__('last', o_))]
    def post(b, lg):
        m = b['cmask']; ci = b['ci'][m]
        for k in cap: cap[k][ci] = st[k].float()[m]
    s = predict(model, data, pre=pre, post=post, tf32=tf32)
    for h in hs: h.remove()
    return s, cap
def wfc_flat(Z, data):
    fid = torch.from_numpy(np.repeat(np.arange(data.n), data.M_np)).to(dev)
    cnt = torch.from_numpy(data.M_np).to(dev).double()
    mu = torch.zeros((data.n, Z.shape[1]), dtype=torch.float64, device=dev).index_add_(0, fid, Z.double()) / cnt[:, None]
    return (Z.double() - mu[fid])
def fit(Zc, yc):
    S = Zc.T @ Zc; v = Zc.T @ yc
    return torch.linalg.solve(S + 1e-6 * torch.trace(S) / S.shape[0] * torch.eye(S.shape[0], device=dev, dtype=S.dtype), v)
out = []
for s in (951, 952, 953, 954):
    v = f'w5_e10_kd05w192_s{s}'
    model, _ = build(v, dev)
    sv, cv = capture(model, va, True); sh, ch = capture(model, ho, False)
    nv0, nh0 = split_ndcg(va, sv), split_ndcg(ho, sh)
    yh = wfc_flat(torch.from_numpy(sh).to(dev)[:, None], ho)[:, 0]
    rows = {}
    Xh = wfc_flat(ch['X'], ho)
    ev, U = torch.linalg.eigh(Xh.T @ Xh); U = U.flip(1)
    for name, Zh, Zv in (('X', Xh, cv['X'].double()), ('c0', wfc_flat(ch['c0'], ho), cv['c0'].double()), ('last', wfc_flat(ch['last'], ho), cv['last'].double()),
                         ('X top-1 PC', Xh @ U[:, :1], cv['X'].double() @ U[:, :1]), ('X top-8 PC', Xh @ U[:, :8], cv['X'].double() @ U[:, :8])):
        beta = fit(Zh, yh)
        r2 = float(1 - ((yh - Zh @ beta) ** 2).sum() / (yh ** 2).sum())
        pv = (Zv @ beta).float().cpu().numpy(); ph = (Zh @ beta).float().cpu().numpy()
        dv = split_ndcg(va, pv) - nv0; dh = split_ndcg(ho, ph) - nh0
        rows[name] = (r2, dv, dh)
    yc = wfc_flat(torch.from_numpy(ho.target_np).to(dev)[:, None], ho)[:, 0]
    beta = fit(Xh, yc)
    pv = (cv['X'].double() @ beta).float().cpu().numpy(); ph = (Xh @ beta).float().cpu().numpy()
    rows['click-LS(X)'] = (float('nan'), split_ndcg(va, pv) - nv0, split_ndcg(ho, ph) - nh0)
    for k, (r2, dv, dh) in rows.items():
        print(f'{v} {k:12s} logit R2 (holdout fit) {r2:.3f} | val NDCG delta vs member {dv.mean():+.5f}+-{se(dv):.5f} | holdout {dh.mean():+.5f}+-{se(dh):.5f}', flush=True)
    print(f'{v} member NDCG val {nv0.mean():.5f} holdout {nh0.mean():.5f}', flush=True)
    out.append(rows)
print('== mean over the 4 members (per-feed mean of the deltas)')
for k in out[0]:
    dv = np.mean([o[k][1] for o in out], 0); dh = np.mean([o[k][2] for o in out], 0)
    print(f'{k:12s} logit R2 {np.nanmean([o[k][0] for o in out]):.3f} | val {dv.mean():+.5f}+-{se(dv):.5f} | holdout {dh.mean():+.5f}+-{se(dh):.5f}')
