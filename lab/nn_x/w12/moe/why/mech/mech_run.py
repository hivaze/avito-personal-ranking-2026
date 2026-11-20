"""w12/moe/why/mech: mechanism analysis of the MoE / depth screen checkpoints (GPU forward passes with hooks; read-only).
  python mech_run.py <group> <seeds csv>     group in ref | A | B | C  (selected checkpoints w5_e10_<arm>_s<seed>.pt)
Per model, on the FULL val split (14,423 feeds) and the FULL holdout (30,000 feeds):
  intact pass (instrumented; reproduces the stored predictions bit-exactly, checked):
    per candidate block: RMS of block input / attention update / dense FFN output / routed output (raw and within-feed centred),
      cos(routed, dense FFN); attention entropy per head (normalised by log #valid keys), max prob, user-token / self mass;
    within-feed-centred second moments of X (normalised dense input), c0 (token entering block 1), every block output and the head hidden
      -> effective rank + linear-probe R^2 of the (within-feed-centred) final logit from [layer] and [layer, X];
    routing records (arm A: top-2 experts / gates per candidate and block; arm B: per head top-2 for the V router on key tokens and the
      O router on candidates; sigmoid gate moments) saved for the CPU specialisation / cross-seed analysis (mech_mi.py).
  ablation passes (per-feed NDCG saved; delta vs intact computed later, paired per feed):
    every model: drop each candidate block (identity), zero its attention update, zero its dense FFN; drop the history-encoder block.
    A: zero routed experts (all blocks / each block), random top-2 routing, constant mean-gate mixture, zero dense+routed FFN per block.
    B: zero routed V+O / V only / O only (all), zero per block, random top-2 routing, constant mean-gate mixture.
"""
import json
import math
import sys
import time
import types

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, '<repo>/lab/nn_x/w12/moe/why/mech')
from mech_common import *  # noqa
from model_moe import RoutedFFN, SwitchVO  # noqa

group = sys.argv[1]
seeds = [int(s) for s in sys.argv[2].split(',')]
ARM = dict(ref='kd05w192', A='kd05w192moeffn', B='kd05w192moevo', C='kd05w192b3ff1')[group]
dev = torch.device('cuda:0')
OUTD = OUT / group
OUTD.mkdir(parents=True, exist_ok=True)
CTX = {}


def wfc(t, m):
    """within-feed centring over real candidates; t [B,M,D] float, m [B,M] bool -> centred, zero on padding."""
    mf = m[..., None].float()
    mu = (t * mf).sum(1, keepdim=True) / mf.sum(1, keepdim=True).clamp_min(1)
    return (t - mu) * mf


class Acc(dict):
    def add(self, k, v):
        v = v.detach() if torch.is_tensor(v) else v
        self[k] = self[k] + v if k in self else v


# ------------------------------------------------------------------ patched routed modules (intact mode == original forward, verified)
def moe_forward(self, x, tmask=None, shared=None):
    mode = self._mode
    if mode == 'zero':
        return torch.zeros_like(x)
    D = x.shape[-1]
    E, w = self.n_exp, self.width
    xf = x.reshape(-1, D)
    with torch.autocast('cuda', enabled=False):
        s = torch.sqrt(F.softplus(self.router(xf.float())))
        z = s + self.rbias[0]
        sel = z.topk(self.k, -1).indices
        if mode == 'random':
            sel = torch.rand(s.shape, device=s.device, generator=CTX['gen']).topk(self.k, -1).indices
        g = s.gather(-1, sel)
        g = self.scale * g / g.sum(-1, keepdim=True).clamp_min(1e-9)
        gate = torch.zeros_like(s).scatter(-1, sel, g)
        if mode == 'meanmix':
            gate = self._gbar[CTX['split']][None].expand_as(s).contiguous()
    h = F.gelu(xf @ self.w1.reshape(E * w, D).t())
    h = h.view(-1, E, w) * gate.to(h.dtype)[..., None]
    out = self.dp(h.reshape(-1, E * w)) @ self.w2.reshape(E * w, D)
    if self._rec is not None:
        m = CTX['b']['cmask'].reshape(-1)
        ci = CTX['b']['ci'].reshape(-1)[m]
        rec = self._rec
        rec['sel'][ci] = sel[m].to(torch.int8)
        rec['gate'][ci] = g[m].to(torch.float16)
        rec['s'][ci] = s[m].to(torch.float16)
        CTX['acc'].add(f'{self._name}.gsum', gate[m].sum(0).double())
        CTX['acc'].add(f'{self._name}.n', float(m.sum()))
    return out.view(x.shape)


def vo_route(self, r, x, gi, tmask):
    mode = self._mode[gi]
    with torch.autocast('cuda', enabled=False):
        s = torch.sigmoid(r(x.float())).view(*x.shape[:-1], self.h, self.E)
        z = s + self.rbias[gi]
        sel = z.topk(self.k, -1).indices
        if mode == 'random':
            sel = torch.rand(s.shape, device=s.device, generator=CTX['gen']).topk(self.k, -1).indices
        gate = torch.zeros_like(s).scatter(-1, sel, s.gather(-1, sel))
        if mode == 'meanmix':
            gate = self._gbar[CTX['split']][gi].expand_as(s).contiguous()
        if mode == 'zero':
            gate = torch.zeros_like(s)
    if self._rec is not None:
        if gi == 0:  # key tokens [B, K]: canonical index = hist row (history keys) or n_hist + feed (user token)
            m = tmask
            idx = CTX['key_idx'][m]
        else:
            m = CTX['b']['cmask']
            idx = CTX['b']['ci'][m]
        rec = self._rec[gi]
        rec['sel'][idx] = sel[m].to(torch.int8)
        rec['s'][idx] = s[m].to(torch.float16)
        zs = z[m].sort(-1, descending=True).values
        a = CTX['acc']
        nm = f'{self._name}.{"VO"[gi]}'
        a.add(nm + '.gsum', gate[m].sum(0).double())
        a.add(nm + '.ssum', s[m].sum(0).double())
        a.add(nm + '.s2sum', (s[m].double() ** 2).sum(0))
        a.add(nm + '.selgsum', s.gather(-1, sel)[m].sum(0).double())
        a.add(nm + '.margin', (zs[..., self.k - 1] - zs[..., self.k]).sum(0).double())
        a.add(nm + '.n', float(m.sum()))
    return gate


def vo_value(self, ctx, kmask, v):
    B, K, D = ctx.shape
    h, E, dh = self.h, self.E, self.dh
    gv = self._route(self.rv, ctx, 0, kmask)
    ve = (ctx @ self.wv.permute(1, 0, 2).reshape(D, h * E * dh)).view(B, K, h, E, dh)
    rv = (ve * gv.to(ve.dtype)[..., None]).sum(3)
    if self._rec is not None:
        mf = kmask[..., None, None].float()
        CTX['acc'].add(f'{self._name}.V.rout2', (rv.float() ** 2 * mf).sum().double())
        CTX['acc'].add(f'{self._name}.V.dense2', (v.float() ** 2 * mf).sum().double())
        CTX['acc'].add(f'{self._name}.V.cos', (F.cosine_similarity(rv.float(), v.float(), dim=-1).mean(-1) * kmask.float()).sum().double())
    return rv


def vo_output(self, o, x, qmask, out):
    B, N, D = x.shape
    h, E, dh = self.h, self.E, self.dh
    go = self._route(self.ro, x, 1, qmask)
    oe = (o[:, :, :, None, :] * go.to(o.dtype)[..., None]).reshape(B, N, h * E * dh)
    ro = oe @ self.wo.reshape(h * E * dh, D)
    if self._rec is not None:
        m = CTX['b']['cmask']
        mf = m[..., None].float()
        a = CTX['acc']
        a.add(f'{self._name}.O.rout2', (ro.float() ** 2 * mf).sum().double())
        a.add(f'{self._name}.O.dense2', (out.float() ** 2 * mf).sum().double())
        a.add(f'{self._name}.O.rout2c', (wfc(ro.float(), m) ** 2).sum().double())
        a.add(f'{self._name}.O.dense2c', (wfc(out.float(), m) ** 2).sum().double())
        a.add(f'{self._name}.O.cos', (F.cosine_similarity(ro.float(), out.float(), dim=-1) * m.float()).sum().double())
    return ro


def setup_model(model):
    blocks = []
    for i in range(len(model.cross)):
        blocks += [(f'cross.{i}', model.cross[i]), (f'sets.{i}', model.sets[i])]
    for n, blk in blocks:
        blk._name = n
        if blk.moe is not None:
            blk.moe._name, blk.moe._mode, blk.moe._rec, blk.moe._gbar = n, 'intact', None, {}
            blk.moe.forward = types.MethodType(moe_forward, blk.moe)
        vo = blk.attn.vo
        if vo is not None:
            vo._name, vo._mode, vo._rec, vo._gbar = n, ['intact', 'intact'], None, {}
            vo._route = types.MethodType(vo_route, vo)
            vo.value = types.MethodType(vo_value, vo)
            vo.output = types.MethodType(vo_output, vo)
    return blocks


# ------------------------------------------------------------------ instrumentation hooks (intact pass only)
def attn_stats(attn, x, ctx, kmask, bias, name, is_cross):
    B, N, D = x.shape
    K = ctx.shape[1]
    h = attn.h
    dh = D // h
    with torch.autocast('cuda', enabled=False):
        q = attn.q(x.float()).view(B, N, h, dh).transpose(1, 2)
        k = (attn.k(ctx.float()) if attn.split_kv else attn.kv(ctx.float())[..., :D]).view(B, K, h, dh).transpose(1, 2)
        sc = q @ k.transpose(-1, -2) / math.sqrt(dh)
        if bias is not None and hasattr(attn, 'bias_scale'):
            sc = sc + bias[:, None].float() * attn.bias_scale[None, :, None, None].float()
        sc = sc.masked_fill(~kmask[:, None, None, :], float('-inf'))
        p = torch.softmax(sc, -1)
        ent = -(p * p.clamp_min(1e-30).log()).sum(-1)  # [B,h,N]
        nv = kmask.sum(-1).float().clamp_min(2)[:, None, None]
        m = CTX['b']['cmask'][:, None, :].float().expand(B, h, N)
        a = CTX['acc']
        a.add(f'{name}.att.ent', (ent / nv.log() * m).sum((0, 2)).double())
        a.add(f'{name}.att.entbits', (ent / math.log(2) * m).sum((0, 2)).double())
        a.add(f'{name}.att.maxp', (p.max(-1).values * m).sum((0, 2)).double())
        a.add(f'{name}.att.effk', (ent.exp() / nv * m).sum((0, 2)).double())
        if is_cross:
            a.add(f'{name}.att.user', (p[..., 0] * m).sum((0, 2)).double())
        else:
            a.add(f'{name}.att.self', (torch.diagonal(p, dim1=-2, dim2=-1) * m).sum((0, 2)).double())
        a.add(f'{name}.att.n', m.sum((0, 2)).double())


def install_hooks(model, blocks):
    hs = []
    a = lambda: CTX['acc']

    def blk_hook(name):
        def f(mod, args, out):
            m = CTX['b']['cmask']
            mf = m[..., None].float()
            xin, xo = args[0].float(), out.float()
            if name == blocks[0][0]:
                CTX['layers']['c0'] = wfc(xin, m)[m]
            CTX['layers'][name] = wfc(xo, m)[m]
            for tag, t in (('in', xin), ('out', xo), ('upd', xo - xin)):
                a().add(f'{name}.{tag}2', (t ** 2 * mf).sum().double())
                a().add(f'{name}.{tag}2c', (wfc(t, m) ** 2).sum().double())
            a().add(f'{name}.ntok', float(m.sum()))
        return f

    def sub_hook(name, tag):
        def f(mod, args, out):
            m = CTX['b']['cmask']
            t = out.float()
            a().add(f'{name}.{tag}2', (t ** 2 * m[..., None].float()).sum().double())
            a().add(f'{name}.{tag}2c', (wfc(t, m) ** 2).sum().double())
            CTX['sub'][(name, tag)] = t
            if tag == 'moe':
                fo = CTX['sub'].get((name, 'ff'))
                a().add(f'{name}.cos_moe_ff', (F.cosine_similarity(t, fo, dim=-1) * m.float()).sum().double())
        return f

    def att_hook(name, is_cross):
        def f(mod, args, out):
            x, ctx, kmask, bias = args[0], args[1], args[2], (args[3] if len(args) > 3 else None)
            attn_stats(mod, x, ctx, kmask, bias, name, is_cross)
        return f

    for name, blk in blocks:
        hs.append(blk.register_forward_hook(blk_hook(name)))
        hs.append(blk.attn.register_forward_hook(sub_hook(name, 'att')))
        hs.append(blk.attn.register_forward_hook(att_hook(name, name.startswith('cross'))))
        hs.append(blk.ff.register_forward_hook(sub_hook(name, 'ff')))
        if blk.moe is not None:
            hs.append(blk.moe.register_forward_hook(sub_hook(name, 'moe')))

    def x_hook(mod, args, out):
        m = CTX['b']['cmask']
        CTX['layers']['X'] = wfc(args[0].float(), m)[m]

    def head_hook(mod, args, out):
        m = CTX['b']['cmask']
        CTX['layers']['head.h'] = wfc(out.float(), m)[m]
    hs.append(model.c_mlp.register_forward_hook(x_hook))
    hs.append(model.head[2].register_forward_hook(head_hook))
    return hs


def make_pre(data):
    def pre(b):
        CTX['b'] = b
        CTX['layers'] = {}
        CTX['sub'] = {}
        if 'hmask' in b:
            Hx = b['hmask'].shape[1]
            hi = data.hist_off[b['fidx']][:, None] + torch.arange(Hx, device=dev)[None]
            ui = data.n_hist + CTX['feed_local'][b['fidx']]
            CTX['key_idx'] = torch.cat([ui[:, None], hi], 1)
    return pre


def post_instr(b, lg):
    m = b['cmask']
    y = wfc(lg.float()[..., None], m)[m].double()
    a = CTX['acc']
    X = CTX['layers']['X'].double()
    a.add('yy', (y ** 2).sum())
    for ln, Z in CTX['layers'].items():
        Zc = torch.cat([Z.double(), X], 1) if ln != 'X' else X
        a.add(f'L.{ln}.S', Zc.T @ Zc)
        a.add(f'L.{ln}.v', (Zc * y).sum(0))
    a.add('ntok', float(m.sum()))


def summarize_layers(acc, d_x):
    res = {}
    yy = float(acc['yy'])
    for k in [k for k in acc if k.startswith('L.') and k.endswith('.S')]:
        ln = k[2:-2]
        S, v = acc[k], acc[f'L.{ln}.v']
        D = S.shape[0] - (d_x if ln != 'X' else 0)
        out = {}
        for tag, ix in (('self', slice(0, D)), ('withX', slice(0, S.shape[0]))):
            if ln == 'X' and tag == 'withX':
                continue
            Ss, vs = S[ix, ix], v[ix]
            reg = 1e-8 * torch.trace(Ss) / Ss.shape[0]
            beta = torch.linalg.solve(Ss + reg * torch.eye(Ss.shape[0], device=Ss.device, dtype=Ss.dtype), vs)
            out[f'r2_{tag}'] = float(beta @ vs / yy)
        evt, U = torch.linalg.eigh(S[:D, :D])
        evt, U = evt.flip(0).clamp_min(1e-30), U.flip(1)
        contrib = ((U.T @ v[:D]) ** 2 / evt).cpu().numpy() / yy  # logit R^2 carried by each principal direction (descending variance)
        out['r2_topk'] = {int(k): float(contrib[:k].sum()) for k in (1, 2, 4, 8, 16, 32, 64, D) if k <= D}
        ev = evt.clamp_min(0).cpu().numpy()
        p = ev / ev.sum()
        cs = np.cumsum(p)
        out.update(dim=int(D), pr=float(ev.sum() ** 2 / (ev ** 2).sum()), erank=float(np.exp(-(p[p > 0] * np.log(p[p > 0])).sum())),
                   k90=int(np.searchsorted(cs, 0.90) + 1), k99=int(np.searchsorted(cs, 0.99) + 1), top1=float(p[0]),
                   spectrum=[float(x) for x in p[:64]])
        res[ln] = out
    return res


def summarize_acc(acc, blocks, model):
    res = {}
    for name, blk in blocks:
        n = float(acc[f'{name}.ntok'])
        r = {}
        for tag in ('in', 'out', 'upd', 'att', 'ff', 'moe'):
            if f'{name}.{tag}2' in acc:
                r[f'rms_{tag}'] = math.sqrt(float(acc[f'{name}.{tag}2']) / n)
                r[f'rmsc_{tag}'] = math.sqrt(float(acc[f'{name}.{tag}2c']) / n)
        if f'{name}.cos_moe_ff' in acc:
            r['cos_moe_ff'] = float(acc[f'{name}.cos_moe_ff']) / n
        na = acc[f'{name}.att.n']
        for t in ('ent', 'entbits', 'maxp', 'effk', 'user', 'self'):
            if f'{name}.att.{t}' in acc:
                r[f'att_{t}'] = (acc[f'{name}.att.{t}'] / na).cpu().tolist()
        if blk.moe is not None:
            r['moe_gbar'] = (acc[f'{name}.gsum'] / acc[f'{name}.n']).cpu().tolist()
            r['moe_rbias'] = blk.moe.rbias[0].cpu().tolist()
        if blk.attn.vo is not None:
            for R in 'VO':
                k = f'{name}.{R}'
                nn_ = acc[k + '.n']
                sm = acc[k + '.ssum'] / nn_
                r[f'vo_{R}_gbar'] = (acc[k + '.gsum'] / nn_).cpu().tolist()
                r[f'vo_{R}_smean'] = sm.cpu().tolist()
                r[f'vo_{R}_sstd'] = (acc[k + '.s2sum'] / nn_ - sm ** 2).clamp_min(0).sqrt().cpu().tolist()
                r[f'vo_{R}_selg'] = (acc[k + '.selgsum'] / nn_).cpu().tolist()
                r[f'vo_{R}_margin'] = (acc[k + '.margin'] / nn_).cpu().tolist()
                if R == 'V':
                    r['vo_V_rms_ratio'] = math.sqrt(float(acc[k + '.rout2']) / float(acc[k + '.dense2']))
                    r['vo_V_cos'] = float(acc[k + '.cos']) / float(nn_)
                else:
                    r['vo_O_rms_ratio'] = math.sqrt(float(acc[k + '.rout2']) / float(acc[k + '.dense2']))
                    r['vo_O_rmsc_ratio'] = math.sqrt(float(acc[k + '.rout2c']) / float(acc[k + '.dense2c']))
                    r['vo_O_cos'] = float(acc[k + '.cos']) / float(nn_)
        res[name] = r
    return res


# ------------------------------------------------------------------ ablation context managers (forward hooks replacing outputs)
def ablate(model, blocks, spec):
    """spec: list of (kind, block name or None). Returns hook handles; modes on routed modules are set in place (reset by reset_modes)."""
    hs = []
    bd = dict(blocks)
    for kind, name in spec:
        if kind == 'drop':
            mod = model.hist_enc[0] if name == 'hist_enc.0' else bd[name]
            hs.append(mod.register_forward_hook(lambda m_, a_, o_: a_[0]))
        elif kind == 'zero_attn':
            hs.append(bd[name].attn.register_forward_hook(lambda m_, a_, o_: torch.zeros_like(o_)))
        elif kind == 'zero_ff':
            hs.append(bd[name].ff.register_forward_hook(lambda m_, a_, o_: torch.zeros_like(o_)))
        elif kind in ('moe_zero', 'moe_random', 'moe_meanmix'):
            for n_, blk in blocks:
                if blk.moe is not None and (name is None or n_ == name):
                    blk.moe._mode = kind[4:]
        elif kind.startswith('vo_'):
            _, which, mode = kind.split('_')  # vo_<V|O|VO>_<zero|random|meanmix>
            for n_, blk in blocks:
                if blk.attn.vo is not None and (name is None or n_ == name):
                    for gi, R in enumerate('VO'):
                        if R in which:
                            blk.attn.vo._mode[gi] = mode
    return hs


def reset_modes(blocks):
    for _, blk in blocks:
        if blk.moe is not None:
            blk.moe._mode = 'intact'
        if blk.attn.vo is not None:
            blk.attn.vo._mode = ['intact', 'intact']


def ablation_list(blocks, model):
    L = [('hist_enc.drop', [('drop', 'hist_enc.0')])]
    for n, _ in blocks:
        L += [(f'{n}.drop', [('drop', n)]), (f'{n}.zero_attn', [('zero_attn', n)]), (f'{n}.zero_ff', [('zero_ff', n)])]
    if group == 'A':
        L += [('moe.zero_all', [('moe_zero', None)]), ('moe.random_all', [('moe_random', None)]), ('moe.meanmix_all', [('moe_meanmix', None)])]
        for n, _ in blocks:
            L += [(f'{n}.moe_zero', [('moe_zero', n)]), (f'{n}.zero_ff+moe', [('zero_ff', n), ('moe_zero', n)]),
                  (f'{n}.moe_random', [('moe_random', n)])]
    if group == 'B':
        for which in ('VO', 'V', 'O'):
            L += [(f'vo.{which}.zero_all', [(f'vo_{which}_zero', None)])]
        L += [('vo.VO.random_all', [('vo_VO_random', None)]), ('vo.VO.meanmix_all', [('vo_VO_meanmix', None)]),
              ('vo.V.random_all', [('vo_V_random', None)]), ('vo.O.random_all', [('vo_O_random', None)])]
        for n, blk in blocks:
            if blk.attn.vo is not None:
                L += [(f'{n}.vo_zero', [('vo_VO_zero', n)])]
    return L


def wf_rel_perturb(data, s0, s1):
    """sqrt( sum_feeds sum_c (d - mean_f d)^2 / sum_feeds sum_c (s0 - mean_f s0)^2 ) over all real candidates."""
    fid = np.repeat(np.arange(data.n), data.M_np)
    def cen(x):
        x = x.astype(np.float64)
        return x - (np.bincount(fid, x, data.n) / data.M_np)[fid]
    a, b = cen(s0), cen(s1)
    rho = float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))
    # (raw relative change, scale-free residual sqrt(1 - rho^2) / rho: = r for i.i.d. noise of relative size r)
    return float(np.sqrt(((b - a) ** 2).sum() / (a * a).sum())), float(np.sqrt(max(0.0, 1 - rho ** 2)) / rho)


def main():
    t00 = time.time()
    m0, cfg = build(f'w5_e10_{ARM}_s{seeds[0]}', dev)
    va = load_val(dev, cfg['xt_names'])
    ho = load_hold(dev, cfg['xt_names'])
    SPL = dict(val=(va, True), hold=(ho, False))
    for d_ in (va, ho):
        assert (d_.cand_off_np == np.concatenate([[0], np.cumsum(d_.M_np)[:-1]])).all(), 'non-contiguous candidate layout'
    del m0
    for seed in seeds:
        v = f'w5_e10_{ARM}_s{seed}'
        model, cfg = build(v, dev)
        blocks = setup_model(model)
        pv_s, ph_s = pred_paths(v)
        stored = dict(val=np.load(pv_s), hold=np.load(ph_s))
        res = dict(variant=v, group=group, blocks=[n for n, _ in blocks], params=sum(p.numel() for p in model.parameters()))
        nd = {}
        scores0 = {}
        for sp, (data, tf32) in SPL.items():
            CTX['split'] = sp
            CTX['feed_local'] = torch.arange(data.n, device=dev)
            CTX['acc'] = Acc()
            CTX['gen'] = torch.Generator(device=dev)
            # routing records (canonical order)
            for n, blk in blocks:
                if blk.moe is not None:
                    blk.moe._rec = dict(sel=torch.full((data.n_cand, blk.moe.k), -1, dtype=torch.int8, device=dev),
                                        gate=torch.zeros((data.n_cand, blk.moe.k), dtype=torch.float16, device=dev),
                                        s=torch.zeros((data.n_cand, blk.moe.n_exp), dtype=torch.float16, device=dev))
                if blk.attn.vo is not None:
                    vo = blk.attn.vo
                    nk = data.n_hist + data.n
                    vo._rec = [dict(sel=torch.full((nk, vo.h, vo.k), -1, dtype=torch.int8, device=dev),
                                    s=torch.zeros((nk, vo.h, vo.E), dtype=torch.float16, device=dev)),
                               dict(sel=torch.full((data.n_cand, vo.h, vo.k), -1, dtype=torch.int8, device=dev),
                                    s=torch.zeros((data.n_cand, vo.h, vo.E), dtype=torch.float16, device=dev))]
            hs = install_hooks(model, blocks)
            t0 = time.time()
            s0 = predict(model, data, pre=make_pre(data), post=post_instr, tf32=tf32)
            for h in hs:
                h.remove()
            dmax = float(np.abs(s0 - stored[sp]).max())
            print(f'[{v} {sp}] intact pass {time.time() - t0:.1f}s, max|pred - stored| {dmax:.2e}', flush=True)
            assert dmax == 0.0, 'instrumented intact pass does not reproduce the stored predictions'
            scores0[sp] = s0
            acc = CTX['acc']
            res[f'{sp}.blocks'] = summarize_acc(acc, blocks, model)
            res[f'{sp}.layers'] = summarize_layers(acc, cfg['n_dense'])
            # save routing records, set mean gates for the meanmix ablation
            for n, blk in blocks:
                if blk.moe is not None:
                    rec = blk.moe._rec
                    extra = dict(s=rec['s'].cpu().numpy()) if sp == 'val' else {}
                    np.savez_compressed(OUTD / f'route_{v}_{sp}_{n}.npz', sel=rec['sel'].cpu().numpy(), gate=rec['gate'].cpu().numpy(), **extra)
                    blk.moe._gbar[sp] = (acc[f'{n}.gsum'] / acc[f'{n}.n']).float()
                    blk.moe._rec = None
                if blk.attn.vo is not None:
                    vo = blk.attn.vo
                    extra = dict(sV=vo._rec[0]['s'].cpu().numpy(), sO=vo._rec[1]['s'].cpu().numpy()) if sp == 'val' else {}
                    np.savez_compressed(OUTD / f'route_{v}_{sp}_{n}.npz', selV=vo._rec[0]['sel'].cpu().numpy(), selO=vo._rec[1]['sel'].cpu().numpy(), **extra)
                    vo._gbar[sp] = torch.stack([(acc[f'{n}.{R}.gsum'] / acc[f'{n}.{R}.n']).float() for R in 'VO'])
                    vo._rec = None
            nd[f'{sp}.intact'] = split_ndcg(data, s0).astype(np.float32)
        # ablation passes
        abl = {}
        for tag, spec in ablation_list(blocks, model):
            for sp, (data, tf32) in SPL.items():
                CTX['split'] = sp
                CTX['gen'] = torch.Generator(device=dev).manual_seed(20261119)
                hs = ablate(model, blocks, spec)
                s1 = predict(model, data, tf32=tf32)
                for h in hs:
                    h.remove()
                reset_modes(blocks)
                n1 = split_ndcg(data, s1).astype(np.float32)
                nd[f'{sp}.{tag}'] = n1
                d = n1.astype(np.float64) - nd[f'{sp}.intact']
                rp, rr = wf_rel_perturb(data, scores0[sp], s1)
                abl[f'{sp}.{tag}'] = dict(d=float(d.mean()), se=se(d), rel_perturb=rp, resid=rr)
            print(f'  {tag:28s} val {abl["val." + tag]["d"]:+.5f}+-{abl["val." + tag]["se"]:.5f} (pert {abl["val." + tag]["rel_perturb"]:.3f} resid {abl["val." + tag]["resid"]:.3f}) | '
                  f'hold {abl["hold." + tag]["d"]:+.5f}+-{abl["hold." + tag]["se"]:.5f} (pert {abl["hold." + tag]["rel_perturb"]:.3f} resid {abl["hold." + tag]["resid"]:.3f})', flush=True)
        res['ablations'] = abl
        res['ndcg_intact'] = {sp: float(nd[f'{sp}.intact'].mean()) for sp in SPL}
        np.savez_compressed(OUTD / f'ndcg_{v}.npz', **nd)
        (OUTD / f'res_{v}.json').write_text(json.dumps(res, indent=1))
        print(f'[{v}] done, {time.time() - t00:.0f}s total', flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
