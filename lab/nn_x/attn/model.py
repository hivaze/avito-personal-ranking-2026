"""Neural feed ranker: candidate tokens + history target attention + candidate set attention.

Per feed:
  user token      Linear(normalized user_emb)
  history tokens  Linear(item_emb) + emb(event_type) + emb(event_source) + emb(category) + emb(recency idx)
                  -> L_h pre-LN transformer layers over [user, history]
  candidate tokens Linear([item_emb, item_emb*user_emb]) + emb(category) + MLP(dense features)
  for each of L blocks:  cross-attention cand -> [user, history]  (target attention, multi-head)
                         self-attention over the feed's candidates (no positional encoding: set model)
  head            MLP(token) + linear "wide" part on dense features -> logit

Dense features are computed on the GPU inside the batch (no precomputed files needed):
  f1..f5 raw, within-feed z-score and mid-rank percentile of f1..f5, item-emb missing flag, item norm,
  cos/dot with user emb, cosine similarity of the candidate to history items (max over all / last 8 /
  last 32, mean, soft-max pooling, max per event type), category affinity with history (counts total and
  per event type, share, recency of first same-category event), feed category share, log M, log H,
  within-feed z/rank of key similarity features.

Position tower (train only): additive logit g(pos) + g'(M-1-pos), dropped at inference (PAL).
"""
import contextlib
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

N_ET = 10
N_ES = 32  # event_source max is 30 in train, 29 in test
N_CAT = 54
MAX_POS = 301


# ---------------------------------------------------------------------------- dense features
def _zscore(x, mask, cnt):
    xm = (x * mask).sum(1, keepdim=True) / cnt
    xv = (((x - xm) * mask) ** 2).sum(1, keepdim=True) / cnt
    return (x - xm) / (xv.sqrt() + 1e-4)


def _pctrank(x, mask, cnt):
    """Mid-rank percentile of x within each feed (ties get the average rank). x [B,M]."""
    xi = x[:, :, None]
    xj = x[:, None, :]
    mj = mask[:, None, :]
    less = ((xj < xi) & mj).sum(-1).float()
    eq = ((xj == xi) & mj).sum(-1).float() - 1.0
    return (less + 0.5 * eq) / (cnt - 1).clamp_min(1)


@torch.no_grad()
def dense_features(b, n_dense=None):
    """Returns [B,M,F] float32 features computed from a gathered batch dict.
    n_dense == N_DENSE + 2 appends the user-0 regime features (K5): is_user0, is_user0 * log(1+H)."""
    cmask = b["cmask"]
    mf = cmask.float()
    cnt = mf.sum(1, keepdim=True)
    ce_raw, he_raw, ue_raw = b["ce_raw"], b["he_raw"], b["ue_raw"]
    cn = ce_raw.norm(dim=-1)
    ce = ce_raw / cn.clamp_min(1e-3)[..., None]
    he = he_raw / he_raw.norm(dim=-1).clamp_min(1e-3)[..., None]
    un = ue_raw / ue_raw.norm(dim=-1, keepdim=True).clamp_min(1e-3)
    b["ce"], b["he"], b["un"] = ce, he, un
    feats = []
    cf = b["cf"]
    for k in range(5):
        x = cf[..., k]
        feats += [x, _zscore(x, mf, cnt), _pctrank(x, cmask, cnt)]
    cmiss = b["cmiss"].float()
    feats += [cmiss, torch.log1p(cn) * (1 - cmiss)]
    cos_u = torch.einsum("bmd,bd->bm", ce, un)
    dot_u = torch.einsum("bmd,bd->bm", ce_raw, ue_raw) / 1e4
    feats += [cos_u, dot_u]
    # candidate x history cosine matrix
    hv = b["hemb_ok"]  # [B,K] event exists and has embedding
    S = torch.einsum("bmd,bkd->bmk", ce, he)
    neg = torch.tensor(-1.5, device=S.device)
    Sv = torch.where(hv[:, None, :], S, neg)
    K = S.shape[-1]
    nh = hv.float().sum(-1, keepdim=True)
    smax = Sv.max(-1).values
    smax8 = Sv[:, :, :8].max(-1).values
    smax32 = Sv[:, :, :32].max(-1).values
    smean = (S * hv[:, None, :]).sum(-1) / nh.clamp_min(1)
    lse = torch.logsumexp(torch.where(hv[:, None, :], 5 * S, torch.tensor(-1e4, device=S.device)), -1) / 5
    lse = torch.where(nh > 0, lse - torch.log(nh.clamp_min(1)) / 5, neg)
    top3 = Sv.topk(min(3, K), dim=-1).values.mean(-1)
    feats += [smax, smax8, smax32, smean, lse, top3]
    het = b["het"]
    for t in range(1, N_ET + 1):
        mt = hv & (het == t)
        feats.append(torch.where(mt[:, None, :], S, neg).max(-1).values)
    # category affinity
    hm = b["hmask"]
    E = (b["ccat"][:, :, None] == b["hcat"][:, None, :]) & hm[:, None, :]
    ccnt = E.float().sum(-1)
    H = b["H"].float()[:, None]
    feats += [torch.log1p(ccnt), ccnt / H.clamp_min(1)]
    for t in range(1, N_ET + 1):
        feats.append(torch.log1p((E & (het == t)[:, None, :]).float().sum(-1)))
    ar = torch.arange(K, device=S.device)
    first = torch.where(E, ar, K).min(-1).values.float()
    feats.append(torch.where(first < K, 1.0 / (1.0 + first), torch.zeros_like(first)))
    fcs = ((b["ccat"][:, :, None] == b["ccat"][:, None, :]) & cmask[:, None, :]).float().sum(-1) / cnt
    feats += [fcs, torch.log(cnt).expand_as(fcs), torch.log1p(H).expand_as(fcs)]
    # within-feed relative versions of key similarity features
    for x in (cos_u, smax, lse, top3, torch.log1p(ccnt)):
        feats += [_zscore(x, mf, cnt), _pctrank(x, cmask, cnt)]
    if n_dense is not None and n_dense >= N_DENSE + 2:
        u0 = b["user0"].float()[:, None].expand_as(fcs)
        feats += [u0, u0 * torch.log1p(H).expand_as(fcs)]
    X = torch.stack(feats, -1)
    if n_dense is not None and n_dense > N_DENSE + 2:  # train-stage extra features (collaborative signals)
        X = torch.cat([X, b["xt"]], -1)
    return torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


N_DENSE = 15 + 2 + 2 + 6 + N_ET + 2 + N_ET + 1 + 3 + 10


# ---------------------------------------------------------------------------- nnx helpers
_EXTRA = [0]


@contextlib.contextmanager
def extra_init(offset=7919):
    """Build new (treatment) modules under a forked CPU RNG so every baseline parameter gets exactly the baseline
    init. Each call gets a distinct seed (counter reset in Ranker.__init__), so e.g. the 5 attention gates differ."""
    _EXTRA[0] += 1
    with torch.random.fork_rng(devices=[]):  # the model is built on CPU, then moved with .to(dev)
        torch.manual_seed(torch.initial_seed() + offset + 1009 * _EXTRA[0])
        yield


class MoEFF(nn.Module):
    """A5: fine-grained MoE FFN with a shared expert (DeepSeekMoE / DeepSeek-V3 / Ling 2.0 routing).
    y = SwiGLU_shared(x) + sum_{i in TopK(s + b)} (s_i / sum_TopK s) SwiGLU_i(x), s = sigmoid(W_r x) in fp32.
    b (buffer, no gradient) only affects selection; train.py calls update_bias() after each optimizer step
    (aux-loss-free balancing: b_i += gamma * sign(mean_load - load_i)). Dense dispatch: all experts in one matmul,
    multiplied by the sparse gate."""

    def __init__(self, d, n_exp=8, topk=2, drop=0.1, w_exp=64, w_shared=128, gamma=1e-3):
        super().__init__()
        self.n, self.k, self.we, self.gamma = n_exp, topk, w_exp, gamma
        self.sh13 = nn.Linear(d, 2 * w_shared, bias=False)
        self.sh2 = nn.Linear(w_shared, d, bias=False)
        self.e13 = nn.Linear(d, 2 * n_exp * w_exp, bias=False)
        self.e2 = nn.Linear(n_exp * w_exp, d, bias=False)
        self.router = nn.Linear(d, n_exp, bias=False)
        self.register_buffer("rbias", torch.zeros(n_exp))
        self.dp = nn.Dropout(drop)
        self.last_load = None
        self.record = False
        self.stats = {}

    def forward(self, x, qmask=None):
        a, g = self.sh13(x).chunk(2, -1)
        y = self.sh2(self.dp(F.silu(a) * g))
        with torch.autocast("cuda", enabled=False):
            s = torch.sigmoid(self.router(x.float()))
            sel = (s + self.rbias).topk(self.k, -1).indices
            w = s.gather(-1, sel)
            w = w / w.sum(-1, keepdim=True)
            gate = torch.zeros_like(s).scatter(-1, sel, w)
        if self.training or self.record:
            with torch.no_grad():
                oh = F.one_hot(sel, self.n).sum(-2).float()  # [..., n]
                if qmask is not None:
                    oh = oh * qmask[..., None].float()
                self.last_load = oh.reshape(-1, self.n).sum(0)
                if self.record:
                    self.stats = dict(sel=sel, s=s)
        a, g = self.e13(x).chunk(2, -1)
        hdn = (F.silu(a) * g).view(*x.shape[:-1], self.n, self.we) * gate.to(a.dtype)[..., None]
        return y + self.e2(self.dp(hdn.reshape(*x.shape[:-1], self.n * self.we)))

    @torch.no_grad()
    def update_bias(self):
        if self.last_load is None:
            return
        ld = self.last_load
        self.rbias += self.gamma * torch.sign(ld.mean() - ld)
        self.last_load = None


class HistRetrieval(nn.Module):
    """A4: TabR-style candidate-conditioned retrieval over the candidate's top-m history items by raw cosine.
    keys k = W_K e (normalized raw embeddings, 128 -> dk, no bias); logit = -||k_c - k_j||^2 + a S_cj + beta[type_j]
    plus a learned sink logit with zero value; values v_cj = LN(E_type[t_j] + E_src[s_j] + E_rec[log2 bucket of j]
    + T(k_c - k_j)), T = Linear(128->d, no bias) o Dropout o ReLU o Linear(dk->128); out = Z(sum_j p_cj v_cj),
    Z zero-init (exact no-op at init). The LN sits on the values (not on r as in the design draft) so that sink mass
    really shrinks the read instead of being normalised away."""

    def __init__(self, d, m=16, dk=64, drop=0.1, vals="all"):
        super().__init__()
        self.m = m
        self.vals = vals  # "all" | "meta" (no T(k_c - k_j) value term) | "diff" (only T(k_c - k_j)): dissection arms
        self.wk = nn.Linear(128, dk, bias=False)
        self.a = nn.Parameter(torch.tensor(4.0))
        self.beta = nn.Embedding(N_ET + 1, 1)
        nn.init.zeros_(self.beta.weight)
        self.sink = nn.Parameter(torch.zeros(1))
        self.e_type = nn.Embedding(N_ET + 1, d)
        self.e_src = nn.Embedding(N_ES, d)
        self.e_rec = nn.Embedding(9, d)
        self.t1 = nn.Linear(dk, 128)
        self.t2 = nn.Linear(128, d, bias=False)
        self.dp = nn.Dropout(drop)
        self.v_ln = nn.LayerNorm(d)
        self.Z = nn.Linear(d, d)
        nn.init.zeros_(self.Z.weight)
        nn.init.zeros_(self.Z.bias)
        self.record = False
        self.stats = {}

    def forward(self, ce, he, hv, het, hes, S, cmiss):
        B, M, _ = ce.shape
        K = he.shape[1]
        m = min(self.m, K)
        top, idx = S.float().masked_fill(~hv[:, None, :], -2.0).topk(m, -1)  # [B,M,m]
        valid = (top > -1.5) & ~cmiss[..., None]
        kc = self.wk(ce)  # [B,M,dk]
        kh = self.wk(he)  # [B,K,dk]
        flat = (idx + (torch.arange(B, device=ce.device) * K)[:, None, None]).reshape(-1)
        khg = kh.reshape(B * K, -1)[flat].view(B, M, m, -1)
        diff = kc[:, :, None, :] - khg  # [B,M,m,dk]
        tg = het.reshape(-1)[flat].view(B, M, m)
        sg = hes.reshape(-1)[flat].view(B, M, m)
        rb = torch.floor(torch.log2(1.0 + idx.float())).long().clamp_max(8)
        with torch.autocast("cuda", enabled=False):
            lg = -(diff.float() ** 2).sum(-1) + self.a * top + self.beta(tg).squeeze(-1)
            lg = lg.masked_fill(~valid, float("-inf"))
            lg = torch.cat([self.sink.expand(B, M, 1), lg], -1)
            p = torch.softmax(lg, -1)  # [B,M,m+1], the sink is always finite
        if self.vals == "meta":
            v = self.e_type(tg) + self.e_src(sg) + self.e_rec(rb)
        elif self.vals == "diff":
            v = self.t2(self.dp(F.relu(self.t1(diff))))
        else:
            v = self.e_type(tg) + self.e_src(sg) + self.e_rec(rb) + self.t2(self.dp(F.relu(self.t1(diff))))
        v = self.v_ln(v)
        r = (p[..., 1:, None].to(v.dtype) * v).sum(2)  # [B,M,d]
        out = self.Z(r)
        if self.record:
            self.stats = dict(sink=p[..., 0].detach(), out_norm=out.float().norm(dim=-1).detach(),
                              smax=top[..., 0].detach(), valid=valid.any(-1).detach())
        return out


# ---------------------------------------------------------------------------- blocks
class Attn(nn.Module):
    def __init__(self, d, heads, drop, sim_bias=False, gate="none", qk_norm=False, ssmax=False, n_ref=64):
        super().__init__()
        self.h = heads
        self.q = nn.Linear(d, d)
        self.kv = nn.Linear(d, 2 * d)
        self.o = nn.Linear(d, d)
        self.drop = drop
        if sim_bias == 1:  # per-head scale of an additive content-similarity bias (DIN-like target attention)
            self.bias_scale = nn.Parameter(torch.linspace(0.0, 8.0, heads))
        elif sim_bias == 2:  # rich bias: similarity scale + prior per (head, key event type), same-category, recency
            self.sim_a = nn.Parameter(torch.linspace(0.0, 8.0, heads)[:, None].repeat(1, N_ET + 1))
            self.sim_c = nn.Parameter(torch.zeros(heads, N_ET + 1))
            self.sim_e = nn.Parameter(torch.zeros(heads))
            self.sim_r = nn.Parameter(torch.zeros(heads, 257))
        self.gate = gate
        if gate == "elem":  # A1: Y = softmax(QK^T/sqrt(dh) + B) V * sigmoid(X W_g + b_g), elementwise per head
            with extra_init():
                self.g = nn.Linear(d, d)
        self.qk_norm, self.ssmax = qk_norm, ssmax
        if qk_norm:  # A3: per-head RMSNorm of q and k (shared across heads, as in Qwen3)
            self.qn = nn.RMSNorm(d // heads, eps=1e-6)
            self.kn = nn.RMSNorm(d // heads, eps=1e-6)
        if ssmax:  # A3: length-aware temperature, q <- q * s_h * log(n_valid_keys)
            self.ss = nn.Parameter(torch.full((heads,), 1.0 / math.log(n_ref)))
        self.record = False
        self.stats = {}

    def forward(self, x, ctx, kmask, bias=None, extra_bias=None):
        B, N, D = x.shape
        K = ctx.shape[1]
        h = self.h
        q = self.q(x).view(B, N, h, D // h).transpose(1, 2)
        k, v = self.kv(ctx).view(B, K, 2, h, D // h).unbind(2)
        k, v = k.transpose(1, 2), v.transpose(1, 2)
        if self.qk_norm:
            dh = q.shape[-1]  # cast the gain to the activation dtype so the fused rms_norm kernel is used
            q = F.rms_norm(q, (dh,), self.qn.weight.to(q.dtype), self.qn.eps)
            k = F.rms_norm(k, (dh,), self.kn.weight.to(k.dtype), self.kn.eps)
        if self.ssmax:
            n = kmask.sum(-1).clamp_min(2).float()
            q = q * (self.ss[None, :, None, None] * torch.log(n)[:, None, None, None]).to(q.dtype)
        if bias is not None and hasattr(self, "bias_scale"):
            am = bias[:, None] * self.bias_scale[None, :, None, None]
            if extra_bias is not None:  # A2: FoX forgetting bias [B,h,1,K], fp32
                am = am + extra_bias
            am = am.masked_fill(~kmask[:, None, None, :], float("-inf")).to(q.dtype)
        elif bias is not None and hasattr(self, "sim_a"):
            S, kt, same = bias  # [B,N,K] cosine, [B,K] key type (0=user, 1..10 event type), [B,N,K] same category
            a = self.sim_a[:, kt].permute(1, 0, 2)[:, :, None, :]  # [B,h,1,K]
            c = self.sim_c[:, kt].permute(1, 0, 2)[:, :, None, :]
            am = S[:, None].float() * a + c + self.sim_r[None, :, None, :K] \
                + same[:, None].float() * self.sim_e[None, :, None, None]
            am = am.masked_fill(~kmask[:, None, None, :], float("-inf")).to(q.dtype)
        elif extra_bias is not None:
            am = extra_bias.masked_fill(~kmask[:, None, None, :], float("-inf")).to(q.dtype)
        else:
            am = kmask[:, None, None, :]
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=am,
                                           dropout_p=self.drop if self.training else 0.0)
        gt = torch.sigmoid(self.g(x)).view(B, N, h, D // h).transpose(1, 2) if self.gate == "elem" else None
        if self.record:
            self._record(q, k, am, gt)
        if gt is not None:
            o = o * gt
        return self.o(o.transpose(1, 2).reshape(B, N, D))

    @torch.no_grad()
    def _record(self, q, k, am, gt):
        """Diagnostics: attention probabilities recomputed in fp32 (SDPA does not return them)."""
        sc = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(q.shape[-1])
        sc = sc.masked_fill(~am, float("-inf")) if am.dtype == torch.bool else sc + am.float()
        p = torch.softmax(sc, -1)  # [B,h,N,K]
        ent = -(p * torch.log(p.clamp_min(1e-12))).sum(-1)
        self.stats = dict(p0=p[..., 0], ent=ent, pmax=p.max(-1).values,
                          gate=None if gt is None else gt.float().mean(-1))  # gate [B,h,N]


class Block(nn.Module):
    """Pre-LN transformer block; self-attention if ctx is None else cross-attention."""

    def __init__(self, d, heads, drop, ff=2, sim_bias=False, gate="none", qk_norm=False, ssmax=False, n_ref=64,
                 moe=0, moe_topk=2):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.lnc = nn.LayerNorm(d)
        self.attn = Attn(d, heads, drop, sim_bias, gate, qk_norm, ssmax, n_ref)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff * d), nn.GELU(), nn.Dropout(drop), nn.Linear(ff * d, d))
        if moe:  # A5 (the dense FFN above is still built first so the baseline RNG stream is unchanged)
            with extra_init():
                self.ff = MoEFF(d, moe, moe_topk, drop)
        self.moe = bool(moe)
        self.dr = nn.Dropout(drop)

    def forward(self, x, kmask, ctx=None, bias=None, extra_bias=None, qmask=None):
        hx = self.ln1(x)
        c = hx if ctx is None else self.lnc(ctx)
        x = x + self.dr(self.attn(hx, c, kmask, bias, extra_bias))
        y = self.ln2(x)
        return x + self.dr(self.ff(y, qmask) if self.moe else self.ff(y))


class Ranker(nn.Module):
    def __init__(self, d=128, heads=4, hist_layers=1, blocks=2, drop=0.1, use_hist=True, use_set=True,
                 use_pos_tower=True, n_dense=N_DENSE, aux_pos=False, ff=2, sim_bias=False,
                 pos_mode="both", domain=False, intent=False, xt_names=None, id_buckets=0, id_dim=32,
                 id_drop=0.5, arch="tf", attn_gate="none", fox=False, qk_norm=False, ssmax=False, retr_m=0,
                 moe=0, moe_topk=2, retr_vals="all"):
        super().__init__()
        _EXTRA[0] = 0
        self.cfg = dict(d=d, heads=heads, hist_layers=hist_layers, blocks=blocks, drop=drop, use_hist=use_hist,
                        use_set=use_set, use_pos_tower=use_pos_tower, n_dense=n_dense, aux_pos=aux_pos, ff=ff, sim_bias=sim_bias, pos_mode=pos_mode, domain=domain,
                        intent=intent, xt_names=xt_names, id_buckets=id_buckets, id_dim=id_dim, id_drop=id_drop, arch=arch,
                        attn_gate=attn_gate, fox=fox, qk_norm=qk_norm, ssmax=ssmax, retr_m=retr_m, moe=moe,
                        moe_topk=moe_topk, retr_vals=retr_vals)
        ak = dict(gate=attn_gate, qk_norm=qk_norm, ssmax=ssmax)
        self.arch = arch  # "tf" (default), "gru" (bi-GRU history encoder), "din" (DIN MLP attention + DCNv2 head)
        self.id_buckets, self.id_drop = id_buckets, id_drop
        if id_buckets:  # N4: hashed item-ID residual (zero-init: cold items == dropped IDs)
            self.id_emb = nn.Embedding(id_buckets, id_dim)
            nn.init.zeros_(self.id_emb.weight)
            self.id_c = nn.Linear(id_dim, d)
            self.id_h = nn.Linear(id_dim, d)
        self.sim_bias = sim_bias
        self.pos_mode = pos_mode  # "both": g(pos)+g'(M-1-pos); "rev": only end-of-feed term; "pos": only g(pos)
        self.use_hist, self.use_set, self.use_pos_tower = use_hist, use_set, use_pos_tower
        self.register_buffer("f_mean", torch.zeros(n_dense))
        self.register_buffer("f_std", torch.ones(n_dense))
        self.cat = nn.Embedding(N_CAT, d)
        self.c_lin = nn.Linear(256, d)
        self.c_miss = nn.Parameter(torch.zeros(d))
        self.c_mlp = nn.Sequential(nn.Linear(n_dense, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.c_ln = nn.LayerNorm(d)
        self.extra = None
        self.domain = domain
        self.intent = intent and use_hist
        if self.intent:  # N3: category-intent head on the encoded user token, exposed to the candidates
            self.intent_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, N_CAT))
            self.intent_proj = nn.Linear(1, d)
            self.intent_scale = nn.Parameter(torch.zeros(1))
        if arch == "gru":
            self.gru = nn.GRU(d, d // 2, batch_first=True, bidirectional=True)
            self.gru_ln = nn.LayerNorm(d)
        if arch == "din":  # 2-head additive (MLP) target attention + SiLU pooling channel + DCNv2 cross network
            self.din_h, self.din_da = 2, 16
            self.din_q = nn.Linear(d, self.din_h * self.din_da)
            self.din_k = nn.Linear(d, self.din_h * self.din_da)
            self.din_w = nn.Parameter(torch.randn(self.din_h, self.din_da) * 0.1)
            self.din_s = nn.Parameter(torch.tensor([2.0, 6.0]))
            self.din_v = nn.Linear(d, d)
            self.din_ln = nn.LayerNorm(d)
            self.dcn = nn.ModuleList([nn.Linear(4 * d, 4 * d) for _ in range(2)])
            self.din_out = nn.Sequential(nn.LayerNorm(4 * d), nn.Linear(4 * d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.infer_domain = 1  # domain used when the batch has no "dom": 0 production, 1 random-like, 2 exploration
        if domain:  # K2: feed-level traffic-domain token added to candidate tokens and to the head input
            self.dom = nn.Embedding(3, d)
            nn.init.normal_(self.dom.weight, std=0.02)
        self.u_lin = nn.Linear(128, d)
        if use_hist:
            self.h_lin = nn.Linear(128, d)
            self.h_miss = nn.Parameter(torch.zeros(d))
            self.et = nn.Embedding(N_ET + 1, d)
            self.es = nn.Embedding(N_ES, d)
            self.rec = nn.Embedding(256, d)
            self.h_ln = nn.LayerNorm(d)
            self.hist_enc = nn.ModuleList([Block(d, heads, drop, ff, n_ref=192, **ak) for _ in range(hist_layers)])
            self.cross = nn.ModuleList([Block(d, heads, drop, ff, sim_bias, n_ref=192, moe=moe, moe_topk=moe_topk, **ak)
                                        for _ in range(blocks)])
        else:
            self.u_cross = nn.ModuleList([Block(d, heads, drop, ff) for _ in range(blocks)])
        if use_set:
            self.sets = nn.ModuleList([Block(d, heads, drop, ff, n_ref=64, moe=moe, moe_topk=moe_topk, **ak)
                                       for _ in range(blocks)])
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.aux_pos = aux_pos
        if aux_pos:  # auxiliary head: predict the production display position (distillation of prod ranker)
            self.aux_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.wide = nn.Linear(n_dense, 1)
        if use_pos_tower:
            self.pos_emb = nn.Embedding(MAX_POS, 1)
            self.rev_emb = nn.Embedding(MAX_POS, 1)
            nn.init.zeros_(self.pos_emb.weight)
            nn.init.zeros_(self.rev_emb.weight)
        self.fox = None
        if fox and use_hist:  # A2: FoX data-dependent forgetting bias on the history target attention
            with extra_init():
                self.fox = nn.ModuleList([nn.Linear(d, heads) for _ in range(blocks)])
            f0 = torch.tensor([0.999, 0.995, 0.98, 0.95]) if heads == 4 else torch.linspace(0.999, 0.95, heads)
            for lin in self.fox:  # exactly static (ALiBi-like spread of horizons ~1000/200/50/20 events) at init
                nn.init.zeros_(lin.weight)
                with torch.no_grad():
                    lin.bias.copy_(torch.logit(f0))
        self.retr = None
        if retr_m and use_hist:  # A4: TabR-style retrieval values
            with extra_init():
                self.retr = HistRetrieval(d, retr_m, drop=drop, vals=retr_vals)
        self.record = False
        self.stats = {}

    def add_extra(self, n_extra):
        """Attach a zero-initialized branch that injects extra per-candidate features into the candidate
        token (used when fine-tuning on val with precomputed handcrafted / transductive features)."""
        d = self.cfg["d"]
        self.extra = nn.Sequential(nn.Linear(n_extra, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        nn.init.zeros_(self.extra[2].weight)
        nn.init.zeros_(self.extra[2].bias)
        return self

    def _din(self, c, ctx, kmask, sbias, u):
        """DIN-style additive attention (per head: w . GELU(Wq c + Wk h) + scale * cos(c, h)), softmax pooling plus
        a non-normalised SiLU pooling channel, then a 2-layer DCNv2 cross network over [c, soft, silu, user]."""
        B, M, D = c.shape
        Kc = ctx.shape[1]
        nh, da = self.din_h, self.din_da
        q = self.din_q(c).view(B, M, 1, nh, da)
        k = self.din_k(ctx).view(B, 1, Kc, nh, da)
        sc = (F.gelu(q + k) * self.din_w).sum(-1)  # [B,M,K,nh]
        if sbias is not None:
            sc = sc + sbias[..., None] * self.din_s
        km = kmask[:, None, :, None]
        v = self.din_v(ctx).view(B, Kc, nh, D // nh)
        att = torch.softmax(sc.float().masked_fill(~km, -1e4), 2).to(v.dtype)
        soft = torch.einsum("bmkn,bknd->bmnd", att, v).reshape(B, M, D)
        cnt = kmask.sum(1).clamp_min(1)[:, None, None, None].to(v.dtype)
        sil = torch.einsum("bmkn,bknd->bmnd", (F.silu(sc) * km).to(v.dtype) / cnt, v).reshape(B, M, D)
        x0 = torch.cat([c, self.din_ln(soft), self.din_ln(sil), u.expand(B, M, D)], -1)
        x = x0
        for lin in self.dcn:
            x = x0 * lin(x) + x
        return c + self.din_out(x)

    def _id_vec(self, item):
        """Hashed item-ID embedding with ID-dropout (whole vector zeroed for a fraction id_drop in training)."""
        hsh = (item.long() * 2654435761 + 97) % self.id_buckets
        v = self.id_emb(hsh)
        if self.training and self.id_drop > 0:
            v = v * (torch.rand(v.shape[:-1], device=v.device) >= self.id_drop)[..., None]
        return v

    def set_norm(self, mean, std):
        self.f_mean.copy_(mean)
        self.f_std.copy_(std.clamp_min(1e-3))

    def forward(self, b, X=None, use_pos=False):
        if X is None:
            X = dense_features(b, self.cfg["n_dense"])
        X = ((X - self.f_mean) / self.f_std).clamp(-6, 6)
        cmask = b["cmask"]
        ce, un = b["ce"], b["un"]
        c = self.c_lin(torch.cat([ce, ce * un[:, None, :]], -1))
        c = c + self.cat(b["ccat"]) + self.c_mlp(X) + b["cmiss"][..., None].float() * self.c_miss
        if self.id_buckets:
            c = c + self.id_c(self._id_vec(b["citem"]))
        c = self.c_ln(c)
        dvec = None
        if self.domain:
            dom = b.get("dom")
            if dom is None:
                dom = F.one_hot(torch.full((c.shape[0],), self.infer_domain, device=c.device), 3).float()
            dvec = (dom.float() @ self.dom.weight)[:, None, :]
            c = c + dvec
        if self.extra is not None and "xe" in b:  # extra precomputed features (fine-tune stage only)
            c = c + self.extra(b["xe"])
        u = self.u_lin(un)[:, None, :]  # [B,1,d]
        B = c.shape[0]
        if self.use_hist:
            he = b["he"]
            K = he.shape[1]
            h = self.h_lin(he) + self.et(b["het"]) + self.es(b["hes"]) + self.cat(b["hcat"]) \
                + self.rec(torch.arange(K, device=he.device))[None] + (~b["hemb_ok"])[..., None].float() * self.h_miss
            if self.id_buckets:
                h = h + self.id_h(self._id_vec(b["hitem"]))
            h = self.h_ln(h)
            if self.arch == "gru":  # bidirectional GRU over the (newest-first) history instead of self-attention
                lens = b["H"].clamp_min(1).cpu()
                pk = nn.utils.rnn.pack_padded_sequence(h, lens, batch_first=True, enforce_sorted=False)
                with torch.autocast("cuda", enabled=False):
                    go, _ = self.gru(nn.utils.rnn.PackedSequence(pk.data.float(), pk.batch_sizes, pk.sorted_indices,
                                                                pk.unsorted_indices))
                go, _ = nn.utils.rnn.pad_packed_sequence(go, batch_first=True, total_length=K)
                h = self.gru_ln(h + go.to(h.dtype))
            ctx = torch.cat([u, h], 1)
            kmask = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=c.device), b["hmask"]], 1)
            if self.arch == "tf":
                for blk in self.hist_enc:
                    ctx = blk(ctx, kmask)
        else:
            ctx = u
            kmask = torch.ones(B, 1, dtype=torch.bool, device=c.device)
        lp_c = None
        if self.intent:
            with torch.autocast("cuda", enabled=False):
                logp_cat = torch.log_softmax(self.intent_head(ctx[:, 0].float()), -1)  # [B, N_CAT]
            self.intent_logp = logp_cat
            lp_c = logp_cat.gather(1, b["ccat"])  # log P(candidate category | user, history)
            c = c + self.intent_proj(lp_c[..., None])
        crosses = self.cross if self.use_hist else self.u_cross
        sbias = None
        S = None
        if (self.sim_bias or self.retr is not None) and self.use_hist:
            S = torch.einsum("bmd,bkd->bmk", ce, b["he"]) * b["hemb_ok"][:, None, :]
        if self.sim_bias and self.use_hist:  # raw cosine(cand, user) and cosine(cand, history item)
            sbias = torch.cat([torch.einsum("bmd,bd->bm", ce, un)[..., None], S], -1)
            if self.sim_bias == 2:
                kt = torch.cat([torch.zeros_like(b["het"][:, :1]), b["het"] * b["hmask"]], 1)
                same = (b["ccat"][:, :, None] == b["hcat"][:, None, :]) & b["hmask"][:, None, :]
                same = torch.cat([torch.zeros_like(same[:, :, :1]), same], -1)
                sbias = (sbias, kt, same)
        if self.arch == "din":
            c = self._din(c, ctx, kmask, sbias, u)
            crosses = []
        if self.retr is not None:
            r = self.retr(ce, b["he"], b["hemb_ok"], b["het"], b["hes"], S, b["cmiss"])
            if self.record:
                self.stats["retr_ratio"] = (r.float().norm(dim=-1) / c.float().norm(dim=-1).clamp_min(1e-6)).detach()
            c = c + r
        dfox = None
        if self.fox is not None:  # D_j = sum_{l<j} log sigmoid(w_h . x_l + b_h) over newer events (fp32 cumsum)
            with torch.autocast("cuda", enabled=False):
                hm = b["hmask"][..., None].float()
                dfox = []
                for lin in self.fox:
                    logf = F.logsigmoid(lin(ctx[:, 1:].float())) * hm  # [B,K,h]
                    Dx = logf.cumsum(1) - logf
                    dfox.append(torch.cat([torch.zeros_like(Dx[:, :1]), Dx], 1).permute(0, 2, 1)[:, :, None, :])
            if self.record:
                self.stats["fox_D"] = [x.detach() for x in dfox]
        for i in range(len(crosses)):
            c = crosses[i](c, kmask, ctx, sbias, extra_bias=None if dfox is None else dfox[i], qmask=cmask)
            if self.use_set:
                c = self.sets[i](c, cmask, qmask=cmask)
        # the head runs in fp32: bf16 logits are quantized (step 1/64 around |x|~2-4), which creates many
        # within-feed ties, and ties are scored pessimistically by the metric (cost ~0.001-0.002 NDCG)
        if dvec is not None:
            c = c + dvec
        with torch.autocast("cuda", enabled=False):
            logit = self.head(c.float()).squeeze(-1) + self.wide(X.float()).squeeze(-1)
            if lp_c is not None:
                logit = logit + self.intent_scale * lp_c
        if self.aux_pos and self.training:
            self.aux_out = self.aux_head(c).squeeze(-1).float()
        if use_pos and self.use_pos_tower and "pos" in b:
            pos = b["pos"].clamp(0, MAX_POS - 1)
            rev = (b["M"][:, None] - 1 - pos).clamp(0, MAX_POS - 1)
            if self.pos_mode in ("both", "pos"):
                logit = logit + self.pos_emb(pos).squeeze(-1)
            if self.pos_mode in ("both", "rev"):
                logit = logit + self.rev_emb(rev).squeeze(-1)
        return logit.float()


_IDCG = None


def listwise_loss(logit, y, mask, w=None, norm="batch", denom=None):
    """Softmax cross-entropy over the feed with target distribution y/sum(y); optional per-feed weights w [B].
    norm: "batch" = weighted mean over the feeds of the batch (batches of long feeds hold fewer feeds, so each long
          feed gets more gradient weight); "feed" = sum / denom (constant, e.g. max_feeds -> every feed weighs the
          same); "idcg" = like "feed" with extra per-feed weight K/IDCG(K) (K = #positives), i.e. target y/IDCG(K),
          the NDCG-consistent target of Ravikumar et al. 2011."""
    global _IDCG
    lg = logit.masked_fill(~mask, -1e4)
    logp = torch.log_softmax(lg, dim=1)
    ys = y.sum(1)
    valid = (ys > 0).float()
    l = -(y * logp).sum(1) / ys.clamp_min(1)
    w = torch.ones_like(l) if w is None else w
    w = w * valid
    if norm == "batch":
        return (l * w).sum() / w.sum().clamp_min(1e-6)
    if norm == "idcg":
        if _IDCG is None or _IDCG.device != l.device:
            _IDCG = torch.cat([torch.ones(1), torch.cumsum(1.0 / torch.log2(torch.arange(2, 400).float()), 0)]).to(l.device)
        k = ys.long().clamp(0, 398)
        w = w * ys / _IDCG[k].clamp_min(1e-6) * (k > 0)
    return (l * w).sum() / denom


def bce_loss(logit, y, mask):
    l = F.binary_cross_entropy_with_logits(logit, y, reduction="none")
    return (l * mask).sum() / mask.sum()
