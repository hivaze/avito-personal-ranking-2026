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


# ---------------------------------------------------------------------------- blocks
@contextlib.contextmanager
def extra_init(offset=7919):
    """Build new (treatment-only) modules under a forked CPU RNG, so the baseline parameters get exactly the
    baseline init and an A/B differs only by the treatment."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(torch.initial_seed() + offset)
        yield


class Attn(nn.Module):
    def __init__(self, d, heads, drop, sim_bias=False, split_kv=False):
        super().__init__()
        self.h = heads
        self.q = nn.Linear(d, d)
        self.kv = nn.Linear(d, 2 * d)
        self.split_kv = split_kv
        if split_kv:  # C2 (Muon): separate k and v matrices (orthogonalised separately), same init as the fused kv
            with extra_init():
                self.k, self.v = nn.Linear(d, d), nn.Linear(d, d)
            with torch.no_grad():
                for j, lin in enumerate((self.k, self.v)):
                    lin.weight.copy_(self.kv.weight[j * d:(j + 1) * d])
                    lin.bias.copy_(self.kv.bias[j * d:(j + 1) * d])
            del self.kv
        self.o = nn.Linear(d, d)
        self.drop = drop
        if sim_bias == 1:  # per-head scale of an additive content-similarity bias (DIN-like target attention)
            self.bias_scale = nn.Parameter(torch.linspace(0.0, 8.0, heads))
        elif sim_bias == 2:  # rich bias: similarity scale + prior per (head, key event type), same-category, recency
            self.sim_a = nn.Parameter(torch.linspace(0.0, 8.0, heads)[:, None].repeat(1, N_ET + 1))
            self.sim_c = nn.Parameter(torch.zeros(heads, N_ET + 1))
            self.sim_e = nn.Parameter(torch.zeros(heads))
            self.sim_r = nn.Parameter(torch.zeros(heads, 257))

    def forward(self, x, ctx, kmask, bias=None):
        B, N, D = x.shape
        K = ctx.shape[1]
        h = self.h
        q = self.q(x).view(B, N, h, D // h).transpose(1, 2)
        if self.split_kv:
            k, v = self.k(ctx).view(B, K, h, D // h), self.v(ctx).view(B, K, h, D // h)
        else:
            k, v = self.kv(ctx).view(B, K, 2, h, D // h).unbind(2)
        k, v = k.transpose(1, 2), v.transpose(1, 2)
        if bias is not None and hasattr(self, "bias_scale"):
            am = bias[:, None] * self.bias_scale[None, :, None, None]
            am = am.masked_fill(~kmask[:, None, None, :], float("-inf")).to(q.dtype)
        elif bias is not None and hasattr(self, "sim_a"):
            S, kt, same = bias  # [B,N,K] cosine, [B,K] key type (0=user, 1..10 event type), [B,N,K] same category
            a = self.sim_a[:, kt].permute(1, 0, 2)[:, :, None, :]  # [B,h,1,K]
            c = self.sim_c[:, kt].permute(1, 0, 2)[:, :, None, :]
            am = S[:, None].float() * a + c + self.sim_r[None, :, None, :K] \
                + same[:, None].float() * self.sim_e[None, :, None, None]
            am = am.masked_fill(~kmask[:, None, None, :], float("-inf")).to(q.dtype)
        else:
            am = kmask[:, None, None, :]
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=am,
                                           dropout_p=self.drop if self.training else 0.0)
        return self.o(o.transpose(1, 2).reshape(B, N, D))


class Block(nn.Module):
    """Pre-LN transformer block; self-attention if ctx is None else cross-attention."""

    def __init__(self, d, heads, drop, ff=2, sim_bias=False, split_kv=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.lnc = nn.LayerNorm(d)
        self.attn = Attn(d, heads, drop, sim_bias, split_kv)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff * d), nn.GELU(), nn.Dropout(drop), nn.Linear(ff * d, d))
        self.dr = nn.Dropout(drop)

    def forward(self, x, kmask, ctx=None, bias=None):
        hx = self.ln1(x)
        c = hx if ctx is None else self.lnc(ctx)
        x = x + self.dr(self.attn(hx, c, kmask, bias))
        return x + self.dr(self.ff(self.ln2(x)))


class Ranker(nn.Module):
    def __init__(self, d=128, heads=4, hist_layers=1, blocks=2, drop=0.1, use_hist=True, use_set=True,
                 use_pos_tower=True, n_dense=N_DENSE, aux_pos=False, ff=2, sim_bias=False,
                 pos_mode="both", domain=False, intent=False, xt_names=None, id_buckets=0, id_dim=32,
                 id_drop=0.5, arch="tf", split_kv=False):
        super().__init__()
        self.cfg = dict(d=d, heads=heads, hist_layers=hist_layers, blocks=blocks, drop=drop, use_hist=use_hist,
                        use_set=use_set, use_pos_tower=use_pos_tower, n_dense=n_dense, aux_pos=aux_pos, ff=ff, sim_bias=sim_bias, pos_mode=pos_mode, domain=domain,
                        intent=intent, xt_names=xt_names, id_buckets=id_buckets, id_dim=id_dim, id_drop=id_drop, arch=arch)
        if split_kv:  # stored only when on, so baseline cfgs stay identical to the original code's
            self.cfg["split_kv"] = True
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
            self.hist_enc = nn.ModuleList([Block(d, heads, drop, ff, split_kv=split_kv) for _ in range(hist_layers)])
            self.cross = nn.ModuleList([Block(d, heads, drop, ff, sim_bias, split_kv) for _ in range(blocks)])
        else:
            self.u_cross = nn.ModuleList([Block(d, heads, drop, ff, split_kv=split_kv) for _ in range(blocks)])
        if use_set:
            self.sets = nn.ModuleList([Block(d, heads, drop, ff, split_kv=split_kv) for _ in range(blocks)])
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
        if self.sim_bias and self.use_hist:  # raw cosine(cand, user) and cosine(cand, history item)
            S = torch.einsum("bmd,bkd->bmk", ce, b["he"]) * b["hemb_ok"][:, None, :]
            sbias = torch.cat([torch.einsum("bmd,bd->bm", ce, un)[..., None], S], -1)
            if self.sim_bias == 2:
                kt = torch.cat([torch.zeros_like(b["het"][:, :1]), b["het"] * b["hmask"]], 1)
                same = (b["ccat"][:, :, None] == b["hcat"][:, None, :]) & b["hmask"][:, None, :]
                same = torch.cat([torch.zeros_like(same[:, :, :1]), same], -1)
                sbias = (sbias, kt, same)
        if self.arch == "din":
            c = self._din(c, ctx, kmask, sbias, u)
            crosses = []
        for i in range(len(crosses)):
            c = crosses[i](c, kmask, ctx, sbias)
            if self.use_set:
                c = self.sets[i](c, cmask)
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
