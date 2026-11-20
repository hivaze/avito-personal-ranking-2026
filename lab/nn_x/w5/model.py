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
def dense_features(b, n_dense=None, xt_rel=None, xt_drop=0.0, train=False, xt_idcols=None, xt_relonly=None):
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
        xt = b["xt"]
        if train and xt_drop > 0:  # ID-input dropout: blank all ID-based inputs of a fraction of candidates (cold)
            keep = (torch.rand(xt.shape[:2], device=xt.device) >= xt_drop).float()[..., None]
            if xt_idcols is None:
                xt = xt * keep
            else:  # only the ID-based columns (co-visitation / popularity), content-based inputs stay
                cm = torch.zeros(xt.shape[-1], device=xt.device)
                cm[xt_idcols] = 1.0
                xt = xt * (1 - cm + cm * keep)
        if xt_rel:  # within-feed pct rank and z-score of selected raw columns, computed after the blanking
            rel = []
            for j in xt_rel:
                rel += [_pctrank(xt[..., j], cmask, cnt), _zscore(xt[..., j], mf, cnt)]
            if xt_relonly:  # columns used only through their within-feed rank/z (raw scale differs across splits)
                keep = [i for i in range(xt.shape[-1]) if i not in set(xt_relonly)]
                xt = xt[..., keep]
            xt = torch.cat([xt, torch.stack(rel, -1)], -1)
        X = torch.cat([X, xt], -1)
    return torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


N_DENSE = 15 + 2 + 2 + 6 + N_ET + 2 + N_ET + 1 + 3 + 10


_EXTRA = [0]


@contextlib.contextmanager
def extra_init(offset=7919):
    """Build treatment-only modules under a forked CPU RNG so that every baseline parameter keeps exactly its
    baseline init (paired comparisons); each call gets a distinct seed (counter reset in Ranker.__init__)."""
    _EXTRA[0] += 1
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(torch.initial_seed() + offset + 1009 * _EXTRA[0])
        yield


class HistRetrieval(nn.Module):
    """nnx A4 (ported): TabR-style candidate-conditioned retrieval over the candidate's top-m history items by raw
    cosine. keys k = W_K e (128 -> dk); logit = -||k_c - k_j||^2 + a S_cj + beta[type_j], plus a learned sink with zero
    value; values v = LN(E_type + E_src + E_rec(log2 bucket of j) + T(k_c - k_j)); out = Z(sum_j p_j v_j), Z zero-init."""

    def __init__(self, d, m=16, dk=64, drop=0.1, prod=False, uniq=False, tcnt=False):
        super().__init__()
        self.m = m
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
        # nn_x/r6 options, created LAST (still inside the caller's extra_init fork) so the parameters above keep their init
        self.prod, self.uniq = prod, uniq
        if prod:  # r6 arm a4w: DIN activation-unit product term T3(k_c * k_j), zero-init (exact no-op at init)
            self.t3 = nn.Linear(dk, d, bias=False)
            nn.init.zeros_(self.t3.weight)
        if uniq:  # r6 arm a4u: repeat count of the retrieved item in the history (log2 bucket), zero-init
            self.e_cnt = nn.Embedding(9, d)
            nn.init.zeros_(self.e_cnt.weight)
        self.tcnt = tcnt
        if tcnt:  # a4u+: per-event-type occurrence counts of the retrieved ITEM over all its copies (dedup keeps only the
            # newest copy's type; 17.8% of distinct val items lose >= 1 older type, e.g. a contact 2/3 behind a newer view)
            self.e_tc = nn.Linear(N_ET + 1, d, bias=False)  # value term, zero-init
            nn.init.zeros_(self.e_tc.weight)
            self.b_tc = nn.Linear(N_ET + 1, 1, bias=False)  # logit prior term (like beta[type]), zero-init
            nn.init.zeros_(self.b_tc.weight)

    def forward(self, ce, he, hv, het, hes, S, cmiss, cnt=None, tc=None):
        B, M, _ = ce.shape
        K = he.shape[1]
        m = min(self.m, K)
        top, idx = S.float().masked_fill(~hv[:, None, :], -2.0).topk(m, -1)  # [B,M,m]
        valid = (top > -1.5) & ~cmiss[..., None]
        kc = self.wk(ce)
        kh = self.wk(he)
        flat = (idx + (torch.arange(B, device=ce.device) * K)[:, None, None]).reshape(-1)
        khg = kh.reshape(B * K, -1)[flat].view(B, M, m, -1)
        diff = kc[:, :, None, :] - khg
        tg = het.reshape(-1)[flat].view(B, M, m)
        sg = hes.reshape(-1)[flat].view(B, M, m)
        rb = torch.floor(torch.log2(1.0 + idx.float())).long().clamp_max(8)
        tcg = torch.log1p(tc.reshape(B * K, -1)[flat].view(B, M, m, -1)) if self.tcnt else None
        with torch.autocast("cuda", enabled=False):
            lg = -(diff.float() ** 2).sum(-1) + self.a * top + self.beta(tg).squeeze(-1)
            if self.tcnt:
                lg = lg + self.b_tc(tcg).squeeze(-1)
            lg = lg.masked_fill(~valid, float("-inf"))
            lg = torch.cat([self.sink.expand(B, M, 1), lg], -1)
            p = torch.softmax(lg, -1)
        v = self.e_type(tg) + self.e_src(sg) + self.e_rec(rb) + self.t2(self.dp(F.relu(self.t1(diff))))
        if self.prod:
            v = v + self.t3(kc[:, :, None, :] * khg)
        if self.uniq:  # cnt [B,K] occurrences of event j's item in the (capped) history, >= 1 on valid events
            cg = cnt.reshape(-1)[flat].view(B, M, m)
            v = v + self.e_cnt(torch.floor(torch.log2(cg.clamp_min(1).float())).long().clamp(0, 8))
        if self.tcnt:
            v = v + self.e_tc(tcg)
        v = self.v_ln(v)
        r = (p[..., 1:, None].to(v.dtype) * v).sum(2)
        return self.Z(r)


def history_dedup(hidx, hv, het=None):
    """r6 arm a4u. hidx [B,K] item-table row of each history event, hv [B,K] event valid with an embedding.
    Returns first [B,K] (valid and no NEWER event (smaller index; history is newest-first) has the same item) and
    cnt [B,K] (number of valid events with the same item, including itself; 0 on invalid events)."""
    K = hidx.shape[1]
    eq = (hidx[:, :, None] == hidx[:, None, :]) & hv[:, :, None] & hv[:, None, :]  # [B,K,K]
    older = torch.ones(K, K, dtype=torch.bool, device=hidx.device).tril(-1)  # [j,k] = k < j (k is newer than j)
    first = hv & ~(eq & older).any(-1)
    if het is None:
        return first, eq.sum(-1)
    with torch.autocast("cuda", enabled=False):  # tc [B,K,N_ET+1]: per-type count over all valid copies of event k's item
        tc = torch.bmm(eq.float(), F.one_hot(het.clamp(0, N_ET), N_ET + 1).float() * hv[..., None])
    return first, eq.sum(-1), tc


class CandRetrieval(nn.Module):
    """r6 arm cset (R4). TabR-style retrieval over each candidate's top-m most similar candidates of the SAME feed by raw
    cosine (self, padded and embedding-less candidates excluded). keys k = W_K e; logit = -||k_i - k_j||^2 + a S_ij plus
    a learned zero-value sink; values v_j = LN(P c_j + T(k_i - k_j)) with c_j the neighbour's current token (content,
    dense and collaborative inputs); out = Z(sum_j p_j v_j), Z zero-init (exact no-op at init). Uses no labels; every
    input is known at request time, and all gathers stay inside the feed's own row (no cross-feed information)."""

    def __init__(self, d, m=8, dk=64, drop=0.1):
        super().__init__()
        self.m = m
        self.wk = nn.Linear(128, dk, bias=False)
        self.a = nn.Parameter(torch.tensor(4.0))
        self.sink = nn.Parameter(torch.zeros(1))
        self.p = nn.Linear(d, d, bias=False)
        self.t1 = nn.Linear(dk, 128)
        self.t2 = nn.Linear(128, d, bias=False)
        self.dp = nn.Dropout(drop)
        self.v_ln = nn.LayerNorm(d)
        self.Z = nn.Linear(d, d)
        nn.init.zeros_(self.Z.weight)
        nn.init.zeros_(self.Z.bias)

    def forward(self, ce, c, cmask, cmiss):
        B, M, _ = ce.shape
        m = min(self.m, M - 1)
        if m < 1:
            return torch.zeros_like(c)
        ok = cmask & ~cmiss  # [B,M] usable as a neighbour
        with torch.autocast("cuda", enabled=False):
            S = torch.einsum("bmd,bnd->bmn", ce.float(), ce.float())
            eye = torch.eye(M, dtype=torch.bool, device=ce.device)[None]
            S = S.masked_fill(~ok[:, None, :] | eye, -2.0)
            top, idx = S.topk(m, -1)  # [B,M,m]
        valid = (top > -1.5) & ok[..., None]
        kc = self.wk(ce)
        flat = (idx + (torch.arange(B, device=ce.device) * M)[:, None, None]).reshape(-1)
        kg = kc.reshape(B * M, -1)[flat].view(B, M, m, -1)
        cg = c.reshape(B * M, -1)[flat].view(B, M, m, -1)
        diff = kc[:, :, None, :] - kg
        with torch.autocast("cuda", enabled=False):
            lg = -(diff.float() ** 2).sum(-1) + self.a * top
            lg = lg.masked_fill(~valid, float("-inf"))
            lg = torch.cat([self.sink.expand(B, M, 1), lg], -1)
            pr = torch.softmax(lg, -1)
        v = self.v_ln(self.p(cg) + self.t2(self.dp(F.relu(self.t1(diff)))))
        r = (pr[..., 1:, None].to(v.dtype) * v).sum(2)
        return self.Z(r)


# ---------------------------------------------------------------------------- blocks
class Attn(nn.Module):
    def __init__(self, d, heads, drop, sim_bias=False, gate=False, split_kv=False, attn_drop=None):
        super().__init__()
        self.h = heads
        self.attn_drop = attn_drop  # r6 arm adrop0: attention-probability dropout (None = same as drop)
        self.gate = gate
        if gate == "early":  # (v11_gate_e10 layout: gate built first, changes the RNG stream of the other inits)
            self.g = nn.Linear(d, d)
        self.q = nn.Linear(d, d)
        self.kv = nn.Linear(d, 2 * d)
        self.split_kv = split_kv
        if split_kv:  # Muon: separate k and v matrices (orthogonalised separately), same init as the fused kv
            with extra_init():
                self.k, self.v = nn.Linear(d, d), nn.Linear(d, d)
            with torch.no_grad():
                for j, lin in enumerate((self.k, self.v)):
                    lin.weight.copy_(self.kv.weight[j * d:(j + 1) * d])
                    lin.bias.copy_(self.kv.bias[j * d:(j + 1) * d])
            del self.kv
        self.o = nn.Linear(d, d)
        if gate and gate != "early":  # gated attention (nnx A1): output * sigmoid(W_g x + b_g), elementwise per head
            with extra_init():
                self.g = nn.Linear(d, d)
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
        pdrop = self.drop if self.attn_drop is None else self.attn_drop
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=am, dropout_p=pdrop if self.training else 0.0)
        if self.gate:
            o = o * torch.sigmoid(self.g(x)).view(B, N, h, D // h).transpose(1, 2)
        return self.o(o.transpose(1, 2).reshape(B, N, D))


class Block(nn.Module):
    """Pre-LN transformer block; self-attention if ctx is None else cross-attention."""

    def __init__(self, d, heads, drop, ff=2, sim_bias=False, gate=False, split_kv=False, attn_drop=None):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.lnc = nn.LayerNorm(d)
        self.attn = Attn(d, heads, drop, sim_bias, gate, split_kv, attn_drop)
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
                 id_drop=0.5, arch="tf", id_hist_only=False, xt_rel=None, xt_drop=0.0, xt_idcols=None,
                 xt_relonly=None, attn_gate=False, split_kv=False, retr_m=0,
                 attn_drop=None, retr_prod=False, retr_uniq=False, cretr_m=0, set_sim_bias=False, typed_cross=False,
                 retr_tcnt=False, hist_ext=0, hist_ext_src=None):
        super().__init__()
        self.cfg = dict(d=d, heads=heads, hist_layers=hist_layers, blocks=blocks, drop=drop, use_hist=use_hist,
                        use_set=use_set, use_pos_tower=use_pos_tower, n_dense=n_dense, aux_pos=aux_pos, ff=ff, sim_bias=sim_bias, pos_mode=pos_mode, domain=domain,
                        intent=intent, xt_names=xt_names, id_buckets=id_buckets, id_dim=id_dim, id_drop=id_drop, arch=arch,
                        id_hist_only=id_hist_only, xt_rel=xt_rel, xt_drop=xt_drop, xt_idcols=xt_idcols,
                        xt_relonly=xt_relonly, attn_gate=attn_gate, split_kv=split_kv, retr_m=retr_m)
        # nn_x/r6 options: stored in cfg ONLY when not at their default, so a flag-off r6 checkpoint has exactly the nn/
        # cfg (loadable by nn/model.py) and every flag-off code path below is the nn/ one
        r6 = dict(attn_drop=attn_drop, retr_prod=retr_prod, retr_uniq=retr_uniq, cretr_m=cretr_m,
                  set_sim_bias=set_sim_bias, typed_cross=typed_cross, retr_tcnt=retr_tcnt)
        r6_def = dict(attn_drop=None, retr_prod=False, retr_uniq=False, cretr_m=0, set_sim_bias=False, typed_cross=False,
                      retr_tcnt=False)
        self.cfg.update({k: v for k, v in r6.items() if v != r6_def[k]})
        # gap1117 lifelong (extension tokens from the same user's earlier snapshot): in cfg ONLY when on, so a flag-off
        # checkpoint keeps exactly the old cfg / state dict and every flag-off code path below is the old one
        self.hist_ext = int(hist_ext or 0)
        if self.hist_ext:
            assert use_hist and arch == "tf" and sim_bias != 2 and not id_buckets, \
                "--hist_ext: tf history encoder, sim_bias 0/1, no item-ID residual"
            self.cfg.update(hist_ext=self.hist_ext, hist_ext_src=hist_ext_src or {})
        assert not (retr_prod or retr_uniq) or (retr_m and use_hist), "--retr_prod/--retr_uniq need --retr_m"
        assert not retr_tcnt or retr_uniq, "--retr_tcnt needs --retr_uniq"
        assert not typed_cross or (use_hist and arch != "din" and blocks >= 1), "--typed_cross needs cross blocks"
        self.retr_uniq, self.typed_cross, self.set_sim_bias = retr_uniq, typed_cross, set_sim_bias
        _EXTRA[0] = 0
        self.xt_drop = xt_drop
        self.id_hist_only = id_hist_only  # item-ID embeddings only for history tokens (no candidate-ID shortcut)
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
        self.retr = None
        if retr_m and use_hist:  # nnx A4: TabR-style retrieval values (zero-init output: exact no-op at init)
            with extra_init():
                self.retr = HistRetrieval(d, retr_m, drop=drop, prod=retr_prod, uniq=retr_uniq, tcnt=retr_tcnt)
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
            self.hist_enc = nn.ModuleList([Block(d, heads, drop, ff, gate=attn_gate, split_kv=split_kv,
                                                 attn_drop=attn_drop) for _ in range(hist_layers)])
            self.cross = nn.ModuleList([Block(d, heads, drop, ff, sim_bias, attn_gate, split_kv, attn_drop=attn_drop)
                                        for _ in range(blocks)])
        else:
            self.u_cross = nn.ModuleList([Block(d, heads, drop, ff, attn_drop=attn_drop) for _ in range(blocks)])
        if use_set:  # r6 set_sim_bias: sim_bias=1 adds only a deterministic linspace parameter (no RNG draw)
            self.sets = nn.ModuleList([Block(d, heads, drop, ff, sim_bias=1 if set_sim_bias else False, gate=attn_gate,
                                             split_kv=split_kv, attn_drop=attn_drop) for _ in range(blocks)])
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
        self.cretr = None
        if cretr_m:  # r6 arm cset: built LAST under its own RNG fork (no baseline parameter or later RNG draw changes)
            with extra_init(offset=104729):
                self.cretr = CandRetrieval(d, cretr_m, drop=drop)
        if self.hist_ext:  # gap1117 lifelong: position codes of extension tokens (positions 256 .. 256 + hist_ext - 1), own RNG
            with extra_init(offset=130363):  # fork, built after everything else: no baseline parameter changes its init
                self.rec_x = nn.Embedding(self.hist_ext, d)

    @torch.no_grad()
    def init_hist_ext(self, mode="row255"):
        """gap1117 lifelong: start values of the extension position codes when a checkpoint trained without them is re-cooled.
        row255   : every extension slot starts as the oldest stored slot (rec row 255) = the convention of the refill replay's
                   untrained pass d1 (reports/SEG_1115/refill.md: d1 - a = +0.0009 +- 0.0006 pooled; the measured start point)
        shift128 : extension slot j starts as stored slot 256 - hist_ext + j (the deep rows reused in order)
        deep_mean: the mean of rows 192..255 rescaled to their mean norm (their shared 'old event' component)"""
        W, k = self.rec.weight, self.hist_ext
        if mode == "row255":
            self.rec_x.weight.copy_(W[255].expand(k, -1))
        elif mode == "shift128":
            self.rec_x.weight.copy_(W[256 - k:256])
        elif mode == "deep_mean":
            m = W[192:256].mean(0)
            self.rec_x.weight.copy_((m * W[192:256].norm(dim=1).mean() / m.norm().clamp_min(1e-6)).expand(k, -1))
        else:
            raise ValueError(mode)

    def _pos(self, K, device):
        """position codes of K history tokens: stored slots 0..255 from rec, extension slots from rec_x."""
        if K <= 256 or not self.hist_ext:
            return self.rec(torch.arange(K, device=device))
        return torch.cat([self.rec.weight, self.rec_x(torch.arange(K - 256, device=device))], 0)

    def _hist_view(self, b):
        """gap1117 lifelong: a shallow copy of the batch whose history keys span the stored block AND the extension block
        (gather() keys x*); used by the token path only (history encoder, cross-attention and its cosine bias, top-16
        retrieval). The dense inputs were computed from the stored keys alone (dense_features never sees x*)."""
        if "xe_raw" not in b:
            return b
        assert b["he"].shape[1] == 256, "extension tokens need the stored block at full width"
        xr = b["xe_raw"]
        bh = dict(b)
        bh["he"] = torch.cat([b["he"], xr / xr.norm(dim=-1, keepdim=True).clamp_min(1e-3)], 1)
        for k_h, k_x in (("hemb_ok", "xemb_ok"), ("hmask", "xmask"), ("het", "xet"), ("hes", "xes"), ("hcat", "xcat"),
                         ("hidx", "xidx")):
            bh[k_h] = torch.cat([b[k_h], b[k_x]], 1)
        return bh

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

    def dense(self, b):
        """Dense features with this model's xt options (ID-input dropout only in training mode)."""
        return dense_features(b, self.cfg["n_dense"], self.cfg.get("xt_rel"), self.xt_drop, self.training,
                              self.cfg.get("xt_idcols"), self.cfg.get("xt_relonly"))

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
            X = self.dense(b)
        X = ((X - self.f_mean) / self.f_std).clamp(-6, 6)
        cmask = b["cmask"]
        ce, un = b["ce"], b["un"]
        c = self.c_lin(torch.cat([ce, ce * un[:, None, :]], -1))
        c = c + self.cat(b["ccat"]) + self.c_mlp(X) + b["cmiss"][..., None].float() * self.c_miss
        if self.id_buckets and not self.id_hist_only:
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
        bh = self._hist_view(b) if self.hist_ext else b  # gap1117 lifelong: stored + extension tokens (== b when off)
        if self.use_hist:
            he = bh["he"]
            K = he.shape[1]
            h = self.h_lin(he) + self.et(bh["het"]) + self.es(bh["hes"]) + self.cat(bh["hcat"]) \
                + (self.rec(torch.arange(K, device=he.device)) if not self.hist_ext else self._pos(K, he.device))[None] \
                + (~bh["hemb_ok"])[..., None].float() * self.h_miss
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
            kmask = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=c.device), bh["hmask"]], 1)
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
            S = torch.einsum("bmd,bkd->bmk", ce, bh["he"]) * bh["hemb_ok"][:, None, :]
        if self.retr is not None and not self.retr_uniq:  # top-16 over stored AND extension events (recency bucket 8 from 255 on)
            c = c + self.retr(ce, bh["he"], bh["hemb_ok"], bh["het"], bh["hes"], S, b["cmiss"])
        elif self.retr is not None:  # r6 a4u: one slot per distinct item (newest occurrence), fp32 scores, repeat count
            tc = None
            if self.retr.tcnt:
                first, cnt, tc = history_dedup(bh["hidx"], bh["hemb_ok"], bh["het"])
            else:
                first, cnt = history_dedup(bh["hidx"], bh["hemb_ok"])
            with torch.autocast("cuda", enabled=False):
                S32 = torch.einsum("bmd,bkd->bmk", ce.float(), bh["he"].float()) * bh["hemb_ok"][:, None, :]
            c = c + self.retr(ce, bh["he"], first, bh["het"], bh["hes"], S32, b["cmiss"], cnt=cnt, tc=tc)
        if self.cretr is not None:
            c = c + self.cretr(ce, c, cmask, b["cmiss"])
        if self.sim_bias and self.use_hist:  # raw cosine(cand, user) and cosine(cand, history item)
            sbias = torch.cat([torch.einsum("bmd,bd->bm", ce, un)[..., None], S], -1)
            if self.sim_bias == 2:
                kt = torch.cat([torch.zeros_like(bh["het"][:, :1]), bh["het"] * bh["hmask"]], 1)
                same = (b["ccat"][:, :, None] == bh["hcat"][:, None, :]) & bh["hmask"][:, None, :]
                same = torch.cat([torch.zeros_like(same[:, :, :1]), same], -1)
                sbias = (sbias, kt, same)
        if self.arch == "din":
            c = self._din(c, ctx, kmask, sbias, u)
            crosses = []
        sset = None
        if self.set_sim_bias and self.use_set:  # r6 cset: cosine prior between candidates (self and missing -> 0)
            M_ = ce.shape[1]
            sset = torch.einsum("bmd,bnd->bmn", ce, ce).masked_fill(
                torch.eye(M_, dtype=torch.bool, device=ce.device)[None], 0.0)
        kmask_t = None
        if self.typed_cross:  # r6 typed: last cross block sees the user token + target actions (et 1 from source 1)
            kmask_t = kmask.clone()
            kmask_t[:, 1:] &= (bh["het"] == 1) & (bh["hes"] == 1)
        for i in range(len(crosses)):
            km = kmask_t if (kmask_t is not None and i == len(crosses) - 1) else kmask
            c = crosses[i](c, km, ctx, sbias)
            if self.use_set:
                c = self.sets[i](c, cmask) if sset is None else self.sets[i](c, cmask, bias=sset)
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
