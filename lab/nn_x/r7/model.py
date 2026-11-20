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

nn_x/r7 copy (reports/r7_build.md): every r7 arm (and every arm ported from nn_x/r6) sits behind a Ranker keyword whose
default reproduces nn/model.py exactly; non-default keywords are the only extra cfg keys. Treatment modules are built AFTER
all baseline modules, under forked RNGs (extra_init), so every baseline parameter keeps its baseline init.
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

    def __init__(self, d, m=16, dk=64, drop=0.1, prod=False, uniq=False):
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
        # nn_x/r6 options (ported verbatim), created LAST (still inside the caller's extra_init fork)
        self.prod, self.uniq = prod, uniq
        if prod:  # r6 arm a4w: DIN activation-unit product term T3(k_c * k_j), zero-init (exact no-op at init)
            self.t3 = nn.Linear(dk, d, bias=False)
            nn.init.zeros_(self.t3.weight)
        if uniq:  # r6 arm a4u: repeat count of the retrieved item in the history (log2 bucket), zero-init
            self.e_cnt = nn.Embedding(9, d)
            nn.init.zeros_(self.e_cnt.weight)

    def forward(self, ce, he, hv, het, hes, S, cmiss, cnt=None):
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
        with torch.autocast("cuda", enabled=False):
            lg = -(diff.float() ** 2).sum(-1) + self.a * top + self.beta(tg).squeeze(-1)
            lg = lg.masked_fill(~valid, float("-inf"))
            lg = torch.cat([self.sink.expand(B, M, 1), lg], -1)
            p = torch.softmax(lg, -1)
        v = self.e_type(tg) + self.e_src(sg) + self.e_rec(rb) + self.t2(self.dp(F.relu(self.t1(diff))))
        if self.prod:
            v = v + self.t3(kc[:, :, None, :] * khg)
        if self.uniq:  # cnt [B,K] occurrences of event j's item in the (capped) history, >= 1 on valid events
            cg = cnt.reshape(-1)[flat].view(B, M, m)
            v = v + self.e_cnt(torch.floor(torch.log2(cg.clamp_min(1).float())).long().clamp(0, 8))
        v = self.v_ln(v)
        r = (p[..., 1:, None].to(v.dtype) * v).sum(2)
        return self.Z(r)


def history_dedup(hidx, hv):
    """r6 arm a4u (ported). hidx [B,K] item-table row of each history event, hv [B,K] event valid with an embedding.
    Returns first [B,K] (valid and no NEWER event (smaller index; history is newest-first) has the same item) and
    cnt [B,K] (number of valid events with the same item, including itself; 0 on invalid events)."""
    K = hidx.shape[1]
    eq = (hidx[:, :, None] == hidx[:, None, :]) & hv[:, :, None] & hv[:, None, :]  # [B,K,K]
    older = torch.ones(K, K, dtype=torch.bool, device=hidx.device).tril(-1)  # [j,k] = k < j (k is newer than j)
    first = hv & ~(eq & older).any(-1)
    return first, eq.sum(-1)


class CandRetrieval(nn.Module):
    """r6 arm cset (R4, ported). TabR-style retrieval over each candidate's top-m most similar candidates of the SAME feed
    by raw cosine (self, padded and embedding-less candidates excluded). keys k = W_K e; logit = -||k_i - k_j||^2 + a S_ij
    plus a learned zero-value sink; values v_j = LN(P c_j + T(k_i - k_j)) with c_j the neighbour's current token;
    out = Z(sum_j p_j v_j), Z zero-init (exact no-op at init). All gathers stay inside the feed's own row."""

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


# ---------------------------------------------------------------------------- r7 modules (all flag-gated)
class Derf(nn.Module):
    """r7 derf: point-wise normalisation-free replacement of a LayerNorm (Derf, arXiv 2512.10938):
    y = weight * erf(alpha * x + s) + bias, alpha = 0.5, s = 0 (learnable scalars), weight = 1, bias = 0. No RNG."""

    def __init__(self, d, alpha0=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(alpha0)))
        self.s = nn.Parameter(torch.tensor(0.0))
        self.weight = nn.Parameter(torch.ones(d))
        self.bias = nn.Parameter(torch.zeros(d))

    def forward(self, x):
        return self.weight * torch.erf(self.alpha * x.float() + self.s) + self.bias


class SwiGLU(nn.Module):
    """r7 swiglu: o(dropout(silu(g x) * u x)); g and u are separate Linears (Muon orthogonalises each)."""

    def __init__(self, d, hidden, drop):
        super().__init__()
        self.g = nn.Linear(d, hidden)
        self.u = nn.Linear(d, hidden)
        self.o = nn.Linear(hidden, d)
        self.dp = nn.Dropout(drop)

    def forward(self, x):
        return self.o(self.dp(F.silu(self.g(x)) * self.u(x)))


class PBLD(nn.Module):
    """r7 pbld: RealMLP periodic embeddings of the standardised dense inputs X [B,M,F]:
    per_f = W2_f^T cos(2 pi w1_f x_f + b1_f) + b2_f (k frequencies -> n_out per feature), out = P(concat_f per_f) with P
    zero-init (added to c_mlp's first-layer pre-activation: exact baseline at init)."""

    def __init__(self, n_feat, k, out, sigma=0.1, n_out=3):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(n_feat, k) * sigma)
        self.b1 = nn.Parameter(torch.empty(n_feat, k).uniform_(-math.pi, math.pi))
        bound = 1.0 / math.sqrt(k)  # nn.Linear default (kaiming_uniform, a=sqrt(5)) for fan_in = k
        self.W2 = nn.Parameter(torch.empty(n_feat, k, n_out).uniform_(-bound, bound))
        self.b2 = nn.Parameter(torch.zeros(n_feat, n_out))
        self.P = nn.Linear(n_feat * n_out, out, bias=False)
        nn.init.zeros_(self.P.weight)

    def forward(self, X):
        z = torch.cos(2 * math.pi * self.w1 * X.float()[..., None] + self.b1)  # [B,M,F,k]
        per = torch.einsum("bmfk,fkj->bmfj", z, self.W2) + self.b2  # [B,M,F,n_out]
        return self.P(per.flatten(-2))


class HCUnit(nn.Module):
    """r7 hc: one dynamic Hyper-Connection (arXiv 2409.19606) around one residual sublayer, n streams H [B,M,n,d].
    static A_m = e_{k mod n}, A_r = I, B = 1 (== Pre-LN); dynamic part s * tanh(RMSNorm(H) W) with W = 0, s = 0.01.
    read: h0 = sum_j Am_j H_j;  write: H'_k = Bk_k * y + sum_j Ar_jk H_j. Element-wise fp32 arithmetic (no bf16/TF32
    matmul on the stream), so a fresh unit is bit-exactly the Pre-LN residual update."""

    def __init__(self, d, n, k):
        super().__init__()
        self.n = n
        am = torch.zeros(n)
        am[k % n] = 1.0
        self.A_m = nn.Parameter(am)
        self.A_r = nn.Parameter(torch.eye(n))
        self.B = nn.Parameter(torch.ones(n))
        self.W_m = nn.Parameter(torch.zeros(d, 1))
        self.W_r = nn.Parameter(torch.zeros(d, n))
        self.W_b = nn.Parameter(torch.zeros(d, 1))
        self.g = nn.Parameter(torch.ones(d))  # RMSNorm gain
        self.s_a = nn.Parameter(torch.tensor(0.01))
        self.s_b = nn.Parameter(torch.tensor(0.01))

    def coeffs(self, H):
        n = self.n
        with torch.autocast("cuda", enabled=False):
            Hf = H.float()
            Hn = Hf * torch.rsqrt(Hf.pow(2).mean(-1, keepdim=True) + 1e-6) * self.g
            t = torch.tanh(Hn @ torch.cat([self.W_m, self.W_r, self.W_b], 1))  # one matmul: [B,M,n,1+n+1]
            Am = self.A_m[:, None] + self.s_a * t[..., :1]  # [B,M,n,1]
            Ar = self.A_r + self.s_a * t[..., 1:n + 1]  # [B,M,n(j),n(k)]
            Bk = self.B[:, None] + self.s_b * t[..., n + 1:]  # [B,M,n,1]
        return Am, Ar, Bk

    @staticmethod
    def read(H, Am):
        return (Am * H).sum(-2)

    def write(self, H, Ar, Bk, y):
        # sum_j Ar[j,k] H_j as one broadcast product + reduction over j (element-wise fp32: exact at init)
        return Bk * y[..., None, :].float() + (Ar[..., None] * H[..., :, None, :]).sum(-3)


def hc_readout(H):
    """Mean over the n streams; pairwise tree for n = 2/4/8 so that n identical streams give exactly the stream."""
    n = H.shape[-2]
    if n in (1, 2, 4, 8):
        parts = [H[..., j, :] for j in range(n)]
        while len(parts) > 1:
            parts = [parts[i] + parts[i + 1] for i in range(0, len(parts), 2)]
        return parts[0] * (1.0 / n)
    return H.mean(-2)


class FTok(nn.Module):
    """r7 ftok: FT-Transformer-lite over feature-group tokens. Per group Linear(n_g, d) + group embedding, plus a CLS
    token, through one Block over the G+1 tokens of each candidate (per candidate: no cross-candidate mixing);
    out = Zf(CLS output), Zf zero-init (exact baseline at init)."""

    def __init__(self, d, heads, drop, ff, gate, split_kv, groups):
        super().__init__()
        self.groups = [list(g) for g in groups]
        self.inp = nn.ModuleList([nn.Linear(len(g), d) for g in self.groups])
        self.gemb = nn.Parameter(torch.randn(len(self.groups), d) * 0.02)
        self.cls = nn.Parameter(torch.randn(d) * 0.02)
        self.blk = Block(d, heads, drop, ff, gate=gate, split_kv=split_kv)
        self.Zf = nn.Linear(d, d)
        nn.init.zeros_(self.Zf.weight)
        nn.init.zeros_(self.Zf.bias)
        for i, g in enumerate(self.groups):
            self.register_buffer(f"gidx{i}", torch.tensor(g, dtype=torch.long), persistent=False)

    def forward(self, X):
        B, M, _ = X.shape
        d = self.cls.shape[0]
        T = torch.stack([lin(X[..., getattr(self, f"gidx{i}")]) for i, lin in enumerate(self.inp)], -2) + self.gemb
        T = torch.cat([self.cls.to(T.dtype).expand(B, M, 1, d), T], -2)
        G1 = T.shape[-2]
        Tf = T.reshape(B * M, G1, d)
        # no key mask (all G+1 tokens are real). Chunks of <= 32,768 rows: fused SDPA kernels put the batch on a CUDA grid
        # axis limited to 65,535 (B*M reaches 65,536 in predict() and under batch2s)
        CH = 32768
        out = torch.cat([self.blk(Tf[i:i + CH], None)[:, 0] for i in range(0, Tf.shape[0], CH)], 0).reshape(B, M, d)
        return self.Zf(out)


# FT-lite feature groups over the 104 standardised inputs of the cvnd_pre2s layout (reports/r7_research.md §5.12)
FTOK_GROUPS_PRE2S = [list(range(0, 15)),  # f1..f5 raw / z / pct
                     list(range(15, 35)) + list(range(51, 61)),  # content similarity (+ within-feed rel versions)
                     list(range(35, 48)),  # category affinity
                     list(range(48, 51)) + [61, 62],  # feed context + user0
                     [63, 64, 65, 94, 95],  # co-visitation (+ within-feed rank/z of seq_cv_invd20)
                     [66, 67, 68, 69, 70, 96, 97],  # leave-user-out popularity (+ rank/z of it_hist_all_luo)
                     list(range(71, 91)),  # neighbour context (ctx_nd / ctx_hn / ctx_hs)
                     [91, 92, 93] + list(range(98, 104))]  # pre2 encoder (+ rank/z of pre2_fe_s/new_s/hmax32)


# ---------------------------------------------------------------------------- blocks
class Attn(nn.Module):
    diff = False  # r7 diff2 (set by Ranker._r7_build)
    attn_drop = None  # r6 adrop0: attention-probability dropout (None = same as drop)

    def __init__(self, d, heads, drop, sim_bias=False, gate=False, split_kv=False, attn_drop=None):
        super().__init__()
        self.h = heads
        self.gate = gate
        self.attn_drop = attn_drop  # r6 adrop0: attention-probability dropout (None = same as drop)
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
            am = None if kmask is None else kmask[:, None, None, :]  # None: no padding (r7 ftok tokens) -> fused SDPA
        pdrop = self.drop if self.attn_drop is None else self.attn_drop
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=am, dropout_p=pdrop if self.training else 0.0)
        if self.diff:  # r7 diff2 (DIFF-V2): second query map, shared k/v, own sim-bias scale; o = o1 - sigmoid(lam x) o2
            q2 = self.q2(x).view(B, N, h, D // h).transpose(1, 2)
            if bias is not None and hasattr(self, "bias_scale"):
                am2 = bias[:, None] * self.bias_scale2[None, :, None, None]
                am2 = am2.masked_fill(~kmask[:, None, None, :], float("-inf")).to(q.dtype)
            else:
                am2 = am
            o2 = F.scaled_dot_product_attention(q2, k, v, attn_mask=am2, dropout_p=pdrop if self.training else 0.0)
            lam = torch.sigmoid(self.lam(x)).view(B, N, h, 1).transpose(1, 2)  # [B,h,N,1]
            if not self.training:
                self.lam_mean = lam.detach().float().mean((0, 2, 3))  # diagnostics: mean lambda per head (last batch)
            o = o - lam.to(o.dtype) * o2
        if self.gate:
            o = o * torch.sigmoid(self.g(x)).view(B, N, h, D // h).transpose(1, 2)
        return self.o(o.transpose(1, 2).reshape(B, N, D))


class Block(nn.Module):
    """Pre-LN transformer block; self-attention if ctx is None else cross-attention.
    r7 options (class defaults = nn/ behaviour; set by Ranker._r7_build): periln output norms ln_ao / ln_fo, postln
    (Post-LN + DeepNorm residual scale alpha). attn_sub / ff_sub expose the two residual branches (used by hc)."""
    # plain (non-Module) class defaults only: a Module-valued class attribute would shadow a registered submodule of the
    # same name (nn.Module.__getattr__ is a fallback), so the Peri-LN norms are gated by the bool `periln`
    general = False  # True when periln / postln is on (the flag-off path below is the nn/ code verbatim)
    periln = False
    postln_alpha = None

    def __init__(self, d, heads, drop, ff=2, sim_bias=False, gate=False, split_kv=False, attn_drop=None):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.lnc = nn.LayerNorm(d)
        self.attn = Attn(d, heads, drop, sim_bias, gate, split_kv, attn_drop)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff * d), nn.GELU(), nn.Dropout(drop), nn.Linear(ff * d, d))
        self.dr = nn.Dropout(drop)

    def forward(self, x, kmask, ctx=None, bias=None):
        if not self.general:  # nn/ path (verbatim)
            hx = self.ln1(x)
            c = hx if ctx is None else self.lnc(ctx)
            x = x + self.dr(self.attn(hx, c, kmask, bias))
            return x + self.dr(self.ff(self.ln2(x)))
        if self.postln_alpha is not None:  # r7 postln: x = LN(alpha x + F(x)) after each sublayer (DeepNorm)
            c = x if ctx is None else self.lnc(ctx)
            x = self.ln1(self.postln_alpha * x + self.dr(self.attn(x, c, kmask, bias)))
            return self.ln2(self.postln_alpha * x + self.dr(self.ff(x)))
        x = x + self.attn_sub(x, kmask, ctx, bias)
        return x + self.ff_sub(x)

    def attn_sub(self, x, kmask, ctx=None, bias=None):
        hx = self.ln1(x)
        c = hx if ctx is None else self.lnc(ctx)
        a = self.attn(hx, c, kmask, bias)
        if self.periln:  # r7 periln: output norm on the attention branch
            a = self.ln_ao(a)
        return self.dr(a)

    def ff_sub(self, x):
        f = self.ff(self.ln2(x))
        if self.periln:  # r7 periln: output norm on the FFN branch
            f = self.ln_fo(f)
        return self.dr(f)


class Ranker(nn.Module):
    def __init__(self, d=128, heads=4, hist_layers=1, blocks=2, drop=0.1, use_hist=True, use_set=True,
                 use_pos_tower=True, n_dense=N_DENSE, aux_pos=False, ff=2, sim_bias=False,
                 pos_mode="both", domain=False, intent=False, xt_names=None, id_buckets=0, id_dim=32,
                 id_drop=0.5, arch="tf", id_hist_only=False, xt_rel=None, xt_drop=0.0, xt_idcols=None,
                 xt_relonly=None, attn_gate=False, split_kv=False, retr_m=0,
                 attn_drop=None, retr_prod=False, retr_uniq=False, cretr_m=0, set_sim_bias=False, typed_cross=False,
                 res_drop=None, pbld=0, hc=0, dwa=False, diff_attn="", ffn="gelu", norm="ln", hist_extra=0,
                 periln=False, ftok=0, ftok_groups=None, postln=False):
        super().__init__()
        self.cfg = dict(d=d, heads=heads, hist_layers=hist_layers, blocks=blocks, drop=drop, use_hist=use_hist,
                        use_set=use_set, use_pos_tower=use_pos_tower, n_dense=n_dense, aux_pos=aux_pos, ff=ff, sim_bias=sim_bias, pos_mode=pos_mode, domain=domain,
                        intent=intent, xt_names=xt_names, id_buckets=id_buckets, id_dim=id_dim, id_drop=id_drop, arch=arch,
                        id_hist_only=id_hist_only, xt_rel=xt_rel, xt_drop=xt_drop, xt_idcols=xt_idcols,
                        xt_relonly=xt_relonly, attn_gate=attn_gate, split_kv=split_kv, retr_m=retr_m)
        # nn_x/r6 + nn_x/r7 options: stored in cfg ONLY when not at their default, so a flag-off checkpoint has exactly
        # the nn/ cfg (loadable by nn/model.py) and every flag-off code path below is the nn/ one
        if ftok and ftok_groups is None:
            assert ftok == len(FTOK_GROUPS_PRE2S), "--ftok: only the 8-group cvnd_pre2s layout is defined"
            ftok_groups = FTOK_GROUPS_PRE2S
        opt = dict(attn_drop=attn_drop, retr_prod=retr_prod, retr_uniq=retr_uniq, cretr_m=cretr_m,
                   set_sim_bias=set_sim_bias, typed_cross=typed_cross, res_drop=res_drop, pbld=pbld, hc=hc, dwa=dwa,
                   diff_attn=diff_attn, ffn=ffn, norm=norm, hist_extra=hist_extra, periln=periln, ftok=ftok,
                   ftok_groups=ftok_groups, postln=postln)
        opt_def = dict(attn_drop=None, retr_prod=False, retr_uniq=False, cretr_m=0, set_sim_bias=False, typed_cross=False,
                       res_drop=None, pbld=0, hc=0, dwa=False, diff_attn="", ffn="gelu", norm="ln", hist_extra=0,
                       periln=False, ftok=0, ftok_groups=None, postln=False)
        self.cfg.update({k: v for k, v in opt.items() if v != opt_def[k]})
        assert not (retr_prod or retr_uniq) or (retr_m and use_hist), "--retr_prod/--retr_uniq need --retr_m"
        assert not typed_cross or (use_hist and arch != "din" and blocks >= 1), "--typed_cross needs cross blocks"
        tf_stream = use_hist and use_set and arch == "tf"
        assert not (hc or dwa) or tf_stream, "--hc/--dwa need the tf candidate stream (history + set blocks)"
        assert sum(bool(x) for x in (hc, dwa, postln)) <= 1, "--hc, --dwa and --postln are mutually exclusive"
        assert not postln or (ffn == "gelu" and norm == "ln" and not periln), "--postln: LayerNorm + GELU blocks only"
        assert not diff_attn or (diff_attn == "cross" and use_hist and arch != "din"), "--diff_attn cross needs cross blocks"
        assert not hist_extra or (use_hist and arch == "tf"), "--hist_extra needs the tf history encoder"
        assert not ftok or n_dense == 104, "--ftok: groups are defined for the 104-input cvnd_pre2s layout"
        assert not pbld or n_dense > 0
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
                self.retr = HistRetrieval(d, retr_m, drop=drop, prod=retr_prod, uniq=retr_uniq)
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
        self._r7_build(d, heads, drop, ff, attn_gate, split_kv, n_dense, res_drop, pbld, hc, dwa, diff_attn, ffn, norm,
                       hist_extra, periln, ftok_groups if ftok else None, postln, blocks)

    # ------------------------------------------------------------------ r7 arms (all after the baseline modules)
    def _r7_build(self, d, heads, drop, ff, attn_gate, split_kv, n_dense, res_drop, pbld, hc, dwa, diff_attn, ffn, norm,
                  hist_extra, periln, ftok_groups, postln, blocks):
        """Builds every r7 treatment AFTER all baseline modules. Modules that draw random numbers are built under
        extra_init (forked CPU RNG), so every baseline parameter keeps its nn/ init and the global RNG stream after the
        constructor is unchanged. Deterministic ones (HC, DWA, Derf, Peri-LN, DeepNorm scaling) draw nothing."""
        self.pbld = self.hc = self.dwa = self.ftok = None
        new_hist = []
        if hist_extra:  # hl2: extra history layer(s) after the baseline one; outputs zeroed at the end (exact no-op)
            for _ in range(hist_extra):
                with extra_init():
                    blk = Block(d, heads, drop, ff, gate=attn_gate, split_kv=split_kv, attn_drop=self.cfg.get("attn_drop"))
                self.hist_enc.append(blk)
                new_hist.append(blk)
        if ftok_groups:
            assert sorted(i for g in ftok_groups for i in g) == list(range(n_dense)), "ftok groups must partition X"
            with extra_init():
                self.ftok = FTok(d, heads, drop, ff, attn_gate, split_kv, ftok_groups)
        if pbld:
            with extra_init():
                self.pbld = PBLD(n_dense, pbld, self.c_mlp[0].out_features)
        if hc:
            self.hc = nn.ModuleList([HCUnit(d, hc, k) for k in range(4 * blocks)])
        if dwa:
            ps = []
            for j in range(2 * blocks):
                v = torch.zeros(j + 2)
                v[-1] = 1.0
                ps.append(nn.Parameter(v))
            self.dwa = nn.ParameterList(ps)
        all_blocks = [m for m in self.modules() if isinstance(m, Block)]
        if ffn == "swiglu":  # matched parameters: 3 d h + 2 h + d == 2 ff d^2 + ff d + d
            hid = int(round((2 * ff * d * d + ff * d) / (3 * d + 2)))
            for blk in all_blocks:
                with extra_init():
                    blk.ff = SwiGLU(d, hid, drop)
        if diff_attn == "cross":
            for blk in self.cross:
                a = blk.attn
                with extra_init():
                    a.q2 = nn.Linear(d, d)
                    a.lam = nn.Linear(d, a.h)
                if hasattr(a, "bias_scale"):
                    a.bias_scale2 = nn.Parameter(torch.linspace(0.0, 8.0, a.h))
                a.diff = True
        if norm == "derf":  # block norms (lnc only where it is used: cross blocks) and the head norms
            cross_ids = {id(b) for b in getattr(self, "cross", [])}
            for blk in all_blocks:
                blk.ln1, blk.ln2 = Derf(d), Derf(d)
                if id(blk) in cross_ids:
                    blk.lnc = Derf(d)
            self.head[0] = Derf(d)
            if self.aux_pos:
                self.aux_head[0] = Derf(d)
        if periln:
            for blk in all_blocks:
                blk.ln_ao, blk.ln_fo = nn.LayerNorm(d), nn.LayerNorm(d)
                blk.periln = blk.general = True
        if postln:  # DeepNorm (arXiv 2203.00555, encoder): alpha = (2N)^(1/4), beta = (8N)^(-1/4) on v, o, ff[0], ff[3]
            groups = [(list(self.hist_enc), len(self.hist_enc))] if self.use_hist and self.arch == "tf" else []
            cand = list(getattr(self, "cross", getattr(self, "u_cross", []))) + list(getattr(self, "sets", []))
            groups.append((cand, len(cand)))
            for blks, N in groups:
                alpha, beta = (2 * N) ** 0.25, (8 * N) ** -0.25
                for blk in blks:
                    blk.postln_alpha, blk.general = alpha, True
                    with torch.no_grad():
                        vw = blk.attn.v.weight if blk.attn.split_kv else blk.attn.kv.weight[d:]
                        for w in (vw, blk.attn.o.weight, blk.ff[0].weight, blk.ff[3].weight):
                            w.mul_(beta)
        for blk in new_hist:  # hl2: zero the residual-branch outputs (after any ffn swap) -> exact no-op at init
            with torch.no_grad():
                for lin in (blk.attn.o, blk.ff[3] if isinstance(blk.ff, nn.Sequential) else blk.ff.o):
                    lin.weight.zero_()
                    lin.bias.zero_()
        if res_drop is not None:  # resdrop0: residual, FFN-hidden and retrieval-T dropout (attention probs unchanged)
            for m in self.modules():
                if isinstance(m, Block):
                    m.dr.p = res_drop
                    (m.ff[2] if isinstance(m.ff, nn.Sequential) else m.ff.dp).p = res_drop
                elif isinstance(m, (HistRetrieval, CandRetrieval)):
                    m.dp.p = res_drop

    def input_proj_names(self):
        """r7 muon_in: Linear weights that read fixed, non-learned input vectors (dense features / content embeddings)."""
        names = ["c_mlp.0.weight", "c_lin.weight", "u_lin.weight"]
        if self.use_hist:
            names.append("h_lin.weight")
        if self.retr is not None:
            names.append("retr.wk.weight")
        if self.cretr is not None:
            names.append("cretr.wk.weight")
        if self.pbld is not None:
            names.append("pbld.P.weight")
        if self.ftok is not None:
            names += [f"ftok.inp.{i}.weight" for i in range(len(self.ftok.inp))]
        return names

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
        if self.pbld is None:
            c = c + self.cat(b["ccat"]) + self.c_mlp(X) + b["cmiss"][..., None].float() * self.c_miss
        else:  # r7 pbld: periodic embeddings join c_mlp's first-layer pre-activation (zero-init P: exact at init)
            cm = self.c_mlp[2](self.c_mlp[1](self.c_mlp[0](X) + self.pbld(X)))
            c = c + self.cat(b["ccat"]) + cm + b["cmiss"][..., None].float() * self.c_miss
        if self.ftok is not None:  # r7 ftok: feature-group token transformer, zero-init output
            c = c + self.ftok(X)
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
        if self.retr is not None and not self.retr_uniq:
            c = c + self.retr(ce, b["he"], b["hemb_ok"], b["het"], b["hes"], S, b["cmiss"])
        elif self.retr is not None:  # r6 a4u: one slot per distinct item (newest occurrence), fp32 scores, repeat count
            first, cnt = history_dedup(b["hidx"], b["hemb_ok"])
            with torch.autocast("cuda", enabled=False):
                S32 = torch.einsum("bmd,bkd->bmk", ce.float(), b["he"].float()) * b["hemb_ok"][:, None, :]
            c = c + self.retr(ce, b["he"], first, b["het"], b["hes"], S32, b["cmiss"], cnt=cnt)
        if self.cretr is not None:
            c = c + self.cretr(ce, c, cmask, b["cmiss"])
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
        sset = None
        if self.set_sim_bias and self.use_set:  # r6 cset: cosine prior between candidates (self and missing -> 0)
            M_ = ce.shape[1]
            sset = torch.einsum("bmd,bnd->bmn", ce, ce).masked_fill(
                torch.eye(M_, dtype=torch.bool, device=ce.device)[None], 0.0)
        kmask_t = None
        if self.typed_cross:  # r6 typed: last cross block sees the user token + target actions (et 1 from source 1)
            kmask_t = kmask.clone()
            kmask_t[:, 1:] &= (b["het"] == 1) & (b["hes"] == 1)
        if self.hc is not None:  # r7 hc: n residual streams over the 4*blocks candidate sublayers
            n = self.hc[0].n
            H = c.float()[..., None, :].expand(*c.shape[:-1], n, c.shape[-1])
            k = 0
            for i in range(len(crosses)):
                km = kmask_t if (kmask_t is not None and i == len(crosses) - 1) else kmask
                subs = [lambda h, i=i, km=km: crosses[i].attn_sub(h, km, ctx, sbias), crosses[i].ff_sub,
                        lambda h, i=i: self.sets[i].attn_sub(h, cmask, None, sset), self.sets[i].ff_sub]
                for sub in subs:
                    unit = self.hc[k]
                    Am, Ar, Bk = unit.coeffs(H)
                    H = unit.write(H, Ar, Bk, sub(unit.read(H, Am)))
                    k += 1
            c = hc_readout(H)
        elif self.dwa is not None:  # r7 dwa: DenseFormer depth-weighted average after every block
            ys = [c]
            for i in range(len(crosses)):
                km = kmask_t if (kmask_t is not None and i == len(crosses) - 1) else kmask
                for j, f in enumerate((lambda x: crosses[i](x, km, ctx, sbias),
                                       lambda x: self.sets[i](x, cmask) if sset is None else self.sets[i](x, cmask, bias=sset))):
                    ys.append(f(c))
                    w = self.dwa[2 * i + j]
                    acc = w[0] * ys[0]
                    for t in range(1, len(ys)):
                        acc = acc + w[t] * ys[t]
                    c = acc
        else:
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


@torch.no_grad()
def r7_diagnostics(model):
    """Scale quantities of reports/r7_research.md §2.4 (top sim-bias scale per cross block, c_ln gain, retr.a, cross0 W_O
    RMS / spectral norm, c_mlp.0 spectral norm) plus every arm's own scalars. Information only (SCALE-SUSPECT readout)."""
    out = {}
    sd = lambda w: float(torch.linalg.matrix_norm(w.float(), ord=2))  # noqa: E731
    for i, blk in enumerate(getattr(model, "cross", [])):
        if hasattr(blk.attn, "bias_scale"):
            out[f"cross{i}_bias_scale"] = [round(float(x), 3) for x in blk.attn.bias_scale]
        if hasattr(blk.attn, "bias_scale2"):
            out[f"cross{i}_bias_scale2"] = [round(float(x), 3) for x in blk.attn.bias_scale2]
        if getattr(blk.attn, "lam_mean", None) is not None:
            out[f"cross{i}_lambda_mean"] = [round(float(x), 4) for x in blk.attn.lam_mean]
    out["c_ln_gain"] = float(model.c_ln.weight.abs().mean())
    if model.retr is not None:
        out["retr_a"] = float(model.retr.a)
    if hasattr(model, "cross"):
        w = model.cross[0].attn.o.weight
        out["cross0_WO_rms"] = float(w.float().pow(2).mean().sqrt())
        out["cross0_WO_spec"] = sd(w)
    out["c_mlp0_spec"] = sd(model.c_mlp[0].weight)
    for n in ("c_lin", "u_lin", "h_lin"):
        if hasattr(model, n):
            out[f"{n}_spec"] = sd(getattr(model, n).weight)
    if model.retr is not None:
        out["retr_wk_spec"] = sd(model.retr.wk.weight)
    if model.hc is not None:
        for k, u in enumerate(model.hc):
            off = u.A_r - torch.diag(torch.diagonal(u.A_r))
            out[f"hc{k}"] = dict(A_m=[round(float(x), 3) for x in u.A_m], A_r_offdiag_absmean=float(off.abs().mean()),
                                 A_r_diag=[round(float(x), 3) for x in torch.diagonal(u.A_r)],
                                 B=[round(float(x), 3) for x in u.B], s_a=float(u.s_a), s_b=float(u.s_b),
                                 W_rms=float(torch.cat([u.W_m.flatten(), u.W_r.flatten(), u.W_b.flatten()]).pow(2).mean().sqrt()))
    if model.dwa is not None:
        out["dwa_alpha"] = [[round(float(x), 4) for x in p] for p in model.dwa]
    derf = {n: (round(float(m.alpha), 4), round(float(m.s), 4), round(float(m.weight.abs().mean()), 4))
            for n, m in model.named_modules() if isinstance(m, Derf)}
    if derf:
        out["derf_alpha_s_gain"] = derf
    if model.pbld is not None:
        out["pbld_P_fro"] = float(model.pbld.P.weight.norm())
        out["pbld_w1_absmean"] = float(model.pbld.w1.abs().mean())
    if model.ftok is not None:
        out["ftok_Zf_fro"] = float(model.ftok.Zf.weight.norm())
    if model.cfg.get("hist_extra"):
        out["hist_extra_o_fro"] = [float(b.attn.o.weight.norm()) for b in list(model.hist_enc)[-model.cfg["hist_extra"]:]]
    peri = {n: round(float(m.weight.abs().mean()), 4) for n, m in model.named_modules()
            if n.endswith(("ln_ao", "ln_fo"))}
    if peri:
        out["periln_gain"] = peri
    return out


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
