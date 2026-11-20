# Ranker model of the input-side group (2026-11-11). All five tricks in it were rejected: piecewise-linear
# embeddings of numeric inputs (NumEmbPLE), a propensity head supervised by history events, semantic-ID codes
# of items, exponentially weighted history summaries (_ts_feats) and a weight-sharing mini-ensemble (TabM).
# Kept because it shows how each null result was implemented.
# Used through train.py of lab/nn_x/feat with the flags --num_emb ple, --prop_head 0.05, --sid, --ts_feats,
# --tabm 4 (launcher sh/run.sh; 2 epochs per run on one GPU). Not run on its own.
# Results of all arms: ../results/feat_results.tsv.
# Kept as it ran: paths and names refer to the lab tree (see lab/ and docs/glossary.md).
# Original: lab/nn_x/feat/model.py
#
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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@contextlib.contextmanager
def extra_init(offset=7919):
    """Build new (treatment) modules under a forked CPU RNG, so the baseline parameters keep exactly the baseline
    init and an A/B differs only by the treatment."""
    with torch.random.fork_rng(devices=[]):  # the model is built on CPU, then .to(dev)
        torch.manual_seed(torch.initial_seed() + offset)
        yield

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


EXTRA_COLS = {"ts": 12, "sid": 5}  # named dense-extra blocks (appended after the user-0 pair, before xt)


def n_extra_cols(extras):
    return sum(EXTRA_COLS[e] for e in (extras or ()))


def _ts_feats(S, hv, hm, E, het, hes, mf, cnt):
    """B4: multi-scale recency-decayed (EWMA over the recency index k, newest first) history features.
    For tau in (4, 16, 64): w_k = exp(-k/tau) * valid_k;
      dmean = sum w S / sum w; dlse = (1/5) log(sum w exp(5S) / sum w); dcat = sum w 1[same cat] / sum w (hmask).
    + dlse_16 and dcat_16 over type1&src1 events only, + within-feed z-score of dlse_16 (12 columns).
    Empty sums -> -1.5 (similarities) / 0 (shares)."""
    K = S.shape[-1]
    k = torch.arange(K, device=S.device, dtype=torch.float32)
    neg = torch.tensor(-1.5, device=S.device)
    Sf = S.float()
    Ef = E.float()
    t11 = (het == 1) & (hes == 1)
    out = []
    keep = {}

    def block(tau, vmask, cmask_h):
        lw = -k / tau  # log w for valid events
        w = torch.exp(lw)[None, :] * vmask.float()  # [B,K]
        sw = w.sum(-1, keepdim=True)  # [B,1]
        dmean = torch.einsum("bmk,bk->bm", Sf, w) / sw.clamp_min(1e-12)
        lse = torch.logsumexp(torch.where(vmask[:, None, :], 5 * Sf + lw[None, None, :], torch.tensor(-1e4, device=S.device)), -1)
        dlse = (lse - torch.log(sw.clamp_min(1e-12))) / 5
        has = sw > 0
        dmean = torch.where(has, dmean, neg)
        dlse = torch.where(has, dlse, neg)
        wc = torch.exp(lw)[None, :] * cmask_h.float()
        swc = wc.sum(-1, keepdim=True)
        dcat = torch.einsum("bmk,bk->bm", Ef, wc) / swc.clamp_min(1e-12)
        dcat = torch.where(swc > 0, dcat, torch.zeros_like(dcat))
        return dmean, dlse, dcat

    for tau in (4.0, 16.0, 64.0):
        dmean, dlse, dcat = block(tau, hv, hm)
        out += [dmean, dlse, dcat]
        if tau == 16.0:
            keep["dlse16"] = dlse
    _, dlse_t, dcat_t = block(16.0, hv & t11, hm & t11)
    out += [dlse_t, dcat_t, _zscore(keep["dlse16"], mf, cnt)]
    return out


def _sid_feats(b, hv, mf, cnt):
    """B3: semantic-ID code-match features: log1p #hist same c1, log1p #hist same (c1,c2), log1p #type1&src1 hist
    same (c1,c2), 1/(1+first k with same (c1,c2)) (0 if none), within-feed z of log1p #hist same (c1,c2)."""
    cv = ~b["cmiss"]
    v = hv[:, None, :] & cv[:, :, None]
    E1 = (b["cc1"][:, :, None] == b["hc1"][:, None, :]) & v
    E2 = (b["cpair"][:, :, None] == b["hpair"][:, None, :]) & v
    t11 = ((b["het"] == 1) & (b["hes"] == 1))[:, None, :]
    n1 = torch.log1p(E1.float().sum(-1))
    n2 = torch.log1p(E2.float().sum(-1))
    n2t = torch.log1p((E2 & t11).float().sum(-1))
    K = E2.shape[-1]
    ar = torch.arange(K, device=E2.device)
    first = torch.where(E2, ar, K).min(-1).values.float()
    rec = torch.where(first < K, 1.0 / (1.0 + first), torch.zeros_like(first))
    return [n1, n2, n2t, rec, _zscore(n2, mf, cnt)]


@torch.no_grad()
def dense_features(b, n_dense=None, extras=()):
    """Returns [B,M,F] float32 features computed from a gathered batch dict.
    n_dense == N_DENSE + 2 (+ extra cols) appends the user-0 regime features (K5): is_user0, is_user0 * log(1+H).
    extras: named blocks (EXTRA_COLS) appended after the user-0 pair and before xt."""
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
    nx = n_extra_cols(extras)
    if n_dense is not None and n_dense - nx >= N_DENSE + 2:
        u0 = b["user0"].float()[:, None].expand_as(fcs)
        feats += [u0, u0 * torch.log1p(H).expand_as(fcs)]
    for name in (extras or ()):
        if name == "ts":
            feats += _ts_feats(S, hv, hm, E, het, b["hes"], mf, cnt)
        elif name == "sid":
            feats += _sid_feats(b, hv, mf, cnt)
        else:
            raise ValueError(name)
    X = torch.stack(feats, -1)
    if n_dense is not None and n_dense > N_DENSE + 2 + nx:  # train-stage extra features (collaborative signals)
        X = torch.cat([X, b["xt"]], -1)
    return torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


N_DENSE = 15 + 2 + 2 + 6 + N_ET + 2 + N_ET + 1 + 3 + 10


# ---------------------------------------------------------------------------- blocks
class Attn(nn.Module):
    def __init__(self, d, heads, drop, sim_bias=False):
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

    def forward(self, x, ctx, kmask, bias=None):
        B, N, D = x.shape
        K = ctx.shape[1]
        h = self.h
        q = self.q(x).view(B, N, h, D // h).transpose(1, 2)
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

    def __init__(self, d, heads, drop, ff=2, sim_bias=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.lnc = nn.LayerNorm(d)
        self.attn = Attn(d, heads, drop, sim_bias)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff * d), nn.GELU(), nn.Dropout(drop), nn.Linear(ff * d, d))
        self.dr = nn.Dropout(drop)

    def forward(self, x, kmask, ctx=None, bias=None):
        hx = self.ln1(x)
        c = hx if ctx is None else self.lnc(ctx)
        x = x + self.dr(self.attn(hx, c, kmask, bias))
        return x + self.dr(self.ff(self.ln2(x)))


def ple_edges_from_sample(Xs, T=16, force_cols=(0, 3, 6, 9, 12), pm_thresh=0.01, delta=1e-3, min_gap=5e-4):
    """B1: per-feature PLE bin edges from a sample of standardized+clamped features Xs [N,F].
    Edges = deduplicated quantiles at levels t/T; for the raw f1..f5 columns (force_cols) every point mass
    (value with sample frequency >= pm_thresh: floor/ceiling of the interpolated bins) is forced in as an edge,
    together with a step edge p+delta (p-delta for the maximum) so the point mass sits alone in a narrow bin and
    the jump in target rate is representable. Edges closer than min_gap are merged (forced ones kept); if more than
    T+1 remain, the non-forced edge with the smallest neighbour gap is dropped. Padded at the front with zero-width
    (masked) bins to exactly T+1 edges. Returns float32 [F, T+1]."""
    Xs = Xs.float().cpu()
    N, F = Xs.shape
    srt = Xs.sort(0).values
    qi = torch.round(torch.linspace(0, 1, T + 1) * (N - 1)).long()
    Q = srt[qi].numpy()  # [T+1, F]
    out = np.zeros((F, T + 1), np.float32)
    for j in range(F):
        col = srt[:, j].numpy()
        lo, hi = float(col[0]), float(col[-1])
        forced = {lo, hi}
        if j in force_cols:
            vals, cnts = np.unique(col, return_counts=True)
            for p in vals[cnts >= pm_thresh * N]:
                p = float(p)
                forced.add(p)
                forced.add(p - delta if p >= hi else p + delta)
        edges = sorted(set(float(v) for v in Q[:, j]) | forced)
        merged = []
        for e in edges:  # merge near-duplicates (float noise), keeping forced edges
            if merged and e - merged[-1] < min_gap:
                if e in forced and merged[-1] not in forced:
                    merged[-1] = e
                continue
            merged.append(e)
        edges = merged
        while len(edges) > T + 1:
            gaps = [(min(edges[i] - edges[i - 1], edges[i + 1] - edges[i]), i)
                    for i in range(1, len(edges) - 1) if edges[i] not in forced]
            if not gaps:
                edges = edges[:T] + [edges[-1]]
                break
            edges.pop(min(gaps)[1])
        edges = [edges[0]] * (T + 1 - len(edges)) + edges
        out[j] = edges
    return torch.from_numpy(out)


class NumEmbPLE(nn.Module):
    """B1: piecewise-linear numeric embeddings, rtdl version B (Gorishniy et al. 2022):
    phi_j(x) = x w_j + beta_j + sum_t e_t(x) v_{j,t},  e_t(x) = clamp((x - b_{t-1}) / (b_t - b_{t-1}), 0, 1)
    (first valid bin unclamped below, last bin unclamped above, zero-width padding bins masked). v is zero-init,
    so training starts as a plain linear embedding. Output: flatten over features -> [.., F * d_e]."""

    def __init__(self, n_feat, n_bins=16, d_e=12):
        super().__init__()
        self.F, self.T, self.d_e = n_feat, n_bins, d_e
        self.register_buffer("ple_edges", torch.linspace(-3, 3, n_bins + 1).repeat(n_feat, 1))
        lim = d_e ** -0.5
        self.lin_w = nn.Parameter(torch.empty(n_feat, d_e).uniform_(-lim, lim))
        self.lin_b = nn.Parameter(torch.empty(n_feat, d_e).uniform_(-lim, lim))
        self.ple_v = nn.Parameter(torch.zeros(n_feat, n_bins, d_e))

    def set_edges(self, edges):
        self.ple_edges.copy_(edges.to(self.ple_edges))

    def encode(self, x):
        """x [..., F] -> e [..., F, T] (fp32)."""
        ed = self.ple_edges
        left, width = ed[:, :-1], ed[:, 1:] - ed[:, :-1]  # [F,T]
        valid = width > 0
        T = self.T
        t = torch.arange(T, device=ed.device)
        first = torch.where(valid, t, T).min(-1).values  # first valid bin per feature
        lo = torch.where(t[None, :] == first[:, None], torch.tensor(-1e9, device=ed.device), torch.zeros_like(left))
        hi = torch.where(t[None, :] == T - 1, torch.tensor(1e9, device=ed.device), torch.ones_like(left))
        r = (x.float()[..., None] - left) / width.clamp_min(1e-6)
        return torch.minimum(torch.maximum(r, lo), hi) * valid

    def forward(self, x):
        e = self.encode(x)  # [...,F,T]
        sh = x.shape[:-1]
        e2 = e.reshape(-1, self.F, self.T).transpose(0, 1)  # [F,N,T]
        nl = torch.bmm(e2, self.ple_v)  # [F,N,d_e] (bf16 under autocast)
        phi = x.float()[..., None] * self.lin_w + self.lin_b + nl.transpose(0, 1).reshape(*sh, self.F, self.d_e)
        return phi.reshape(*sh, self.F * self.d_e)

    def diag(self):
        return self.ple_v.detach().float().norm(dim=(1, 2))  # [F] norm of the non-linear part per feature


N_PROP = 11  # 0 = type1&src1, 1 = type1&src!=1, 2..10 = event types 2..10


def prop_class(het, hes):
    return torch.where(het == 1, torch.where(hes == 1, 0, 1), het)


def _insert_in_cols(lin, p0, n_new, offset):
    """Return a copy of Linear `lin` with n_new input columns inserted at position p0. Existing columns keep their
    (baseline) values; the new ones get the default fan-in-scaled uniform init under a forked RNG."""
    with extra_init(offset), torch.no_grad():
        new = nn.Linear(lin.in_features + n_new, lin.out_features)
        bound = 1.0 / math.sqrt(lin.in_features)
        new.weight[:, p0:p0 + n_new].uniform_(-bound, bound)
        new.weight[:, :p0] = lin.weight[:, :p0]
        new.weight[:, p0 + n_new:] = lin.weight[:, p0:]
        new.bias.copy_(lin.bias)
    return new


class Ranker(nn.Module):
    def __init__(self, d=128, heads=4, hist_layers=1, blocks=2, drop=0.1, use_hist=True, use_set=True,
                 use_pos_tower=True, n_dense=N_DENSE, aux_pos=False, ff=2, sim_bias=False,
                 pos_mode="both", domain=False, intent=False, xt_names=None, id_buckets=0, id_dim=32,
                 id_drop=0.5, arch="tf", dense_extras=(), num_emb="none", ple_bins=16, ple_dim=12,
                 prop_head=False, sid=False, sid_k1=256, sid_k2=64, sid_dim=32, sid_p2=0.5, sid_pz=0.2, tabm=0):
        super().__init__()
        self.cfg = dict(d=d, heads=heads, hist_layers=hist_layers, blocks=blocks, drop=drop, use_hist=use_hist,
                        use_set=use_set, use_pos_tower=use_pos_tower, n_dense=n_dense, aux_pos=aux_pos, ff=ff, sim_bias=sim_bias, pos_mode=pos_mode, domain=domain,
                        intent=intent, xt_names=xt_names, id_buckets=id_buckets, id_dim=id_dim, id_drop=id_drop, arch=arch,
                        dense_extras=tuple(dense_extras or ()), num_emb=num_emb, ple_bins=ple_bins, ple_dim=ple_dim,
                        prop_head=prop_head, sid=sid, sid_k1=sid_k1, sid_k2=sid_k2, sid_dim=sid_dim, sid_p2=sid_p2,
                        sid_pz=sid_pz, tabm=tabm)
        self.tabm = tabm
        self.dense_extras = tuple(dense_extras or ())
        self.num_emb, self.prop_head, self.sid = num_emb, prop_head, sid
        self.sid_p2, self.sid_pz = sid_p2, sid_pz
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
        nx = n_extra_cols(dense_extras)
        n_base = n_dense - nx  # extras columns sit at [p0, p0 + nx), before the xt columns
        p0 = n_base - len(xt_names or [])
        self.c_mlp = nn.Sequential(nn.Linear(n_base, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        if nx:  # baseline columns keep the baseline init (same RNG stream), new columns from a forked RNG
            self.c_mlp[0] = _insert_in_cols(self.c_mlp[0], p0, nx, 404)
        if num_emb == "ple":  # B1: c_mlp[0] reads the flattened PLE embeddings (built under a forked RNG)
            with extra_init(101):
                self.ple = NumEmbPLE(n_dense, ple_bins, ple_dim)
                self.c_mlp[0] = nn.Linear(n_dense * ple_dim, 2 * d)
        if prop_head:  # B2: history-supervised item propensity head (event-type distribution of an item)
            with extra_init(202):
                self.prop = nn.Sequential(nn.Linear(128 + 32, 128), nn.GELU(), nn.Linear(128, N_PROP))
                self.prop_cat = nn.Embedding(N_CAT, 32)
                self.prop_c = nn.Linear(N_PROP, d)
                self.prop_w = nn.Linear(N_PROP, 1)
                for m in (self.prop_c, self.prop_w):
                    nn.init.zeros_(m.weight)
                    nn.init.zeros_(m.bias)
            self.prop_loss = None
        if sid:  # B3: semantic-ID code embeddings (2-level residual k-means cells of the content embedding)
            with extra_init(303):
                self.sid_emb1 = nn.Embedding(sid_k1 + 1, sid_dim)
                self.sid_emb2 = nn.Embedding(sid_k1 * sid_k2 + 1, sid_dim)
                self.sid_zc = nn.Linear(2 * sid_dim, d)
                self.sid_zh = nn.Linear(2 * sid_dim, d)
                for m in (self.sid_zc, self.sid_zh):
                    nn.init.zeros_(m.weight)
                    nn.init.zeros_(m.bias)
        if tabm:  # B5: TabM-mini BatchEnsemble adapters (random +-1 init) on the candidate inputs
            with extra_init(505):
                self.tabm_r1 = nn.Parameter(torch.randint(0, 2, (tabm, 256)).float() * 2 - 1)
                self.tabm_r2 = nn.Parameter(torch.randint(0, 2, (tabm, n_dense)).float() * 2 - 1)
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
            self.hist_enc = nn.ModuleList([Block(d, heads, drop, ff) for _ in range(hist_layers)])
            self.cross = nn.ModuleList([Block(d, heads, drop, ff, sim_bias) for _ in range(blocks)])
        else:
            self.u_cross = nn.ModuleList([Block(d, heads, drop, ff) for _ in range(blocks)])
        if use_set:
            self.sets = nn.ModuleList([Block(d, heads, drop, ff) for _ in range(blocks)])
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.aux_pos = aux_pos
        if aux_pos:  # auxiliary head: predict the production display position (distillation of prod ranker)
            self.aux_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.wide = nn.Linear(n_base, 1)
        if nx:
            self.wide = _insert_in_cols(self.wide, p0, nx, 405)
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

    def dense(self, b):
        return dense_features(b, self.cfg["n_dense"], self.dense_extras)

    def _sid_vec(self, c1, pair):
        """[E1(c1); E2(pair)] with SID-dropout in training: E2 zeroed with p=sid_p2, whole vector with p=sid_pz."""
        e1, e2 = self.sid_emb1(c1), self.sid_emb2(pair)
        if self.training:
            sh = c1.shape
            e2 = e2 * (torch.rand(sh, device=c1.device) >= self.sid_p2)[..., None]
            z = torch.cat([e1, e2], -1)
            return z * (torch.rand(sh, device=c1.device) >= self.sid_pz)[..., None]
        return torch.cat([e1, e2], -1)

    def prop_logits(self, e, cat):
        """B2 propensity head logits (fp32) for normalized item embeddings e [...,128] and categories."""
        with torch.autocast("cuda", enabled=False):
            return self.prop(torch.cat([e.float(), self.prop_cat(cat).float()], -1))

    def prop_history_ce(self, b):
        """Mean CE of the propensity head over valid history tokens (+ counts for diagnostics)."""
        vm = b["hemb_ok"]
        lg = self.prop_logits(b["he"][vm], b["hcat"][vm])
        cls = prop_class(b["het"][vm], b["hes"][vm])
        return F.cross_entropy(lg, cls), cls

    def set_norm(self, mean, std):
        self.f_mean.copy_(mean)
        self.f_std.copy_(std.clamp_min(1e-3))

    def forward(self, b, X=None, use_pos=False):
        if X is None:
            X = self.dense(b)
        X = ((X - self.f_mean) / self.f_std).clamp(-6, 6)
        cmask = b["cmask"]
        ce, un = b["ce"], b["un"]
        c_in = torch.cat([ce, ce * un[:, None, :]], -1)
        Xk = X
        ntm = self.tabm
        rep = (lambda t: t.repeat_interleave(ntm, 0)) if ntm else (lambda t: t)
        if ntm:  # member i sees r1_i * input and r2_i * X; members are laid out as rows b*k + i
            c_in = (c_in[:, None] * self.tabm_r1[None, :, None, :]).flatten(0, 1)
            Xk = (X[:, None] * self.tabm_r2[None, :, None, :]).flatten(0, 1)
        c = self.c_lin(c_in)
        xm = self.ple(Xk) if self.num_emb == "ple" else Xk
        c = c + rep(self.cat(b["ccat"]) + b["cmiss"][..., None].float() * self.c_miss) + self.c_mlp(xm)
        if self.id_buckets:
            c = c + self.id_c(self._id_vec(b["citem"]))
        lp_prop = None
        if self.prop_head:
            if self.training:
                self.prop_loss, _ = self.prop_history_ce(b)
            with torch.no_grad():  # detached: a clean supervised feature
                lp_prop = torch.log_softmax(self.prop_logits(ce, b["ccat"]), -1)  # [B,M,11] fp32
            c = c + rep(self.prop_c(lp_prop))
        if self.sid:
            c = c + rep(self.sid_zc(self._sid_vec(b["cc1"], b["cpair"])))
        c = self.c_ln(c)
        dvec = None
        if self.domain:
            dom = b.get("dom")
            if dom is None:
                dom = F.one_hot(torch.full((X.shape[0],), self.infer_domain, device=c.device), 3).float()
            dvec = rep((dom.float() @ self.dom.weight)[:, None, :])
            c = c + dvec
        if self.extra is not None and "xe" in b:  # extra precomputed features (fine-tune stage only)
            c = c + rep(self.extra(b["xe"]))
        u = self.u_lin(un)[:, None, :]  # [B,1,d]
        B = X.shape[0]  # feeds (TabM: c holds B*k member rows)
        if self.use_hist:
            he = b["he"]
            K = he.shape[1]
            h = self.h_lin(he) + self.et(b["het"]) + self.es(b["hes"]) + self.cat(b["hcat"]) \
                + self.rec(torch.arange(K, device=he.device))[None] + (~b["hemb_ok"])[..., None].float() * self.h_miss
            if self.id_buckets:
                h = h + self.id_h(self._id_vec(b["hitem"]))
            if self.sid:
                h = h + self.sid_zh(self._sid_vec(b["hc1"], b["hpair"]))
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
        if ntm:  # the history encoder ran once; share its output across the k members
            ctx, kmask, cmask = rep(ctx), rep(kmask), rep(cmask)
            if sbias is not None:
                sbias = rep(sbias) if torch.is_tensor(sbias) else tuple(rep(t) for t in sbias)
        for i in range(len(crosses)):
            c = crosses[i](c, kmask, ctx, sbias)
            if self.use_set:
                c = self.sets[i](c, cmask)
        # the head runs in fp32: bf16 logits are quantized (step 1/64 around |x|~2-4), which creates many
        # within-feed ties, and ties are scored pessimistically by the metric (cost ~0.001-0.002 NDCG)
        if dvec is not None:
            c = c + dvec
        with torch.autocast("cuda", enabled=False):
            logit = self.head(c.float()).squeeze(-1) + rep(self.wide(X.float()).squeeze(-1))
            if lp_prop is not None:
                logit = logit + rep(self.prop_w(lp_prop).squeeze(-1))
            if ntm:  # members [B,k,M]; prediction = mean of member logits (fp32)
                logit = logit.view(-1, ntm, logit.shape[-1])
                self.member_logits = logit if (self.training or getattr(self, "keep_members", False)) else None
                logit = logit.mean(1)
            if lp_c is not None:
                logit = logit + self.intent_scale * lp_c
        if self.aux_pos and self.training:
            self.aux_out = self.aux_head(c).squeeze(-1).float()
            if ntm:
                self.aux_out = self.aux_out.view(-1, ntm, self.aux_out.shape[-1])  # [B,k,M]
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
