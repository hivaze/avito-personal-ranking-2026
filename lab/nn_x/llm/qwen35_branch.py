"""Qwen3.5 (hybrid Gated DeltaNet + gated softmax attention) as the soft-token trunk, with the exact 'branch' layout.

Why: in Qwen3.5 3 of every 4 layers are Gated DeltaNet (linear attention). HF's DeltaNet ignores custom 4D masks
(it only zeroes padding), so appending M candidates to [user, history] would make candidate c see candidates
1..c-1 in storage order (order-dependent scores, label-free but leaky across candidates). The branch layout makes every
candidate exactly "the next token after the history", permutation-equivariant, in one pass:

  prefix  = [pad.. pad, sink, user, h_old .. h_new]  (LEFT-padded to P = n_pre + max H; zero pads keep the recurrent
            state and the conv window exactly as at a true sequence start, since all trunk maps are bias-free and map
            0 -> 0. sink = amendment S1, the checkpoint's own frozen <|im_start|> row (id 248045), as in model_llm;
            n_pre = 2; the massive-activation analysis behind the row choice was done on Qwen3-0.6B only)
  DeltaNet layer: run the standard chunked gated delta rule on the prefix -> final state S_H [B,h,dk,dv] and the last
            3 pre-conv inputs; each candidate c then takes ONE recurrent step from S_H with its own conv window
            [x_{P-3}, x_{P-2}, x_{P-1}, x_c]:
              o_c = a_c (S_H^T q_c - b_c (S_H^T k_c)(k_c . q_c)) + b_c v_c (k_c . q_c)
            (a = exp(g) decay, b = beta; q, k L2-normed, q scaled by dk^-1/2; then the gated RMSNorm and out_proj).
  Full-attention layer (every 4th): the L1 mask of model_llm.build_layout (candidates see user + history + self;
            L2: + other candidates), position ids sink 0, user 1, history 2..H+1, every candidate H+2 (text mRoPE: 3
            equal streams; without the sink everything shifts down by one).
Verified on CPU (smoke_qwen35.py): a candidate's hidden state equals HF's own Qwen3_5TextModel forward over the
unpadded sequence [sink, user, history, c]; scores are permutation-equivariant; the naive causal append is not.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from model_llm import LORA_TARGETS, LLMRanker, build_layout


def _l2norm(x, eps=1e-6):  # == transformers' l2norm used with use_qk_l2norm_in_kernel=True
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)


def branch_deltanet(mod, x, P):
    """Qwen3_5GatedDeltaNet over x = [prefix (P tokens, left-padded) | candidates]; candidates branch from S_H."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule
    B, T, _ = x.shape
    xp, xc = x[:, :P], x[:, P:]
    Kd, Vd, hk, hv = mod.key_dim, mod.value_dim, mod.head_k_dim, mod.head_v_dim
    w = mod.conv1d.weight  # [C, 1, 4]
    ksz = w.shape[-1]
    # ---- prefix: the module's own computation (causal conv + chunked gated delta rule) with the final state
    mixed_p = mod.in_proj_qkv(xp).transpose(1, 2)  # [B, C, P]
    conv_p = F.silu(F.conv1d(mixed_p.to(w.dtype), w, mod.conv1d.bias, padding=ksz - 1, groups=mixed_p.shape[1])[..., :P])
    q, k, v = torch.split(conv_p.transpose(1, 2), [Kd, Kd, Vd], dim=-1)
    q, k, v = q.reshape(B, P, -1, hk), k.reshape(B, P, -1, hk), v.reshape(B, P, -1, hv)
    beta = mod.in_proj_b(xp).sigmoid()
    g = -mod.A_log.float().exp() * F.softplus(mod.in_proj_a(xp).float() + mod.dt_bias)
    rep = mod.num_v_heads // mod.num_k_heads
    if rep > 1:
        q, k = q.repeat_interleave(rep, dim=2), k.repeat_interleave(rep, dim=2)
    o_p, S = torch_chunk_gated_delta_rule(q, k, v, g=g, beta=beta, initial_state=None, output_final_state=True,
                                          use_qk_l2norm_in_kernel=True)
    z_p = mod.in_proj_z(xp).reshape(B, P, -1, hv)
    out_p = mod.out_proj(mod.norm(o_p.reshape(-1, hv), z_p.reshape(-1, hv)).reshape(B, P, -1))
    if T == P:
        return out_p
    # ---- candidates: one recurrent step each from S_H, conv window = last (ksz-1) prefix inputs + own input
    Mx = T - P
    mixed_c = mod.in_proj_qkv(xc)  # [B, Mx, C]
    last = F.pad(mixed_p, (ksz - 1, 0))[..., -(ksz - 1):] if P < ksz - 1 else mixed_p[..., -(ksz - 1):]  # [B,C,3]
    wc = w[:, 0, :].to(mixed_c.dtype)  # [C, 4]
    pre = (last * wc[None, :, : ksz - 1]).sum(-1)  # [B, C]
    conv_c = pre[:, None, :] + mixed_c * wc[:, ksz - 1]
    if mod.conv1d.bias is not None:
        conv_c = conv_c + mod.conv1d.bias
    conv_c = F.silu(conv_c)
    qc, kc, vc = torch.split(conv_c, [Kd, Kd, Vd], dim=-1)
    qc, kc, vc = qc.reshape(B, Mx, -1, hk), kc.reshape(B, Mx, -1, hk), vc.reshape(B, Mx, -1, hv)
    if rep > 1:
        qc, kc = qc.repeat_interleave(rep, dim=2), kc.repeat_interleave(rep, dim=2)
    qc, kc, vc = _l2norm(qc.float()) * hk ** -0.5, _l2norm(kc.float()), vc.float()
    bc = mod.in_proj_b(xc).float().sigmoid()  # [B, Mx, h]
    ac = (-mod.A_log.float().exp() * F.softplus(mod.in_proj_a(xc).float() + mod.dt_bias)).exp()
    S = S.float()  # [B, h, dk, dv]
    Sq = torch.einsum("bhkv,bmhk->bmhv", S, qc)
    Sk = torch.einsum("bhkv,bmhk->bmhv", S, kc)
    kq = (kc * qc).sum(-1, keepdim=True)  # [B, Mx, h, 1]
    o_c = ac[..., None] * (Sq - bc[..., None] * Sk * kq) + bc[..., None] * vc * kq
    o_c = o_c.to(o_p.dtype)
    z_c = mod.in_proj_z(xc).reshape(B, Mx, -1, hv)
    out_c = mod.out_proj(mod.norm(o_c.reshape(-1, hv), z_c.reshape(-1, hv)).reshape(B, Mx, -1))
    return torch.cat([out_p, out_c.to(out_p.dtype)], 1)


def left_layout(Hc, M, Mx, layout="L1", n_pre=1):
    """Left-padded prefix layout: P = n_pre + max H; feed b's sink (n_pre = 2) at P-2-H_b, user at P-1-H_b, history
    (old->new) up to P-1, candidates at P..P+M_b-1. Returns P, T, pos_ids [B,T], bool mask [B,1,T,T], type flags."""
    dev = Hc.device
    B = Hc.shape[0]
    P = n_pre + int(Hc.max())
    T = P + Mx
    t = torch.arange(T, device=dev)[None, :].expand(B, T)
    start = (P - n_pre - Hc)[:, None]
    is_ctx = (t >= start) & (t < P)
    is_cand = (t >= P) & (t < P + M[:, None])
    pos = torch.where(is_ctx, t - start, torch.where(is_cand, Hc[:, None] + n_pre, torch.zeros_like(t)))
    i, j = t[:, :, None], t[:, None, :]
    eye = i == j
    ctx_i, ctx_j = is_ctx[:, :, None], is_ctx[:, None, :]
    cand_i, cand_j = is_cand[:, :, None], is_cand[:, None, :]
    m = (ctx_i & ctx_j & (j <= i)) | (cand_i & ctx_j) | eye
    if layout == "L2":
        m = m | (cand_i & cand_j)
    return P, T, pos, m[:, None], is_ctx, is_cand


class Qwen35Ranker(LLMRanker):
    """LLMRanker with a Qwen3.5 text trunk run through the exact branch layout (same projectors, heads, loss)."""
    LORA_TARGETS = LORA_TARGETS + ["in_proj_qkv", "in_proj_z", "out_proj"]
    left_padded = True  # rows = n_pre + max H + max M: batch with FeedStore.token_batches_left (reports/llm_large.md)

    def _check_cfg(self, hf_cfg):
        assert hf_cfg.model_type == "qwen3_5", hf_cfg.model_type

    def _load_trunk(self, path, hf_cfg, random_init, tdtype, attn_impl):
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
        if random_init:
            tc = hf_cfg.text_config
            tc._attn_implementation = attn_impl
            m = Qwen3_5ForCausalLM(tc).to(torch.float32)
        else:
            m = Qwen3_5ForCausalLM.from_pretrained(path, dtype=tdtype, attn_implementation=attn_impl)
        lm = m.model  # Qwen3_5TextModel (vision tower, MTP head and lm_head are not loaded / dropped)
        del m
        return lm

    def assemble_left(self, u_tok, h_tok, c_tok, Hc, M):
        B, K, D = h_tok.shape
        Mx = c_tok.shape[1]
        dev = h_tok.device
        P = self.n_pre + int(Hc.max())
        kk = torch.arange(K, device=dev)[None, :].expand(B, K)
        hval = kk < Hc[:, None]
        mm = torch.arange(Mx, device=dev)[None, :].expand(B, Mx)
        cval = mm < M[:, None]
        bh = torch.arange(B, device=dev)[:, None]
        ar = torch.arange(B, device=dev)
        rows = [ar, bh.expand(B, K)[hval], bh.expand(B, Mx)[cval]]
        cols = [P - 1 - Hc, (P - 1 - kk)[hval], (P + mm)[cval]]
        vals = [u_tok, h_tok[hval], c_tok[cval]]
        if self.sink:  # S1: the frozen sink right before the user token
            rows.append(ar)
            cols.append(P - 2 - Hc)
            vals.append(self.sink_emb[None].expand(B, D))
        vals = torch.cat([v.to(h_tok.dtype) for v in vals])
        return torch.zeros(B, P + Mx, D, dtype=vals.dtype, device=dev).index_put((torch.cat(rows), torch.cat(cols)), vals)

    def trunk(self, seq, pos, mask, P):
        lm = self.lm
        pos3 = pos[None].expand(3, -1, -1)
        cos, sin = lm.rotary_emb(seq, pos3)
        h = seq
        for layer in lm.layers:
            def f(h, layer=layer):
                r = h
                x = layer.input_layernorm(h)
                if layer.block_type == "linear_attention":
                    x = branch_deltanet(layer.linear_attn, x, P)
                else:
                    x, _ = layer.self_attn(hidden_states=x, position_embeddings=(cos, sin), attention_mask=mask)
                h = r + x
                return h + layer.mlp(layer.post_attention_layernorm(h))
            h = checkpoint(f, h, use_reentrant=False) if (self.grad_ckpt and self.training) else f(h)
        return lm.norm(h)

    def forward(self, b, X=None):
        if X is None:
            X = self.dense(b)
        u_tok, h_tok, c_tok, Xn = self.tokens(b, X)
        Hc, M = b["H"], b["M"]
        Mx = c_tok.shape[1]
        P, T, pos, mask, _, _ = left_layout(Hc, M, Mx, self.cfg["layout"], self.n_pre)
        seq = self.assemble_left(u_tok, h_tok, c_tok, Hc, M)
        out = self.trunk(seq, pos, mask, P)
        hc = out[:, P:P + Mx]
        with torch.autocast(device_type=hc.device.type, enabled=False):
            hc = hc.float()
            logit = self.head(hc).squeeze(-1) + self.wide(Xn.float()).squeeze(-1)
            self.aux_out = self.aux_head(hc).squeeze(-1) if self.training else None
        return logit


__all__ = ["Qwen35Ranker", "branch_deltanet", "left_layout", "build_layout", "nn"]
