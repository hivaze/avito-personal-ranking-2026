"""Sequence encoder (SASRec/ARGUS-style) pretrained on all user histories.

Token t = ItemTower(item_t) + E_et[et_t] + E_es[es_t]; a learned BOS token is prepended so that empty histories
still have a state. Causal pre-LN transformer with RoPE over the sequence old -> new.
ItemTower(i) = MLP(normalized int8 content emb) (or a learned 'missing' vector) + E_cat[cat] + P * E_id[slot(i)]
(E_id only for items with corpus count >= min_cnt; ID-dropout during training so the content path works alone).
Heads: q_k = normalize(W_k h_t), one per target type k (fe = next new feed-originated open, new = next new item);
logit(h, i) = scale * <q_k, normalize(W_out ItemTower(i))>.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_ET, N_ES, N_CAT = 11, 32, 54


def rope_cache(L, hd, device, base=10000.0):
    inv = 1.0 / (base ** (torch.arange(0, hd, 2, device=device).float() / hd))
    t = torch.arange(L, device=device).float()
    f = torch.outer(t, inv)
    return f.cos(), f.sin()


def apply_rope(x, cos, sin):  # x [B,h,L,hd]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c, s = cos[None, None, :x.shape[2]], sin[None, None, :x.shape[2]]
    out = torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)
    return out.flatten(-2)


class Block(nn.Module):
    def __init__(self, d, nh, drop):
        super().__init__()
        self.nh = nh
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(drop)
        self.pdrop = drop

    def forward(self, x, cos, sin):
        B, L, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, L, 3, self.nh, d // self.nh).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.pdrop if self.training else 0.0)
        x = x + self.drop(self.o(a.transpose(1, 2).reshape(B, L, d)))
        return x + self.drop(self.ff(self.ln2(x)))


class SeqEncoder(nn.Module):
    def __init__(self, n_id, d=256, nl=3, nh=4, d_id=64, drop=0.05, heads=("fe", "new"), max_len=320, id_bias=False):
        super().__init__()
        self.d = d
        self.content = nn.Sequential(nn.Linear(128, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.missing = nn.Parameter(torch.zeros(d))
        self.cat = nn.Embedding(N_CAT, d)
        self.n_id = n_id
        self.id_emb = nn.Embedding(n_id + 1, d_id, sparse=True, padding_idx=0) if n_id > 0 else None
        if self.id_emb is not None:
            nn.init.normal_(self.id_emb.weight, std=0.02)
            with torch.no_grad():
                self.id_emb.weight[0].zero_()
            self.id_proj = nn.Linear(d_id, d, bias=False)
        self.id_bias = nn.Embedding(n_id + 1, 1, sparse=True, padding_idx=0) if (id_bias and n_id > 0) else None
        if self.id_bias is not None:
            nn.init.zeros_(self.id_bias.weight)
        self.et = nn.Embedding(N_ET, d)
        self.es = nn.Embedding(N_ES, d)
        self.bos = nn.Parameter(torch.randn(d) * 0.02)
        self.blocks = nn.ModuleList([Block(d, nh, drop) for _ in range(nl)])
        self.lnf = nn.LayerNorm(d)
        self.heads = nn.ModuleDict({k: nn.Linear(d, d) for k in heads})
        self.out_proj = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(20.0)))
        cos, sin = rope_cache(max_len, d // nh, "cpu")
        self.register_buffer("rcos", cos, persistent=False)
        self.register_buffer("rsin", sin, persistent=False)

    def item_rep(self, emb_i8, has, cat, slot, id_drop=0.0):
        """emb_i8 int8 [...,128], has bool, cat long, slot long (0 = no id)."""
        e = emb_i8.float()
        e = e / e.norm(dim=-1, keepdim=True).clamp_min(1e-3)
        r = self.content(e)
        r = torch.where(has[..., None], r, self.missing.to(r.dtype))
        r = r + self.cat(cat)
        if self.id_emb is not None:
            if id_drop > 0 and self.training:
                slot = torch.where(torch.rand(slot.shape, device=slot.device) < id_drop, torch.zeros_like(slot), slot)
            r = r + self.id_proj(self.id_emb(slot))
        return r

    def bias(self, slot):
        if self.id_bias is None:
            return torch.zeros(slot.shape, device=slot.device)
        return self.id_bias(slot.long()).squeeze(-1).float()

    def out_vec(self, rep):
        return F.normalize(self.out_proj(rep).float(), dim=-1)

    def encode(self, rep, et, es):
        """rep [B,L,d] item reps of the (right-padded) sequence, et/es long [B,L]. Returns h [B,L+1,d] where
        position 0 is BOS and position t+1 is the state after event t."""
        x = rep + self.et(et) + self.es(es)
        B = x.shape[0]
        x = torch.cat([self.bos.to(x.dtype).expand(B, 1, -1), x], dim=1)
        for blk in self.blocks:
            x = blk(x, self.rcos, self.rsin)
        return self.lnf(x)

    def query(self, h, head):
        return F.normalize(self.heads[head](h).float(), dim=-1)
