# The soft-token LLM ranker. A local Qwen3 checkpoint is the trunk; the user, the history events and the
# candidates enter as projected vectors through inputs_embeds, with no vocabulary token except one frozen
# sink token. In layout L1 each candidate attends to the user, the whole history and itself, and all
# candidates share one position id, so the scores do not depend on candidate order. LoRA rank 32 on the
# trunk, fp32 score head and auxiliary head, the ranker's own listwise loss.
# Imported by train_llm.py; it is not run on its own.
# Kept as it ran: paths and names refer to the lab tree (see lab/ and docs/glossary.md).
# Original: lab/nn_x/llm/model_llm.py
#
"""Soft-token LLM ranker: a local Qwen3 checkpoint as the trunk over projected (non-language) tokens.

Sequence per feed (soft tokens via inputs_embeds; the only vocabulary row used is the frozen sink):
  [sink] [user] [h_old ... h_new] [c_1 ... c_M] [pad ...]
  sink token  = amendment S1 (2026-11-12): ONE frozen token at position 0, identical for every feed = a row of the
                checkpoint's own embedding matrix (captured before embed_tokens is dropped; the random-init control
                uses the same row of its own init). Reason (review): without it the user token at position 0 became
                Qwen's massive-activation attention sink from ~layer 3 on (max|h| ~5e3, content-independent: cos
                0.99999 across users at layer 13). sink=False restores the old layout.
                DEFAULT ROW = <|im_start|> (151644), NOT the <|endoftext|> (151643) named in LLM_FEASIBILITY 5.2: with
                <|endoftext|> the token right after it is treated as a new document start and still becomes massive
                (CPU, init: 8/12 soft user tokens, 5/12 even for REAL vocabulary rows; layer-13 user cosine 0.99999 ->
                the declared S1 acceptance (< 0.99) fails). With <|im_start|> (the canonical start of every Qwen chat
                sequence): 0/12, cosine max 0.936. Chosen on init-time hidden-state geometry only (no labels, no GPU);
                sink_token="endoftext" (or "<|endoftext|>") reproduces the declared row. See reports/llm_build.md "Amendments".
  user token  = proj(L2-normed user_emb || is_user0 || domain mixture)
  history j   = proj(LN(Linear(L2-normed item emb) + E_cat + E_event_type + E_event_source + E_recency + miss))
  candidate c = proj(the nn/ ranker's own candidate token: Linear([e, e*u]) + E_cat + MLP(dense X) + miss, LN, + domain)
                X = nn/model.dense_features (61 base + 2 user-0 + cvnd_pre2s train-stage inputs = 104 for cvnd_pre2s)
  Every projector ends in LayerNorm(d_model) whose gain is initialised to the RMS of the checkpoint's token
  embeddings, so the soft tokens enter the trunk on the scale its first layer was trained on (ORCA "align").

Layout L1 (default): sink+user+history causal; candidate c attends to sink + user + all history + itself only;
  position ids sink 0, user 1, history 2..H+1, EVERY candidate H+2 (without the sink: user 0, history 1..H,
  candidates H+1) -> each candidate is computed exactly as "the next token after the history"; scores are
  permutation-equivariant over candidates and all M candidates are scored in one pass.
Layout L2: as L1, plus candidates attend to each other (still permutation-equivariant, shared position id).
Labels / production positions are never inputs (the model reads only the keys listed in INPUT_KEYS).

Heads (fp32): score = MLP(LN(h_c)) + wide(X); K9 aux = MLP'(LN(h_c)) (production-order distillation, train only).
Trainable: LoRA (r=32, alpha=32, dropout 0.05, B=0 init) on q/k/v/o/gate/up/down of every layer, all RMSNorm gains,
projectors and heads. random_init=True builds the same architecture from the config with HF init, fully trainable.
"""
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from data_llm import dense_features

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # weights are local; never touch the network

N_ET, N_ES, N_CAT, MAX_REC = 10, 32, 54, 256
BACKBONES = {"qwen3-0.6b": "Qwen/Qwen3-0.6B", "qwen3-4b-instruct": "Qwen/Qwen3-4B-Instruct-2507",
             "qwen3-8b": "Qwen/Qwen3-8B", "qwen3.5-2b": "Qwen/Qwen3.5-2B", "qwen3.5-4b": "Qwen/Qwen3.5-4B",
             "qwen3.5-9b": "Qwen/Qwen3.5-9B"}
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
# keys of a gathered batch the model may read (no "y", no "pos")
INPUT_KEYS = {"user0", "fidx", "M", "H", "cmask", "ci", "hmask", "ce_raw", "cmiss", "ccat", "cf", "he_raw", "hemb_ok",
              "hcat", "het", "hes", "ue_raw", "dom", "xt", "ce", "he", "un"}


def build_layout(Hc, M, Tx, layout="L1", float_mask_dtype=None, Mx=None, n_pre=1):
    """Token index maps, position ids and the attention mask for a padded batch.
    Hc [B] history lengths (already clamped), M [B] candidate counts, n_pre = context tokens before the history
    (1 = [user]; 2 = [sink, user], amendment S1). Token t of feed b is: 0..n_pre-1 sink/user, n_pre..n_pre+Hc-1 history
    (old -> new), n_pre+Hc..n_pre+Hc+M-1 candidates, the rest padding. Returns pos_ids [B,T], mask [B,1,T,T] (bool:
    True = may attend; or additive float if float_mask_dtype is given), cand_tok [B,Mx] token index of each candidate
    slot. Every context token (sink included) is causal, so every real token may attend to the sink."""
    dev = Hc.device
    B = Hc.shape[0]
    t = torch.arange(Tx, device=dev)[None, :].expand(B, Tx)
    c0 = n_pre + Hc  # first candidate token == the shared candidate position id
    T = c0 + M
    is_ctx = t < c0[:, None]
    is_cand = (t >= c0[:, None]) & (t < T[:, None])
    pos_ids = torch.where(is_ctx, t, c0[:, None])  # candidates (and pads) all at n_pre + H
    i, j = t[:, :, None], t[:, None, :]
    eye = i == j
    ctx_i, ctx_j = is_ctx[:, :, None], is_ctx[:, None, :]
    cand_i, cand_j = is_cand[:, :, None], is_cand[:, None, :]
    m = (ctx_i & ctx_j & (j <= i)) | (cand_i & ctx_j) | eye  # pads attend only to themselves (no NaN rows)
    if layout == "L2":
        m = m | (cand_i & cand_j)
    m = m[:, None]
    if float_mask_dtype is not None:
        m = torch.zeros(m.shape, dtype=float_mask_dtype, device=dev).masked_fill(~m, torch.finfo(float_mask_dtype).min)
    Mx = int(M.max()) if Mx is None else Mx  # Mx = padded candidate width of the batch (slots >= M are masked)
    cand_tok = (c0[:, None] + torch.arange(Mx, device=dev)[None, :]).clamp_max(Tx - 1)
    return pos_ids, m, cand_tok, is_ctx, is_cand


class ScaledLN(nn.LayerNorm):
    """LayerNorm whose gain starts at `rms`: outputs have per-token RMS == rms at init (Qwen embedding scale)."""

    def __init__(self, d, rms):
        super().__init__(d)
        with torch.no_grad():
            self.weight.fill_(float(rms))


class LLMRanker(nn.Module):
    def __init__(self, backbone="qwen3-0.6b", n_dense=104, xt_rel=None, xt_idcols=None, xt_relonly=None,
                 layout="L1", d_in=256, lora_r=32, lora_alpha=32, lora_dropout=0.05, random_init=False,
                 dtype="fp32", attn_impl="sdpa", grad_ckpt=False, head_hidden=256, sink=True,
                 sink_token="<|im_start|>"):
        super().__init__()
        self.cfg = dict(backbone=backbone, n_dense=n_dense, xt_rel=xt_rel, xt_idcols=xt_idcols, xt_relonly=xt_relonly,
                        layout=layout, d_in=d_in, lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                        random_init=random_init, dtype=dtype, attn_impl=attn_impl, head_hidden=head_hidden, sink=sink,
                        sink_token=sink_token)
        from transformers import AutoConfig
        path = BACKBONES.get(backbone, backbone)
        tdtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[dtype]
        hf_cfg = AutoConfig.from_pretrained(path)
        self._check_cfg(hf_cfg)
        lm = self._load_trunk(path, hf_cfg, random_init, tdtype, attn_impl)
        D = lm.config.hidden_size
        self.D = D
        with torch.no_grad():
            self.emb_rms = float(lm.embed_tokens.weight.float().pow(2).mean().sqrt())
            # S1: frozen sink = this checkpoint's (or this random init's) own `sink_token` row; a buffer, never trained
            self.sink = bool(sink)
            self.n_pre = 2 if self.sink else 1
            self.sink_id = sink_token_id(path, hf_cfg, sink_token) if self.sink else None
            sink_row = lm.embed_tokens.weight[self.sink_id].float().clone() if self.sink else torch.zeros(D)
        self.register_buffer("sink_emb", sink_row)
        lm.embed_tokens = nn.Embedding(1, D)  # vocabulary never used (inputs_embeds only); frees ~155M-389M params
        lm.embed_tokens.weight.requires_grad_(False)
        self.attn_impl = attn_impl
        self.random_init = random_init
        if not random_init and lora_r > 0:
            from peft import LoraConfig, inject_adapter_in_model
            lcfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                              target_modules=self.LORA_TARGETS, bias="none", init_lora_weights=True)
            lm = inject_adapter_in_model(lcfg, lm)
            for n, p in lm.named_parameters():
                p.requires_grad_("lora_" in n or n.endswith("norm.weight"))
                if p.requires_grad and p.dtype != torch.float32:
                    p.data = p.data.float()  # trainable params keep fp32 master copies
        else:
            for n, p in lm.named_parameters():
                p.requires_grad_(n != "embed_tokens.weight")
        self.grad_ckpt = grad_ckpt
        if grad_ckpt:
            lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        lm.config.use_cache = False
        self.lm = lm
        r = self.emb_rms
        # ---- candidate token: the ranker's construction (nn/model.Ranker.forward) at width d_in
        self.register_buffer("f_mean", torch.zeros(n_dense))
        self.register_buffer("f_std", torch.ones(n_dense))
        self.cat = nn.Embedding(N_CAT, d_in)
        self.c_lin = nn.Linear(256, d_in)
        self.c_miss = nn.Parameter(torch.zeros(d_in))
        self.c_mlp = nn.Sequential(nn.Linear(n_dense, 2 * d_in), nn.GELU(), nn.Linear(2 * d_in, d_in))
        self.c_ln = nn.LayerNorm(d_in)
        self.dom = nn.Embedding(3, d_in)
        nn.init.normal_(self.dom.weight, std=0.02)
        self.c_out = nn.Sequential(nn.Linear(d_in, D), ScaledLN(D, r))
        # ---- history tokens
        self.h_lin = nn.Linear(128, d_in)
        self.h_miss = nn.Parameter(torch.zeros(d_in))
        self.et = nn.Embedding(N_ET + 1, d_in)
        self.es = nn.Embedding(N_ES, d_in)
        self.rec = nn.Embedding(MAX_REC, d_in)
        self.h_ln = nn.LayerNorm(d_in)
        self.h_out = nn.Sequential(nn.Linear(d_in, D), ScaledLN(D, r))
        # ---- user token: user emb + is_user0 + domain mixture
        self.u_proj = nn.Sequential(nn.Linear(128 + 1 + 3, d_in), nn.LayerNorm(d_in), nn.Linear(d_in, D), ScaledLN(D, r))
        # ---- heads (fp32)
        self.head = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, head_hidden), nn.GELU(), nn.Linear(head_hidden, 1))
        self.aux_head = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, head_hidden), nn.GELU(), nn.Linear(head_hidden, 1))
        self.wide = nn.Linear(n_dense, 1)
        self.infer_domain = 1  # as the ranker: val/test/holdout-without-dom scored as random-like traffic
        self.aux_out = None

    # ------------------------------------------------------------------ trunk (overridden for Qwen3.5)
    LORA_TARGETS = LORA_TARGETS
    left_padded = False  # Qwen3 rows are [sink, user, H, cands, pad]: FeedStore.token_batches (unchanged)

    def _check_cfg(self, hf_cfg):
        assert hf_cfg.model_type == "qwen3", ("Qwen3.5 (hybrid Gated DeltaNet) ignores custom masks in 3/4 of its "
                                              "layers; use qwen35_branch.Qwen35Ranker for it")

    def _load_trunk(self, path, hf_cfg, random_init, tdtype, attn_impl):
        from transformers import AutoModel
        if random_init:  # same architecture, HF init (normal std initializer_range), fp32 master weights
            hf_cfg._attn_implementation = attn_impl
            return AutoModel.from_config(hf_cfg, attn_implementation=attn_impl, dtype=torch.float32)
        return AutoModel.from_pretrained(path, dtype=tdtype, attn_implementation=attn_impl)

    # ------------------------------------------------------------------ helpers
    def set_norm(self, mean, std):
        self.f_mean.copy_(mean)
        self.f_std.copy_(std.clamp_min(1e-3))

    def dense(self, b):
        c = self.cfg
        return dense_features(b, c["n_dense"], c["xt_rel"], 0.0, False, c["xt_idcols"], c["xt_relonly"])

    def param_groups(self):
        """(name, param) lists: 'trunk' = LoRA / RMSNorm gains (or all trunk weights for random init); 'proj' = the rest."""
        trunk, proj = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (trunk if n.startswith("lm.") else proj).append((n, p))
        return trunk, proj

    STATE_BUFFERS = ("f_mean", "f_std", "sink_emb")

    def trainable_state(self):
        """Everything not reproducible from the checkpoint: trainable tensors + feature stats + the S1 sink row."""
        return {n: p.detach().to("cpu", copy=True) for n, p in self.named_parameters() if p.requires_grad} | \
               {n: b.detach().to("cpu", copy=True) for n, b in self.named_buffers() if n in self.STATE_BUFFERS}

    def fp32_copy(self):
        """Amendment I3: the same trained model with fp32 trunk weights (re-loaded from the checkpoint, so frozen
        weights are exact; not an upcast of rounded buffers) and this model's trainable state. Returned on CPU, eval."""
        return fp32_from_state(type(self), self.cfg, self.trainable_state())

    def layout_of(self, Hc, M, Tx, Mx, float_mask_dtype=None, layout=None):
        return build_layout(Hc, M, Tx, layout or self.cfg["layout"], float_mask_dtype, Mx, self.n_pre)

    # ------------------------------------------------------------------ forward
    def tokens(self, b, X):
        """Soft tokens: user [B,D], history [B,K,D] (newest-first, as gathered), candidates [B,Mx,D]; normalised X."""
        Xn = ((X - self.f_mean) / self.f_std).clamp(-6, 6)
        ce, un = b["ce"], b["un"]
        B = ce.shape[0]
        dom = b.get("dom")
        if dom is None:
            dom = F.one_hot(torch.full((B,), self.infer_domain, device=ce.device), 3).float()
        dom = dom.float()
        c = self.c_lin(torch.cat([ce, ce * un[:, None, :]], -1)) + self.cat(b["ccat"]) + self.c_mlp(Xn) \
            + b["cmiss"][..., None].float() * self.c_miss
        c = self.c_ln(c) + (dom @ self.dom.weight)[:, None, :]
        c_tok = self.c_out(c)
        u_tok = self.u_proj(torch.cat([un, b["user0"].float()[:, None], dom], -1))
        he = b["he"]
        K = he.shape[1]
        rec = torch.arange(K, device=he.device).clamp_max(MAX_REC - 1)
        h = self.h_lin(he) + self.et(b["het"]) + self.es(b["hes"]) + self.cat(b["hcat"]) + self.rec(rec)[None] \
            + (~b["hemb_ok"])[..., None].float() * self.h_miss
        h_tok = self.h_out(self.h_ln(h))
        return u_tok, h_tok, c_tok, Xn

    def assemble(self, u_tok, h_tok, c_tok, Hc, M):
        """Scatter tokens into the padded sequence [B,T,D]: [sink,] user, history reversed to old -> new, candidates."""
        B, K, D = h_tok.shape
        Mx = c_tok.shape[1]
        npre = self.n_pre
        Tx = int((npre + Hc + M).max())
        Tx = -(-Tx // 16) * 16  # align for the SDPA kernels; extra pad tokens attend only to themselves
        dev = h_tok.device
        kk = torch.arange(K, device=dev)[None, :].expand(B, K)
        hval = kk < Hc[:, None]
        mm = torch.arange(Mx, device=dev)[None, :].expand(B, Mx)
        cval = mm < M[:, None]
        bh = torch.arange(B, device=dev)[:, None]
        ar = torch.arange(B, device=dev)
        rows = [ar, bh.expand(B, K)[hval], bh.expand(B, Mx)[cval]]
        cols = [torch.full((B,), npre - 1, dtype=torch.long, device=dev), (npre - 1 + Hc[:, None] - kk)[hval],
                (npre + Hc[:, None] + mm)[cval]]
        vals = [u_tok, h_tok[hval], c_tok[cval]]
        if self.sink:  # the same frozen row at position 0 of every feed
            rows.append(ar)
            cols.append(torch.zeros(B, dtype=torch.long, device=dev))
            vals.append(self.sink_emb[None].expand(B, D))
        rows, cols = torch.cat(rows), torch.cat(cols)
        vals = torch.cat([v.to(h_tok.dtype) for v in vals])
        seq = torch.zeros(B, Tx, D, dtype=vals.dtype, device=dev).index_put((rows, cols), vals)
        return seq, Tx

    def forward(self, b, X=None):
        if X is None:
            X = self.dense(b)
        u_tok, h_tok, c_tok, Xn = self.tokens(b, X)
        Hc, M = b["H"], b["M"]  # H already clamped to max_hist by gather()
        seq, Tx = self.assemble(u_tok, h_tok, c_tok, Hc, M)
        fdt = None if self.attn_impl in ("sdpa", "flex_attention") else torch.float32
        Mx = c_tok.shape[1]
        pos_ids, mask, cand_tok, _, _ = self.layout_of(Hc, M, Tx, Mx, fdt)
        if fdt is not None and seq.dtype != torch.float32:
            mask = mask.to(seq.dtype)
        out = self.lm(inputs_embeds=seq, attention_mask={"full_attention": mask}, position_ids=pos_ids).last_hidden_state
        hc = out.gather(1, cand_tok[..., None].expand(-1, -1, out.shape[-1]))
        with torch.autocast(device_type=hc.device.type, enabled=False):
            hc = hc.float()
            logit = self.head(hc).squeeze(-1) + self.wide(Xn.float()).squeeze(-1)
            self.aux_out = self.aux_head(hc).squeeze(-1) if self.training else None
        return logit


QWEN3_SPECIAL = {"<|endoftext|>": 151643, "<|im_start|>": 151644, "<|im_end|>": 151645}


def sink_token_id(path, hf_cfg, token="<|im_start|>"):
    """Id of a (special) token in the checkpoint's own local tokenizer; Qwen3 ids are asserted against QWEN3_SPECIAL."""
    from transformers import AutoTokenizer
    token = token if token.startswith("<|") else f"<|{token}|>"  # shell-safe aliases: endoftext, im_start, im_end
    tk = AutoTokenizer.from_pretrained(path)
    tid = tk.convert_tokens_to_ids(token)
    vocab = getattr(hf_cfg, "text_config", hf_cfg).vocab_size
    assert isinstance(tid, int) and 0 <= tid < vocab and tid != tk.unk_token_id, (path, token, tid)
    if hf_cfg.model_type == "qwen3" and token in QWEN3_SPECIAL:
        assert tid == QWEN3_SPECIAL[token], (token, tid)
    return tid


def fp32_from_state(cls, cfg, state):
    """Rebuild a ranker of class `cls` from `cfg` with fp32 trunk weights and load a trainable_state() into it."""
    m = cls(**{**cfg, "dtype": "fp32"})
    res = m.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in state.items()}, strict=False)
    assert not res.unexpected_keys, res.unexpected_keys
    trained = {n for n, p in m.named_parameters() if p.requires_grad} | set(cls.STATE_BUFFERS)
    assert not trained & set(res.missing_keys), sorted(trained & set(res.missing_keys))[:5]
    return m.eval()


def count_params(model):
    tot = sum(p.numel() for p in model.parameters())
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora = sum(p.numel() for n, p in model.named_parameters() if "lora_" in n)
    body = sum(p.numel() for n, p in model.lm.named_parameters() if "lora_" not in n and "embed_tokens" not in n)
    return dict(total=tot, trainable=tr, lora=lora, body=body)


def lr_lambda_cosine(warmup, total, floor=0.1):
    def f(s):
        if s < warmup:
            return (s + 1) / warmup
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warmup) / max(1, total - warmup))))
    return f
