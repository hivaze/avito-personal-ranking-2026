"""ENCODER-DOSE (C) step 2: a ZERO-INIT GATED K/V adapter over the frozen encoder's 64-d states for the ranker's 2 cross-attention
blocks (ledger nn_x/abl/ledger/ledger.md section 10.C; SOLVER_POWER_1113 section 6.3 item 2 (C)). nn_x/abl/model.py and data.py
stay untouched (the abl_retr.py pattern: hooks); everything here is OFF unless setup_kv64() is called.

Mechanism. Ranker.forward builds the cross-attention context ctx = [user token; history slots] ([B, 1 + K, d], newest-first history)
and calls crosses[i](c, kmask, ctx, sbias) for the 2 cross blocks; Block.forward applies LayerNorm(ctx) and the (split) k / v
projections. The adapter adds, per block i, a state term to ctx BEFORE that block:
    ctx_i = ctx + tanh(g_i) * A_i(LN64(s))        s [B, 1 + K, 64]: the 64-d PCA state of each history slot (slot k of the ranker =
                                                   raw row hist_off + k, newest-first; the export stores exactly that order), the
                                                   learned USER-slot vector u0 for position 0, and the learned NULL token for slots of
                                                   feeds without exported states (kv_off -1: train rows on or before Sep 11) and for
                                                   padded slots.
    g_i = 0 at init  ->  tanh(0) = 0  ->  ctx_i == ctx bit-for-bit (0.0 * finite == 0.0; x + 0.0 == x), so the ranker at init is
    BIT-IDENTICAL to the base; with the adapter absent (flag off) no hook exists and the code path is the base's.
The parameters (A_i: Linear(64, d) per block, g_i scalar per block, u0 and null 64-d) are created under model.extra_init (a forked
RNG) AFTER the Ranker, so the base parameters and the training-batch sequence are unchanged (the retr_cat precedent).

Wiring (setup_kv64): a forward pre-hook on the Ranker stashes the batch's "kv" [B, K, 64] fp16 / "kv_ok" [B] bool; a forward
pre-hook (with_kwargs) on each cross Block replaces its ctx argument by ctx + adapter(i, ...). The batch keys come from
kv_data.attach_kv64(sd) (a gather wrapper over SplitData; see kv_data.py).

Pass rule (ledger 10.C, verbatim from SOLVER_POWER 6.3): GO only if val >= +0.0015 with holdout > 0 and stale-val / pmix vetoes >= -1 SE.
"""
import torch
import torch.nn as nn

from model import extra_init  # nn_x/abl/model.py (read-only import)

KV_OFFSET = 7368787  # extra_init seed offset of the adapter parameters (distinct from retr_cat 15485863 and cretr 104729)
D_STATE = 64


class KVAdapter(nn.Module):
    def __init__(self, d, n_blocks=2, d_state=D_STATE, seed=KV_OFFSET):
        super().__init__()
        with extra_init(seed):
            self.ln = nn.LayerNorm(d_state)
            self.proj = nn.ModuleList([nn.Linear(d_state, d) for _ in range(n_blocks)])
            self.gate = nn.Parameter(torch.zeros(n_blocks))  # tanh(0) = 0: exact no-op at init
            self.user0 = nn.Parameter(torch.zeros(d_state))
            self.null = nn.Parameter(torch.zeros(d_state))
        self.n_blocks, self.d_state = n_blocks, d_state

    def states(self, kv, kv_ok, hmask):
        """kv [B, K, 64] (any float dtype), kv_ok [B] bool, hmask [B, K] bool -> s [B, 1 + K, 64] fp32 with u0 / null filled in."""
        B, K, _ = kv.shape
        use = (kv_ok[:, None] & hmask)[..., None]
        s = torch.where(use, kv.float(), self.null[None, None, :].expand(B, K, -1))
        return torch.cat([self.user0[None, None, :].expand(B, 1, -1), s], 1)

    def forward(self, i, kv, kv_ok, hmask):
        s = self.states(kv, kv_ok, hmask)
        return torch.tanh(self.gate[i]) * self.proj[i](self.ln(s))


def setup_kv64(model, log=print):
    """Attach the adapter to a Ranker (after its construction) and wire the hooks. Returns the model."""
    assert getattr(model, "kv_adapter", None) is None, "kv64 adapter already attached"
    assert model.use_hist and model.arch != "din", "--kv64 needs the history cross-attention blocks"
    d = model.cross[0].ln1.normalized_shape[0]
    dev = model.cross[0].ln1.weight.device  # the adapter is built AFTER Ranker(...).to(dev): put it on the model's device
    model.kv_adapter = KVAdapter(d, n_blocks=len(model.cross)).to(dev)
    model.cfg["kv64"] = 1
    stash = {}

    def pre_ranker(mod, args, kwargs):
        b = args[0] if args else kwargs["b"]
        stash["kv"], stash["kv_ok"], stash["hmask"] = b["kv"], b["kv_ok"], b["hmask"]

    def make_pre_block(i):
        def pre_block(mod, args, kwargs):
            args = list(args)
            ctx = args[2] if len(args) > 2 else kwargs.get("ctx")
            if ctx is None:
                return None
            add = model.kv_adapter(i, stash["kv"], stash["kv_ok"], stash["hmask"]).to(ctx.dtype)
            if len(args) > 2:
                args[2] = ctx + add
            else:
                kwargs["ctx"] = ctx + add
            return tuple(args), kwargs
        return pre_block

    model._kv_hooks = [model.register_forward_pre_hook(pre_ranker, with_kwargs=True)]
    for i, blk in enumerate(model.cross):
        model._kv_hooks.append(blk.register_forward_pre_hook(make_pre_block(i), with_kwargs=True))
    n = sum(p.numel() for p in model.kv_adapter.parameters())
    log(f"kv64: zero-init gated K/V adapter attached to {len(model.cross)} cross blocks ({n} parameters; gates {model.kv_adapter.gate.tolist()}; "
        f"bit-identical to the base at init)")
    return model


def remove_kv64(model):
    for h in getattr(model, "_kv_hooks", []):
        h.remove()
    model._kv_hooks = []
    if hasattr(model, "kv_adapter"):
        del model.kv_adapter
    model.cfg.pop("kv64", None)
    return model


def load_ranker_kv64(ck, dev="cpu"):
    """Ranker from a checkpoint dict saved by a --kv64 run: cfg['kv64'] is popped, the adapter re-attached (hooks included) and the
    full state_dict (base + kv_adapter.*) loaded. Readers of such checkpoints (holdout_eval / stale_pred / finetune) must also attach
    the states to their SplitData (kv_data.attach_kv64) because every batch needs b['kv'] / b['kv_ok']."""
    from model import Ranker
    cfg = dict(ck["cfg"])
    kv = cfg.pop("kv64", 0)
    model = Ranker(**cfg).to(dev)
    if kv:
        setup_kv64(model, log=lambda *a: None)
        model.to(dev)
    model.load_state_dict(ck["state_dict"])
    return model
