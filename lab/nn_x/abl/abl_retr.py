"""nn_x/abl second-wave arm retr_cat (reports/ABLATION_PLAN_1113.md section 3.3 row 2, pre-declared in
nn_x/abl/ledger/ledger.md section 6): a SECOND HistRetrieval over the candidate's SAME-CATEGORY history items, added to
the candidate token at the very point where the base retrieval (retr_m 16 + attn_gate) is added, with a zero-initialised
output projection Z (exact no-op at init). OFF at the train.py default `--retr_cat 0`; train.py calls setup_retr_cat only
behind `if a.retr_cat:` (tests/test_abl.py: AST proof + end-to-end identity), and model.py stays byte-identical to nn_x/w5.

Mechanism (HistRetrievalCat = model.HistRetrieval with the same parameters, the same init and the inherited forward):
    same[b,i,j] = (ccat[b,i] == hcat[b,j]) & hv[b,j]      hv = the base retrieval's valid-event mask (b['hemb_ok'])
    S_cat       = S.masked_fill(~same, -2.0)              the inherited forward masks ~hv to -2.0 as well, takes the top-m
                                                          and keeps only scores > -1.5, so ONLY same-category items are
                                                          valid; the others get logit -inf (p exactly 0)
    out         = Z(sum_j p_j v_j) * has[b,i]             has = the candidate has an embedding and >= 1 same-category valid
                                                          item; otherwise the module outputs exactly 0 (not Z.bias)
  logit = -||k_c - k_j||^2 + a S_cj + beta[type_j] + sink; values LN(E_type + E_src + E_rec + T(k_c - k_j)); m slots.
  The dropout inside the value MLP draws from a DEDICATED generator (GenDropout, seeded from torch.initial_seed()), so the
  global RNG stream of the other dropout sites is exactly the flag-off stream: at init (Z = 0) flag-ON is bit-identical
  to flag-OFF in train mode as well as in eval mode. The parameters are created under model.extra_init (forked CPU RNG)
  AFTER the Ranker, so every baseline parameter keeps its init; train.py builds the optimisers after setup_retr_cat, so
  they are optimised exactly like the base retrieval's (hidden 2-D Linear weights -> Muon, the rest AdamW).
Wiring (no model.py change): model.retr_cat = the module; a forward pre-hook on the Ranker stashes the batch dict (for
  ccat / hcat), a forward hook on model.retr adds retr_cat(...) to the base retrieval's output (same positional args).
Checkpoints: the state_dict gains the retr_cat.* keys and cfg['retr_cat'] = m. Ranker(**cfg) then fails loudly on the
  unknown key: load such a checkpoint with load_ranker(ck, dev) (f_replay.py / the tests do).
"""
import torch
import torch.nn as nn

from model import HistRetrieval, Ranker, extra_init

CAT_OFFSET = 15485863  # extra_init seed offset of the retr_cat parameters (distinct from the r6 cretr offset 104729)


class GenDropout(nn.Module):
    """nn.Dropout semantics (Bernoulli(1-p) keep mask scaled by 1/(1-p) in training, identity in eval) drawn from a
    dedicated generator, so the module never advances the global (CPU or CUDA) RNG stream."""

    def __init__(self, p, seed):
        super().__init__()
        self.p, self.seed, self.gen = float(p), int(seed), None

    def forward(self, x):
        if not self.training or self.p <= 0:
            return x
        if self.gen is None or self.gen.device != x.device:
            self.gen = torch.Generator(device=x.device)
            self.gen.manual_seed(self.seed)
        keep = torch.rand(x.shape, generator=self.gen, device=x.device, dtype=torch.float32) >= self.p
        return x * keep.to(x.dtype) / (1.0 - self.p)

    def extra_repr(self):
        return f"p={self.p}, dedicated generator"


class HistRetrievalCat(HistRetrieval):
    """HistRetrieval restricted to the candidate's same-category history items (see the module docstring)."""

    def __init__(self, d, m=16, drop=0.1, seed=0):
        super().__init__(d, m, drop=drop)
        self.dp = GenDropout(drop, seed)  # replaces nn.Dropout (no parameters, no RNG draw at construction)

    def forward(self, ce, he, hv, het, hes, S, cmiss, same):  # noqa: D102 (same, [B,M,K] bool: same category & valid)
        has = same.any(-1) & ~cmiss  # [B,M]: at least one same-category valid item and the candidate has an embedding
        out = super().forward(ce, he, hv, het, hes, S.masked_fill(~same, -2.0), cmiss)
        return out * has[..., None].to(out.dtype)


def same_cat_mask(b, hv):
    """[B,M,K] bool: history event j is valid (hv) and has the category of candidate i."""
    return (b["ccat"][:, :, None] == b["hcat"][:, None, :]) & hv[:, None, :]


def attach_retr_cat(model, m):
    """Create model.retr_cat (m slots) and wire it behind model.retr; records cfg['retr_cat'] = m. Returns the model."""
    assert model.retr is not None, "retr_cat needs the base retrieval (retr_m > 0 with the history encoder)"
    assert not getattr(model, "retr_uniq", False), "retr_cat: base retrieval over all events only (not --retr_uniq)"
    assert not hasattr(model, "retr_cat"), "retr_cat already attached"
    assert int(m) > 0
    dev = model.retr.Z.weight.device
    # extra_init forks only the CPU RNG, but its torch.manual_seed also re-seeds the CUDA generator (not restored): wrap it
    # in a fork that includes the model's CUDA device, so the dropout stream of the following training step is exactly the
    # flag-off stream (GPU test G2: first-batch logits bit-identical). Init stays deterministic per seed.
    devs = [dev.index if dev.index is not None else torch.cuda.current_device()] if dev.type == "cuda" else []
    with torch.random.fork_rng(devices=devs):
        with extra_init(offset=CAT_OFFSET):
            cat = HistRetrievalCat(model.cfg["d"], int(m), drop=model.cfg["drop"], seed=torch.initial_seed() + CAT_OFFSET + 1)
    model.retr_cat = cat.to(dev)
    model.cfg["retr_cat"] = int(m)

    def stash(mod, args):  # Ranker forward pre-hook: the batch dict (ccat / hcat) for the retr hook below
        mod.retr_cat._b = args[0]

    def add(mod, args, out):  # model.retr forward hook: out + retr_cat(...) at the same point of Ranker.forward
        b, cat._b = cat._b, None
        ce, he, hv, het, hes, S, cmiss = args[:7]
        return out + cat(ce, he, hv, het, hes, S, cmiss, same_cat_mask(b, hv))

    cat._b = None
    model.register_forward_pre_hook(stash)
    model.retr.register_forward_hook(add)
    return model


def setup_retr_cat(model, a, log):
    """train.py hook (called only behind `if a.retr_cat:`, after the Ranker is built and normalised, before the optimisers)."""
    assert a.retr_m > 0 and not a.no_hist, "--retr_cat needs the base retrieval (--retr_m > 0, history encoder on)"
    assert not a.retr_uniq, "--retr_cat: not combined with --retr_uniq (the base retrieval must see every event)"
    attach_retr_cat(model, a.retr_cat)
    n = sum(p.numel() for p in model.retr_cat.parameters())
    log(f"abl retr_cat: second HistRetrieval over the candidate's SAME-CATEGORY history items, m {a.retr_cat}, zero-init Z "
        f"(exact no-op at init; dedicated dropout generator), {n / 1e6:.3f}M params; cfg retr_cat={a.retr_cat}")
    return model


def load_ranker(ck, dev):
    """Ranker from a checkpoint dict, with retr_cat re-attached when cfg['retr_cat'] is set (Ranker(**cfg) would fail)."""
    cfg = dict(ck["cfg"])
    m = cfg.pop("retr_cat", 0)
    model = Ranker(**cfg).to(dev)
    if m:
        attach_retr_cat(model, m)
    model.load_state_dict(ck["state_dict"])
    return model
