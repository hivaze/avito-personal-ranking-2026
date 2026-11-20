"""Unit tests for the nn_x/r6 arms (reports/r6_design.md). <= 12 threads, read-only on all shared data.

  CUDA_VISIBLE_DEVICES=N OMP_NUM_THREADS=12 <repo>/.venv/bin/python nn_x/r6/tests/test_r6.py   # GPU
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 <repo>/.venv/bin/python nn_x/r6/tests/test_r6.py  # CPU
GPU mode (2026-11-12 wave-2 build; user rule: NN work on GPUs): same checks on cuda:0 with deterministic kernels (math
SDPA backend, no TF32, torch deterministic algorithms), so the bit-identity checks stay exact. Models are still BUILT on
the CPU (T1 RNG checks) and moved to the device afterwards.

Checks, per arm (base = the sx_base_s23 cfg = current recipe, cvnd_pre2s, d128/4h/2 blocks, retr16 + gate, split k/v):
  T1 init: every baseline parameter keeps exactly its baseline init (same seed) and the global RNG state after model
     construction is unchanged (forked RNG for new modules); flag-off cfg == nn/ cfg (nn/model.py can load it).
  T2 flag-off equality vs nn/model.py on a real train batch with the trained sx_base_s23 weights: eval logits, train-mode
     loss (listwise feed-norm + K9 aux, same dropout seed) and every gradient bit-identical.
  T3 zero-init no-op at warm start (a4w, cretr part of cset) / size of the change for the non-zero-init arms (info).
  T4 shapes and masks: valid logits do not depend on padded candidates or padded history events.
  T5 gradient flow into every new parameter (after a small perturbation of the zero-init output layers), finite grads.
  T6 permutation equivariance within a feed (real batch; and a tie-free jittered batch for the top-k arms).
  T7 no cross-feed leakage: perturbing every input of feed 0 leaves every other feed's logits unchanged.
  T8 parameter counts.  T9 CPU throughput (fwd+bwd) relative to the baseline.
  T10 history_dedup on a hand example; a4u retrieves distinct items only (+ duplicate share of the baseline top-16).
  T11 typed_cross: the last cross block gets kmask & (et==1 & es==1); earlier blocks the full mask.
  T12 CLI: train.py's parser with the sx_base command line leaves every r6 flag at its OFF default.
Writes nn_x/r6/logs/test_r6.log (+ .json).
"""
import ast
import argparse
import copy
import importlib.util
import json
import os
import sys
import time

_CVD = os.environ.get("CUDA_VISIBLE_DEVICES", "x")
assert _CVD == "" or _CVD in tuple(str(i) for i in range(8)), \
    "run with CUDA_VISIBLE_DEVICES='' (CPU) or exactly one GPU id (GPU mode)"
if _CVD:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # deterministic cuBLAS (before CUDA init)
NT = min(12, int(os.environ.get("OMP_NUM_THREADS", "12")))
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(NT)
DEV = torch.device("cuda:0" if _CVD else "cpu")
if DEV.type == "cuda":  # exact (bit-identity) checks need deterministic kernels on the GPU
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)  # -> math SDPA (deterministic forward/backward, same dropout RNG use)
HERE = os.path.dirname(os.path.abspath(__file__))
R6 = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(1, R6)
import warnings  # noqa: E402

warnings.filterwarnings("ignore", message=".*CUDA is not available.*")
warnings.filterwarnings("ignore", message=".*device_type of 'cuda'.*")
import model as M6  # noqa: E402  (nn_x/r6/model.py)
from mini_split import MiniSplit, pick_train_feeds  # noqa: E402

ROOT = "<repo>"
_spec = importlib.util.spec_from_file_location("ref_model", f"{ROOT}/nn/model.py")
MREF = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(MREF)  # the live nn/model.py, read-only (it imports only torch)

BASE_CK = f"{ROOT}/nn_x/solvers/ckpt/sx_base_s23.pt"  # the 2-epoch screen control (same args as the r6 queue)
ARMS = {
    "wd01": {},  # optimizer-only arm (--wd 0.1): no model change
    "adrop0": {"attn_drop": 0.0},
    "a4u": {"retr_uniq": True},
    "a4w": {"retr_m": 32, "retr_prod": True},
    "cset": {"cretr_m": 8, "set_sim_bias": True},
    "typed": {"typed_cross": True},
}
LOG = open(f"{R6}/logs/test_r6.log", "w")
RES = {}
FAIL = []


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + "\n")
    LOG.flush()


def check(name, ok, info=""):
    P(f"  [{'PASS' if ok else 'FAIL'}] {name} {info}")
    RES.setdefault("checks", []).append(dict(name=name, ok=bool(ok), info=str(info)))
    if not ok:
        FAIL.append(name)


def train_loss(Mod, model, b, prand_b, aux_w=0.5, tau=10.0, denom=512):
    """Exactly the nn/train.py loss for the current recipe (softmax, loss_norm feed, no prand_base, K9 listwise aux
    weighted by 1 - p_random; nn/train.py:282-307)."""
    X = model.dense(b)
    lg = model(b, X, use_pos=True)
    y, m = b["y"], b["cmask"]
    loss = Mod.listwise_loss(lg, y, m, None, norm="feed", denom=denom)
    tgt = torch.softmax((-b["pos"].float() / tau).masked_fill(~m, -1e4), 1) * m
    lp = torch.log_softmax(model.aux_out.masked_fill(~m, -1e4), 1)
    fw_aux = 1 - prand_b
    loss = loss + aux_w * (-(tgt * lp).sum(1) * fw_aux).sum() / fw_aux.sum().clamp_min(1e-6)
    return loss, lg


def fresh(b):
    """Deep copy of a gathered batch (model.dense writes ce/he/un into the dict)."""
    return {k: (v.clone() if torch.is_tensor(v) else copy.deepcopy(v)) for k, v in b.items()
            if k not in ("ce", "he", "un")}


def build(Mod, cfg, seed=23):
    torch.manual_seed(seed)
    return Mod.Ranker(**cfg)


def perturb_zero_init(model, scale=0.02, seed=5):
    """Make zero-init output layers non-zero so gradients reach every parameter upstream of them."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if any(n.startswith(k) for k in ("cretr.Z.", "retr.t3.", "retr.e_cnt.")) and p.abs().max() == 0:
                p.add_((torch.randn(p.shape, generator=g) * scale).to(p.device))


def main():
    t_all = time.time()
    ck = torch.load(BASE_CK, map_location="cpu", weights_only=False)
    base_cfg = dict(ck["cfg"])
    sd = ck["state_dict"]
    P(f"== nn_x/r6 unit tests  ({time.ctime()}, device {DEV}, CPU threads {NT})")
    P(f"base cfg from {BASE_CK}: d {base_cfg['d']} heads {base_cfg['heads']} blocks {base_cfg['blocks']} "
      f"hist_layers {base_cfg['hist_layers']} n_dense {base_cfg['n_dense']} retr_m {base_cfg['retr_m']} "
      f"gate {base_cfg['attn_gate']} split_kv {base_cfg['split_kv']} sim_bias {base_cfg['sim_bias']}")

    # ------------------------------------------------------------------ T12 CLI defaults (static, no GPU, no common.py)
    P("\n== T12 train.py CLI: sx_base command line -> every r6 flag OFF")
    src = open(f"{R6}/train.py").read()
    tree = ast.parse(src)
    fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "parse"][0]
    import cd_arms  # train.py's --pool_p default (wave-2 flags, tests/test_cdw2.py)
    ns = {"argparse": argparse, "POOL_P_DEFAULT": cd_arms.POOL_P_DEFAULT}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "train_parse", "exec"), ns)
    sx_cmd = ("--variant r6_x --no_pos --aux_pos 0.5 --aux_mode listwise --aux_tau 10 --sim_bias --user0 --domain "
              "--holdout 30000 --select_on holdout --loss_norm feed --xt cvnd_pre2s --xt_drop 0 --muon --lr 1e-3 "
              "--sched wsd --cooldown_frac 0.2 --cooldown_mix 0.1 --aux_anneal 0.1 --retr_m 16 --attn_gate --epochs 2 "
              "--eval_every 1000 --seed 23").split()
    sys.argv = ["train.py"] + sx_cmd
    a = ns["parse"]()
    off = dict(attn_drop=a.attn_drop, retr_prod=a.retr_prod, retr_uniq=a.retr_uniq, cretr_m=a.cretr_m,
               set_sim_bias=a.set_sim_bias, typed_cross=a.typed_cross, drop_cd=a.drop_cd, pool_p=a.pool_p,
               aug_cand=a.aug_cand, aug_hist=a.aug_hist)
    check("r6 flags at OFF defaults", off == dict(attn_drop=-1.0, retr_prod=False, retr_uniq=False, cretr_m=0,
                                                   set_sim_bias=False, typed_cross=False, drop_cd=-1.0, pool_p=0.5,
                                                   aug_cand=0.0, aug_hist=0.0), off)
    sxa = ck["args"]
    same = {k: (getattr(a, k), sxa[k]) for k in sxa if k not in ("variant",) and hasattr(a, k) and getattr(a, k) != sxa[k]}
    check("all sx_base_s23 args reproduced by the r6 command line", not same, same or "identical")
    extra = sorted(set(vars(a)) - set(sxa))
    P(f"  r6-only arg keys (all OFF): {extra}")

    # ------------------------------------------------------------------ T1 init equality + RNG
    P("\n== T1 init: baseline parameters and global RNG stream unchanged by every arm")
    ref = build(MREF, base_cfg)
    rng_ref = torch.get_rng_state()
    r6off = build(M6, base_cfg)
    check("flag-off r6 cfg == nn/ cfg", r6off.cfg == ref.cfg)
    check("flag-off r6 state_dict == nn/ (keys and values)",
          list(r6off.state_dict()) == list(ref.state_dict())
          and all(torch.equal(r6off.state_dict()[k], v) for k, v in ref.state_dict().items()))
    check("flag-off global RNG state after construction", torch.equal(torch.get_rng_state(), rng_ref))
    RES["params"] = {"base": sum(p.numel() for p in ref.parameters())}
    for arm, delta in ARMS.items():
        cfg = {**base_cfg, **delta}
        mdl = build(M6, cfg)
        same_rng = torch.equal(torch.get_rng_state(), rng_ref)
        rsd, asd = ref.state_dict(), mdl.state_dict()
        shared = [k for k in rsd if k in asd and rsd[k].shape == asd[k].shape]
        changed = [k for k in shared if not torch.equal(rsd[k], asd[k])]
        new = [k for k in asd if k not in rsd or asd[k].shape != rsd[k].shape]
        missing = [k for k in rsd if k not in asd]
        n_new = sum(asd[k].numel() for k in new)
        RES["params"][arm] = sum(p.numel() for p in mdl.parameters())
        ok = (not changed and not missing and same_rng) if "retr_m" not in delta else (same_rng and not missing)
        info = f"changed {len(changed)} missing {len(missing)} new {len(new)} ({n_new} params) rng_same {same_rng}"
        if "retr_m" in delta:  # retr_m 32 keeps every tensor shape; report which differ (none expected: m is no param)
            info += f" | retr_m 32: differing shared tensors {changed[:3]}"
            ok = ok and not changed
        check(f"{arm}: baseline init unchanged", ok, info)
        P(f"     new keys: {new[:8]}{' ...' if len(new) > 8 else ''}")
        check(f"{arm}: cfg stores only non-default r6 keys", set(mdl.cfg) - set(ref.cfg) == set(delta) - {"retr_m"},
              sorted(set(mdl.cfg) - set(ref.cfg)))

    # ------------------------------------------------------------------ real data
    P("\n== real train batch (mini split, memory-mapped; read-only)")
    t0 = time.time()
    feeds = pick_train_feeds(n_each=48, seed=0)
    ms = MiniSplit("train", feeds, xt_names=base_cfg["xt_names"]).to(DEV)
    P(f"  {ms.n} feeds, {ms.n_cand} cands, {ms.n_hist} hist rows, table {tuple(ms.table.shape)}, "
      f"M {ms.M_np.min()}-{ms.M_np.max()}, H==0 feeds {(ms.H_np == 0).sum()}, user0 {int(ms.user0.sum())}, "
      f"{time.time() - t0:.1f}s")
    # two real training-shaped batches: short feeds (M<=64) and long feeds (M>=150)
    idx_s = np.where(ms.M_np <= 64)[0][:96]
    idx_l = np.where(ms.M_np >= 150)[0][:40]
    bs = ms.gather(idx_s, with_pos=True)
    bl = ms.gather(idx_l, with_pos=True)
    P(f"  batch S: B {bs['cmask'].shape[0]} Mx {bs['cmask'].shape[1]} Hx {bs['hmask'].shape[1]};  "
      f"batch L: B {bl['cmask'].shape[0]} Mx {bl['cmask'].shape[1]} Hx {bl['hmask'].shape[1]}")
    pr_s, pr_l = ms.prand[bs["fidx"]], ms.prand[bl["fidx"]]

    # ------------------------------------------------------------------ T2 flag-off equality with trained weights
    P("\n== T2 flag-off r6 vs nn/model.py, trained sx_base_s23 weights, real batches")
    ref.load_state_dict(sd)
    r6off.load_state_dict(sd)
    ref.to(DEV), r6off.to(DEV)
    for nm, b, prb in (("S", bs, pr_s), ("L", bl, pr_l)):
        ref.eval(), r6off.eval()
        with torch.no_grad():
            l_ref, l_r6 = ref(fresh(b)), r6off(fresh(b))
        check(f"eval logits identical (batch {nm})", torch.equal(l_ref, l_r6),
              f"max|d| {float((l_ref - l_r6).abs().max()):.2e}")
        ref.train(), r6off.train()
        torch.manual_seed(77)
        lo_ref, _ = train_loss(MREF, ref, fresh(b), prb)
        ref.zero_grad(set_to_none=True)
        lo_ref.backward()
        torch.manual_seed(77)
        lo_r6, _ = train_loss(M6, r6off, fresh(b), prb)
        r6off.zero_grad(set_to_none=True)
        lo_r6.backward()
        g_ok = all((p1.grad is None and p2.grad is None) or torch.equal(p1.grad, p2.grad)
                   for p1, p2 in zip(ref.parameters(), r6off.parameters()))
        check(f"train-mode loss identical (batch {nm}, dropout on, same seed)", torch.equal(lo_ref, lo_r6),
              f"{lo_ref.item():.6f} vs {lo_r6.item():.6f}")
        check(f"all gradients identical (batch {nm})", g_ok)
    ref.eval()

    # ------------------------------------------------------------------ per-arm model tests
    with torch.no_grad():
        base_logits = {nm: ref(fresh(b)) for nm, b in (("S", bs), ("L", bl))}
    arm_models = {}
    for arm, delta in ARMS.items():
        if not delta:
            continue
        P(f"\n== arm {arm} {delta}")
        cfg = {**base_cfg, **delta}
        mdl = build(M6, cfg)
        missing, unexpected = mdl.load_state_dict(sd, strict=False)
        mdl.to(DEV)
        check(f"{arm}: warm start loads every baseline tensor", not unexpected,
              f"new (not in ckpt): {len(missing)}")
        mdl.eval()
        # T3 warm-start no-op / size of change
        with torch.no_grad():
            lg = {nm: mdl(fresh(b)) for nm, b in (("S", bs), ("L", bl))}
        for nm in ("S", "L"):
            d = (lg[nm] - base_logits[nm])[(bs if nm == "S" else bl)["cmask"]]
            if arm == "adrop0":
                check(f"{arm}: warm start is an exact no-op in eval (batch {nm})", float(d.abs().max()) == 0.0,
                      f"max|d| {float(d.abs().max()):.2e}")
            else:
                P(f"  info {arm}: warm-start eval logit change (batch {nm}) mean|d| {float(d.abs().mean()):.4f} "
                  f"max|d| {float(d.abs().max()):.4f}")
        if arm == "a4w":  # the product half alone is an exact no-op (t3 zero-init); retr_m 32 changes the retrieval set
            m2 = build(M6, {**base_cfg, "retr_prod": True})
            m2.load_state_dict(sd, strict=False)
            m2.to(DEV)
            m2.eval()
            with torch.no_grad():
                d = (m2(fresh(bs)) - base_logits["S"])[bs["cmask"]]
            check("a4w: retr_prod alone is an exact no-op at warm start", float(d.abs().max()) == 0.0,
                  f"max|d| {float(d.abs().max()):.2e}")
        if arm == "cset":  # the cretr half alone is an exact no-op (Z zero-init); the set bias half is not
            m2 = build(M6, {**base_cfg, "cretr_m": 8})
            m2.load_state_dict(sd, strict=False)
            m2.to(DEV)
            m2.eval()
            with torch.no_grad():
                d = (m2(fresh(bs)) - base_logits["S"])[bs["cmask"]]
            check("cset: cretr alone is an exact no-op at warm start", float(d.abs().max()) == 0.0,
                  f"max|d| {float(d.abs().max()):.2e}")
        # T4 shapes and masks
        b = fresh(bl)
        with torch.no_grad():
            out = mdl(fresh(b))
        check(f"{arm}: logits shape [B, Mx] and finite", tuple(out.shape) == tuple(b["cmask"].shape)
              and bool(torch.isfinite(out[b["cmask"]]).all()), tuple(out.shape))
        bp = fresh(bl)
        g = torch.Generator().manual_seed(11)
        pc = ~bp["cmask"]
        bp["ce_raw"][pc] = (torch.randn(int(pc.sum()), 128, generator=g) * 50).to(DEV)
        bp["cf"][pc] = (torch.randn(int(pc.sum()), 5, generator=g) * 9).to(DEV)
        bp["ccat"][pc] = torch.randint(0, 54, (int(pc.sum()),), generator=g).to(DEV)
        if "xt" in bp:
            bp["xt"][pc] = (torch.randn(int(pc.sum()), bp["xt"].shape[-1], generator=g) * 3).to(DEV, bp["xt"].dtype)
        bp["cmiss"][pc] = (torch.rand(int(pc.sum()), generator=g) < 0.5).to(DEV)
        ph = ~bp["hmask"]
        nph = int(ph.sum())
        bp["he_raw"][ph] = (torch.randn(nph, 128, generator=g) * 50).to(DEV)
        bp["het"][ph] = torch.randint(1, 11, (nph,), generator=g).to(DEV)
        bp["hes"][ph] = torch.randint(0, 30, (nph,), generator=g).to(DEV)
        bp["hcat"][ph] = torch.randint(0, 54, (nph,), generator=g).to(DEV)
        bp["hidx"][ph] = torch.randint(0, 10, (nph,), generator=g).to(DEV, bp["hidx"].dtype)
        with torch.no_grad():
            out_p = mdl(bp)
        dm = float((out_p - out)[b["cmask"]].abs().max())
        check(f"{arm}: valid logits independent of padded candidates/history", dm == 0.0, f"max|d| {dm:.2e}")
        # T7 cross-feed leakage
        bx = fresh(bl)
        bx["ce_raw"][0] = torch.randn_like(bx["ce_raw"][0]) * 30
        bx["cf"][0] = torch.randn_like(bx["cf"][0]) * 5
        bx["ccat"][0] = torch.randint(0, 54, bx["ccat"][0].shape, generator=g).to(DEV)
        if "xt" in bx:
            bx["xt"][0] = torch.randn_like(bx["xt"][0])
        bx["he_raw"][0] = torch.randn_like(bx["he_raw"][0]) * 30
        bx["het"][0] = torch.randint(1, 11, bx["het"][0].shape, generator=g).to(DEV)
        bx["hes"][0] = torch.ones_like(bx["hes"][0])
        bx["hidx"][0] = torch.randint(0, 5, bx["hidx"][0].shape, generator=g).to(DEV, bx["hidx"].dtype)
        bx["ue_raw"][0] = torch.randn_like(bx["ue_raw"][0]) * 30
        with torch.no_grad():
            out_x = mdl(bx)
        d0 = float((out_x[0] - out[0])[b["cmask"][0]].abs().max())
        dr = float((out_x[1:] - out[1:])[b["cmask"][1:]].abs().max())
        check(f"{arm}: no cross-feed leakage (feed 0 perturbed)", dr == 0.0 and d0 > 0,
              f"other feeds max|d| {dr:.2e}; feed 0 max|d| {d0:.2e}")
        # T6 permutation equivariance within feeds
        for tag, jitter in (("real", 0.0), ("jittered", 1e-3)):
            b0 = fresh(bl)
            if jitter:  # break exact embedding ties (duplicate items) that make top-k order arbitrary
                gj = torch.Generator().manual_seed(3)
                b0["ce_raw"] = b0["ce_raw"] + torch.randn(b0["ce_raw"].shape, generator=gj).to(DEV) * jitter * (~b0["cmiss"])[..., None]
            Bn, Mx = b0["cmask"].shape
            gp = torch.Generator().manual_seed(9)
            Pm = torch.arange(Mx).repeat(Bn, 1)
            for r in range(Bn):
                mr = int(b0["M"][r])
                Pm[r, :mr] = torch.randperm(mr, generator=gp)
            Pm = Pm.to(DEV)
            b1 = fresh(b0)
            for k in ("ci", "ce_raw", "cmiss", "ccat", "cf", "y", "xt", "pos", "cmask"):
                if k in b1:
                    v = b1[k]
                    ix = Pm.view(Bn, Mx, *([1] * (v.dim() - 2))).expand_as(v) if v.dim() > 2 else Pm
                    b1[k] = v.gather(1, ix)
            with torch.no_grad():
                o0, o1 = mdl(fresh(b0)), mdl(b1)
            dperm = float((o1 - o0.gather(1, Pm))[b0["cmask"]].abs().max())
            tol = 1e-4
            if tag == "real" and arm == "cset" and dperm > tol:
                P(f"  info cset: real-batch equivariance max|d| {dperm:.2e} (exact duplicate embeddings in a feed make "
                  f"the top-k neighbour order arbitrary; the jittered check below is the test)")
            else:
                check(f"{arm}: permutation equivariance within feeds ({tag})", dperm <= tol, f"max|d| {dperm:.2e}")
        # T5 gradient flow
        mdl.train()
        perturb_zero_init(mdl)
        torch.manual_seed(1)
        lo, _ = train_loss(M6, mdl, fresh(bl), pr_l)
        mdl.zero_grad(set_to_none=True)
        lo.backward()
        newp = [(n, p) for n, p in mdl.named_parameters() if n in missing]
        if arm == "cset":
            newp += [(n, p) for n, p in mdl.named_parameters() if n.endswith("bias_scale") and n.startswith("sets.")]
        dead = [n for n, p in newp if p.grad is None or float(p.grad.abs().sum()) == 0.0]
        fin = all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in mdl.parameters())
        check(f"{arm}: gradient reaches every new parameter, all grads finite", not dead and fin,
              f"{len(newp)} new tensors; zero-grad: {dead}")
        if arm in ("a4w", "cset", "a4u"):  # at the TRUE init the zero-init output layer itself must get a gradient
            m0 = build(M6, cfg)
            m0.load_state_dict(sd, strict=False)
            m0.to(DEV)
            m0.train()
            torch.manual_seed(1)
            lo0, _ = train_loss(M6, m0, fresh(bl), pr_l)
            lo0.backward()
            gz = {n: float(p.grad.abs().sum()) for n, p in m0.named_parameters()
                  if n in ("cretr.Z.weight", "retr.t3.weight", "retr.e_cnt.weight") and p.grad is not None}
            check(f"{arm}: zero-init output layer gets a gradient at the true init", all(v > 0 for v in gz.values())
                  and len(gz) > 0, gz)
        mdl.eval()
        arm_models[arm] = mdl

    # ------------------------------------------------------------------ adrop0 train-mode semantics
    P("\n== adrop0 semantics")
    m_a = build(M6, {**base_cfg, "attn_drop": 0.1})  # explicit = drop -> identical to the baseline in train mode
    m_a.load_state_dict(sd)
    m_a.to(DEV)
    ref.train(), m_a.train()
    torch.manual_seed(5)
    o_ref = ref(fresh(bs))
    torch.manual_seed(5)
    o_a = m_a(fresh(bs))
    check("attn_drop=0.1 (== drop) is bit-identical to the baseline in train mode", torch.equal(o_ref, o_a))
    m_0 = arm_models["adrop0"]
    m_0.train()
    torch.manual_seed(5)
    o_0 = m_0(fresh(bs))
    check("attn_drop=0 changes train-mode outputs (dropout removed)", not torch.equal(o_ref, o_0),
          f"mean|d| {float((o_ref - o_0)[bs['cmask']].abs().mean()):.4f}")
    nd = [(n, mod.p) for n, mod in m_0.named_modules() if isinstance(mod, torch.nn.Dropout)]
    check("adrop0 keeps residual/FFN/retrieval dropout at 0.1", all(p == 0.1 for _, p in nd), f"{len(nd)} Dropout")
    at = [mod.attn_drop for mod in m_0.modules() if isinstance(mod, M6.Attn)]
    check("adrop0 sets attention-probability dropout 0 in every Attn", all(x == 0.0 for x in at), f"{len(at)} Attn")
    ref.eval()

    # ------------------------------------------------------------------ T10 dedup
    P("\n== T10 history_dedup / a4u distinct retrieval")
    hidx = torch.tensor([[5, 3, 5, -1, 3, 7, 5]])
    hv = torch.tensor([[1, 1, 1, 0, 1, 1, 0]], dtype=torch.bool)
    first, cnt = M6.history_dedup(hidx, hv)
    check("hand example: first occurrence (newest) and counts",
          first.tolist() == [[True, True, False, False, False, True, False]] and cnt.tolist() == [[2, 2, 2, 0, 2, 1, 0]],
          f"first {first.int().tolist()} cnt {cnt.tolist()}")
    stats = {}
    for nm, b in (("S", bs), ("L", bl)):
        b2 = fresh(b)
        M6.dense_features(b2)
        S = torch.einsum("bmd,bkd->bmk", b2["ce"], b2["he"]) * b2["hemb_ok"][:, None, :]
        m = 16
        _, ib = S.masked_fill(~b2["hemb_ok"][:, None, :], -2.0).topk(m, -1)
        it = b2["hidx"].gather(1, ib.flatten(1)).view_as(ib)
        topv = S.masked_fill(~b2["hemb_ok"][:, None, :], -2.0).gather(2, ib)
        ok = (topv > -1.5) & ~b2["cmiss"][..., None] & b2["cmask"][..., None]
        dup = torch.zeros_like(ok)
        for j in range(1, m):
            dup[..., j] = (it[..., j:j + 1] == it[..., :j]).any(-1)
        first, _ = M6.history_dedup(b2["hidx"], b2["hemb_ok"])
        _, iu = S.float().masked_fill(~first[:, None, :], -2.0).topk(m, -1)
        tu = S.float().masked_fill(~first[:, None, :], -2.0).gather(2, iu)
        itu = b2["hidx"].gather(1, iu.flatten(1)).view_as(iu)
        oku = (tu > -1.5) & ~b2["cmiss"][..., None] & b2["cmask"][..., None]
        dupu = torch.zeros_like(oku)
        for j in range(1, m):
            dupu[..., j] = ((itu[..., j:j + 1] == itu[..., :j]) & oku[..., :j]).any(-1)
        share = float((dup & ok).sum() / ok.sum().clamp_min(1))
        rows = float(((dup & ok).any(-1) & b2["cmask"]).sum() / b2["cmask"].sum())
        stats[nm] = dict(dup_slot_share=share, rows_with_dup=rows, uniq_dup_slots=int((dupu & oku).sum()))
        check(f"a4u retrieves only distinct items (batch {nm})", int((dupu & oku).sum()) == 0,
              f"baseline top-16: {share:.1%} of valid slots are repeats of a higher slot; {rows:.1%} of candidate rows "
              f"have >= 1 repeated item")
    RES["dedup_stats"] = stats

    # ------------------------------------------------------------------ T11 typed cross mask
    P("\n== T11 typed_cross key masks")
    mt = arm_models["typed"]
    seen = []
    hooks = [blk.register_forward_pre_hook(lambda mod, args: seen.append(args[1].clone())) for blk in mt.cross]
    b2 = fresh(bl)
    with torch.no_grad():
        mt(b2)
    for h in hooks:
        h.remove()
    full = torch.cat([torch.ones(b2["hmask"].shape[0], 1, dtype=torch.bool, device=DEV), b2["hmask"]], 1)
    tgt = full.clone()
    tgt[:, 1:] &= (b2["het"] == 1) & (b2["hes"] == 1)
    check("cross[0] gets the full key mask, cross[-1] the target-action mask",
          torch.equal(seen[0], full) and torch.equal(seen[-1], tgt),
          f"keys kept in the typed view: {float(tgt[:, 1:].sum() / full[:, 1:].sum()):.1%} of valid events")

    # ------------------------------------------------------------------ T13 optimizer groups + one step (train.py:217-231)
    P("\n== T13 train.py optimizer construction (Muon for hidden 2-D Linear weights, AdamW rest) + one step per arm")
    for arm, mdl in [("base", ref)] + list(arm_models.items()):
        mdl.train()
        wd = 0.1 if arm == "wd01" else 0.02
        muon_names = {f"{mn}.weight" for mn, mod in mdl.named_modules()
                      if isinstance(mod, torch.nn.Linear) and min(mod.weight.shape) > 1}
        rest = [(n, p) for n, p in mdl.named_parameters() if n not in muon_names]
        decay = [p for n, p in rest if p.ndim >= 2 and "emb" not in n]
        no_decay = [p for n, p in rest if not (p.ndim >= 2 and "emb" not in n)]
        opts = [torch.optim.AdamW([{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}],
                                  lr=1e-3, betas=(0.9, 0.98)),
                torch.optim.Muon([p for n, p in mdl.named_parameters() if n in muon_names], lr=1e-3, weight_decay=wd,
                                 momentum=0.95, nesterov=True, adjust_lr_fn="match_rms_adamw")]
        before = {n: p.detach().clone() for n, p in mdl.named_parameters()}
        torch.manual_seed(2)
        lo, _ = train_loss(M6, mdl, fresh(bs), pr_s)
        for o in opts:
            o.zero_grad(set_to_none=True)
        lo.backward()
        torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
        for o in opts:
            o.step()
        fin = all(bool(torch.isfinite(p).all()) for p in mdl.parameters())
        newn = [n for n in before if n.startswith(("cretr.", "retr.t3", "retr.e_cnt")) or n.endswith("sets.0.attn.bias_scale")]
        moved = [n for n in newn if not torch.equal(before[n], dict(mdl.named_parameters())[n])]
        new_muon = sorted(n for n in muon_names if n.startswith(("cretr.", "retr.t3")))
        check(f"{arm}: optimizer step finite; new tensors updated", fin and len(moved) == len(newn),
              f"Muon matrices {len(muon_names)} (new: {new_muon}); new tensors moved {len(moved)}/{len(newn)}")
        mdl.eval()
    ref.load_state_dict(sd)

    # ------------------------------------------------------------------ T8 params
    P("\n== T8 parameter counts")
    for k, v in RES["params"].items():
        P(f"  {k:7s} {v:>9,d}  ({v - RES['params']['base']:+,d})")

    # ------------------------------------------------------------------ T9 CPU throughput
    P(f"\n== T9 {DEV.type.upper()} throughput, fwd+bwd of the training loss, {NT} threads (relative cost; fp32, math SDPA "
      f"on GPU: production bf16 ratios differ)")
    thr = {}
    models = {"base": ref, **{k: v for k, v in arm_models.items()}}
    for arm, mdl in models.items():
        mdl.train()
        row = {}
        for nm, b, prb in (("S", bs, pr_s), ("L", bl, pr_l)):
            ts = []
            for rep in range(4):
                bb = fresh(b)
                torch.manual_seed(rep)
                t0 = time.perf_counter()
                lo, _ = train_loss(M6, mdl, bb, prb)
                mdl.zero_grad(set_to_none=True)
                lo.backward()
                if DEV.type == "cuda":
                    torch.cuda.synchronize()
                ts.append(time.perf_counter() - t0)
            row[nm] = float(np.median(ts[1:]))
        thr[arm] = row
        mdl.eval()
    for arm, row in thr.items():
        P(f"  {arm:7s} S {row['S'] * 1e3:7.0f} ms ({row['S'] / thr['base']['S']:.2f}x)   "
          f"L {row['L'] * 1e3:7.0f} ms ({row['L'] / thr['base']['L']:.2f}x)")
    RES["cpu_throughput_s"] = thr
    RES["failed"] = FAIL
    P(f"\n== {len(RES['checks']) - len(FAIL)}/{len(RES['checks'])} checks passed; failed: {FAIL}; "
      f"{time.time() - t_all:.0f}s")
    json.dump(RES, open(f"{R6}/logs/test_r6.json", "w"), indent=1)
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())
