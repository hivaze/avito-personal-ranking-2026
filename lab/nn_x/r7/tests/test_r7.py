"""GPU unit tests for the nn_x/r7 arms (reports/r7_build.md). Read-only on all shared data; writes only nn_x/r7/logs.

  CUDA_VISIBLE_DEVICES=<gpu> OMP_NUM_THREADS=12 <repo>/.venv/bin/python nn_x/r7/tests/test_r7.py [arm ...]

Base = the sx_base_s23 cfg (current 2-epoch screen recipe: cvnd_pre2s, d128/4h/2 blocks/1 hist layer, retr16 + gate,
split k/v, sim_bias 1) and its trained weights. Real train batches from a memory-mapped mini split moved to the GPU.
  T0  CLI: screen.sh's base command line reproduces every sx_base_s23 arg; every r6/r7 flag at its OFF default; each arm's
      screen.sh flags parse to the intended values.
  T1  init: flag-off r7 == nn/ (cfg, state_dict, global RNG state after construction); per arm every baseline tensor keeps
      its nn/ init (postln: exactly beta x the nn/ init on v / o / ff[0] / ff[3]) and the global RNG stream is unchanged.
  T2  flag-off r7 vs nn/model.py with trained weights, training precision (bf16 autocast, TF32): eval logits and train-mode
      loss (same dropout seed) bit-identical; gradients identical up to the GPU's own run-to-run floor (nn/ vs nn/ twice).
  T2b r6-ported arms: r7 vs nn_x/r6/model.py (read-only import), same cfg + seed: state_dict, eval logits, train loss equal.
  T3  exactness at init for the arms declared exact (fresh init and warm start from the trained weights): eval logits
      bit-identical to nn/; size of the change for the others (info).
  T4  shapes and masks (fp32, TF32 off): logits [B, Mx] finite; valid logits independent of padded candidates / history;
      independent of the AMOUNT of padding (a feed scored alone vs padded next to a longer feed).
  T5  gradient flow: every parameter that is used gets a finite gradient; every new parameter a non-zero one (after a small
      perturbation of the zero-init output layers); zero-init output layers get a gradient at the true init.
  T6  permutation equivariance within a feed (real batch; jittered batch for the top-k arms).
  T7  no cross-feed leakage: perturbing every input of feed 0 leaves every other feed's logits unchanged.
  T8  parameter counts.  T9 GPU step time (dense + fwd + bwd + clip + optimiser step on full 32,768-candidate real train
      batches, train.py's optimiser partition) and peak-memory delta vs the baseline; batch2s per-sample cost at 2x batch;
      25-step loss trajectory on 2 real batches (trainability sanity).
  T10 optimiser partition per arm (train.build_opts): Muon / AdamW-decay / AdamW-no-decay membership as specified.
Writes nn_x/r7/logs/test_r7.log and .json.
"""
import copy
import importlib.util
import json
import os
import subprocess
import sys
import time
import types

sys.dont_write_bytecode = True  # importing nn/model.py and nn_x/r6/model.py must never write their __pycache__
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # deterministic cuBLAS for the T2 gradient comparison
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.nn.attention import SDPBackend, sdpa_kernel  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
R7 = os.path.dirname(HERE)
ROOT = "<repo>"
sys.path.insert(0, R7)
sys.path.insert(0, HERE)
assert os.environ.get("CUDA_VISIBLE_DEVICES", "") in tuple(str(i) for i in range(8)), "GPU tests: set one GPU id"
torch.set_num_threads(min(12, int(os.environ.get("OMP_NUM_THREADS", "12"))))

import model as M7  # noqa: E402  (nn_x/r7/model.py)
import train as T7  # noqa: E402  (nn_x/r7/train.py; imports nn_x/r7/common.py: GPU guard + r7 output dirs)
from mini_split import MiniSplit, pick_train_feeds  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MREF = _load("ref_model", f"{ROOT}/nn/model.py")  # read-only import of the production model
M6 = _load("r6_model", f"{ROOT}/nn_x/r6/model.py")  # read-only import of the r6 code (port check)
BASE_CK = f"{ROOT}/nn_x/solvers/ckpt/sx_base_s23.pt"  # the 2-epoch screen control (same args as screen.sh base)
DEV = torch.device("cuda:0")

ARMS = {  # arm -> (model cfg delta, train-arg delta)
    "muon_in": ({}, {"muon_inputs": "adamw"}),
    "batch2s": ({}, {"muon_momentum": 0.90, "adam_betas": "0.8,0.96", "lr": 1.414e-3}),
    "resdrop0": ({"res_drop": 0.0}, {}),
    "pbld": ({"pbld": 16}, {}),
    "hc4": ({"hc": 4}, {}),
    "dwa": ({"dwa": True}, {}),
    "diff2": ({"diff_attn": "cross"}, {}),
    "swiglu": ({"ffn": "swiglu"}, {}),
    "derf": ({"norm": "derf"}, {}),
    "hl2": ({"hist_extra": 1}, {}),
    "periln": ({"periln": True}, {}),
    "ftok": ({"ftok": 8}, {}),
    "postln": ({"postln": True}, {}),
    "a4u": ({"retr_uniq": True}, {}),
    "wd01": ({}, {"wd": 0.1}),
    "cset": ({"cretr_m": 8, "set_sim_bias": True}, {}),
    "typed": ({"typed_cross": True}, {}),
    "adrop0": ({"attn_drop": 0.0}, {}),
    "a4w": ({"retr_m": 32, "retr_prod": True}, {}),
}
R6_ARMS = ("a4u", "cset", "typed", "adrop0", "a4w")
EXACT_FRESH = {"muon_in", "batch2s", "resdrop0", "pbld", "hc4", "dwa", "hl2", "ftok", "wd01", "adrop0", "a4u", "a4w"}
EXACT_WARM = {"muon_in", "batch2s", "resdrop0", "pbld", "hc4", "dwa", "hl2", "ftok", "wd01", "adrop0"}
ZERO_INIT_OUT = ("pbld.P.", "ftok.Zf.", "hist_enc.1.attn.o.", "hist_enc.1.ff.3.", "hc.", "cretr.Z.", "retr.t3.",
                 "retr.e_cnt.")
LOG = open(f"{R7}/logs/test_r7.log", "w")
RES = {"checks": []}
FAIL = []


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + "\n")
    LOG.flush()


def check(name, ok, info=""):
    P(f"  [{'PASS' if ok else 'FAIL'}] {name} {info}")
    RES["checks"].append(dict(name=name, ok=bool(ok), info=str(info)))
    if not ok:
        FAIL.append(name)


def train_loss(Mod, model, b, prand_b, aux_w=0.5, tau=10.0, denom=512):
    """Exactly the nn/train.py loss of the recipe (softmax, loss_norm feed, K9 listwise aux weighted by 1 - p_random),
    under bf16 autocast as in training (dense features outside autocast)."""
    X = model.dense(b)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lg = model(b, X, use_pos=True)
    y, m = b["y"], b["cmask"]
    loss = Mod.listwise_loss(lg, y, m, None, norm="feed", denom=denom)
    tgt = torch.softmax((-b["pos"].float() / tau).masked_fill(~m, -1e4), 1) * m
    lp = torch.log_softmax(model.aux_out.masked_fill(~m, -1e4), 1)
    fw_aux = 1 - prand_b
    loss = loss + aux_w * (-(tgt * lp).sum(1) * fw_aux).sum() / fw_aux.sum().clamp_min(1e-6)
    return loss, lg


def ev(model, b, amp=True):
    model.eval()
    with torch.no_grad():
        if amp:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return model(fresh(b)).float()
        return model(fresh(b)).float()


def fresh(b):
    """Deep copy of a gathered batch (model.dense writes ce/he/un into the dict)."""
    return {k: (v.clone() if torch.is_tensor(v) else copy.deepcopy(v)) for k, v in b.items()
            if k not in ("ce", "he", "un")}


def build(Mod, cfg, seed=23, sd=None):
    torch.manual_seed(seed)
    m = Mod.Ranker(**cfg)
    if sd is not None:
        m.load_state_dict(sd, strict=False)
    return m.to(DEV)


def perturb_zero_init(model, scale=0.02, seed=5):
    """Make zero-init output layers / dynamic HC weights non-zero so gradients reach every parameter upstream of them."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if (n.startswith(ZERO_INIT_OUT) and p.abs().max() == 0) or n.startswith("dwa."):
                p.add_((torch.randn(p.shape, generator=g) * scale).to(p.device))


def tf32(on):
    torch.backends.cuda.matmul.allow_tf32 = on
    torch.backends.cudnn.allow_tf32 = on


def args_for(tdelta):
    a = types.SimpleNamespace(muon=True, muon_inputs="muon", muon_momentum=0.95, adam_betas="0.9,0.98", lr=1e-3,
                              wd=0.02)
    for k, v in tdelta.items():
        setattr(a, k, v)
    return a


def main():
    t_all = time.time()
    only = set(sys.argv[1:])
    arms = {k: v for k, v in ARMS.items() if not only or k in only}
    ck = torch.load(BASE_CK, map_location="cpu", weights_only=False)
    base_cfg = dict(ck["cfg"])
    sd = ck["state_dict"]
    P(f"== nn_x/r7 GPU unit tests ({time.ctime()}, GPU {os.environ['CUDA_VISIBLE_DEVICES']}, "
      f"{torch.cuda.get_device_name(0)}, torch {torch.__version__})")
    P(f"base cfg from {BASE_CK}: d {base_cfg['d']} heads {base_cfg['heads']} blocks {base_cfg['blocks']} "
      f"hist_layers {base_cfg['hist_layers']} n_dense {base_cfg['n_dense']} retr_m {base_cfg['retr_m']} "
      f"gate {base_cfg['attn_gate']} split_kv {base_cfg['split_kv']} sim_bias {base_cfg['sim_bias']}")

    # ------------------------------------------------------------------ T0 CLI
    P("\n== T0 CLI: screen.sh command lines through train.py's parser")
    sh = f"{R7}/screen.sh"

    def parse_arm(arm, stage=""):
        line = subprocess.run(["bash", sh, "flags", arm] + ([stage] if stage else []), capture_output=True, text=True,
                              check=True).stdout.split()
        sys.argv = ["train.py", "--variant", "r7_x", "--seed", "23"] + line
        return T7.parse()

    a0 = parse_arm("base")
    sxa = ck["args"]
    diff = {k: (getattr(a0, k), sxa[k]) for k in sxa if k not in ("variant", "no_test") and hasattr(a0, k)
            and getattr(a0, k) != sxa[k]}
    check("screen.sh base reproduces every sx_base_s23 arg (except variant / no_test)", not diff, diff or "identical")
    offd = {k: getattr(a0, k) for k in T7.R7_MODEL_DEFAULTS}
    check("every r6/r7 model flag at its OFF default", offd == T7.R7_MODEL_DEFAULTS)
    check("optimiser flags at nn/ defaults", (a0.muon_inputs, a0.muon_momentum, a0.adam_betas, a0.res_drop)
          == ("muon", 0.95, "0.9,0.98", -1.0))
    P(f"  r7-only arg keys: {sorted(set(vars(a0)) - set(sxa))}")
    expect = {"muon_in": dict(muon_inputs="adamw"),
              "batch2s": dict(cand_budget=65536, max_feeds=1024, lr=1.414e-3, warmup=250, eval_every=500,
                              muon_momentum=0.90, adam_betas="0.8,0.96"),
              "resdrop0": dict(res_drop=0.0, drop=0.1, attn_drop=-1.0), "pbld": dict(pbld=16), "hc4": dict(hc=4),
              "dwa": dict(dwa=True), "diff2": dict(diff_attn="cross"), "swiglu": dict(ffn="swiglu"),
              "derf": dict(norm="derf"), "hl2": dict(hist_extra=1), "periln": dict(periln=True), "ftok": dict(ftok=8),
              "postln": dict(postln=True), "a4u": dict(retr_uniq=True), "wd01": dict(wd=0.1),
              "cset": dict(cretr_m=8, set_sim_bias=True), "typed": dict(typed_cross=True), "adrop0": dict(attn_drop=0.0),
              "a4w": dict(retr_m=32, retr_prod=True), "drop0": dict(drop=0.0)}
    for arm, exp in expect.items():
        a = parse_arm(arm)
        got = {k: getattr(a, k) for k in exp}
        others = {k: (getattr(a, k), getattr(a0, k)) for k in vars(a0) if k not in exp and getattr(a, k) != getattr(a0, k)}
        check(f"screen.sh {arm}: flags parse as specified, nothing else changes", got == exp and not others,
              f"{got}{' | other diffs ' + str(others) if others else ''}")
    a10 = parse_arm("base", "e10")
    f7 = torch.load(f"{ROOT}/nn/ckpt/v17_pre2s_s101.pt", map_location="cpu", weights_only=False)["args"]
    diff = {k: (getattr(a10, k), f7[k]) for k in f7 if k not in ("variant", "seed", "save_steps") and hasattr(a10, k)
            and getattr(a10, k) != f7[k]}
    check("screen.sh e10 base reproduces the F7 recipe (v17_pre2s_s101 args, except variant/seed/save_steps)", not diff,
          diff or "identical")
    ab = parse_arm("batch2s", "e10")
    check("screen.sh batch2s e10: lr 2.83e-3, eval 1250, save_steps halved",
          (ab.lr, ab.eval_every, ab.save_steps, ab.warmup) == (2.83e-3, 1250, "25000,30000,35000,38750", 250))
    check("screen.sh wd01 e10: wd 0.05 (lr*wd kept)", parse_arm("wd01", "e10").wd == 0.05)

    # ------------------------------------------------------------------ T1 init equality + RNG
    P("\n== T1 init: baseline parameters and global RNG stream unchanged by every arm (CPU construction, seed 23)")
    torch.manual_seed(23)
    ref = MREF.Ranker(**base_cfg)
    rng_ref = torch.get_rng_state()
    torch.manual_seed(23)
    r7off = M7.Ranker(**base_cfg)
    check("flag-off r7 cfg == nn/ cfg", r7off.cfg == ref.cfg)
    check("flag-off global RNG state after construction", torch.equal(torch.get_rng_state(), rng_ref))
    check("flag-off r7 state_dict == nn/ (keys, order and values)",
          list(r7off.state_dict()) == list(ref.state_dict())
          and all(torch.equal(r7off.state_dict()[k], v) for k, v in ref.state_dict().items()))
    check("flag-off parameter order == nn/ (optimiser group order)",
          [n for n, _ in r7off.named_parameters()] == [n for n, _ in ref.named_parameters()])
    RES["params"] = {"base": sum(p.numel() for p in ref.parameters())}
    rsd = ref.state_dict()
    for arm, (mdelta, _) in arms.items():
        cfg = {**base_cfg, **mdelta}
        torch.manual_seed(23)
        mdl = M7.Ranker(**cfg)
        same_rng = torch.equal(torch.get_rng_state(), rng_ref)
        asd = mdl.state_dict()
        shared = [k for k in rsd if k in asd and rsd[k].shape == asd[k].shape]
        changed = [k for k in shared if not torch.equal(rsd[k], asd[k])]
        new = [k for k in asd if k not in rsd or asd[k].shape != rsd[k].shape]
        missing = [k for k in rsd if k not in asd]
        RES["params"][arm] = sum(p.numel() for p in mdl.parameters())
        info = (f"changed {len(changed)} missing {len(missing)} new {len(new)} "
                f"({sum(asd[k].numel() for k in new)} params) rng_same {same_rng}")
        if arm == "postln":
            d = cfg["d"]
            ok_scale = True
            for pre, N in [("hist_enc.", 1)] + [(f"{x}.", 4) for x in ("cross", "sets")]:
                beta = (8 * N) ** -0.25
                for k in changed:
                    if k.startswith(pre):
                        ok_scale &= torch.equal(asd[k], rsd[k] * beta)
            exp_changed = {k for k in rsd if k.endswith(("attn.v.weight", "attn.o.weight", "ff.0.weight", "ff.3.weight"))
                           and k.startswith(("hist_enc.", "cross.", "sets."))}
            ok = same_rng and not missing and set(changed) == exp_changed and ok_scale
            info += f" | DeepNorm-scaled tensors {len(changed)} (expected {len(exp_changed)}), exactly beta x nn/ {ok_scale}"
        elif arm in ("swiglu", "derf"):  # replaced modules: the removed tensors are the only 'missing' ones
            kind = ".ff." if arm == "swiglu" else (".ln1.", ".ln2.", ".lnc.", "head.0.", "aux_head.0.")
            ok = same_rng and not changed and all(any(s in k or k.startswith(s) for s in
                                                      ((kind,) if isinstance(kind, str) else kind)) for k in missing)
            info += f" | replaced {len(missing)} tensors"
        else:
            ok = same_rng and not changed and not missing
        check(f"{arm}: baseline init unchanged", ok, info)
        P(f"     new keys: {new[:6]}{' ...' if len(new) > 6 else ''}")
        exp_keys = set(mdelta) - {"retr_m"} | ({"ftok_groups"} if "ftok" in mdelta else set())
        check(f"{arm}: cfg stores only the non-default keys", set(mdl.cfg) - set(ref.cfg) == exp_keys,
              sorted(set(mdl.cfg) - set(ref.cfg)))
        if mdl.cfg.get("ftok_groups"):
            g = mdl.cfg["ftok_groups"]
            check("ftok: 8 groups partition the 104 inputs, matched to the cvnd_pre2s xt layout",
                  sorted(i for x in g for i in x) == list(range(104)) and len(g) == 8
                  and base_cfg["xt_names"] == T7.XT_PRESETS["cvnd_pre2s"] and base_cfg["n_dense"] == 104,
                  [len(x) for x in g])

    # ------------------------------------------------------------------ real data on the GPU
    P("\n== real train batches (mini split, memory-mapped, moved to the GPU; read-only)")
    t0 = time.time()
    feeds = pick_train_feeds(n_each=48, seed=0)
    ms = MiniSplit("train", feeds, xt_names=base_cfg["xt_names"], device=DEV)
    idx_s = np.where(ms.M_np <= 64)[0][:96]
    idx_l = np.where(ms.M_np >= 150)[0][:40]
    bs = ms.gather(idx_s, with_pos=True)
    bl = ms.gather(idx_l, with_pos=True)
    pr_s, pr_l = ms.prand[bs["fidx"]], ms.prand[bl["fidx"]]
    P(f"  {ms.n} feeds, {ms.n_cand} cands; batch S: B {bs['cmask'].shape[0]} Mx {bs['cmask'].shape[1]} Hx "
      f"{bs['hmask'].shape[1]}; batch L: B {bl['cmask'].shape[0]} Mx {bl['cmask'].shape[1]} Hx {bl['hmask'].shape[1]}; "
      f"{time.time() - t0:.1f}s")

    # ------------------------------------------------------------------ T2 flag-off equality, trained weights
    P("\n== T2 flag-off r7 vs nn/model.py, trained sx_base_s23 weights, training precision (bf16 autocast, TF32 on)")
    tf32(True)
    ref = build(MREF, base_cfg, sd=sd)
    r7off = build(M7, base_cfg, sd=sd)
    ref.load_state_dict(sd)
    r7off.load_state_dict(sd)
    for nm, b, prb in (("S", bs, pr_s), ("L", bl, pr_l)):
        l1, l2 = ev(ref, b), ev(r7off, b)
        check(f"eval logits identical (batch {nm})", torch.equal(l1, l2), f"max|d| {float((l1 - l2).abs().max()):.2e}")
        def gmax(g1, g2):
            return max((0.0 if a is None and b_ is None else float((a - b_).abs().max())) for a, b_ in zip(g1, g2))
        for det in (False, True):  # default kernels (info: run-to-run floor) / deterministic kernels (exact test)
            grads, losses = [], []
            for mdl, Mod in ((ref, MREF), (ref, MREF), (r7off, M7)):
                mdl.train()
                torch.manual_seed(77)
                if det:
                    torch.use_deterministic_algorithms(True, warn_only=True)
                    with sdpa_kernel([SDPBackend.MATH]):
                        lo, _ = train_loss(Mod, mdl, fresh(b), prb)
                        mdl.zero_grad(set_to_none=True)
                        lo.backward()
                    torch.use_deterministic_algorithms(False)
                else:
                    lo, _ = train_loss(Mod, mdl, fresh(b), prb)
                    mdl.zero_grad(set_to_none=True)
                    lo.backward()
                losses.append(lo.detach())
                grads.append([None if p.grad is None else p.grad.detach().clone() for p in mdl.parameters()])
            floor, d = gmax(grads[0], grads[1]), gmax(grads[0], grads[2])
            if not det:
                check(f"train-mode loss identical (batch {nm}, dropout on, same seed)", torch.equal(losses[0], losses[2]),
                      f"{losses[0].item():.7f} vs {losses[2].item():.7f}")
                P(f"  info: default GPU kernels, max|dgrad| r7 vs nn/ {d:.2e}; nn/ vs nn/ rerun {floor:.2e} "
                  f"(non-deterministic backward: SDPA / index_put atomics)")
            else:
                check(f"all gradients bit-identical under deterministic kernels (batch {nm}, math SDPA)",
                      d == 0.0 and floor == 0.0 and torch.equal(losses[0], losses[2]),
                      f"max|dgrad| r7 vs nn/ {d:.2e}; nn/ rerun {floor:.2e}")
    ref.eval()

    # ------------------------------------------------------------------ T2b r6 port
    P("\n== T2b r6-ported arms: nn_x/r7 vs nn_x/r6/model.py (read-only import), same cfg / seed")
    for arm in R6_ARMS:
        if arm not in arms:
            continue
        cfg = {**base_cfg, **arms[arm][0]}
        m6, m7 = build(M6, cfg), build(M7, cfg)
        s6, s7 = m6.state_dict(), m7.state_dict()
        same_sd = list(s6) == list(s7) and all(torch.equal(s6[k], s7[k]) for k in s6)
        m6.load_state_dict(sd, strict=False)
        m7.load_state_dict(sd, strict=False)
        same_ev = all(torch.equal(ev(m6, b), ev(m7, b)) for b in (bs, bl))
        m6.train(), m7.train()
        torch.manual_seed(31)
        lo6, _ = train_loss(M6, m6, fresh(bl), pr_l)
        torch.manual_seed(31)
        lo7, _ = train_loss(M7, m7, fresh(bl), pr_l)
        check(f"{arm}: r7 port == r6 (init state_dict, warm eval logits, train loss)",
              same_sd and same_ev and torch.equal(lo6, lo7), f"sd {same_sd} eval {same_ev} loss {lo6.item():.6f} / "
              f"{lo7.item():.6f}")
        del m6, m7

    # ------------------------------------------------------------------ per-arm model tests
    base_fresh = {nm: ev(build(MREF, base_cfg), b) for nm, b in (("S", bs), ("L", bl))}
    base_warm = {nm: ev(ref, b) for nm, b in (("S", bs), ("L", bl))}
    for arm, (mdelta, tdelta) in arms.items():
        cfg = {**base_cfg, **mdelta}
        P(f"\n== arm {arm} model {mdelta} train {tdelta}")
        tf32(True)
        # T3 exactness at init / warm start
        mf = build(M7, cfg)
        for tag, basel, mdl in (("fresh init", base_fresh, mf), ("warm start", base_warm, None)):
            if mdl is None:
                mdl = build(M7, cfg)
                missing, unexpected = mdl.load_state_dict(sd, strict=False)
                if arm not in ("swiglu", "derf", "postln"):
                    check(f"{arm}: warm start loads every baseline tensor", not unexpected, f"new: {len(missing)}")
                if arm == "postln":  # DeepNorm scaling is an init rule; a warm start keeps the trained weights
                    pass
            ds = []
            for nm, b in (("S", bs), ("L", bl)):
                d = (ev(mdl, b) - basel[nm])[b["cmask"]]
                ds.append((float(d.abs().max()), float(d.abs().mean())))
            exact_set = EXACT_FRESH if tag == "fresh init" else EXACT_WARM
            if arm in exact_set:
                check(f"{arm}: {tag} is an exact no-op (eval logits bit-identical to nn/)", max(x[0] for x in ds) == 0.0,
                      f"max|d| S {ds[0][0]:.2e} L {ds[1][0]:.2e}")
            else:
                P(f"  info {arm}: {tag} eval-logit change mean|d| S {ds[0][1]:.4f} L {ds[1][1]:.4f}; "
                  f"max|d| S {ds[0][0]:.4f} L {ds[1][0]:.4f}")
            RES.setdefault("init_change", {}).setdefault(arm, {})[tag] = ds
        mdl = build(M7, cfg, sd=sd)  # warm-started arm model for the structural tests
        # T4 shapes and masks (fp32, TF32 off)
        tf32(False)
        b = fresh(bl)
        out = ev(mdl, b, amp=False)
        check(f"{arm}: logits shape [B, Mx], finite", tuple(out.shape) == tuple(b["cmask"].shape)
              and bool(torch.isfinite(out[b["cmask"]]).all()), tuple(out.shape))
        bp = fresh(bl)
        g = torch.Generator(device="cpu").manual_seed(11)
        R = lambda *s: torch.randn(*s, generator=g).to(DEV)  # noqa: E731
        pc = ~bp["cmask"]
        npc = int(pc.sum())
        bp["ce_raw"][pc] = R(npc, 128) * 50
        bp["cf"][pc] = R(npc, 5) * 9
        bp["ccat"][pc] = torch.randint(0, 54, (npc,), generator=g).to(DEV)
        bp["xt"][pc] = R(npc, bp["xt"].shape[-1]) * 3
        bp["cmiss"][pc] = (torch.rand(npc, generator=g) < 0.5).to(DEV)
        ph = ~bp["hmask"]
        nph = int(ph.sum())
        bp["he_raw"][ph] = R(nph, 128) * 50
        bp["het"][ph] = torch.randint(1, 11, (nph,), generator=g).to(DEV)
        bp["hes"][ph] = torch.randint(0, 30, (nph,), generator=g).to(DEV)
        bp["hcat"][ph] = torch.randint(0, 54, (nph,), generator=g).to(DEV)
        bp["hidx"][ph] = torch.randint(0, 10, (nph,), generator=g).to(DEV).to(bp["hidx"].dtype)
        out_p = ev(mdl, bp, amp=False)
        dm = float((out_p - out)[b["cmask"]].abs().max())
        check(f"{arm}: valid logits independent of padded candidates / history", dm <= 1e-5, f"max|d| {dm:.2e}")
        # padding amount: feeds scored alone vs inside the long-feed batch (more candidate and history padding).
        # Position-keyed jitter (same for a feed in both batches) breaks exact embedding ties of repeated items, whose
        # top-k order may otherwise depend on the tensor width (a property of the baseline retrieval, reported as info)
        short = idx_s[:6]
        dpads = {}
        for jit in (0.0, 1e-3):
            bb = [ms.gather(short, with_pos=True), ms.gather(np.concatenate([short, idx_l[:2]]), with_pos=True)]
            for x in bb:
                if jit:
                    for key, mk in (("ce_raw", "cmask"), ("he_raw", "hmask")):
                        L_ = x[key].shape[1]
                        wave = torch.sin(torch.arange(L_, device=DEV)[:, None] * 0.37
                                         + torch.arange(128, device=DEV)[None, :] * 1.3)
                        x[key] = x[key] + jit * wave[None] * x[mk][..., None]
            o_a, o_m = ev(mdl, bb[0], amp=False), ev(mdl, bb[1], amp=False)
            Ma = o_a.shape[1]
            dpads[jit] = float((o_m[:len(short), :Ma] - o_a)[bb[0]["cmask"]].abs().max())
        check(f"{arm}: logits independent of the amount of padding (Mx {Ma} -> {o_m.shape[1]}, Hx "
              f"{bb[0]['hmask'].shape[1]} -> {bb[1]['hmask'].shape[1]}; tie-free)", dpads[1e-3] <= 1e-4,
              f"max|d| {dpads[1e-3]:.2e} (unjittered, info: {dpads[0.0]:.2e})")
        # T7 cross-feed leakage
        bx = fresh(bl)
        bx["ce_raw"][0] = torch.randn_like(bx["ce_raw"][0]) * 30
        bx["cf"][0] = torch.randn_like(bx["cf"][0]) * 5
        bx["ccat"][0] = torch.randint(0, 54, bx["ccat"][0].shape, generator=g).to(DEV)
        bx["xt"][0] = torch.randn_like(bx["xt"][0])
        bx["he_raw"][0] = torch.randn_like(bx["he_raw"][0]) * 30
        bx["het"][0] = torch.randint(1, 11, bx["het"][0].shape, generator=g).to(DEV)
        bx["hes"][0] = torch.ones_like(bx["hes"][0])
        bx["hidx"][0] = torch.randint(0, 5, bx["hidx"][0].shape, generator=g).to(DEV).to(bx["hidx"].dtype)
        bx["ue_raw"][0] = torch.randn_like(bx["ue_raw"][0]) * 30
        out_x = ev(mdl, bx, amp=False)
        d0 = float((out_x[0] - out[0])[b["cmask"][0]].abs().max())
        dr = float((out_x[1:] - out[1:])[b["cmask"][1:]].abs().max())
        check(f"{arm}: no cross-feed leakage (feed 0 perturbed)", dr <= 1e-6 and d0 > 0,
              f"other feeds max|d| {dr:.2e}; feed 0 max|d| {d0:.2e}")
        # T6 permutation equivariance within feeds
        for tag, jitter in (("real", 0.0), ("jittered", 1e-3)):
            b0 = fresh(bl)
            if jitter:  # break exact embedding ties (duplicate items) that make a top-k order arbitrary
                gj = torch.Generator(device="cpu").manual_seed(3)
                b0["ce_raw"] = b0["ce_raw"] + (torch.randn(b0["ce_raw"].shape, generator=gj).to(DEV) * jitter
                                               * (~b0["cmiss"])[..., None])
            Bn, Mx = b0["cmask"].shape
            gp = torch.Generator(device="cpu").manual_seed(9)
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
            o0, o1 = ev(mdl, b0, amp=False), ev(mdl, b1, amp=False)
            dperm = float((o1 - o0.gather(1, Pm))[b0["cmask"]].abs().max())
            if tag == "real" and arm == "cset" and dperm > 1e-4:
                P(f"  info cset: real-batch equivariance max|d| {dperm:.2e} (duplicate embeddings make the top-k order "
                  f"arbitrary; the jittered check is the test)")
            else:
                check(f"{arm}: permutation equivariance within feeds ({tag})", dperm <= 1e-4, f"max|d| {dperm:.2e}")
        # T5 gradient flow (training precision)
        tf32(True)
        # arm is wired: after perturbing its zero-init / identity parameters the function changes (exact arms), or it
        # already differs (non-exact arms); dropout arms: train-mode outputs differ at the same dropout seed
        perturb_zero_init(mdl)
        if arm in ("resdrop0", "adrop0", "muon_in", "batch2s", "wd01"):
            if arm in ("resdrop0", "adrop0"):
                ref.train(), mdl.train()
                torch.manual_seed(5)
                o_r = ref(fresh(bs))
                torch.manual_seed(5)
                o_a = mdl(fresh(bs))
                ref.eval()
                drops = {type(m).__name__ + ":" + n.split(".")[-1]: getattr(m, "p", None) for n, m in mdl.named_modules()
                         if isinstance(m, torch.nn.Dropout)}
                pz = sorted({v for v in drops.values()})
                ad = sorted({m.drop if m.attn_drop is None else m.attn_drop for m in mdl.modules()
                             if isinstance(m, M7.Attn)})
                ok_p = (pz == [0.0] and ad == [0.1]) if arm == "resdrop0" else (pz == [0.1] and ad == [0.0])
                check(f"{arm}: active (train-mode outputs differ at the same seed) and dropout sites as specified",
                      not torch.equal(o_r, o_a) and ok_p, f"nn.Dropout p {pz}; attention-probability p {ad}")
        else:
            dact = max(float((ev(mdl, b) - base_warm[nm])[b["cmask"]].abs().max()) for nm, b in (("S", bs), ("L", bl)))
            check(f"{arm}: active (perturbed new parameters change the eval logits)", dact > 1e-4, f"max|d| {dact:.2e}")
        mdl.train()
        torch.manual_seed(1)
        lo, _ = train_loss(M7, mdl, fresh(bl), pr_l)
        mdl.zero_grad(set_to_none=True)
        lo.backward()
        base_names = set(rsd)
        newp = [(n, p) for n, p in mdl.named_parameters() if n not in base_names or p.shape != rsd[n].shape]
        unused_ok = lambda n: (".lnc." in n and not n.startswith("cross.")) or n.startswith("ftok.blk.lnc")  # noqa
        no_grad = [n for n, p in mdl.named_parameters() if p.grad is None and not unused_ok(n)]
        dead = [n for n, p in newp if (p.grad is None or float(p.grad.abs().sum()) == 0.0) and not unused_ok(n)]
        fin = all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in mdl.parameters())
        check(f"{arm}: every used parameter gets a gradient; every new one non-zero; all finite",
              not no_grad and not dead and fin and bool(torch.isfinite(lo)),
              f"{len(newp)} new tensors; no-grad {no_grad[:4]}; zero-grad new {dead[:4]}; loss {lo.item():.4f}")
        m0 = build(M7, cfg, sd=sd)  # at the TRUE init, zero-init output layers must receive a gradient
        m0.train()
        torch.manual_seed(1)
        lo0, _ = train_loss(M7, m0, fresh(bl), pr_l)
        lo0.backward()
        gz = {n: float(p.grad.abs().sum()) for n, p in m0.named_parameters()
              if n.startswith(ZERO_INIT_OUT) and n.endswith("weight") and p.grad is not None and not n.startswith("hc.")}
        if arm == "hc4":
            gz = {n: float(p.grad.abs().sum()) for n, p in m0.named_parameters()
                  if n.startswith("hc.") and n.endswith(("A_m", "A_r", "B")) and p.grad is not None}
        if gz:
            check(f"{arm}: zero-init output / static HC tensors get a gradient at the true init",
                  all(v > 0 for v in gz.values()), f"{len(gz)} tensors, min |g|_1 {min(gz.values()):.2e}")
        del m0, mf
        mdl.eval()
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------ T8 params
    P("\n== T8 parameter counts")
    for k, v in RES["params"].items():
        P(f"  {k:9s} {v:>10,d}  ({v - RES['params']['base']:+,d})")

    # ------------------------------------------------------------------ T10 optimiser partition
    P("\n== T10 optimiser partition (train.build_opts, the code train.py runs)")
    tf32(True)
    part = {}
    for arm in ["base"] + list(arms):
        mdelta, tdelta = ({}, {}) if arm == "base" else arms[arm]
        mdl = build(M7, {**base_cfg, **mdelta})
        opts, info = T7.build_opts(mdl, args_for(tdelta), log=lambda *x: None)
        part[arm] = info
        P(f"  {arm:9s} Muon {len(info['muon']):3d} | AdamW decay {len(info['adamw_decay']):3d} | "
          f"no-decay {len(info['adamw_no_decay']):3d} | Muon momentum {opts[1].defaults['momentum']} | "
          f"betas {opts[0].defaults['betas']}")
    bm = set(part["base"]["muon"])
    check("base: 46 Muon matrices (as logged by the F7 / sx runs)", len(bm) == 46, len(bm))
    if "muon_in" in part:
        inp = {"c_mlp.0.weight", "c_lin.weight", "h_lin.weight", "u_lin.weight", "retr.wk.weight"}
        check("muon_in: Muon 46 -> 41; the 5 input projections move to AdamW decay",
              set(part["muon_in"]["muon"]) == bm - inp and inp <= set(part["muon_in"]["adamw_decay"]),
              len(part["muon_in"]["muon"]))
    if "pbld" in part:
        q = part["pbld"]
        check("pbld: P on Muon; W2 AdamW decay; w1 / b1 / b2 AdamW no-decay",
              "pbld.P.weight" in q["muon"] and "pbld.W2" in q["adamw_decay"]
              and {"pbld.w1", "pbld.b1", "pbld.b2"} <= set(q["adamw_no_decay"]))
    if "hc4" in part:
        q = part["hc4"]
        hcn = [n for n in q["adamw_no_decay"] if n.startswith("hc.")]
        check("hc4: every HC tensor on AdamW no-decay (incl. the 2-D A_r / W_*)",
              not any(n.startswith("hc.") for n in q["muon"] + q["adamw_decay"]) and len(hcn) == 8 * 9, len(hcn))
    if "dwa" in part:
        check("dwa: 4 alpha vectors on AdamW no-decay", sum(n.startswith("dwa.") for n in part["dwa"]["adamw_no_decay"]) == 4)
    if "derf" in part:
        q = part["derf"]
        check("derf: Derf alpha / s / weight / bias on AdamW no-decay",
              not any(n.endswith((".alpha", ".s")) for n in q["muon"] + q["adamw_decay"]))
    if "diff2" in part:
        q = part["diff2"]
        check("diff2: q2 and lam on Muon (+4 matrices), bias_scale2 no-decay",
              len(q["muon"]) == 50 and all(f"cross.{i}.attn.{x}.weight" in q["muon"] for i in (0, 1) for x in ("q2", "lam")))
    if "swiglu" in part:
        check("swiglu: gate / up / out separate Muon matrices (46 - 10 + 15 = 51)", len(part["swiglu"]["muon"]) == 51,
              len(part["swiglu"]["muon"]))

    # ------------------------------------------------------------------ T9 GPU step time, memory, trainability
    P("\n== T9 GPU step time on full real train batches (Mpad 64: 512 feeds x 64 = 32,768 slots; batch2s: 1,024 feeds)")
    del ms
    torch.cuda.empty_cache()
    import polars as pl
    f = pl.read_parquet(f"{ROOT}/data/flat/train/feeds.parquet", columns=["M"])
    Mall = f["M"].to_numpy()
    rng = np.random.default_rng(0)
    big = np.sort(rng.choice(np.where((Mall > 48) & (Mall <= 64))[0], 2048, replace=False))
    mb = MiniSplit("train", big, xt_names=base_cfg["xt_names"], device=DEV)
    full = [mb.gather(np.arange(0, 512), with_pos=True), mb.gather(np.arange(512, 1024), with_pos=True)]
    dbl = [mb.gather(np.arange(0, 1024), with_pos=True), mb.gather(np.arange(1024, 2048), with_pos=True)]
    prf = [mb.prand[x["fidx"]] for x in full]
    prd = [mb.prand[x["fidx"]] for x in dbl]
    P(f"  full batches: {[tuple(x['cmask'].shape) for x in full]} Hx {[x['hmask'].shape[1] for x in full]}; "
      f"double: {[tuple(x['cmask'].shape) for x in dbl]}")

    fstats = [sd[k].to(DEV) for k in ("f_mean", "f_std")]

    # predict()-size batch (cand_budget 65,536 / 1,024 feeds, as common.predict and batch2s training use): every arm
    # must run (FTok attends over B*M rows; fused SDPA kernels cap the batch grid axis at 65,535)
    torch.manual_seed(23)
    mref = M7.Ranker(**base_cfg).to(DEV)
    mref.load_state_dict(sd)
    o_ref = ev(mref, dbl[0])
    for arm, (mdelta, _) in arms.items():
        try:
            mm = build(M7, {**base_cfg, **mdelta}, sd=sd)
            o = ev(mm, dbl[0])
            mm.train()
            torch.manual_seed(3)
            lo, _ = train_loss(M7, mm, fresh(dbl[0]), prd[0], denom=1024)
            lo.backward()
            ok = bool(torch.isfinite(o[dbl[0]["cmask"]]).all()) and bool(torch.isfinite(lo))
            if arm in EXACT_WARM:
                ok = ok and torch.equal(o, o_ref)
            check(f"{arm}: 65,536-candidate batch (1,024 feeds) runs in eval and train, finite"
                  + (", eval == nn/" if arm in EXACT_WARM else ""), ok)
            del mm
        except RuntimeError as e:
            check(f"{arm}: 65,536-candidate batch (1,024 feeds) runs in eval and train", False, str(e)[:200])
        torch.cuda.empty_cache()
    del mref

    def run_steps(cfg, tdelta, batches, prs, n_warm=5, n_time=20, denom=512):
        torch.manual_seed(23)
        mdl = M7.Ranker(**cfg).to(DEV)
        mdl.set_norm(*fstats)
        mdl.train()
        opts, _ = T7.build_opts(mdl, args_for(tdelta), log=lambda *x: None)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        mem0 = torch.cuda.memory_allocated()
        ts, losses = [], []
        for it in range(n_warm + n_time):
            bb = fresh(batches[it % 2])
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            lo, _ = train_loss(M7, mdl, bb, prs[it % 2], denom=denom)
            for o in opts:
                o.zero_grad(set_to_none=True)
            lo.backward()
            torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
            for o in opts:
                o.step()
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
            losses.append(float(lo.detach()))
        peak = (torch.cuda.max_memory_allocated() - mem0) / 2**30
        del mdl, opts
        torch.cuda.empty_cache()
        return float(np.median(ts[n_warm:])), peak, losses

    tf32(True)
    timing = {}
    t_base, m_base, l_base = run_steps(base_cfg, {}, full, prf)
    timing["base"] = dict(ms=t_base * 1e3, rel=1.0, peak_gb=m_base, loss_first=l_base[0], loss_last=np.mean(l_base[-4:]))
    for arm, (mdelta, tdelta) in arms.items():
        if arm == "batch2s":
            t2, m2, l2 = run_steps(base_cfg, tdelta, dbl, prd, denom=1024)
            timing[arm] = dict(ms=t2 * 1e3, rel=t2 / t_base, per_sample_rel=t2 / (2 * t_base), peak_gb=m2,
                               loss_first=l2[0], loss_last=np.mean(l2[-4:]))
            continue
        t, m, losses = run_steps({**base_cfg, **mdelta}, tdelta, full, prf)
        timing[arm] = dict(ms=t * 1e3, rel=t / t_base, peak_gb=m, loss_first=losses[0], loss_last=np.mean(losses[-4:]))
    t_b2, _, _ = run_steps(base_cfg, {}, full, prf)  # baseline again: timing drift on a shared GPU
    timing["base_again"] = dict(ms=t_b2 * 1e3, rel=t_b2 / t_base)
    for arm, r in timing.items():
        extra = f"  per-sample {r['per_sample_rel']:.2f}x" if "per_sample_rel" in r else ""
        tr = f"  loss {r['loss_first']:.3f} -> {r['loss_last']:.3f}" if "loss_first" in r else ""
        pk = f"  peak +{r['peak_gb']:.1f} GB ({r['peak_gb'] - m_base:+.1f})" if "peak_gb" in r else ""
        P(f"  {arm:10s} {r['ms']:7.1f} ms/step ({r['rel']:.2f}x){extra}{pk}{tr}")
        if "loss_first" in r:
            check(f"{arm}: 25 optimiser steps finite and the training loss falls", np.isfinite(r["loss_last"])
                  and r["loss_last"] < r["loss_first"], f"{r['loss_first']:.3f} -> {r['loss_last']:.3f}")
    RES["gpu_timing"] = timing
    RES["partition"] = {k: dict(muon=len(v["muon"]), decay=len(v["adamw_decay"]), no_decay=len(v["adamw_no_decay"]))
                        for k, v in part.items()}
    RES["failed"] = FAIL
    P(f"\n== {len(RES['checks']) - len(FAIL)}/{len(RES['checks'])} checks passed; failed: {FAIL}; "
      f"{time.time() - t_all:.0f}s")
    json.dump(RES, open(f"{R7}/logs/test_r7.json", "w"), indent=1, default=float)
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())
