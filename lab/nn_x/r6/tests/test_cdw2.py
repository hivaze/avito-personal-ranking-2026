"""Unit tests for the r6 wave-2 cooldown-continuation (CD) flags: train.py --drop_cd / --pool_p / --aug_cand / --aug_hist
(helpers in nn_x/r6/cd_arms.py; R6_PLAN section 4.1 "Wave 2"). GPU mode by default (user rule: NN work on GPUs), <= 12
CPU threads, read-only on every shared file.

  CUDA_VISIBLE_DEVICES=N OMP_NUM_THREADS=12 <repo>/.venv/bin/python nn_x/r6/tests/test_cdw2.py
  (TRAIN_PY=<path> / CD_LANE=<path> test staged copies; defaults nn_x/r6/train.py and nn_x/r6/cd_lane.sh)

Model = the CD init itself, nn/ckpt/v17_pre2s_s101@77500.pt (F7 @77.5k, read-only), on real train batches (MiniSplit).
GPU mode uses deterministic kernels (math SDPA, no TF32, deterministic algorithms), as test_r6.py, so equalities are exact.
  W0 code: (a) the cd_lane.sh commands (DRY) parse with train.py's own parser; c0 leaves every wave-2 flag at its OFF
     default and each arm differs from c0 only by its own flags; (b) AST proof of the flag-off path: train.py minus the
     wave-2 statements (each guarded by a condition that is False at the defaults, or side-effect free) and with
     exploration_pool(...) replaced by the old `feeds[prand_np[feeds] > 0.5]` is IDENTICAL to the pre-wave-2 train.py
     (tests/ref_train_pre_cdw2.py.txt); (c) set_dropout / augment_pool_batch are called only inside the training loop,
     behind `a.drop_cd >= 0 and step == t_cool` / `is_pool and aug_on`; no eval/predict code references them.
  W1 flag-off bit-identity (T2-style): the helpers at OFF values leave batch, model and the CPU/CUDA RNG untouched;
     same seed -> identical train-mode logits, loss (listwise feed + K9) and every gradient; set_dropout(model, 0.1)
     (= the trained p) is bit-identical too (the override itself is exact).
  W2 d0/d20: every dropout site set (F7: 11 nn.Dropout + 5 attention-probability sites), complete vs model.py's source;
     weights / cfg / eval logits unchanged; d0: train-mode logits == eval-mode logits (all stochasticity removed), while
     zeroing only nn.Dropout does not (attention-probability dropout is live and covered); d20: measured drop rate ~0.2.
  W3 pp02 (real arrays, numpy): pool sizes p>0.5 = 134,118 (= the c0 logs) and p>0.2 = 197,477 (+47%), nested, M>=50
     only, zero holdout feeds (the 30K holdout is entirely pool-eligible, so the exclusion is material), leak guard fires.
  W4 aug: only cmask / hmask / hemb_ok change (y = y*cmask); every positive kept; kept-negative share ~U(A,1) (mean ~(1+A)/2);
     history drop ~H; no empty feed; inputs of dropped candidates / events have no influence on the kept logits or the
     train-mode loss (exact); loss and gradients finite; the aug arm's guard is True only on pool steps.
Writes nn_x/r6/logs/test_cdw2.log (+ .json).
"""
import argparse
import ast
import copy
import json
import os
import re
import subprocess
import sys
import time

_CVD = os.environ.get("CUDA_VISIBLE_DEVICES", "x")
assert _CVD == "" or _CVD in tuple(str(i) for i in range(8)), \
    "run with CUDA_VISIBLE_DEVICES=<one GPU id> (or '' for CPU)"
if _CVD:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
NT = min(12, int(os.environ.get("OMP_NUM_THREADS", "12")))
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(NT)
DEV = torch.device("cuda:0" if _CVD else "cpu")
if DEV.type == "cuda":
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
HERE = os.path.dirname(os.path.abspath(__file__))
R6 = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(1, R6)
import warnings  # noqa: E402

warnings.filterwarnings("ignore", message=".*CUDA is not available.*")
warnings.filterwarnings("ignore", message=".*device_type of 'cuda'.*")
import cd_arms as CA  # noqa: E402  (nn_x/r6/cd_arms.py)
import model as M6  # noqa: E402
from mini_split import MiniSplit, pick_train_feeds  # noqa: E402

ROOT = "<repo>"
INIT = f"{ROOT}/nn/ckpt/v17_pre2s_s101@77500.pt"
TRAIN_PY = os.environ.get("TRAIN_PY", f"{R6}/train.py")
CD_LANE = os.environ.get("CD_LANE", f"{R6}/cd_lane.sh")
REF_TRAIN = f"{HERE}/ref_train_pre_cdw2.py.txt"
WAVE2_FLAGS = ("--drop_cd", "--pool_p", "--aug_cand", "--aug_hist")
OFF = dict(drop_cd=-1.0, pool_p=0.5, aug_cand=0.0, aug_hist=0.0)
ARM_FLAGS = {"c0": {}, "d0": dict(drop_cd=0.0), "d20": dict(drop_cd=0.2), "pp02": dict(pool_p=0.2),
             "aug": dict(aug_cand=0.6, aug_hist=0.15)}
LOG = open(f"{R6}/logs/test_cdw2.log", "w")
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


def fresh(b):
    return {k: (v.clone() if torch.is_tensor(v) else copy.deepcopy(v)) for k, v in b.items()
            if k not in ("ce", "he", "un")}


def clone(m):
    """deepcopy of a model after a train-mode forward (drops the non-leaf aux_out / intent_logp attributes first)."""
    for mod in m.modules():
        for k in ("aux_out", "intent_logp"):
            mod.__dict__.pop(k, None)
    return copy.deepcopy(m)


def same_batch(b1, b2, keys=None):
    keys = keys or sorted(set(b1) | set(b2))
    return [k for k in keys if not (k in b1 and k in b2 and torch.equal(b1[k], b2[k]))]


def train_loss(model, b, prand_b, aux_w=0.5, tau=10.0, denom=512):
    """train.py's loss for the CD recipe (softmax, loss_norm feed, no prand_base, K9 listwise aux weighted 1 - p)."""
    X = model.dense(b)
    lg = model(b, X, use_pos=True)
    y, m = b["y"], b["cmask"]
    loss = M6.listwise_loss(lg, y, m, None, norm="feed", denom=denom)
    tgt = torch.softmax((-b["pos"].float() / tau).masked_fill(~m, -1e4), 1) * m
    lp = torch.log_softmax(model.aux_out.masked_fill(~m, -1e4), 1)
    fw_aux = 1 - prand_b
    loss = loss + aux_w * (-(tgt * lp).sum(1) * fw_aux).sum() / fw_aux.sum().clamp_min(1e-6)
    return loss, lg


def rng_state():
    s = [torch.get_rng_state()]
    if DEV.type == "cuda":
        s.append(torch.cuda.get_rng_state())
    return s


def rng_same(s1, s2):
    return all(torch.equal(x, y) for x, y in zip(s1, s2))


def load_parse(path):
    """train.py's own parse() (AST-extracted, no import of common.py / no GPU guard)."""
    tree = ast.parse(open(path).read())
    fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "parse"][0]
    ns = {"argparse": argparse, "POOL_P_DEFAULT": CA.POOL_P_DEFAULT}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "train_parse", "exec"), ns)

    def parse(argv):
        sys.argv = ["train.py"] + argv
        return ns["parse"]()
    return parse


# ------------------------------------------------------------------------------------------------ W0 code-level checks
GUARDS = {"aug_on or a.pool_p != POOL_P_DEFAULT", "aug_on", "a.drop_cd >= 0", "a.pool_p != POOL_P_DEFAULT or aug_on",
          "a.drop_cd >= 0 and step == t_cool", "is_pool and aug_on"}


class StripWave2(ast.NodeTransformer):
    """Removes the wave-2 statements of train.py and restores the old pool expression (records what it removed)."""

    def __init__(self):
        self.removed_ifs, self.removed_other, self.pool_calls = [], [], 0

    def visit_ImportFrom(self, n):
        if n.module == "cd_arms":
            self.removed_other.append("import cd_arms")
            return None
        return n

    def visit_Expr(self, n):
        v = n.value
        if (isinstance(v, ast.Call) and isinstance(v.func, ast.Attribute) and v.func.attr == "add_argument"
                and v.args and isinstance(v.args[0], ast.Constant) and v.args[0].value in WAVE2_FLAGS):
            self.removed_other.append(f"add_argument {v.args[0].value}")
            return None
        return self.generic_visit(n)

    def visit_Assign(self, n):
        if len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "aug_on":
            self.removed_other.append("aug_on = " + ast.unparse(n.value))
            return None
        return self.generic_visit(n)

    def visit_Assert(self, n):
        if ast.unparse(n.test).startswith("a.drop_cd < 0 or"):
            self.removed_other.append("assert " + ast.unparse(n.test))
            return None
        return n

    def visit_If(self, n):
        t = ast.unparse(n.test)
        if t in GUARDS:
            self.removed_ifs.append(t)
            return None
        return self.generic_visit(n)

    def visit_Call(self, n):
        if isinstance(n.func, ast.Name) and n.func.id == "exploration_pool":
            self.pool_calls += 1
            return ast.parse("feeds[prand_np[feeds] > 0.5]", mode="eval").body
        return self.generic_visit(n)


def strip_doc(tree):
    if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant):
        tree.body = tree.body[1:]
    return tree


def w0(parse):
    P("\n== W0 code: cd_lane.sh commands, AST flag-off proof, call sites")
    out = subprocess.run(["bash", CD_LANE, "0", " ".join(f"{a}:101" for a in ARM_FLAGS)], capture_output=True, text=True,
                         env={**os.environ, "DRY": "1"})
    cmds = {}
    for line in out.stdout.splitlines():
        if " train.py " in line:
            argv = line.split(" train.py ", 1)[1].split()
            arm = re.search(r"r6_cd_(\w+)_r101", line).group(1)
            cmds[arm] = argv
    check("cd_lane.sh DRY prints one train.py command per arm", sorted(cmds) == sorted(ARM_FLAGS) and out.returncode == 0,
          sorted(cmds))
    ns = {arm: vars(parse(argv)) for arm, argv in cmds.items()}
    c0 = ns["c0"]
    check("c0: every wave-2 flag at its OFF default", {k: c0[k] for k in OFF} == OFF, {k: c0[k] for k in OFF})
    for arm, fl in ARM_FLAGS.items():
        if arm == "c0":
            continue
        diff = {k: (c0[k], v) for k, v in ns[arm].items() if k != "variant" and v != c0[k]}
        want = {k: (OFF[k], v) for k, v in fl.items()}
        check(f"{arm}: differs from c0 only by {fl}", diff == want, diff)
    check("CD harness: --init F7 @77500, loss_norm feed, cooldown_frac 1, mix 0.1, holdout 30000, seed 811",
          c0["init"].endswith("v17_pre2s_s101@77500.pt") and c0["loss_norm"] == "feed" and c0["cooldown_frac"] == 1.0
          and c0["cooldown_mix"] == 0.1 and c0["holdout"] == 30000 and c0["seed"] == 811 and c0["select_on"] == "holdout")
    RES["cd_lane_cmds"] = {k: " ".join(v) for k, v in cmds.items()}

    new_src, old_src = open(TRAIN_PY).read(), open(REF_TRAIN).read()
    st = StripWave2()
    new_t = strip_doc(st.visit(ast.parse(new_src)))
    old_t = strip_doc(ast.parse(old_src))
    ident = ast.dump(new_t) == ast.dump(old_t)
    check("AST: train.py minus the wave-2 statements == pre-wave-2 train.py (flag-off path unchanged)",
          ident and st.pool_calls == 1 and sorted(st.removed_ifs) == sorted(GUARDS),
          f"removed guards {len(st.removed_ifs)}, other {st.removed_other}, exploration_pool calls {st.pool_calls}")
    # every removed guard is False at the defaults (c0); the aug guard is True only on pool steps for the aug arm
    a0 = argparse.Namespace(**c0)
    env0 = dict(a=a0, aug_on=CA.aug_enabled(a0.aug_cand, a0.aug_hist), POOL_P_DEFAULT=CA.POOL_P_DEFAULT, step=0, t_cool=0)
    vals = {g: bool(eval(g, {}, {**env0, "is_pool": ip})) for g in GUARDS for ip in (True,)}
    check("every wave-2 guard is False for c0 (also at step == t_cool and on pool steps)", not any(vals.values()), vals)
    aa = argparse.Namespace(**ns["aug"])
    enva = dict(a=aa, aug_on=CA.aug_enabled(aa.aug_cand, aa.aug_hist), POOL_P_DEFAULT=CA.POOL_P_DEFAULT, step=0, t_cool=0)
    g_pool = eval("is_pool and aug_on", {}, {**enva, "is_pool": True})
    g_reg = eval("is_pool and aug_on", {}, {**enva, "is_pool": False})
    check("aug arm: augmentation guard True on pool steps, False on regular steps", g_pool is True and g_reg is False)
    ad = argparse.Namespace(**ns["d0"])
    check("d0 arm: dropout guard fires at step 0 = t_cool (cooldown_frac 1.0 -> right after --init)",
          eval("a.drop_cd >= 0 and step == t_cool", {}, dict(a=ad, step=0, t_cool=int(round((1 - ad.cooldown_frac) * 100))))
          is True)
    # call sites: inside `for step, fb in enumerate(all_batches)` and under the right guard; nowhere else in nn_x/r6
    tree = ast.parse(new_src)
    parents = {}
    for p_ in ast.walk(tree):
        for ch in ast.iter_child_nodes(p_):
            parents[ch] = p_
    sites = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in ("set_dropout", "augment_pool_batch"):
            chain, x = [], n
            while x in parents:
                x = parents[x]
                if isinstance(x, ast.If):
                    chain.append("if " + ast.unparse(x.test))
                if isinstance(x, ast.For):
                    chain.append("for " + ast.unparse(x.target))
            sites.setdefault(n.func.id, []).append(chain)
    ok = (sites.get("set_dropout") == [["if a.drop_cd >= 0 and step == t_cool", "for (step, fb)"]]
          and sites.get("augment_pool_batch") == [["if is_pool and aug_on", "for (step, fb)"]])
    check("set_dropout / augment_pool_batch: one call each, in the training loop, behind their guards", ok, sites)
    users = sorted(f for f in os.listdir(R6) if f.endswith(".py") and f not in ("cd_arms.py",)
                   and "augment_pool_batch" in open(f"{R6}/{f}").read())
    users = [u for u in users if os.path.join(R6, u) != TRAIN_PY and u != "train.py"]
    check("no other nn_x/r6 module (predict / holdout_eval / finetune / common) uses the augmentation", not users, users)


# ------------------------------------------------------------------------------------------------ W3 pool (numpy)
def w3():
    P("\n== W3 pp02: exploration pool on the real train arrays (train.py holdout and p_random rules)")
    f = pl.read_parquet(f"{ROOT}/data/flat/train/feeds.parquet", columns=["M"])
    M = f["M"].to_numpy().astype(np.int64)
    pr = np.load(f"{ROOT}/eda/labels_position/out/train_prandom.npz")
    assert (pr["fidx"] == np.arange(len(M))).all()
    p = np.nan_to_num(pr["p_random"].astype(np.float32), nan=0.0)
    p[M < 50] = 0.0
    cand = np.where((p > 0.9) & (M >= 50))[0]
    hold = np.sort(np.random.default_rng(12345).choice(cand, min(30000, len(cand)), replace=False))
    feeds = np.setdiff1d(np.arange(len(M)), hold)
    p05 = CA.exploration_pool(feeds, p, 0.5, hold)
    p02 = CA.exploration_pool(feeds, p, 0.2, hold)
    old = feeds[p[feeds] > 0.5]
    check("pool_p 0.5 == the old expression feeds[prand_np[feeds] > 0.5] (array-equal)", np.array_equal(p05, old),
          len(old))
    logn = []
    for r in (101, 102, 103, 104):
        try:
            s = open(f"{R6}/logs/r6_cd_c0_r{r}.log").read()
            logn += [int(x) for x in re.findall(r"exploration pool (\d+) feeds", s)]
        except FileNotFoundError:
            pass
    check("pool_p 0.5 size == the pool of the c0 runs (logs)", len(p05) == 134118 and logn and set(logn) == {len(p05)},
          f"{len(p05)} vs logs {sorted(set(logn))}")
    check("pool_p 0.2: 197,477 feeds = +47% distinct pool feeds (sol_solvers 2.6 / P3: ~197,500)",
          len(p02) == 197477, f"{len(p02)} ({len(p02) / len(p05) - 1:+.1%})")
    check("pool_p 0.5 pool is a subset of the 0.2 pool; the extra feeds have p in (0.2, 0.5]",
          np.isin(p05, p02).all() and bool(((p[np.setdiff1d(p02, p05)] > 0.2) & (p[np.setdiff1d(p02, p05)] <= 0.5)).all()))
    check("both pools: M >= 50 only", M[p05].min() >= 50 and M[p02].min() >= 50, (int(M[p05].min()), int(M[p02].min())))
    n_elig = int((p[hold] > 0.2).sum())
    check("LEAK GUARD: zero holdout feeds in either pool (all 30,000 holdout feeds are pool-eligible by p_random)",
          not np.isin(p05, hold).any() and not np.isin(p02, hold).any() and n_elig == len(hold) == 30000,
          f"holdout feeds with p > 0.2: {n_elig}")
    try:
        CA.exploration_pool(np.union1d(feeds, hold[:3]), p, 0.2, hold)
        fired = False
    except AssertionError:
        fired = True
    check("leak guard fires when a holdout feed reaches the candidate feeds", fired)
    RES["pool"] = dict(p05=len(p05), p02=len(p02), ratio=len(p02) / len(p05))


# ------------------------------------------------------------------------------------------------ model tests
def main():
    t_all = time.time()
    P(f"== r6 wave-2 CD flag tests ({time.ctime()}, device {DEV}, CPU threads {NT})")
    P(f"   train.py: {TRAIN_PY}   cd_lane: {CD_LANE}   init: {INIT}")
    parse = load_parse(TRAIN_PY)
    w0(parse)
    w3()

    ck = torch.load(INIT, map_location="cpu", weights_only=False)
    cfg, sd = dict(ck["cfg"]), ck["state_dict"]
    torch.manual_seed(0)
    base = M6.Ranker(**cfg)
    base.load_state_dict(sd)
    base.to(DEV)
    feeds = pick_train_feeds(n_each=48, seed=0)
    ms = MiniSplit("train", feeds, xt_names=cfg["xt_names"]).to(DEV)
    prn = ms.prand.cpu().numpy()
    idx_pool = np.where((prn > 0.5) & (ms.M_np >= 50))[0][:64]
    idx_reg = np.where((prn < 0.1) & (ms.M_np >= 50) & (ms.M_np <= 200))[0][:48]
    bp = ms.gather(idx_pool, with_pos=True)
    br = ms.gather(idx_reg, with_pos=True)
    pr_p, pr_r = ms.prand[bp["fidx"]], ms.prand[br["fidx"]]
    P(f"\n   F7 cfg: drop {cfg['drop']} arch {cfg['arch']} retr_m {cfg['retr_m']} xt_drop {cfg['xt_drop']} "
      f"id_buckets {cfg['id_buckets']} pos_tower {cfg['use_pos_tower']}; pool batch B {bp['cmask'].shape[0]} Mx "
      f"{bp['cmask'].shape[1]} Hx {bp['hmask'].shape[1]}; regular batch B {br['cmask'].shape[0]}")

    # -------------------------------------------------------------------------------------------- W1 flag-off identity
    P("\n== W1 flag-off bit-identity (T2-style, F7 @77.5k weights, real pool and regular batches)")
    for nm, b in (("pool", bp), ("regular", br)):
        b1 = fresh(b)
        s0 = rng_state()
        r = CA.augment_pool_batch(b1, 0.0, 0.0)
        r2 = CA.augment_pool_batch(b1, 1.0, 0.0)  # q_min >= 1 is also a no-op
        check(f"augment_pool_batch at OFF values: same object, every tensor unchanged, RNG untouched ({nm})",
              r is b1 and r2 is b1 and not same_batch(b1, fresh(b)) and rng_same(s0, rng_state()))
    check("aug_enabled truth table", (not CA.aug_enabled(0.0, 0.0)) and (not CA.aug_enabled(1.0, 0.0))
          and CA.aug_enabled(0.6, 0.0) and CA.aug_enabled(0.0, 0.15) and CA.aug_enabled(0.6, 0.15))
    m_off = clone(base)
    m_eq = clone(base)
    n_sites = CA.set_dropout(m_eq, cfg["drop"])  # override with the trained value: must be exact
    for nm, b, prb in (("pool", bp, pr_p), ("regular", br, pr_r)):
        outs = []
        for mdl in (base, m_off, m_eq):
            mdl.train()
            torch.manual_seed(77)
            bb = fresh(b)
            if mdl is m_off:  # train.py's flag-off path: neither hook runs (guards False); helpers at OFF are no-ops
                CA.augment_pool_batch(bb, 0.0, 0.0)
            lo, lg = train_loss(mdl, bb, prb)
            mdl.zero_grad(set_to_none=True)
            lo.backward()
            outs.append((lo.detach(), lg.detach(), [p.grad.clone() if p.grad is not None else None
                                                    for p in mdl.parameters()]))
        for tag, o in (("flag-off path", outs[1]), (f"set_dropout(p={cfg['drop']})", outs[2])):
            g_ok = all((g1 is None and g2 is None) or (g1 is not None and g2 is not None and torch.equal(g1, g2))
                       for g1, g2 in zip(outs[0][2], o[2]))
            check(f"{tag} vs plain model, {nm} batch, same seed: train logits, loss, all grads identical",
                  torch.equal(outs[0][1], o[1]) and torch.equal(outs[0][0], o[0]) and g_ok,
                  f"loss {outs[0][0].item():.6f} vs {o[0].item():.6f}")
    base.eval()

    # -------------------------------------------------------------------------------------------- W2 dropout
    P("\n== W2 --drop_cd (d0 / d20)")
    src = open(f"{R6}/model.py").read()
    n_mod_defs = len(re.findall(r"nn\.Dropout\(", src))
    n_sdpa = len(re.findall(r"dropout_p=", src))
    other = re.findall(r"F\.dropout\(|functional\.dropout\(|torch\.dropout\(|AlphaDropout|Dropout[123]d|"
                       r"MultiheadAttention|TransformerEncoderLayer|nn\.LSTM", src)
    check("model.py dropout sites: nn.Dropout (Block ff + dr, HistRetrieval.dp, CandRetrieval.dp) and SDPA dropout_p "
          "only; no functional / other dropout", n_mod_defs == 4 and n_sdpa == 1 and not other,
          f"nn.Dropout defs {n_mod_defs}, dropout_p {n_sdpa}, other {other}")
    gru_drop = [m.dropout for m in base.modules() if isinstance(m, (torch.nn.GRU, torch.nn.LSTM))]
    mods, attns = CA.dropout_sites(base)
    check("F7: 11 nn.Dropout modules + 5 attention modules found; no recurrent dropout",
          (len(mods), len(attns)) == (11, 5) and all(d == 0 for d in gru_drop) and n_sites == (11, 5),
          f"{len(mods)} Dropout, {len(attns)} Attn, set_dropout returned {n_sites}")
    with torch.no_grad():
        ev0 = {nm: base(fresh(b)) for nm, b in (("pool", bp), ("regular", br))}
    variants = {}
    for P_ in (0.0, 0.2):
        m = clone(base)
        cfg0 = copy.deepcopy(m.cfg)
        sd0 = {k: v.clone() for k, v in m.state_dict().items()}
        CA.set_dropout(m, P_)
        mods, attns = CA.dropout_sites(m)
        check(f"drop_cd {P_}: every nn.Dropout p and every Attn attention-probability p == {P_}",
              all(x.p == P_ for x in mods) and all(x.attn_drop == P_ for x in attns),
              f"{sorted(set(x.p for x in mods))} / {sorted(set(x.attn_drop for x in attns))}")
        m.eval()
        with torch.no_grad():
            evp = {nm: m(fresh(b)) for nm, b in (("pool", bp), ("regular", br))}
        check(f"drop_cd {P_}: weights, buffers and cfg unchanged; eval logits bit-identical (predict / holdout paths)",
              m.cfg == cfg0 and m.cfg["drop"] == cfg["drop"] and all(torch.equal(sd0[k], v) for k, v in m.state_dict().items())
              and all(torch.equal(ev0[k], evp[k]) for k in ev0))
        variants[P_] = m
    m0 = variants[0.0]
    m0.train()
    with torch.no_grad():
        diffs = []
        for nm, b in (("pool", bp), ("regular", br)):
            bb = fresh(b)
            X = m0.dense(bb)
            lt = m0(bb, X, use_pos=True)
            diffs.append(float((lt - ev0[nm])[b["cmask"]].abs().max()))
    check("d0: train-mode logits == eval-mode logits (every stochastic site removed, incl. attention probabilities)",
          max(diffs) == 0.0, f"max|d| {diffs}")
    mh = clone(base)  # half override: nn.Dropout 0 but attention-probability dropout left at 0.1
    for x in CA.dropout_sites(mh)[0]:
        x.p = 0.0
    mh.train()
    with torch.no_grad():
        torch.manual_seed(3)
        lt = mh(fresh(bp))
    dh = float((lt - ev0["pool"])[bp["cmask"]].abs().max())
    check("zeroing only nn.Dropout leaves train != eval: the attention-probability site is live (set_dropout covers it)",
          dh > 0, f"max|d| {dh:.2e}")
    m20 = variants[0.2]
    m20.train()
    stats = []
    hooks = [x.register_forward_hook(lambda mod, i, o: stats.append(
        (float(((o == 0) & (i[0] != 0)).sum()), float((i[0] != 0).sum())))) for x in CA.dropout_sites(m20)[0]]
    with torch.no_grad():
        torch.manual_seed(5)
        o20 = m20(fresh(bp))
        bref = clone(base)
        bref.train()
        torch.manual_seed(5)
        o10 = bref(fresh(bp))
    for h in hooks:
        h.remove()
    rate = sum(s[0] for s in stats) / max(sum(s[1] for s in stats), 1)
    check("d20: measured nn.Dropout rate ~0.2 (fwd hooks over all 11 modules)", abs(rate - 0.2) < 0.01,
          f"rate {rate:.4f} over {sum(s[1] for s in stats):.0f} elements")
    check("d20: train-mode outputs differ from the p=0.1 model at the same seed", not torch.equal(o20, o10))
    RES["dropout"] = dict(rate20=rate, n_sites=list(n_sites))

    # -------------------------------------------------------------------------------------------- W4 augmentation
    P("\n== W4 --aug_cand 0.6 --aug_hist 0.15 on a real pool batch")
    A, H = 0.6, 0.15
    b0 = fresh(bp)
    b1 = fresh(bp)
    torch.manual_seed(11)
    CA.augment_pool_batch(b1, A, H)
    changed = same_batch(b0, b1)
    check("only cmask, hmask, hemb_ok change (y = y * cmask is unchanged: only negatives are dropped)",
          sorted(changed) == ["cmask", "hemb_ok", "hmask"], changed)
    c0m, c1m = b0["cmask"], b1["cmask"]
    check("candidate mask only shrinks; history masks only shrink",
          bool((c1m <= c0m).all() and (b1["hmask"] <= b0["hmask"]).all() and (b1["hemb_ok"] <= b0["hemb_ok"]).all()))
    pos = (b0["y"] > 0) & c0m
    check("every positive kept; per-feed positive count unchanged; y zero on dropped rows",
          bool((c1m >= pos).all() and torch.equal(b1["y"].sum(1), b0["y"].sum(1)) and (b1["y"][~c1m] == 0).all()),
          f"{int(pos.sum())} positives")
    neg = c0m & (b0["y"] == 0)
    share = ((c1m & neg).sum(1).float() / neg.sum(1).clamp_min(1).float())[neg.sum(1) > 0]
    check("kept-negative share per feed ~ U(0.6, 1): mean ~0.8, all feeds within [0.45, 1]",
          abs(float(share.mean()) - 0.8) < 0.04 and float(share.min()) >= 0.45,
          f"mean {float(share.mean()):.3f} min {float(share.min()):.3f} max {float(share.max()):.3f} ({len(share)} feeds)")
    hd = float((b0["hmask"] & ~b1["hmask"]).sum() / b0["hmask"].sum())
    check("history-event drop rate ~0.15 (hmask and hemb_ok together)", abs(hd - 0.15) < 0.01
          and torch.equal(b1["hemb_ok"], b0["hemb_ok"] & b1["hmask"]), f"rate {hd:.4f}")
    check("no feed loses every candidate", bool(c1m.any(1).all()))
    # dropped candidates / events must have no influence on the kept logits or on the loss
    g = torch.Generator().manual_seed(7)
    bq = fresh(b1)
    dc, dh_ = c0m & ~c1m, b0["hmask"] & ~b1["hmask"]
    nc, nh = int(dc.sum()), int(dh_.sum())
    bq["ce_raw"][dc] = (torch.randn(nc, 128, generator=g) * 50).to(DEV)
    bq["cf"][dc] = (torch.randn(nc, 5, generator=g) * 9).to(DEV)
    bq["ccat"][dc] = torch.randint(0, 54, (nc,), generator=g).to(DEV)
    bq["xt"][dc] = (torch.randn(nc, bq["xt"].shape[-1], generator=g) * 3).to(DEV, bq["xt"].dtype)
    bq["cmiss"][dc] = (torch.rand(nc, generator=g) < 0.5).to(DEV)
    bq["pos"][dc] = torch.randint(0, 300, (nc,), generator=g).to(DEV)
    bq["he_raw"][dh_] = (torch.randn(nh, 128, generator=g) * 50).to(DEV)
    bq["het"][dh_] = torch.randint(1, 11, (nh,), generator=g).to(DEV)
    bq["hes"][dh_] = torch.randint(0, 30, (nh,), generator=g).to(DEV)
    bq["hcat"][dh_] = torch.randint(0, 54, (nh,), generator=g).to(DEV)
    bq["hidx"][dh_] = torch.randint(0, 10, (nh,), generator=g).to(DEV, bq["hidx"].dtype)
    base.eval()
    with torch.no_grad():
        la, lq = base(fresh(b1)), base(fresh(bq))
    dm = float((la - lq)[c1m].abs().max())
    check("eval: kept-candidate logits independent of the dropped candidates' and events' inputs (exact)", dm == 0.0,
          f"max|d| {dm:.2e} ({nc} dropped candidates, {nh} dropped events perturbed)")
    tr_m = clone(base)
    tr_m.train()
    res = []
    for bb in (b1, bq, b0):
        torch.manual_seed(13)
        lo, _ = train_loss(tr_m, fresh(bb), pr_p)
        tr_m.zero_grad(set_to_none=True)
        lo.backward()
        res.append((lo.item(), all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in tr_m.parameters())))
    check("train-mode loss identical with the dropped rows perturbed (same seed); loss and grads finite",
          res[0][0] == res[1][0] and res[0][1] and res[1][1] and np.isfinite(res[0][0]),
          f"loss aug {res[0][0]:.6f} perturbed {res[1][0]:.6f} | unaugmented {res[2][0]:.6f}")
    check("augmentation changes the training loss (it is active)", res[0][0] != res[2][0])
    RES["aug"] = dict(kept_neg_mean=float(share.mean()), hist_drop=hd)
    tr_m.eval()

    RES["failed"] = FAIL
    P(f"\n== {len(RES['checks']) - len(FAIL)}/{len(RES['checks'])} checks passed; failed: {FAIL}; "
      f"{time.time() - t_all:.0f}s")
    json.dump(RES, open(f"{R6}/logs/test_cdw2.json", "w"), indent=1, default=str)
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())
