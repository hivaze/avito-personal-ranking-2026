"""Tests of the nn_x/w10 R4 arm (2026-11-13): nn_x/w10/r4 = a private copy of nn_x/w5 with train.py --dcn_par L / --dcn_rank r /
--dcn_nox (model.py GDCNBranch, Ranker(dcn_par, dcn_rank, dcn_nox)), OFF by default.

  cpu part:  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 .venv/bin/python nn_x/w10/r4/tests/test_r4.py cpu
  gpu part:  CUDA_VISIBLE_DEVICES=<2|3> OMP_NUM_THREADS=12 .venv/bin/python nn_x/w10/r4/tests/test_r4.py gpu

CPU part
  C0 code. (a) module copies: data / cd_arms* / blend / holdout_eval / stale_pred / finetune byte-identical to nn_x/w5; common.py
     AST-equal to nn_x/w5/common.py up to the output-path constant (W5 -> nn_x/w10/r4); the nn_x/w5 sources are unchanged since
     the copy (md5 in tests/w5_source_md5.txt). (b) AST proof: model.py minus the GDCNBranch class, the three __init__ keywords,
     the `self.dcnp = None` assignment, the `if dcn_par:` statement and the `if self.dcnp is not None:` statement in forward
     == nn_x/w5/model.py; train.py minus the three add_argument statements, the `if a.dcn_par:` statement and the three Ranker
     keywords, with the variant assertion restored to the w5 one == nn_x/w5/train.py.
     (c) lane.sh DRY: A:r4 / E:r4 == the nn_x/w5 lane's A:base / E:base (same parser) + exactly {dcn_par 2}; A:r4c adds dcn_nox;
     the nn_x/abl abl_base_s23 / s24 stored args == A_BASE + HYG (the paired screen base); lane.sh refuses GPUs other than 2 / 3.
  C1 flag-off identity: Ranker built from nn_x/w5/model.py and from nn_x/w10/r4/model.py under the same seed and the stage-E cfg
     -> bit-identical state_dict and cfg; dense features + logits on real MiniSplit train batches bit-identical in eval mode
     and in train mode (same RNG state); the r4 Ranker accepts a w5 cfg (no dcn keys).
  C2 flag-on (dcn_par 2, rank 32, with and without X): ZERO-INIT proof: the flag-on model's logits at init are bit-identical
     to the base's in eval mode AND in train mode (no RNG draw in the branch; the base parameters keep their init under the RNG
     fork); cfg stores the dcn keys only when ON; x0 width d + n_dense (nox: d); Muon rule covers exactly the 4 L cross / gate
     matrices and excludes the output layer; at init the output layer receives a non-zero gradient while every U/V/G/LN gradient
     is exactly zero (zero-init output), after one perturbation of the output weight every branch parameter has a finite non-zero
     gradient and the logits differ from the base (the branch is live); padded candidate slots never influence the valid logits
     and one feed never influences another (bitwise); cfg round trip Ranker(**cfg).load_state_dict(strict) reproduces the logits.
GPU part (one GPU, deterministic kernels; test variants w5_t10e2e_* / w10_t10e2e_*, outputs deleted afterwards)
  G1 end-to-end identity (tests/e2e_w5.py wrapper, 39 CD steps from the v17_pre2s_s101@77500 snapshot, cooldown_mix 0.2, evals at
     20 and 39): the frozen nn_x/w5 code (tests/ref_w5, writing into nn_x/w10/r4) vs nn_x/w10/r4/train.py at flags OFF ->
     identical batch sequence, train-mode logits, loss-input logits, logged lines, bit-identical checkpoint and val preds.
  G2 flags ON from scratch (dcn_par 2, 20K train feeds, ~60 steps, evals + hygiene, test preds): runs end to end, writes the
     selected and @last checkpoints / val / holdout / test preds, cfg stores dcn_par 2 / dcn_rank 32 / dcn_nox False; the Muon
     log line counts the 8 extra matrices; holdout_eval.py reproduces the saved holdout preds bitwise; finetune.build_model
     rebuilds the model from the stored cfg and loads the state_dict strictly; logits of the rebuilt model == the checkpoint's.
Writes nn_x/w10/r4/logs/test_r4_<part>.log (+ .json).
"""
import ast
import collections
import hashlib
import json
import os
import re
import subprocess
import sys
import time

PART = sys.argv[1] if len(sys.argv) > 1 else "cpu"
assert PART in ("cpu", "gpu"), "usage: test_r4.py cpu|gpu"
_CVD = os.environ.get("CUDA_VISIBLE_DEVICES", "x")
if PART == "cpu":
    assert _CVD == "", "cpu part: run with CUDA_VISIBLE_DEVICES=''"
else:
    assert _CVD in ("2", "3"), "gpu part: CUDA_VISIBLE_DEVICES=<2|3> (the wave-10 R4 lane)"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
NT = min(12, int(os.environ.get("OMP_NUM_THREADS", "12")))
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(NT)
DEV = torch.device("cuda:0" if PART == "gpu" else "cpu")
if DEV.type == "cuda":
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
HERE = os.path.dirname(os.path.abspath(__file__))
W10 = os.path.dirname(HERE)
ROOT = "<repo>"
W5 = f"{ROOT}/nn_x/w5"
PY = f"{ROOT}/.venv/bin/python"
INIT = f"{ROOT}/nn/ckpt/v17_pre2s_s101@77500.pt"
sys.path.insert(0, HERE)
sys.path.insert(1, W10)
import warnings  # noqa: E402

warnings.filterwarnings("ignore", message=".*CUDA is not available.*")
warnings.filterwarnings("ignore", message=".*device_type of 'cuda'.*")
from mini_split import MiniSplit, pick_train_feeds  # noqa: E402

LOG = open(f"{W10}/logs/test_r4_{PART}.log", "w")
RES = {"checks": []}
FAIL = []
XT = ["seq_cv_invd20", "seq_cv_pop", "seq_cv_norm", "it_hist_all_luo", "it_hist_et1_luo", "it_hist_et2_luo", "it_hist_et3_luo",
      "it_hist_et4_luo", "ctx_nd_hcent", "ctx_nd_knn_f1", "ctx_nd_knn_h", "ctx_nd_knn_h_dev", "ctx_nd_knn_u", "ctx_nd_knn_u_dev",
      "ctx_nd_max", "ctx_nd_top5h", "ctx_nd_top5u", "ctx_nd_ucent", "ctx_hn_top1w", "ctx_hn_rrw", "ctx_hn_bestrk", "ctx_hn_margin",
      "ctx_hn_soft05", "ctx_hn_e1_rrw", "ctx_hs_mutual", "ctx_hs_k128", "ctx_hs_e1", "ctx_hs_t10", "pre2_fe_z", "pre2_new_z",
      "pre2_fe_s", "pre2_new_s", "pre2_hmax32", "has:pre2_fe"]  # = train.py XT_PRESETS["cvnd_pre2s"]
XT_REL = [XT.index(n) for n in ("seq_cv_invd20", "it_hist_all_luo", "pre2_fe_s", "pre2_new_s", "pre2_hmax32")]
XT_IDCOLS = [i for i, n in enumerate(XT) if n.startswith(("seq_cv", "it_hist"))]
XT_RELONLY = [XT.index(n) for n in ("pre2_fe_s", "pre2_new_s", "pre2_hmax32")]
DCN_KEYS = ("dcn_par", "dcn_rank", "dcn_nox")


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


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def load_module(path, name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def base_cfg(Mo, **kw):
    n_dense = Mo.N_DENSE + 2 + len(XT) + 2 * len(XT_REL) - len(XT_RELONLY)
    cfg = dict(d=128, heads=4, hist_layers=1, blocks=2, drop=0.1, use_hist=True, use_set=True, use_pos_tower=False,
               n_dense=n_dense, aux_pos=True, ff=2, sim_bias=1, pos_mode="both", domain=True, intent=False, xt_names=XT,
               id_buckets=0, id_dim=32, id_drop=0.5, arch="tf", id_hist_only=False, xt_rel=XT_REL, xt_drop=0.0,
               xt_idcols=XT_IDCOLS, xt_relonly=XT_RELONLY, attn_gate=True, split_kv=True, retr_m=16)
    cfg.update(kw)
    return cfg


def build(Mo, seed, **kw):
    torch.manual_seed(seed)
    m = Mo.Ranker(**base_cfg(Mo, **kw))
    m.set_norm(torch.zeros(m.cfg["n_dense"]), torch.ones(m.cfg["n_dense"]))
    return m.to(DEV)


def fwd(m, b, train=False, seed=0, pos=False):
    m.train(train)
    torch.manual_seed(seed)
    X = m.dense(b)
    out = m(b, X, use_pos=pos)
    return out.detach(), X


# ================================================================================================= CPU part
def _parse(path):
    return ast.parse(open(path).read())


class StripR4Model(ast.NodeTransformer):
    def __init__(self):
        self.removed = collections.Counter()

    def visit_ClassDef(self, n):
        if n.name == "GDCNBranch":
            self.removed["class GDCNBranch"] += 1
            return None
        return self.generic_visit(n)

    def visit_FunctionDef(self, n):
        if n.name == "__init__" and [a.arg for a in n.args.args[-3:]] == list(DCN_KEYS):
            n.args.args = n.args.args[:-3]
            n.args.defaults = n.args.defaults[:-3]
            self.removed["Ranker.__init__ kwargs dcn_par, dcn_rank, dcn_nox"] += 1
        return self.generic_visit(n)

    def visit_If(self, n):
        t = ast.unparse(n.test)
        if t in ("dcn_par", "self.dcnp is not None"):
            self.removed[f"if {t}"] += 1
            return None
        return self.generic_visit(n)

    def visit_Assign(self, n):
        if ast.unparse(n) == "self.dcnp = None":
            self.removed["self.dcnp = None"] += 1
            return None
        return self.generic_visit(n)


MODEL_EXPECT = collections.Counter({"class GDCNBranch": 1, "Ranker.__init__ kwargs dcn_par, dcn_rank, dcn_nox": 1, "if dcn_par": 1,
                                    "if self.dcnp is not None": 1, "self.dcnp = None": 1})
W5_VARIANT_ASSERT = 'assert a.variant.startswith("w5_"), "nn_x/w5 copy: variant names must start with w5_"'


class StripR4Train(ast.NodeTransformer):
    def __init__(self):
        self.removed = collections.Counter()

    def visit_Expr(self, n):
        v = n.value
        if (isinstance(v, ast.Call) and isinstance(v.func, ast.Attribute) and v.func.attr == "add_argument" and v.args
                and isinstance(v.args[0], ast.Constant) and v.args[0].value in ("--dcn_par", "--dcn_rank", "--dcn_nox")):
            self.removed[f"add_argument {v.args[0].value}"] += 1
            return None
        return self.generic_visit(n)

    def visit_If(self, n):
        if ast.unparse(n.test) == "a.dcn_par":
            self.removed["if a.dcn_par (init assert + log)"] += 1
            return None
        return self.generic_visit(n)

    def visit_Assert(self, n):
        if ast.unparse(n.test) == "a.variant.startswith(('w5_', 'w10_'))":
            self.removed["variant assert w5_|w10_ -> w5_"] += 1
            return ast.parse(W5_VARIANT_ASSERT).body[0]
        return self.generic_visit(n)

    def visit_Call(self, n):
        if isinstance(n.func, ast.Name) and n.func.id == "Ranker":
            kws = [k for k in n.keywords if k.arg in DCN_KEYS]
            if kws:
                self.removed["Ranker(... dcn_par=, dcn_rank=, dcn_nox=)"] += len(kws)
                n.keywords = [k for k in n.keywords if k.arg not in DCN_KEYS]
        return self.generic_visit(n)


TRAIN_EXPECT = collections.Counter({"add_argument --dcn_par": 1, "add_argument --dcn_rank": 1, "add_argument --dcn_nox": 1,
                                    "if a.dcn_par (init assert + log)": 1, "variant assert w5_|w10_ -> w5_": 1,
                                    "Ranker(... dcn_par=, dcn_rank=, dcn_nox=)": 3})


def dry_cmds(script, gpu, jobs, rx, extra_env=None):
    env = {**os.environ, "DRY": "1"}
    env.update(extra_env or {})
    out = subprocess.run(["bash", script, str(gpu), jobs], capture_output=True, text=True, env=env)
    cmds = {}
    for line in out.stdout.splitlines():
        if " train.py " in line:
            m = re.search(rx, line)
            cmds[m.group(0)] = line.split(" train.py ", 1)[1].split()
    return out.returncode, cmds


def train_parser(path):
    import argparse
    tree = ast.parse(open(path).read())
    fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "parse"][0]
    ns = {"argparse": argparse, "POOL_P_DEFAULT": 0.5}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "train_parse", "exec"), ns)

    def parse(argv):
        sys.argv = ["train.py"] + argv
        return vars(ns["parse"]())
    return parse


def diff_ns(a, b, skip=("variant",)):
    return {k: (a.get(k), b.get(k)) for k in set(a) | set(b) if k not in skip and a.get(k) != b.get(k)}


def c0_code():
    P("\n== C0 code: module copies, AST flag-off proof, lane commands")
    for f in ("data.py", "cd_arms.py", "cd_arms4.py", "cd_arms5.py", "cd_arms6.py", "cd_arms7.py", "cd_arms8.py", "blend.py",
              "holdout_eval.py", "stale_pred.py", "finetune.py"):
        check(f"{f} byte-identical to nn_x/w5", md5(f"{W10}/{f}") == md5(f"{W5}/{f}"))
    rec = dict(line.split()[::-1] for line in open(f"{HERE}/w5_source_md5.txt"))
    for f in ("train.py", "model.py", "data.py", "common.py"):
        check(f"nn_x/w5/{f} unchanged since the copy (md5 tests/w5_source_md5.txt)", md5(f"{W5}/{f}") == rec[f"{W5}/{f}"])
    check("nn_x/abl/model.py byte-identical to nn_x/w5/model.py (the screen's paired base lineage)", md5(f"{ROOT}/nn_x/abl/model.py") == md5(f"{W5}/model.py"))
    # common.py: equal up to the W5 path constant
    ca, cb = _parse(f"{W5}/common.py"), _parse(f"{W10}/common.py")
    w5v = [n for n in cb.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "W5"]
    check("common.py: W5 output path -> nn_x/w10/r4", len(w5v) == 1 and ast.unparse(w5v[0].value) == "ROOT / 'nn_x' / 'w10' / 'r4'",
          ast.unparse(w5v[0]) if w5v else "")
    strip = lambda t: ast.unparse(ast.Module(body=[n for n in t.body if not (isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "W5")],  # noqa: E731
                                             type_ignores=[]))
    check("common.py AST-equal to nn_x/w5/common.py apart from the W5 path", strip(ca) == strip(cb))
    # model.py
    st = StripR4Model()
    t10 = st.visit(_parse(f"{W10}/model.py"))
    ast.fix_missing_locations(t10)
    check("model.py AST proof: removed set == expected", st.removed == MODEL_EXPECT, dict(st.removed))
    check("model.py minus the R4 statements == nn_x/w5/model.py (ast.unparse)", ast.unparse(t10) == ast.unparse(_parse(f"{W5}/model.py")))
    st = StripR4Train()
    t10 = st.visit(_parse(f"{W10}/train.py"))
    ast.fix_missing_locations(t10)
    check("train.py AST proof: removed set == expected", st.removed == TRAIN_EXPECT, dict(st.removed))
    check("train.py minus the R4 statements == nn_x/w5/train.py (ast.unparse)", ast.unparse(t10) == ast.unparse(_parse(f"{W5}/train.py")))
    # lane commands
    parse = train_parser(f"{W10}/train.py")
    rc, c10 = dry_cmds(f"{W10}/lane.sh", 2, "A:r4:23 A:r4:24 A:r4c:23 A:r4c:24 E:r4:951 E:r4:952", r"w(5|10)_\w+")
    rc5, c5 = dry_cmds(f"{W5}/lane.sh", 2, "A:base:23 A:base:24 E:base:951 E:base:952", r"w5_\w+")
    check("lane.sh DRY: 6 w10 commands, 4 w5 reference commands", rc == 0 and rc5 == 0 and len(c10) == 6 and len(c5) == 4, sorted(c10))
    want = {"dcn_par": (0, 2)}
    for s in (23, 24):
        d = diff_ns(parse(c5[f"w5_base_s{s}"]), parse(c10[f"w10_r4_s{s}"]))
        check(f"A:r4:{s} == w5 A:base:{s} + exactly {{dcn_par 2}}", d == want, d)
        d = diff_ns(parse(c10[f"w10_r4_s{s}"]), parse(c10[f"w10_r4c_s{s}"]))
        check(f"A:r4c:{s} == A:r4:{s} + dcn_nox", d == {"dcn_nox": (False, True)}, d)
    for s in (951, 952):
        d = diff_ns(parse(c5[f"w5_e10_base_s{s}"]), parse(c10[f"w5_e10_r4_s{s}"]))
        check(f"E:r4:{s} == w5 E:base:{s} + exactly {{dcn_par 2}}", d == want, d)
    for s in (23, 24):
        abl = json.load(open(f"{ROOT}/nn_x/abl/logs/abl_base_s{s}.json"))["args"]
        a10 = parse(c10[f"w10_r4_s{s}"])
        d = diff_ns(abl, a10, skip=("variant", "dcn_par"))
        only_abl = {k: abl[k] for k in abl if k not in a10}
        only_10 = {k: a10[k] for k in a10 if k not in abl}
        ok = (all(k in only_abl or k in only_10 for k in d) and only_abl == {"xt_exclude": "", "xt_rel_extra": ""}
              and only_10 == {"rdrop_cd": 0.0, "sam_cd": 0.0, "dcn_par": 2, "dcn_rank": 32, "dcn_nox": False})
        check(f"abl_base_s{s} stored args == A_BASE + HYG (the screen's paired base; abl-only / r4-only flags at their off values)",
              ok and abl["epochs"] == 2.0 and abl["seed"] == s and abl["cooldown_mix"] == 0.02, d)
    refused = [subprocess.run(["bash", f"{W10}/lane.sh", str(g), "A:r4:23"], capture_output=True, text=True,
                              env={**os.environ, "DRY": "1"}).returncode != 0 for g in (0, 1, 4, 5, 6, 7)]
    check("lane.sh refuses GPUs N <ids>", all(refused), refused)
    check("bash -n lane.sh", subprocess.run(["bash", "-n", f"{W10}/lane.sh"]).returncode == 0)


def perturb_padded(b, g):
    """Random inputs on every PADDED history slot (hmask False) and PADDED candidate slot (cmask False); valid slots untouched."""
    b = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in b.items()}
    hp, cp = ~b["hmask"], ~b["cmask"]
    rnd = lambda shape, dtype=torch.float32: torch.randn(shape, generator=g, dtype=dtype)  # noqa: E731
    he = b["he_raw"]
    he[hp] = rnd((int(hp.sum()), he.shape[-1])) * 50
    b["het"][hp] = torch.randint(1, 11, (int(hp.sum()),), generator=g)
    b["hes"][hp] = torch.randint(0, 32, (int(hp.sum()),), generator=g)
    b["hcat"][hp] = torch.randint(0, 54, (int(hp.sum()),), generator=g)
    ce = b["ce_raw"]
    ce[cp] = rnd((int(cp.sum()), ce.shape[-1])) * 50
    b["cf"][cp] = rnd((int(cp.sum()), b["cf"].shape[-1]))
    b["ccat"][cp] = torch.randint(0, 54, (int(cp.sum()),), generator=g)
    b["xt"][cp] = rnd((int(cp.sum()), b["xt"].shape[-1]), b["xt"].dtype)
    if "pos" in b:
        b["pos"][cp] = torch.randint(0, 300, (int(cp.sum()),), generator=g)
    return b


def perturb_feed(b, j, g):
    """Random inputs on every slot (valid and padded) of feed row j."""
    b = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in b.items()}
    rnd = lambda shape, dtype=torch.float32: torch.randn(shape, generator=g, dtype=dtype)  # noqa: E731
    b["he_raw"][j] = rnd(b["he_raw"].shape[1:]) * 50
    b["het"][j] = torch.randint(1, 11, b["het"].shape[1:], generator=g)
    b["hes"][j] = torch.randint(0, 32, b["hes"].shape[1:], generator=g)
    b["hcat"][j] = torch.randint(0, 54, b["hcat"].shape[1:], generator=g)
    b["ce_raw"][j] = rnd(b["ce_raw"].shape[1:]) * 50
    b["cf"][j] = rnd(b["cf"].shape[1:])
    b["ccat"][j] = torch.randint(0, 54, b["ccat"].shape[1:], generator=g)
    b["xt"][j] = rnd(b["xt"].shape[1:], b["xt"].dtype)
    b["ue_raw"][j] = rnd(b["ue_raw"].shape[1:]) * 50
    return b


def c1_identity(Mo5, Mo10, batches):
    P("\n== C1 flag-off identity: nn_x/w5/model.py vs nn_x/w10/r4/model.py (same seed, stage-E cfg)")
    m5, m10 = build(Mo5, 7), build(Mo10, 7)
    s5, s10 = m5.state_dict(), m10.state_dict()
    check("state_dict keys and values bit-identical", s5.keys() == s10.keys() and all(torch.equal(s5[k], s10[k]) for k in s5),
          f"{len(s5)} tensors")
    check("cfg identical (no dcn keys at flags off)", m5.cfg == m10.cfg and not any(k in m10.cfg for k in DCN_KEYS), sorted(set(m10.cfg) - set(m5.cfg)))
    m10b = Mo10.Ranker(**m5.cfg)
    check("r4 Ranker(**w5 cfg) builds (finetune / holdout_eval / stale_pred path) and matches the state_dict keys",
          m10b.state_dict().keys() == s5.keys() and m10b.dcnp is None)
    ok_e = ok_t = ok_x = True
    for i, b in enumerate(batches):
        o5, X5 = fwd(m5, b, False, 11)
        o10, X10 = fwd(m10, b, False, 11)
        ok_x &= torch.equal(X5, X10)
        ok_e &= torch.equal(o5, o10)
        o5, _ = fwd(m5, b, True, 13 + i, pos=True)
        o10, _ = fwd(m10, b, True, 13 + i, pos=True)
        ok_t &= torch.equal(o5, o10) and torch.equal(m5.aux_out, m10.aux_out)
    check(f"dense features bit-identical on {len(batches)} real train batches", ok_x)
    check("eval-mode logits bit-identical", ok_e)
    check("train-mode logits + aux_out bit-identical (same RNG state, dropout on)", ok_t)


def c2_on(Mo5, Mo10, batches):
    P("\n== C2 flags ON (dcn_par 2, rank 32; with X and nox): zero-init proof, Muon set, gradients, isolation, round trip")
    g = torch.Generator().manual_seed(123)
    mb = build(Mo5, 7)
    m = build(Mo10, 7, dcn_par=2, dcn_rank=32)
    mn = build(Mo10, 7, dcn_par=2, dcn_rank=32, dcn_nox=True)
    nb, n_on, n_nox = (sum(p.numel() for p in x.parameters()) for x in (mb, m, mn))
    nd = m.cfg["n_dense"]
    check("cfg stores dcn_par / dcn_rank / dcn_nox only when ON", m.cfg.get("dcn_par") == 2 and m.cfg.get("dcn_rank") == 32
          and m.cfg.get("dcn_nox") is False and mn.cfg.get("dcn_nox") is True and not any(k in mb.cfg for k in DCN_KEYS),
          f"params base {nb / 1e6:.4f}M, dcn2 {n_on / 1e6:.4f}M (+{n_on - nb}), dcn2nox {n_nox / 1e6:.4f}M (+{n_nox - nb})")
    D = 128 + nd
    exp_on = 2 * 128 + 2 * (2 * D * 32 + D + 2 * 32 * D + D) + D + 1  # LN(d) + L * (V, U(+b), Gv, Gu(+b)) + out
    check(f"branch parameter count == LN(d) + 2 layers of rank-32 cross + gate on D = d + n_dense = {D}, + out",
          n_on - nb == exp_on and n_nox - nb == 2 * 128 + 2 * (2 * 128 * 32 + 128 + 2 * 32 * 128 + 128) + 128 + 1, f"{n_on - nb} vs {exp_on}")
    check("base parameters (the shared ones) keep exactly their flag-off init under the RNG fork",
          all(torch.equal(mb.state_dict()[k], m.state_dict()[k]) for k in mb.state_dict()))
    muon = {f"{mn_}.weight" for mn_, mod in m.named_modules() if isinstance(mod, torch.nn.Linear) and min(mod.weight.shape) > 1}
    dm = sorted(k for k in muon if k.startswith("dcnp."))
    check("Muon rule (train.py) covers exactly the 8 cross / gate matrices (v, u, gv, gu x 2 layers) and excludes dcnp.out",
          len(dm) == 8 and all(k.split(".")[1] in ("v", "u", "gv", "gu") for k in dm) and "dcnp.out.weight" not in muon, dm)
    check("output layer zero-init", float(m.dcnp.out.weight.abs().sum()) == 0.0 and float(m.dcnp.out.bias.abs().sum()) == 0.0)
    # zero-init proof: logits at init == base, eval and train mode
    ok_e = ok_t = ok_en = ok_tn = True
    for i, b in enumerate(batches):
        ob, _ = fwd(mb, b, False, 11)
        o, _ = fwd(m, b, False, 11)
        on, _ = fwd(mn, b, False, 11)
        ok_e &= torch.equal(ob, o)
        ok_en &= torch.equal(ob, on)
        ob, _ = fwd(mb, b, True, 13 + i, pos=True)
        o, _ = fwd(m, b, True, 13 + i, pos=True)
        ok_t &= torch.equal(ob, o) and torch.equal(mb.aux_out, m.aux_out)
        on, _ = fwd(mn, b, True, 13 + i, pos=True)
        ok_tn &= torch.equal(ob, on)
    check("ZERO-INIT: eval-mode logits at init bit-identical to the base (dcn2)", ok_e)
    check("ZERO-INIT: train-mode logits + aux_out at init bit-identical to the base (same RNG state; no RNG draw in the branch)", ok_t)
    check("ZERO-INIT: the same for dcn2 nox (eval + train)", ok_en and ok_tn)
    # gradients at init: out gets a gradient, everything else in the branch exactly zero
    b = batches[0]
    m.train()
    torch.manual_seed(5)
    Xb = m.dense(b)
    lg = m(b, Xb, use_pos=True)
    loss = Mo10.listwise_loss(lg, b["y"], b["cmask"], norm="feed", denom=512) + 0.5 * m.aux_out.pow(2).mean()
    loss.backward()
    gs = {n: p.grad for n, p in m.named_parameters() if n.startswith("dcnp.")}
    check("backward at init: dcnp.out.weight receives a finite non-zero gradient", gs["dcnp.out.weight"] is not None
          and torch.isfinite(gs["dcnp.out.weight"]).all() and float(gs["dcnp.out.weight"].abs().sum()) > 0,
          f"|g out| {float(gs['dcnp.out.weight'].abs().sum()):.3e}")
    inner = [n for n in gs if not n.startswith("dcnp.out.")]
    check("backward at init: every other branch gradient (U/V/G/LN) is exactly zero (zero-init output)",
          all(gs[n] is None or float(gs[n].abs().sum()) == 0.0 for n in inner), f"{len(inner)} tensors")
    base_shared_grad_ok = all((p.grad is None) or torch.isfinite(p.grad).all() for n, p in m.named_parameters() if not n.startswith("dcnp."))
    check("backward at init: shared-parameter gradients finite", base_shared_grad_ok)
    m.zero_grad(set_to_none=True)
    # perturb the output weight -> the branch is live: all its parameters get gradients, logits differ from the base
    with torch.no_grad():
        m.dcnp.out.weight.copy_(torch.randn_like(m.dcnp.out.weight, ) * 0.05)
    torch.manual_seed(5)
    lg = m(b, m.dense(b), use_pos=True)
    loss = Mo10.listwise_loss(lg, b["y"], b["cmask"], norm="feed", denom=512)
    loss.backward()
    gs = {n: p.grad for n, p in m.named_parameters() if n.startswith("dcnp.")}
    check("after perturbing the output weight: every branch parameter has a finite non-zero gradient",
          all(g_ is not None and torch.isfinite(g_).all() and float(g_.abs().sum()) > 0 for g_ in gs.values()), f"{len(gs)} tensors")
    m.zero_grad(set_to_none=True)
    ob, _ = fwd(mb, b, False, 11)
    o, _ = fwd(m, b, False, 11)
    check("live branch: eval logits differ from the base, finite, shape [B, M]", o.shape == ob.shape and not torch.equal(ob, o)
          and torch.isfinite(o[b["cmask"]]).all(), f"max|d| {float((o - ob)[b['cmask']].abs().max()):.3e}")
    # padded slots and feed isolation (eval mode, bitwise), with the live branch
    ok_pad = ok_feed = True
    info = []
    for i, bb in enumerate(batches):
        o1, _ = fwd(m, bb, False, 11)
        o2, _ = fwd(m, perturb_padded(bb, g), False, 11)
        ok_pad &= torch.equal(o1[bb["cmask"]], o2[bb["cmask"]])
        info.append(f"{float((o1 - o2)[bb['cmask']].abs().max()):.1e}")
        j = int(bb["cmask"].shape[0] // 2)
        o3, _ = fwd(m, perturb_feed(bb, j, g), False, 11)
        keep = torch.ones(bb["cmask"].shape[0], dtype=torch.bool)
        keep[j] = False
        ok_feed &= torch.equal(o1[keep][bb["cmask"][keep]], o3[keep][bb["cmask"][keep]]) and not torch.equal(o1[j][bb["cmask"][j]], o3[j][bb["cmask"][j]])
    check("padded history + padded candidate slots never influence the valid logits (live branch, bitwise)", ok_pad, " ".join(info))
    check("one feed never influences another (perturbing feed j: others bitwise equal, feed j changes)", ok_feed)
    # cfg round trip (live branch)
    m2 = Mo10.Ranker(**m.cfg)
    m2.load_state_dict(m.state_dict(), strict=True)
    m2.set_norm(m.f_mean, m.f_std)
    ok_rt = all(torch.equal(fwd(m, bb, False, 11)[0], fwd(m2, bb, False, 11)[0]) for bb in batches)
    check("cfg round trip: Ranker(**cfg) + load_state_dict(strict) reproduces the logits", ok_rt)
    # x0 width: hook
    seen = {}
    h = m.dcnp.register_forward_hook(lambda mod, inp, out: seen.update(c=inp[0].shape[-1], X=inp[1].shape[-1], o=out.shape))
    fwd(m, b, False, 11)
    h.remove()
    check(f"branch inputs: c width 128, X width n_dense = {nd}, output [B, M]", seen.get("c") == 128 and seen.get("X") == nd
          and tuple(seen.get("o", ())) == tuple(b["cmask"].shape), seen)


# ================================================================================================= GPU part
E2E = ("--init {init} --epochs 0.004 --lr 2e-3 --warmup 0 --sched wsd --cooldown_frac 1.0 --muon --aux_pos 0.5 "
       "--aux_mode listwise --aux_tau 10 --aux_anneal 0.1 --cooldown_mix 0.2 --loss_norm feed --holdout 30000 "
       "--select_on holdout --eval_every 20 --seed 811 --no_test")
ON = ("--no_pos --aux_pos 0.5 --aux_mode listwise --aux_tau 10 --sim_bias --user0 --domain --holdout 30000 --select_on holdout "
      "--loss_norm feed --xt cvnd_pre2s --xt_drop 0 --muon --lr 1e-3 --sched wsd --cooldown_frac 0.2 --cooldown_mix 0.02 "
      "--aux_anneal 0.1 --retr_m 16 --attn_gate --epochs 1 --train_feeds 20000 --eval_every 25 --select_min_step 25 --cliplog 10 "
      "--save_last --save_holdout --seed 3 --dcn_par 2")
SCR = os.environ.get("T10_SCRATCH", "/tmp/scratchpad/t10")
os.makedirs(SCR, exist_ok=True)


def outputs(v):
    fs = [f"{W10}/logs/{v}.log", f"{W10}/logs/{v}.json", f"{W10}/logs/{v}_cliplog.npz"]  # (.stdout is written by lane.sh, not train.py)
    for x in (v, f"{v}@last"):
        fs += [f"{W10}/ckpt/{x}.pt", f"{W10}/preds/nn_{x}_trainonly_val.npy", f"{W10}/preds/nn_{x}_trainonly_test.npy",
               f"{W10}/preds/holdout/nn_{x}_holdout.npy"]
    return fs


def clean(v):
    for f in outputs(v):
        if os.path.exists(f):
            os.remove(f)


def run_e2e(wrapper, train_file, v, args):
    rec = f"{SCR}/rec_{v}.npz"
    cmd = [PY, wrapper, train_file, rec, "--variant", v] + args
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=W10, env={**os.environ, "OMP_NUM_THREADS": str(NT)})
    open(f"{SCR}/{v}.stdout", "w").write(r.stdout + "\n" + r.stderr)
    P(f"  ran {v} ({time.time() - t0:.0f}s, rc {r.returncode})")
    assert r.returncode == 0, r.stderr[-3000:]
    return dict(np.load(rec, allow_pickle=True)), open(f"{W10}/logs/{v}.log").read()


def metric_lines(log):
    return [re.sub(r" [\d.]+ min", "", ln) for ln in log.splitlines() if ln.startswith("step ") or ln.startswith("BEST")]


def ck_equal(a, b):
    ca = torch.load(a, map_location="cpu", weights_only=False)["state_dict"]
    cb = torch.load(b, map_location="cpu", weights_only=False)["state_dict"]
    return ca.keys() == cb.keys() and all(torch.equal(ca[k], cb[k]) for k in ca)


def g1_identity():
    P("\n== G1 end-to-end identity: frozen nn_x/w5 code (tests/ref_w5) vs nn_x/w10/r4 at flags OFF (39 CD steps)")
    names = ["w5_t10e2e_ref_off", "w5_t10e2e_new_off"]
    for v in names:
        clean(v)
    ra, la = run_e2e(f"{HERE}/ref_w5/tests/e2e_w5.py", f"{HERE}/ref_w5/train.py", names[0], E2E.format(init=INIT).split())
    rb, lb = run_e2e(f"{HERE}/e2e_w5.py", f"{W10}/train.py", names[1], E2E.format(init=INIT).split())
    nb = len(ra["off"]) - 1
    check("identical training-batch sequence (incl. pool steps)", np.array_equal(ra["fidx"], rb["fidx"]) and np.array_equal(ra["off"], rb["off"])
          and nb == 39, f"{nb} batches")
    check("identical train-mode model logits and loss-input logits", np.array_equal(ra["lg_model"], rb["lg_model"])
          and np.array_equal(ra["lg_loss"], rb["lg_loss"]))
    ma, mb = metric_lines(la), metric_lines(lb)
    check("identical logged loss / val / holdout / BEST lines", ma == mb and len(ma) >= 3, mb[-1] if mb else "")
    check("bit-identical checkpoint and val preds", ck_equal(f"{W10}/ckpt/{names[0]}.pt", f"{W10}/ckpt/{names[1]}.pt")
          and np.array_equal(np.load(f"{W10}/preds/nn_{names[0]}_trainonly_val.npy"), np.load(f"{W10}/preds/nn_{names[1]}_trainonly_val.npy")))
    ck = torch.load(f"{W10}/ckpt/{names[1]}.pt", map_location="cpu", weights_only=False)
    check("flag-off checkpoint cfg carries no dcn keys; args carry dcn_par 0", not any(k in ck["cfg"] for k in DCN_KEYS) and ck["args"]["dcn_par"] == 0)
    for v in names:
        clean(v)


def g2_on():
    P("\n== G2 flags ON from scratch (dcn_par 2, 20K train feeds, 1 epoch, hygiene, test preds)")
    v = "w10_t10e2e_on"
    clean(v)
    rec = f"{SCR}/rec_{v}.npz"
    cmd = [PY, f"{HERE}/e2e_w5.py", f"{W10}/train.py", rec, "--variant", v] + ON.split()
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=W10, env={**os.environ, "OMP_NUM_THREADS": str(NT)})
    open(f"{SCR}/{v}.stdout", "w").write(r.stdout + "\n" + r.stderr)
    P(f"  ran {v} ({time.time() - t0:.0f}s, rc {r.returncode})")
    check("flags-ON run completes (rc 0)", r.returncode == 0, r.stderr[-1500:] if r.returncode else "")
    if r.returncode:
        return
    fs = outputs(v)
    missing = [f for f in fs if not os.path.exists(f)]
    check("selected + @last checkpoints, val / holdout / test preds, log / json / cliplog written", not missing, missing)
    ck = torch.load(f"{W10}/ckpt/{v}.pt", map_location="cpu", weights_only=False)
    check("checkpoint cfg stores dcn_par 2, dcn_rank 32, dcn_nox False",
          ck["cfg"].get("dcn_par") == 2 and ck["cfg"].get("dcn_rank") == 32 and ck["cfg"].get("dcn_nox") is False)
    rc_ = dict(np.load(rec, allow_pickle=True))
    log = open(f"{W10}/logs/{v}.log").read()
    check("training ran >= 40 steps with finite losses; val / holdout NDCG logged", int(rc_["n_model"]) >= 40
          and all(np.isfinite(float(x)) for x in re.findall(r"loss ([\d.]+)", log)) and "holdout_ndcg" in log, f"{int(rc_['n_model'])} steps")
    mu = re.search(r"muon: (\d+) matrices", log)
    mb = re.search(r"muon: (\d+) matrices", open(f"{ROOT}/nn_x/abl/logs/abl_base_s23.log").read())  # the base recipe's count (46)
    check(f"Muon log line counts the 8 extra matrices (base {mb.group(1)} -> {int(mb.group(1)) + 8})",
          mu is not None and int(mu.group(1)) == int(mb.group(1)) + 8, mu.group(0) if mu else "no muon line")
    check("wave-10 R4 log line present", "wave-10 R4: --dcn_par 2" in log)
    sd = ck["state_dict"]
    check("the trained branch is live (output weight non-zero after training)", float(sd["dcnp.out.weight"].abs().sum()) > 0)
    # holdout_eval.py reproduces the saved holdout preds (same deterministic settings)
    hp = f"{W10}/preds/holdout/nn_{v}_holdout.npy"
    mine = np.load(hp)
    os.rename(hp, f"{SCR}/save_holdout_copy.npy")
    det = ("import os, runpy, sys, torch; torch.use_deterministic_algorithms(True, warn_only=True); "
           "torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False); "
           "torch.backends.cuda.enable_cudnn_sdp(False); sys.path.insert(0, os.getcwd()); "
           f"sys.argv = ['holdout_eval.py', '{v}']; runpy.run_path('holdout_eval.py', run_name='__main__')")
    r2 = subprocess.run([PY, "-c", det], capture_output=True, text=True, cwd=W10,
                        env={**os.environ, "OMP_NUM_THREADS": str(NT), "CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
    he = np.load(hp) if os.path.exists(hp) else None
    check("holdout_eval.py (Ranker(**cfg) rebuild) == --save_holdout preds of the r4 checkpoint (bitwise)",
          r2.returncode == 0 and he is not None and np.array_equal(mine, he),
          f"max|d| {float(np.abs(mine - he).max()) if he is not None else float('nan'):.2e}" if r2.returncode == 0 else r2.stderr[-500:])
    # finetune.build_model rebuilds from the stored cfg
    ft = ("import os, sys, torch, numpy as np; sys.path.insert(0, os.getcwd()); import finetune as F; "
          f"ck = torch.load('ckpt/{v}.pt', map_location='cuda:0', weights_only=False); m = F.build_model(ck, torch.device('cuda:0')); "
          "sd = m.state_dict(); c = ck['state_dict']; ok = set(sd) == set(c) and all(torch.equal(sd[k].cpu(), c[k].cpu()) for k in sd if k != 'dom.weight') "
          "and torch.equal(sd['dom.weight'][:2].cpu(), c['dom.weight'][:2].cpu()) and torch.equal(sd['dom.weight'][2].cpu(), c['dom.weight'][1].cpu()); "
          "print('FT_BUILD_OK', ok, m.cfg.get('dcn_par'), len(m.dcnp.v), m.dcnp.n_dense)")  # build_model sets dom.weight[2] = dom.weight[1] (exploration domain)
    r3 = subprocess.run([PY, "-c", ft], capture_output=True, text=True, cwd=W10, env={**os.environ, "OMP_NUM_THREADS": str(NT)})
    check("finetune.build_model rebuilds the r4 model from the stored cfg and loads the state_dict strictly",
          r3.returncode == 0 and f"FT_BUILD_OK True 2 2 {ck['cfg']['n_dense']}" in r3.stdout, (r3.stdout.strip().splitlines() or [r3.stderr[-500:]])[-1])
    clean(v)


if __name__ == "__main__":
    t0 = time.time()
    P(f"test_r4.py {PART} {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} torch {torch.__version__} threads {NT}")
    if PART == "cpu":
        c0_code()
        Mo5 = load_module(f"{W5}/model.py", "model_w5")
        Mo10 = load_module(f"{W10}/model.py", "model_w10")
        feeds = pick_train_feeds(n_each=24, seed=9)
        ms = MiniSplit("train", feeds, xt_names=XT)
        rng = np.random.default_rng(0)
        bs = ms.batches(np.arange(ms.n), cand_budget=4096, max_feeds=48, shuffle=True, rng=rng)
        bs = sorted(bs, key=lambda x: -len(x))[:4]
        batches = [ms.gather(fb, with_pos=True) for fb in bs]
        P(f"  batches: {[tuple(b['cmask'].shape) + (b['hmask'].shape[1],) for b in batches]} (B, Mpad, Kpad)")
        c1_identity(Mo5, Mo10, batches)
        c2_on(Mo5, Mo10, batches)
    else:
        g1_identity()
        g2_on()
    P(f"\n{'ALL PASS' if not FAIL else 'FAILURES: ' + str(FAIL)} ({len(RES['checks'])} checks, {time.time() - t0:.0f}s)")
    RES["fail"] = FAIL
    json.dump(RES, open(f"{W10}/logs/test_r4_{PART}.json", "w"), indent=1)
    sys.exit(1 if FAIL else 0)
