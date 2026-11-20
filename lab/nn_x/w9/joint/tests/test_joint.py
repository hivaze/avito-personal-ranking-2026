"""Tests of the nn_x/w9 JOINT arm (2026-11-13): nn_x/w9/joint = a private copy of nn_x/w5 with train.py --joint_layers N /
--joint_hq {0,1} (model.py JointAttn / JointBlock / Ranker._joint), OFF by default.

  cpu part:  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 .venv/bin/python nn_x/w9/joint/tests/test_joint.py cpu
  gpu part:  CUDA_VISIBLE_DEVICES=<2|3> OMP_NUM_THREADS=12 .venv/bin/python nn_x/w9/joint/tests/test_joint.py gpu

CPU part
  C0 code. (a) module copies: data / cd_arms* / blend / holdout_eval / stale_pred / finetune byte-identical to nn_x/w5; common.py
     AST-equal to nn_x/w5/common.py up to the output-path constant (W5 -> nn_x/w9/joint); the nn_x/w5 sources are unchanged since
     the copy (md5 in tests/w5_source_md5.txt). (b) AST proof: model.py minus the JointAttn / JointBlock classes, Ranker._joint,
     the two `if joint_layers:` statements, the `self.joint, self.joint_hq = ...` assignment, the `if self.joint is not None:`
     statement in forward and the two __init__ keywords == nn_x/w5/model.py; train.py minus the two add_argument statements, the
     --init assertion and the two Ranker keywords, with the variant assertion restored to the w5 one == nn_x/w5/train.py.
     (c) lane.sh DRY: A:joint / E:joint == the nn_x/w5 lane's A:base / E:base (same parser) + exactly {joint_layers 4, blocks 0,
     hist_layers 0}; jointhq0 adds joint_hq 0; the nn_x/abl abl_base_s23 / s24 stored args == A_BASE + HYG (the paired screen base);
     lane.sh refuses GPUs other than 2 / 3.
  C1 flag-off identity: Ranker built from nn_x/w5/model.py and from nn_x/w9/joint/model.py under the same seed and the stage-E
     cfg -> bit-identical state_dict and cfg; dense features + logits on real MiniSplit train batches bit-identical in eval mode
     and in train mode (same RNG state); the w9 Ranker accepts a w5 cfg (no joint keys).
  C2 flag-on smoke (joint_layers 2 / 4, blocks 0, hist_layers 0, sim_bias 1, retr_m 16, attn_gate, split_kv): shapes, finite logits
     and gradients (seg_bias / seg get gradients); the joint mask == key validity for every query (joint_hq 1) and additionally blocks
     candidate keys for user / history queries (joint_hq 0); PADDED history slots and PADDED candidate slots never influence the
     valid logits (random perturbation of every padded input -> bit-identical valid logits); one feed never influences another
     (perturbing feed j -> feed i bit-identical); joint_hq 0: the [user; history] rows after the joint stack are independent of the
     feed's candidates (dropping a candidate leaves them bit-identical) while joint_hq 1 changes them; cfg round trip
     Ranker(**cfg).load_state_dict(strict) reproduces the logits; a flag-off cfg carries no joint keys; parameter counts.
GPU part (one GPU, deterministic kernels; test variants w5_t9e2e_* / w9_t9e2e_*, outputs deleted afterwards)
  G1 end-to-end identity (tests/e2e_w5.py wrapper, 39 CD steps from the v17_pre2s_s101@77500 snapshot, cooldown_mix 0.2, evals at
     20 and 39): the frozen nn_x/w5 code (tests/ref_w5, writing into nn_x/w9/joint) vs nn_x/w9/joint/train.py at flags OFF ->
     identical batch sequence, train-mode logits, loss-input logits, logged lines, bit-identical checkpoint and val preds.
  G2 flags ON from scratch (joint_layers 2, 20K train feeds, ~60 steps, evals + hygiene, test preds): runs end to end, writes the
     selected and @last checkpoints / val / holdout / test preds, cfg stores joint_layers 2 and joint_hq True;
     holdout_eval.py reproduces the saved holdout preds bitwise (same kernel settings); finetune.build_model rebuilds the model from
     the stored cfg and loads the state_dict strictly; logits of the rebuilt model == the checkpoint's model.
Writes nn_x/w9/joint/logs/test_joint_<part>.log (+ .json).
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
assert PART in ("cpu", "gpu"), "usage: test_joint.py cpu|gpu"
_CVD = os.environ.get("CUDA_VISIBLE_DEVICES", "x")
if PART == "cpu":
    assert _CVD == "", "cpu part: run with CUDA_VISIBLE_DEVICES=''"
else:
    assert _CVD in ("2", "3"), "gpu part: CUDA_VISIBLE_DEVICES=<2|3> (the wave-9 JOINT lane)"
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
W9 = os.path.dirname(HERE)
ROOT = "<repo>"
W5 = f"{ROOT}/nn_x/w5"
PY = f"{ROOT}/.venv/bin/python"
INIT = f"{ROOT}/nn/ckpt/v17_pre2s_s101@77500.pt"
sys.path.insert(0, HERE)
sys.path.insert(1, W9)
import warnings  # noqa: E402

warnings.filterwarnings("ignore", message=".*CUDA is not available.*")
warnings.filterwarnings("ignore", message=".*device_type of 'cuda'.*")
from mini_split import MiniSplit, pick_train_feeds  # noqa: E402

LOG = open(f"{W9}/logs/test_joint_{PART}.log", "w")
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


class StripJointModel(ast.NodeTransformer):
    def __init__(self):
        self.removed = collections.Counter()

    def visit_ClassDef(self, n):
        if n.name in ("JointAttn", "JointBlock"):
            self.removed[f"class {n.name}"] += 1
            return None
        return self.generic_visit(n)

    def visit_FunctionDef(self, n):
        if n.name == "_joint":
            self.removed["def Ranker._joint"] += 1
            return None
        if n.name == "__init__" and [a.arg for a in n.args.args[-2:]] == ["joint_layers", "joint_hq"]:
            n.args.args = n.args.args[:-2]
            n.args.defaults = n.args.defaults[:-2]
            self.removed["Ranker.__init__ kwargs joint_layers, joint_hq"] += 1
        return self.generic_visit(n)

    def visit_If(self, n):
        t = ast.unparse(n.test)
        if t in ("joint_layers", "self.joint is not None"):
            self.removed[f"if {t}"] += 1
            return None
        return self.generic_visit(n)

    def visit_Assign(self, n):
        if ast.unparse(n).startswith("self.joint, self.joint_hq = "):
            self.removed["self.joint, self.joint_hq = ..."] += 1
            return None
        return self.generic_visit(n)


MODEL_EXPECT = collections.Counter({"class JointAttn": 1, "class JointBlock": 1, "def Ranker._joint": 1,
                                    "Ranker.__init__ kwargs joint_layers, joint_hq": 1, "if joint_layers": 2,
                                    "if self.joint is not None": 1, "self.joint, self.joint_hq = ...": 1})
W5_VARIANT_ASSERT = 'assert a.variant.startswith("w5_"), "nn_x/w5 copy: variant names must start with w5_"'


class StripJointTrain(ast.NodeTransformer):
    def __init__(self):
        self.removed = collections.Counter()

    def visit_Expr(self, n):
        v = n.value
        if (isinstance(v, ast.Call) and isinstance(v.func, ast.Attribute) and v.func.attr == "add_argument" and v.args
                and isinstance(v.args[0], ast.Constant) and v.args[0].value in ("--joint_layers", "--joint_hq")):
            self.removed[f"add_argument {v.args[0].value}"] += 1
            return None
        return self.generic_visit(n)

    def visit_Assert(self, n):
        t = ast.unparse(n.test)
        if t == "not (a.init and a.joint_layers)":
            self.removed["assert not (a.init and a.joint_layers)"] += 1
            return None
        if t == "a.variant.startswith(('w5_', 'w9_'))":
            self.removed["variant assert w5_|w9_ -> w5_"] += 1
            return ast.parse(W5_VARIANT_ASSERT).body[0]
        return self.generic_visit(n)

    def visit_Call(self, n):
        if isinstance(n.func, ast.Name) and n.func.id == "Ranker":
            kws = [k for k in n.keywords if k.arg in ("joint_layers", "joint_hq")]
            if kws:
                self.removed["Ranker(... joint_layers=, joint_hq=)"] += len(kws)
                n.keywords = [k for k in n.keywords if k.arg not in ("joint_layers", "joint_hq")]
        return self.generic_visit(n)


TRAIN_EXPECT = collections.Counter({"add_argument --joint_layers": 1, "add_argument --joint_hq": 1,
                                    "assert not (a.init and a.joint_layers)": 1, "variant assert w5_|w9_ -> w5_": 1,
                                    "Ranker(... joint_layers=, joint_hq=)": 2})


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
        check(f"{f} byte-identical to nn_x/w5", md5(f"{W9}/{f}") == md5(f"{W5}/{f}"))
    rec = dict(line.split()[::-1] for line in open(f"{HERE}/w5_source_md5.txt"))
    for f in ("train.py", "model.py", "data.py", "common.py"):
        check(f"nn_x/w5/{f} unchanged since the copy (md5 tests/w5_source_md5.txt)", md5(f"{W5}/{f}") == rec[f"{W5}/{f}"])
    # common.py: equal up to the W5 path constant
    ca, cb = _parse(f"{W5}/common.py"), _parse(f"{W9}/common.py")
    w5v = [n for n in cb.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "W5"]
    check("common.py: W5 output path -> nn_x/w9/joint", len(w5v) == 1 and ast.unparse(w5v[0].value) == "ROOT / 'nn_x' / 'w9' / 'joint'",
          ast.unparse(w5v[0]) if w5v else "")
    strip = lambda t: ast.unparse(ast.Module(body=[n for n in t.body if not (isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "W5")],  # noqa: E731
                                             type_ignores=[]))
    check("common.py AST-equal to nn_x/w5/common.py apart from the W5 path", strip(ca) == strip(cb))
    # model.py
    st = StripJointModel()
    t9 = st.visit(_parse(f"{W9}/model.py"))
    ast.fix_missing_locations(t9)
    check("model.py AST proof: removed set == expected", st.removed == MODEL_EXPECT, dict(st.removed))
    check("model.py minus the JOINT statements == nn_x/w5/model.py (ast.unparse)", ast.unparse(t9) == ast.unparse(_parse(f"{W5}/model.py")))
    st = StripJointTrain()
    t9 = st.visit(_parse(f"{W9}/train.py"))
    ast.fix_missing_locations(t9)
    check("train.py AST proof: removed set == expected", st.removed == TRAIN_EXPECT, dict(st.removed))
    check("train.py minus the JOINT statements == nn_x/w5/train.py (ast.unparse)", ast.unparse(t9) == ast.unparse(_parse(f"{W5}/train.py")))
    # lane commands
    parse = train_parser(f"{W9}/train.py")
    rc, c9 = dry_cmds(f"{W9}/lane.sh", 2, "A:joint:23 A:joint:24 A:jointhq0:23 E:joint:951 E:joint:952", r"w[59]_\w+")
    rc5, c5 = dry_cmds(f"{W5}/lane.sh", 2, "A:base:23 A:base:24 E:base:951 E:base:952", r"w5_\w+")
    check("lane.sh DRY: 5 w9 commands, 4 w5 reference commands", rc == 0 and rc5 == 0 and len(c9) == 5 and len(c5) == 4, sorted(c9))
    want = {"joint_layers": (0, 4), "blocks": (2, 0), "hist_layers": (1, 0)}
    for s in (23, 24):
        d = diff_ns(parse(c5[f"w5_base_s{s}"]), parse(c9[f"w9_joint_s{s}"]))
        check(f"A:joint:{s} == w5 A:base:{s} + exactly {{joint_layers 4, blocks 0, hist_layers 0}}", d == want, d)
    d = diff_ns(parse(c9["w9_joint_s23"]), parse(c9["w9_jointhq0_s23"]))
    check("A:jointhq0 == A:joint + joint_hq 0", d == {"joint_hq": (1, 0)}, d)
    for s in (951, 952):
        d = diff_ns(parse(c5[f"w5_e10_base_s{s}"]), parse(c9[f"w5_e10_joint_s{s}"]))
        check(f"E:joint:{s} == w5 E:base:{s} + exactly {{joint_layers 4, blocks 0, hist_layers 0}}", d == want, d)
    for s in (23, 24):
        abl = json.load(open(f"{ROOT}/nn_x/abl/logs/abl_base_s{s}.json"))["args"]
        a9 = parse(c9[f"w9_joint_s{s}"])
        d = diff_ns(abl, a9, skip=("variant", "joint_layers", "blocks", "hist_layers"))
        only_abl = {k: abl[k] for k in abl if k not in a9}
        only_9 = {k: a9[k] for k in a9 if k not in abl}
        ok = (all(k in only_abl or k in only_9 for k in d) and only_abl == {"xt_exclude": "", "xt_rel_extra": ""}
              and only_9 == {"rdrop_cd": 0.0, "sam_cd": 0.0, "joint_layers": 4, "joint_hq": 1})
        check(f"abl_base_s{s} stored args == A_BASE + HYG (the screen's paired base; abl-only / w9-only flags at their off values)",
              ok and abl["epochs"] == 2.0 and abl["seed"] == s and abl["cooldown_mix"] == 0.02, d)
    refused = [subprocess.run(["bash", f"{W9}/lane.sh", str(g), "A:joint:23"], capture_output=True, text=True,
                              env={**os.environ, "DRY": "1"}).returncode != 0 for g in (0, 1, 4, 5, 6, 7)]
    check("lane.sh refuses GPUs N <ids>", all(refused), refused)
    check("bash -n lane.sh", subprocess.run(["bash", "-n", f"{W9}/lane.sh"]).returncode == 0)


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


def c1_identity(Mo5, Mo9, batches):
    P("\n== C1 flag-off identity: nn_x/w5/model.py vs nn_x/w9/joint/model.py (same seed, stage-E cfg)")
    m5, m9 = build(Mo5, 7), build(Mo9, 7)
    s5, s9 = m5.state_dict(), m9.state_dict()
    check("state_dict keys and values bit-identical", s5.keys() == s9.keys() and all(torch.equal(s5[k], s9[k]) for k in s5),
          f"{len(s5)} tensors")
    check("cfg identical (no joint keys at flags off)", m5.cfg == m9.cfg and "joint_layers" not in m9.cfg, sorted(set(m9.cfg) - set(m5.cfg)))
    m9b = Mo9.Ranker(**m5.cfg)
    check("w9 Ranker(**w5 cfg) builds (finetune / holdout_eval / stale_pred path) and matches the state_dict keys",
          m9b.state_dict().keys() == s5.keys() and m9b.joint is None)
    ok_e = ok_t = ok_x = True
    for i, b in enumerate(batches):
        o5, X5 = fwd(m5, b, False, 11)
        o9, X9 = fwd(m9, b, False, 11)
        ok_x &= torch.equal(X5, X9)
        ok_e &= torch.equal(o5, o9)
        o5, _ = fwd(m5, b, True, 13 + i, pos=True)
        o9, _ = fwd(m9, b, True, 13 + i, pos=True)
        ok_t &= torch.equal(o5, o9) and torch.equal(m5.aux_out, m9.aux_out)
    check(f"dense features bit-identical on {len(batches)} real train batches", ok_x)
    check("eval-mode logits bit-identical", ok_e)
    check("train-mode logits + aux_out bit-identical (same RNG state, dropout on)", ok_t)


def c2_smoke(Mo9, batches):
    P("\n== C2 flags ON smoke (joint_layers 2 / 4, blocks 0, hist_layers 0)")
    g = torch.Generator().manual_seed(123)
    m = build(Mo9, 7, joint_layers=2, blocks=0, hist_layers=0)
    m.joint_debug = True
    n_par = sum(p.numel() for p in m.parameters())
    m4 = build(Mo9, 7, joint_layers=4, blocks=0, hist_layers=0)
    n4 = sum(p.numel() for p in m4.parameters())
    mb = build(Mo9, 7)
    nb = sum(p.numel() for p in mb.parameters())
    check("cfg stores joint_layers / joint_hq only when ON", m.cfg.get("joint_layers") == 2 and m.cfg.get("joint_hq") is True
          and "joint_layers" not in mb.cfg, f"params base {nb / 1e6:.3f}M, joint2 {n_par / 1e6:.3f}M, joint4 {n4 / 1e6:.3f}M")
    P(f"  info: params base {nb / 1e6:.3f}M (hist_enc 1 + cross 2 + sets 2 blocks) | joint2 {n_par / 1e6:.3f}M | joint4 {n4 / 1e6:.3f}M")
    muon = {f"{mn}.weight" for mn, mod in m4.named_modules() if isinstance(mod, torch.nn.Linear) and min(mod.weight.shape) > 1}
    jm = [k for k in muon if k.startswith("joint.")]
    check("Muon matrix set covers the joint q/k/v/o/gate/FFN weights (train.py rule)", len(jm) == 4 * 7, f"{len(jm)} joint matrices in Muon")
    # mask structure
    b = batches[0]
    o, X = fwd(m, b, False, 11)
    B, M = b["cmask"].shape
    K = b["hmask"].shape[1]
    kv = torch.cat([torch.ones(B, 1, dtype=torch.bool), b["hmask"], b["cmask"]], 1)
    al = m.joint_allowed
    check("joint_hq 1: allowed[b, q, k] == key validity for EVERY query row", al.shape == (B, 1 + K + M, 1 + K + M)
          and torch.equal(al, kv[:, None, :].expand_as(al)))
    check("logits shape [B, M], finite", o.shape == (B, M) and torch.isfinite(o[b["cmask"]]).all())
    m0 = build(Mo9, 7, joint_layers=2, blocks=0, hist_layers=0, joint_hq=False)
    m0.joint_debug = True
    fwd(m0, b, False, 11)
    al0 = m0.joint_allowed
    exp0 = kv[:, None, :].expand_as(al0).clone()
    exp0[:, :1 + K, 1 + K:] = False
    check("joint_hq 0: user / history queries blocked from candidate keys, everything else == key validity", torch.equal(al0, exp0))
    # gradients
    m.train()
    torch.manual_seed(5)
    Xb = m.dense(b)
    lg = m(b, Xb, use_pos=True)
    loss = Mo9.listwise_loss(lg, b["y"], b["cmask"], norm="feed", denom=512) + 0.5 * m.aux_out.pow(2).mean()
    loss.backward()
    gs = {n: p.grad for n, p in m.named_parameters() if n.startswith(("joint.", "seg."))}
    check("backward: finite gradients on every joint / seg parameter", all(g_ is not None and torch.isfinite(g_).all() for g_ in gs.values()),
          f"{len(gs)} tensors")
    sb = [n for n in gs if n.endswith("seg_bias")]
    check("seg_bias (zero-init) and seg embedding receive non-zero gradients",
          all(gs[n].abs().sum() > 0 for n in sb) and gs["seg.weight"].abs().sum() > 0,
          f"|grad seg_bias| {[float(gs[n].abs().sum()) for n in sb]}")
    m.zero_grad(set_to_none=True)
    # padded slots and feed isolation (eval mode, bitwise)
    ok_pad = ok_pad_base = ok_feed = True
    info = []
    for i, b in enumerate(batches):
        o1, _ = fwd(m, b, False, 11)
        o2, _ = fwd(m, perturb_padded(b, g), False, 11)
        d = float((o1 - o2)[b["cmask"]].abs().max())
        ok_pad &= torch.equal(o1[b["cmask"]], o2[b["cmask"]])
        info.append(f"{d:.1e}")
        ob1, _ = fwd(mb, b, False, 11)
        ob2, _ = fwd(mb, perturb_padded(b, g), False, 11)
        ok_pad_base &= torch.equal(ob1[b["cmask"]], ob2[b["cmask"]])
        j = int(b["cmask"].shape[0] // 2)
        o3, _ = fwd(m, perturb_feed(b, j, g), False, 11)
        keep = torch.ones(b["cmask"].shape[0], dtype=torch.bool)
        keep[j] = False
        ok_feed &= torch.equal(o1[keep][b["cmask"][keep]], o3[keep][b["cmask"][keep]]) and not torch.equal(o1[j][b["cmask"][j]], o3[j][b["cmask"][j]])
    check("padded history slots + padded candidate slots never influence the valid logits (joint model, bitwise)", ok_pad, " ".join(info))
    check("(sanity) the same holds for the flag-off model", ok_pad_base)
    check("one feed never influences another (perturbing feed j: others bitwise equal, feed j changes)", ok_feed)
    # joint_hq 0: [user; history] rows independent of the candidates
    ok0, ok1 = True, False
    for b in batches:
        b2 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in b.items()}
        last = b2["M"].clamp_min(2) - 1
        b2["cmask"][torch.arange(b2["cmask"].shape[0]), last] = False  # drop the last valid candidate of every feed (M >= 2)
        fwd(m0, b, False, 11)
        x0 = m0.joint_x[:, :1 + K].clone()
        fwd(m0, b2, False, 11)
        ok0 &= torch.equal(x0, m0.joint_x[:, :1 + K])
        fwd(m, b, False, 11)
        x1 = m.joint_x[:, :1 + K].clone()
        fwd(m, b2, False, 11)
        ok1 |= not torch.equal(x1, m.joint_x[:, :1 + K])
    check("joint_hq 0: [user; history] rows after the joint stack are bit-identical when a candidate is dropped", ok0)
    check("joint_hq 1: they change (feed-conditioned history)", ok1)
    # cfg round trip
    m2 = Mo9.Ranker(**m.cfg)
    m2.load_state_dict(m.state_dict(), strict=True)
    m2.set_norm(m.f_mean, m.f_std)
    ok_rt = all(torch.equal(fwd(m, b, False, 11)[0], fwd(m2, b, False, 11)[0]) for b in batches)
    check("cfg round trip: Ranker(**cfg) + load_state_dict(strict) reproduces the logits", ok_rt)
    m.joint_debug = m0.joint_debug = False


# ================================================================================================= GPU part
E2E = ("--init {init} --epochs 0.004 --lr 2e-3 --warmup 0 --sched wsd --cooldown_frac 1.0 --muon --aux_pos 0.5 "
       "--aux_mode listwise --aux_tau 10 --aux_anneal 0.1 --cooldown_mix 0.2 --loss_norm feed --holdout 30000 "
       "--select_on holdout --eval_every 20 --seed 811 --no_test")
ON = ("--no_pos --aux_pos 0.5 --aux_mode listwise --aux_tau 10 --sim_bias --user0 --domain --holdout 30000 --select_on holdout "
      "--loss_norm feed --xt cvnd_pre2s --xt_drop 0 --muon --lr 1e-3 --sched wsd --cooldown_frac 0.2 --cooldown_mix 0.02 "
      "--aux_anneal 0.1 --retr_m 16 --attn_gate --epochs 1 --train_feeds 20000 --eval_every 25 --select_min_step 25 --cliplog 10 "
      "--save_last --save_holdout --seed 3 --joint_layers 2 --blocks 0 --hist_layers 0")
SCR = os.environ.get("T9_SCRATCH", "/tmp/scratchpad/t9")
os.makedirs(SCR, exist_ok=True)


def outputs(v):
    fs = [f"{W9}/logs/{v}.log", f"{W9}/logs/{v}.json", f"{W9}/logs/{v}_cliplog.npz"]  # (.stdout is written by lane.sh, not train.py)
    for x in (v, f"{v}@last"):
        fs += [f"{W9}/ckpt/{x}.pt", f"{W9}/preds/nn_{x}_trainonly_val.npy", f"{W9}/preds/nn_{x}_trainonly_test.npy",
               f"{W9}/preds/holdout/nn_{x}_holdout.npy"]
    return fs


def clean(v):
    for f in outputs(v):
        if os.path.exists(f):
            os.remove(f)


def run_e2e(wrapper, train_file, v, args):
    rec = f"{SCR}/rec_{v}.npz"
    cmd = [PY, wrapper, train_file, rec, "--variant", v] + args
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=W9, env={**os.environ, "OMP_NUM_THREADS": str(NT)})
    open(f"{SCR}/{v}.stdout", "w").write(r.stdout + "\n" + r.stderr)
    P(f"  ran {v} ({time.time() - t0:.0f}s, rc {r.returncode})")
    assert r.returncode == 0, r.stderr[-3000:]
    return dict(np.load(rec, allow_pickle=True)), open(f"{W9}/logs/{v}.log").read()


def metric_lines(log):
    return [re.sub(r" [\d.]+ min", "", ln) for ln in log.splitlines() if ln.startswith("step ") or ln.startswith("BEST")]


def ck_equal(a, b):
    ca = torch.load(a, map_location="cpu", weights_only=False)["state_dict"]
    cb = torch.load(b, map_location="cpu", weights_only=False)["state_dict"]
    return ca.keys() == cb.keys() and all(torch.equal(ca[k], cb[k]) for k in ca)


def g1_identity():
    P("\n== G1 end-to-end identity: frozen nn_x/w5 code (tests/ref_w5) vs nn_x/w9/joint at flags OFF (39 CD steps)")
    names = ["w5_t9e2e_ref_off", "w5_t9e2e_new_off"]
    for v in names:
        clean(v)
    ra, la = run_e2e(f"{HERE}/ref_w5/tests/e2e_w5.py", f"{HERE}/ref_w5/train.py", names[0], E2E.format(init=INIT).split())
    rb, lb = run_e2e(f"{HERE}/e2e_w5.py", f"{W9}/train.py", names[1], E2E.format(init=INIT).split())
    nb = len(ra["off"]) - 1
    check("identical training-batch sequence (incl. pool steps)", np.array_equal(ra["fidx"], rb["fidx"]) and np.array_equal(ra["off"], rb["off"])
          and nb == 39, f"{nb} batches")
    check("identical train-mode model logits and loss-input logits", np.array_equal(ra["lg_model"], rb["lg_model"])
          and np.array_equal(ra["lg_loss"], rb["lg_loss"]))
    ma, mb = metric_lines(la), metric_lines(lb)
    check("identical logged loss / val / holdout / BEST lines", ma == mb and len(ma) >= 3, mb[-1] if mb else "")
    check("bit-identical checkpoint and val preds", ck_equal(f"{W9}/ckpt/{names[0]}.pt", f"{W9}/ckpt/{names[1]}.pt")
          and np.array_equal(np.load(f"{W9}/preds/nn_{names[0]}_trainonly_val.npy"), np.load(f"{W9}/preds/nn_{names[1]}_trainonly_val.npy")))
    ck = torch.load(f"{W9}/ckpt/{names[1]}.pt", map_location="cpu", weights_only=False)
    check("flag-off checkpoint cfg carries no joint keys; args carry joint_layers 0", "joint_layers" not in ck["cfg"] and ck["args"]["joint_layers"] == 0)
    for v in names:
        clean(v)


def g2_on():
    P("\n== G2 flags ON from scratch (joint_layers 2, 20K train feeds, 1 epoch, hygiene, test preds)")
    v = "w9_t9e2e_on"
    clean(v)
    rec = f"{SCR}/rec_{v}.npz"
    cmd = [PY, f"{HERE}/e2e_w5.py", f"{W9}/train.py", rec, "--variant", v] + ON.split()
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=W9, env={**os.environ, "OMP_NUM_THREADS": str(NT)})
    open(f"{SCR}/{v}.stdout", "w").write(r.stdout + "\n" + r.stderr)
    P(f"  ran {v} ({time.time() - t0:.0f}s, rc {r.returncode})")
    check("flags-ON run completes (rc 0)", r.returncode == 0, r.stderr[-1500:] if r.returncode else "")
    if r.returncode:
        return
    fs = outputs(v)
    missing = [f for f in fs if not os.path.exists(f)]
    check("selected + @last checkpoints, val / holdout / test preds, log / json / cliplog written", not missing, missing)
    ck = torch.load(f"{W9}/ckpt/{v}.pt", map_location="cpu", weights_only=False)
    check("checkpoint cfg stores joint_layers 2, joint_hq True, blocks 0, hist_layers 0",
          ck["cfg"].get("joint_layers") == 2 and ck["cfg"].get("joint_hq") is True and ck["cfg"]["blocks"] == 0 and ck["cfg"]["hist_layers"] == 0)
    rc_ = dict(np.load(rec, allow_pickle=True))
    log = open(f"{W9}/logs/{v}.log").read()
    check("training ran >= 40 steps with finite losses; val / holdout NDCG logged", int(rc_["n_model"]) >= 40
          and all(np.isfinite(float(x)) for x in re.findall(r"loss ([\d.]+)", log)) and "holdout_ndcg" in log, f"{int(rc_['n_model'])} steps")
    # holdout_eval.py reproduces the saved holdout preds (same deterministic settings)
    hp = f"{W9}/preds/holdout/nn_{v}_holdout.npy"
    mine = np.load(hp)
    os.rename(hp, f"{SCR}/save_holdout_copy.npy")
    det = ("import os, runpy, sys, torch; torch.use_deterministic_algorithms(True, warn_only=True); "
           "torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False); "
           "torch.backends.cuda.enable_cudnn_sdp(False); sys.path.insert(0, os.getcwd()); "
           f"sys.argv = ['holdout_eval.py', '{v}']; runpy.run_path('holdout_eval.py', run_name='__main__')")
    r2 = subprocess.run([PY, "-c", det], capture_output=True, text=True, cwd=W9,
                        env={**os.environ, "OMP_NUM_THREADS": str(NT), "CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
    he = np.load(hp) if os.path.exists(hp) else None
    check("holdout_eval.py (Ranker(**cfg) rebuild) == --save_holdout preds of the joint checkpoint (bitwise)",
          r2.returncode == 0 and he is not None and np.array_equal(mine, he),
          f"max|d| {float(np.abs(mine - he).max()) if he is not None else float('nan'):.2e}" if r2.returncode == 0 else r2.stderr[-500:])
    # finetune.build_model rebuilds from the stored cfg
    ft = ("import os, sys, torch, numpy as np; sys.path.insert(0, os.getcwd()); import finetune as F; "
          f"ck = torch.load('ckpt/{v}.pt', map_location='cuda:0', weights_only=False); m = F.build_model(ck, torch.device('cuda:0')); "
          "sd = m.state_dict(); c = ck['state_dict']; ok = set(sd) == set(c) and all(torch.equal(sd[k].cpu(), c[k].cpu()) for k in sd if k != 'dom.weight') "
          "and torch.equal(sd['dom.weight'][:2].cpu(), c['dom.weight'][:2].cpu()) and torch.equal(sd['dom.weight'][2].cpu(), c['dom.weight'][1].cpu()); "
          "print('FT_BUILD_OK', ok, m.cfg.get('joint_layers'), len(m.joint), m.joint_hq)")  # build_model sets dom.weight[2] = dom.weight[1] (exploration domain)
    r3 = subprocess.run([PY, "-c", ft], capture_output=True, text=True, cwd=W9, env={**os.environ, "OMP_NUM_THREADS": str(NT)})
    check("finetune.build_model rebuilds the joint model from the stored cfg and loads the state_dict strictly",
          r3.returncode == 0 and "FT_BUILD_OK True 2 2 True" in r3.stdout, (r3.stdout.strip().splitlines() or [r3.stderr[-500:]])[-1])
    clean(v)


if __name__ == "__main__":
    t0 = time.time()
    P(f"test_joint.py {PART} {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} torch {torch.__version__} threads {NT}")
    if PART == "cpu":
        c0_code()
        Mo5 = load_module(f"{W5}/model.py", "model_w5")
        Mo9 = load_module(f"{W9}/model.py", "model_w9")
        feeds = pick_train_feeds(n_each=24, seed=9)
        ms = MiniSplit("train", feeds, xt_names=XT)
        rng = np.random.default_rng(0)
        bs = ms.batches(np.arange(ms.n), cand_budget=4096, max_feeds=48, shuffle=True, rng=rng)
        bs = sorted(bs, key=lambda x: -len(x))[:4]
        batches = [ms.gather(fb, with_pos=True) for fb in bs]
        P(f"  batches: {[tuple(b['cmask'].shape) + (b['hmask'].shape[1],) for b in batches]} (B, Mpad, Kpad)")
        c1_identity(Mo5, Mo9, batches)
        c2_smoke(Mo9, batches)
    else:
        g1_identity()
        g2_on()
    P(f"\n{'ALL PASS' if not FAIL else 'FAILURES: ' + str(FAIL)} ({len(RES['checks'])} checks, {time.time() - t0:.0f}s)")
    RES["fail"] = FAIL
    json.dump(RES, open(f"{W9}/logs/test_joint_{PART}.json", "w"), indent=1)
    sys.exit(1 if FAIL else 0)
