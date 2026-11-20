"""Tests of train.py --rdrop_cd / --sam_cd (WAVE-7 cooldown regularisers R-Drop / SAM; helpers cd_arms8.py) and the lane.sh arms
B:rdrop / B:sam. Pre-change copies: tests/ref_pre_w7/*.txt (md5.txt). Written 2026-11-13 ~05:30 UTC.

  cpu part:  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 .venv/bin/python nn_x/w5/tests/test_w7.py cpu
  gpu part:  CUDA_VISIBLE_DEVICES=<one id> OMP_NUM_THREADS=12 [W7_GPUS="2 6"] .venv/bin/python nn_x/w5/tests/test_w7.py gpu
             (6 e2e subprocesses of 24 CD steps, ~70 s each; with W7_GPUS they run one at a time per listed GPU, each start
             gated on nvidia-smi: < 4 compute processes and >= 60 GB free on that GPU)
CPU part
  W0 code. (a) ref_pre_w7 copies intact; train.py minus the wave-7 statements (tests/strip_w7.py, exact set: cd_arms8 import,
     --rdrop_cd / --sam_cd add_argument, `rdrop, sam = None, None`, `if a.rdrop_cd > 0:`, `if a.sam_cd > 0:`, the loop blocks
     `if rdrop is not None and step >= t_cool:` / `if sam is not None and step >= t_cool:`, the eval-log blocks `if rdrop is not
     None:` / `if sam is not None:`) == the previous train.py (AST, docstrings stripped); parse(): rdrop_cd 0.0, sam_cd 0.0, every
     other default == the previous parser.  (b) lane.sh: DRY commands of EVERY previous job (109 jobs: A base/trunc/auxf 23-24;
     B m02ref/truncp/truncp2/auxf/hzcd/aa0/ap1/m01/m04 101-107; B valcd 101-104 x 0-4; E base/trunc/trunc2/auxf/hzcd 951-954)
     byte-identical to the previous lane.sh; B:rdrop:<r> == B:m02ref:<r> + {rdrop_cd 1.0}, B:sam:<r> == + {sam_cd 0.05}, variant
     w5_cd_{rdrop,sam}_r<r>, CDSEED 900+r, init v17_pre2s_s<r>@77500; A:rdrop / E:sam refused; bash -n.  (c) call sites:
     setup_rdrop / setup_sam once each behind their guards in main(), after the valcd setup and before the loop; rdrop.combine
     once, right after the aux block and right before the zero_grad loop; sam.second_pass once, right after loss.backward() and
     right before clip_grad_norm_; rdrop.summary / sam.summary once each inside the eval block; no other nn_x/w5 module imports
     cd_arms8; cd_arms8 imports only torch and model.{listwise_loss, bce_loss}.  (d) guards: every wave-7 test is False at the
     defaults (any step) and in every existing DRY command.
  W1 replica: main_aux_loss == the EXACT train.py loss statements (AST-extracted from the loop: the trunc If through the aux If)
     exec'd on the same tensors, bitwise, for listwise / mse aux, aux_norm batch / feed, loss_norm feed / batch, prand / no prand.
  W2 R-Drop: symkl_feed == a float64 per-feed reference (and ignores masked logits); RDrop.combine in eval mode (no dropout)
     returns exactly loss1 (KL 0); in train mode (seeded) it returns exactly 0.5 (L1 + L2) + alpha sum valid symKL / max_feeds
     recomputed independently, keeps CLIPLOG's main as 0.5 (CE1 + CE2) + alpha KL, accumulates / resets the KL statistics.
  W3 SAM: after second_pass the parameters are bitwise the pre-perturbation w; ||eps|| == rho; ||eps||/||w|| logged (VALID), the
     INVALID mark at an absurd rho; .grad == the gradient at w + eps of an independent deep copy (bitwise); rho -> 0 gives g1.
  W4 setup guards: refuse alpha / rho <= 0, both flags, --loss bce, --intent, --val_mix, t_cool >= total; log lines on success.
GPU part (deterministic wrapper tests/e2e_w7.py; 24 CD steps from v17_pre2s_s101@77500, cooldown_mix 0.2, evals at 12 / 24, cliplog 6)
  G1 flags OFF, previous train.py (scratch code dir) vs new: identical training-gather sequence, one train forward per step, train-
     mode logits, loss-input logits, logged metric lines, bit-identical checkpoint and val preds.
  G2 --rdrop_cd 1.0 --drop_cd 0 vs --drop_cd 0 (no dropout anywhere): two train forwards per step with bitwise-identical logits,
     logged symKL 0.000000, pass-1 logits and optimiser-entry gradient norms bitwise identical to flags-off for the first two steps
     (the two-pass loss reduces exactly to the one-pass loss); the later divergence (fp residual of the algebraically-zero KL
     gradient, amplified by Muon) is reported: first differing step, max rel grad-norm diff, checkpoint max |diff|.
  G3 --rdrop_cd 1.0 (dropout 0.1) vs flags OFF: same gathers; 48 forwards; pass-2 logits differ at every step; logged mean symKL in
     [0.001, 0.05] nats at both evals; checkpoint differs; @last / holdout files written.
  G4 --sam_cd 0.05 --drop_cd 0 vs --drop_cd 0: same gathers; 48 forwards; at every step's second forward ||params - w|| == 0.05
     (1e-3 relative); at the first optimiser entry of every step the parameters are bitwise w (restored); the first-step line reports ||eps||/||w||
     VALID (< 0.02); first-eval loss identical (the logged loss is the loss at w); optimiser-entry gradient norms differ (g2 != g1);
     checkpoint differs.
Writes nn_x/w5/logs/test_w7_<part>.log (+ .json). Test runs are named w5_tw7_*; their outputs are deleted.
"""
import argparse
import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import types

PART = sys.argv[1] if len(sys.argv) > 1 else "cpu"
assert PART in ("cpu", "gpu"), "usage: test_w7.py cpu|gpu"
_CVD = os.environ.get("CUDA_VISIBLE_DEVICES", "x")
if PART == "cpu":
    assert _CVD == "", "cpu part: run with CUDA_VISIBLE_DEVICES=''"
else:
    assert re.fullmatch(r"\d", _CVD), "gpu part: CUDA_VISIBLE_DEVICES=<exactly one GPU id>"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
NT = min(12, int(os.environ.get("OMP_NUM_THREADS", "12")))
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(NT)
HERE = os.path.dirname(os.path.abspath(__file__))
W5 = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(1, W5)
import warnings  # noqa: E402

warnings.filterwarnings("ignore", message=".*CUDA is not available.*")
import cd_arms as CA  # noqa: E402
import cd_arms8 as C8  # noqa: E402
from model import bce_loss, listwise_loss  # noqa: E402
from strip_w7 import EXPECT as W7_EXPECT, W7_IFS, strip_w7  # noqa: E402

ROOT = "<repo>"
PY = f"{ROOT}/.venv/bin/python"
REF = f"{HERE}/ref_pre_w7"
TRAIN_PY, LANE = f"{W5}/train.py", f"{W5}/lane.sh"
INIT = f"{ROOT}/nn/ckpt/v17_pre2s_s101@77500.pt"
WD = W5
SCR = os.environ.get("T5_SCRATCH", "/tmp/scratchpad/t5")
os.makedirs(SCR, exist_ok=True)
os.makedirs(f"{WD}/logs", exist_ok=True)
LOG = open(f"{WD}/logs/test_w7_{PART}.log", "w")
RES = {"checks": []}
FAIL = []
ROOTS = (101, 102, 103, 104)
ARMS = {"rdrop": ("rdrop_cd", 1.0), "sam": ("sam_cd", 0.05)}


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


def load_parse(path):
    tree = ast.parse(open(path).read())
    fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "parse"][0]
    ns = {"argparse": argparse, "POOL_P_DEFAULT": CA.POOL_P_DEFAULT}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "train_parse", "exec"), ns)

    def parse(argv):
        sys.argv = ["train.py"] + argv
        return ns["parse"]()
    return parse


def strip_doc(tree):
    for n in [tree] + [x for x in ast.walk(tree) if isinstance(x, (ast.FunctionDef, ast.ClassDef))]:
        if n.body and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant) \
                and isinstance(n.body[0].value.value, str):
            n.body = n.body[1:] or [ast.Pass()]
    return tree


def lane_dry(script, jobs, gpu="0"):
    r = subprocess.run(["bash", script, gpu, jobs], capture_output=True, text=True, env={**os.environ, "DRY": "1"})
    return r.returncode, r.stdout, r.stderr


def dry_cmds(out, parse):
    cmds = {}
    for ln in out.splitlines():
        if " train.py " in ln:
            argv = ln.split(" train.py ", 1)[1].split()
            cmds[argv[argv.index("--variant") + 1]] = (vars(parse(argv)), ln)
    return cmds


def diff(a, b):
    return {k: (a[k], b[k]) for k in a if k != "variant" and a[k] != b[k]}


# ------------------------------------------------------------------------------------------------ unit-test fixtures
class FakeModel(torch.nn.Module):
    """train.py's model interface as far as the loss statements need it: forward(b, X, use_pos) -> logits [B, M] and sets
    self.aux_out [B, M] (float); dropout on the hidden layer."""

    def __init__(self, drop=0.0, d=6, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.hid = torch.nn.Linear(d, d)
        self.lin = torch.nn.Linear(d, 1)
        self.aux_head = torch.nn.Linear(d, 1)
        self.dp = torch.nn.Dropout(drop)

    def forward(self, b, X=None, use_pos=False):
        h = self.dp(torch.relu(self.hid(X)))
        self.aux_out = self.aux_head(h).squeeze(-1).float()
        return self.lin(h).squeeze(-1)


def make_batch(B=6, Mx=7, d=6, seed=1, zero_pos_feed=True):
    g = torch.Generator().manual_seed(seed)
    M = torch.randint(2, Mx + 1, (B,), generator=g)
    M[0] = Mx
    m = torch.arange(Mx)[None, :] < M[:, None]
    y = (torch.rand(B, Mx, generator=g) < 0.35).float() * m
    for i in range(B):  # at least one positive per feed
        if y[i].sum() == 0:
            y[i, 0] = 1.0
    if zero_pos_feed:
        y[1] = 0.0  # one feed without a positive (valid = 0 in the losses)
    pos = torch.arange(Mx)[None, :].expand(B, Mx).clone()
    X = torch.randn(B, Mx, d, generator=g)
    b = dict(y=y, cmask=m, pos=pos, M=M, fidx=torch.arange(B))
    prand = torch.rand(B, generator=g)
    return b, X, prand


def make_args(**kw):
    a = dict(loss="softmax", loss_norm="feed", max_feeds=512, aux_pos=0.5, aux_mode="listwise", aux_tau=10.0, aux_norm="batch",
             aux_norm_c=1.695, prand_base=-1.0, intent=0.0, rdrop_cd=0.0, sam_cd=0.0, sched="wsd", val_mix=0.0, trunc_off=0.0,
             trunc_lifts="", cooldown_mix=0.02)
    a.update(kw)
    return argparse.Namespace(**a)


def loop_body():
    tree = ast.parse(open(TRAIN_PY).read())
    main_fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"][0]
    loops = [n for n in ast.walk(main_fn) if isinstance(n, ast.For) and ast.unparse(n.target).strip("()") == "step, fb"]
    assert len(loops) == 1
    return loops[0].body


def idx_if(body, test):
    k = [i for i, s in enumerate(body) if isinstance(s, ast.If) and ast.unparse(s.test) == test]
    assert len(k) == 1, (test, k)
    return k[0]


def loss_statements():
    """The exact train.py loss statements of one step as source text: from `if trunc is not None:` through `if a.aux_pos > 0:`."""
    body = loop_body()
    i0, i1 = idx_if(body, "trunc is not None"), idx_if(body, "a.aux_pos > 0")
    src = "\n".join(ast.unparse(s) for s in body[i0:i1 + 1])
    assert "fw = None if prand is None or a.prand_base < 0 else" in src and "listwise_loss(lg, y, m, fw, norm=a.loss_norm" in src
    return src, body, i0, i1


# ================================================================================================= CPU part
def w0_code():
    P("\n== W0 code identity vs the pre-change copies (tests/ref_pre_w7), lane.sh, call sites, guards")
    rec = dict(line.split()[::-1] for line in open(f"{REF}/md5.txt"))
    okm = {k: hashlib.md5(open(f"{REF}/{k}", "rb").read()).hexdigest() == v for k, v in rec.items()}
    check("ref_pre_w7 copies intact (md5.txt: train.py, lane.sh, test_w5/hzcd/trunc_lifts/valcd.py, strip_valcd/hzcd.py)",
          all(okm.values()) and len(okm) == 8, okm)
    # (a) train.py
    w7_t, removed = strip_w7(ast.parse(open(TRAIN_PY).read()))
    new_t, old_t = strip_doc(w7_t), strip_doc(ast.parse(open(f"{REF}/train.py.txt").read()))
    check("train.py minus the wave-7 statements (exact set) == previous train.py (AST, docstrings stripped)",
          ast.dump(new_t) == ast.dump(old_t) and removed == W7_EXPECT, f"removed {dict(removed)}")
    parse, old_parse = load_parse(TRAIN_PY), load_parse(f"{REF}/train.py.txt")
    a0, o0 = vars(parse(["--variant", "w5_x"])), vars(old_parse(["--variant", "w5_x"]))
    check("parse(): rdrop_cd default 0.0, sam_cd 0.0; every other default == the previous parser",
          a0.get("rdrop_cd") == 0.0 and a0.get("sam_cd") == 0.0
          and {k: v for k, v in a0.items() if k not in ("rdrop_cd", "sam_cd")} == o0)
    # (b) lane.sh
    old_jobs = ([f"A:{a}:{s}" for s in (23, 24) for a in ("base", "trunc", "auxf")]
                + [f"B:{a}:{r}" for r in range(101, 108) for a in ("m02ref", "truncp", "truncp2", "auxf", "hzcd", "aa0", "ap1", "m01", "m04")]
                + [f"B:valcd:{r}:{k}" for r in ROOTS for k in range(5)]
                + [f"E:{a}:{s}" for s in range(951, 955) for a in ("base", "trunc", "trunc2", "auxf", "hzcd")])
    rc_o, out_o, _ = lane_dry(f"{REF}/lane.sh.txt", " ".join(old_jobs))
    rc_n, out_n, _ = lane_dry(LANE, " ".join(old_jobs))
    check(f"lane.sh DRY: all {len(old_jobs)} previous jobs print byte-identical commands to the previous lane.sh",
          rc_o == rc_n == 0 and out_o == out_n and out_n.count(" train.py ") == len(old_jobs), f"{out_n.count(' train.py ')} commands")
    new_jobs = [f"B:{arm}:{r}" for r in ROOTS for arm in ARMS]
    rc2, out2, err2 = lane_dry(LANE, " ".join(new_jobs + [f"B:m02ref:{r}" for r in ROOTS]))
    cmds = dry_cmds(out2, parse)
    P("  DRY commands of the new arms (root 101):")
    for v in ("w5_cd_rdrop_r101", "w5_cd_sam_r101"):
        P("    " + cmds.get(v, (None, "(missing)"))[1])
    RES["dry_new"] = [ln for v, (_, ln) in cmds.items() if "rdrop" in v or "sam" in v]
    okB = rc2 == 0 and len(cmds) == 12
    for r in ROOTS:
        for arm, (flag, val) in ARMS.items():
            v = f"w5_cd_{arm}_r{r}"
            okB &= v in cmds
            if v in cmds:
                t = cmds[v][0]
                okB &= (diff(cmds[f"w5_cd_m02ref_r{r}"][0], t) == {flag: (0.0, val)} and t["seed"] == 900 + r
                        and t["init"] == f"{ROOT}/nn/ckpt/v17_pre2s_s{r}@77500.pt" and t["variant"] == v
                        and cmds[v][1].rstrip().endswith(f"--{flag} {val}"))
    check("lane.sh: B:rdrop:<r> == B:m02ref:<r> + {rdrop_cd 1.0}, B:sam:<r> == + {sam_cd 0.05} (flag appended last), variant "
          "w5_cd_<arm>_r<r>, CDSEED 900+r, init v17_pre2s_s<r>@77500, roots 101-104", okB, err2[-200:])
    bad = {j: lane_dry(LANE, j)[0] for j in ("A:rdrop:23", "E:sam:951", "B:rdrop:100", "B:sam:101:0")}
    check("lane.sh refuses A:rdrop / E:sam / root 100 / a 4th field", all(rc != 0 for rc in bad.values()), bad)
    check("bash -n lane.sh", subprocess.run(["bash", "-n", LANE], capture_output=True).returncode == 0)
    # (c) call sites
    src = open(TRAIN_PY).read()
    tree = ast.parse(src)
    parents = {ch: p_ for p_ in ast.walk(tree) for ch in ast.iter_child_nodes(p_)}

    def chain(n):
        out = []
        while n in parents:
            n = parents[n]
            if isinstance(n, ast.If):
                out.append("if " + ast.unparse(n.test))
            elif isinstance(n, ast.For):
                out.append("for " + ast.unparse(n.target).strip("()"))  # unparse renders tuples as (step, fb)
            elif isinstance(n, ast.FunctionDef):
                out.append("def " + n.name)
        return out

    def calls(pred):
        return [chain(n) for n in ast.walk(tree) if isinstance(n, ast.Call) and pred(n.func)]

    su_r = calls(lambda f: isinstance(f, ast.Name) and f.id == "setup_rdrop")
    su_s = calls(lambda f: isinstance(f, ast.Name) and f.id == "setup_sam")
    check("setup_rdrop / setup_sam: one call each, behind `if a.rdrop_cd > 0` / `if a.sam_cd > 0` in main()",
          su_r == [["if a.rdrop_cd > 0", "def main"]] and su_s == [["if a.sam_cd > 0", "def main"]], f"{su_r} {su_s}")
    cb = calls(lambda f: isinstance(f, ast.Attribute) and f.attr == "combine")
    sp = calls(lambda f: isinstance(f, ast.Attribute) and f.attr == "second_pass")
    sm = calls(lambda f: isinstance(f, ast.Attribute) and f.attr == "summary" and isinstance(f.value, ast.Name) and f.value.id in ("rdrop", "sam"))
    check("rdrop.combine / sam.second_pass once each inside their `... is not None and step >= t_cool` loop guards; rdrop.summary / "
          "sam.summary once each inside the eval block",
          cb == [["if rdrop is not None and step >= t_cool", "for step, fb", "def main"]]
          and sp == [["if sam is not None and step >= t_cool", "for step, fb", "def main"]]
          and sorted(c[0] for c in sm) == ["if rdrop is not None", "if sam is not None"]
          and all(c[1].startswith("if (step + 1) % a.eval_every == 0 or last") and c[2] == "for step, fb" for c in sm), f"{cb} {sp} {sm}")
    body = loop_body()
    i_rd, i_sam = idx_if(body, "rdrop is not None and step >= t_cool"), idx_if(body, "sam is not None and step >= t_cool")
    i_aux = idx_if(body, "a.aux_pos > 0")
    i_bw = [i for i, s in enumerate(body) if ast.unparse(s) == "loss.backward()"]
    i_zg = [i for i, s in enumerate(body) if isinstance(s, ast.For) and ast.unparse(s.target) == "opt" and "zero_grad" in ast.unparse(s)]
    i_clip = [i for i, s in enumerate(body) if isinstance(s, ast.Assign) and "clip_grad_norm_" in ast.unparse(s)]
    check("hook positions: rdrop hook right after the aux block and right before the zero_grad loop; sam hook right after "
          "loss.backward() and right before clip_grad_norm_",
          i_rd == i_aux + 1 and i_zg == [i_rd + 1] and i_bw == [i_sam - 1] and i_clip == [i_sam + 1],
          f"aux {i_aux} rdrop {i_rd} zero_grad {i_zg} backward {i_bw} sam {i_sam} clip {i_clip}")
    rd_setup = [i for i, s in enumerate(body) if False]  # placeholder (setup is outside the loop)
    main_body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"][0].body
    j_val = [i for i, s in enumerate(main_body) if isinstance(s, ast.If) and ast.unparse(s.test) == "a.val_mix > 0"]
    j_init = [i for i, s in enumerate(main_body) if isinstance(s, ast.Assign) and ast.unparse(s).startswith("rdrop, sam = ")]
    j_rd = [i for i, s in enumerate(main_body) if isinstance(s, ast.If) and ast.unparse(s.test) == "a.rdrop_cd > 0"]
    j_sam = [i for i, s in enumerate(main_body) if isinstance(s, ast.If) and ast.unparse(s.test) == "a.sam_cd > 0"]
    j_loop = [i for i, s in enumerate(main_body) if isinstance(s, ast.For) and ast.unparse(s.target).strip("()") == "step, fb"]
    check("setup position: `rdrop, sam = None, None` + the two guards right after the valcd setup, before the training loop",
          j_init == [j_val[0] + 1] and j_rd == [j_val[0] + 2] and j_sam == [j_val[0] + 3] and j_loop[0] > j_sam[0] and not rd_setup,
          f"val {j_val} init {j_init} rd {j_rd} sam {j_sam} loop {j_loop}")
    users = sorted(f for f in os.listdir(W5) if f.endswith(".py") and f not in ("cd_arms8.py", "train.py")
                   and re.search(r"\bcd_arms8\b", open(f"{W5}/{f}").read()))
    imps = sorted({(n.module if isinstance(n, ast.ImportFrom) else n.names[0].name)
                   for n in ast.walk(ast.parse(open(f"{W5}/cd_arms8.py").read())) if isinstance(n, (ast.Import, ast.ImportFrom))})
    check("no other nn_x/w5 module imports cd_arms8; cd_arms8 imports only torch and model", users == [] and imps == ["model", "torch"],
          f"users {users} imports {imps}")
    # (d) guards
    env = dict(a=argparse.Namespace(**a0), rdrop=None, sam=None, t_cool=0)
    vals = {g: bool(eval(g, {}, {**env, "step": s})) for g in W7_IFS for s in (0, 50, 19669)}
    check("every wave-7 guard is False at the defaults (any step)", not any(vals.values()), vals)
    allj = dry_cmds(out_n, parse)
    check(f"rdrop_cd 0 / sam_cd 0 in every one of the {len(allj)} previous DRY commands",
          len(allj) == len(old_jobs) and all(t["rdrop_cd"] == 0.0 and t["sam_cd"] == 0.0 for t, _ in allj.values()))


def w1_replica():
    P("\n== W1 main_aux_loss == the exact train.py loss statements (bitwise)")
    src, body, i0, i1 = loss_statements()
    P(f"  statements {i0}..{i1} of the loop body ({len(src.splitlines())} lines)")
    cases = [dict(), dict(aux_norm="feed"), dict(aux_mode="mse"), dict(loss_norm="batch"), dict(aux_pos=0.0),
             dict(aux_norm="feed", aux_mode="mse")]
    ok_all = True
    for noprand in (False, True):
        for kw in cases:
            a = make_args(**kw)
            model = FakeModel(0.0).eval()
            b, X, prand = make_batch()
            pr = None if noprand else prand
            lg0 = model(b, X, use_pos=True)
            W = 0.37
            ns = dict(torch=torch, listwise_loss=listwise_loss, bce_loss=bce_loss, model=model, b=b, a=a, prand=pr, trunc=None,
                      lg=lg0, cliplog=None, aux_w=lambda s: W, step=0)
            exec(src, {}, ns)
            loss_ref, fw = ns["loss"], ns["fw"]
            loss_new, main_new, lg_new = C8.main_aux_loss(a, model, b, lg0, W, pr, fw, None)
            same = torch.equal(loss_ref, loss_new) and torch.equal(lg_new, lg0)
            ok_all &= same
            P(f"    {'no-prand ' if noprand else ''}{kw or 'harness defaults'}: ref {loss_ref.item():.6f} replica {loss_new.item():.6f} "
              f"{'==' if same else '!='}")
    check("main_aux_loss == exec of the exact train.py statements on the same tensors (12 configurations, bitwise)", ok_all)
    a = make_args()
    model = FakeModel(0.0).eval()
    b, X, prand = make_batch()
    lg0 = model(b, X, use_pos=True)
    loss, main, _ = C8.main_aux_loss(a, model, b, lg0, 0.5, prand, None, None)
    ref_main = listwise_loss(lg0, b["y"], b["cmask"], None, norm="feed", denom=512)
    check("main_aux_loss returns the main CE separately (== listwise_loss feed-normalised) and loss > main with aux_pos > 0",
          torch.equal(main, ref_main) and loss.item() > main.item())


def ref_symkl(lg1, lg2, m):
    out = []
    for i in range(lg1.shape[0]):
        l1, l2 = lg1[i][m[i]].double(), lg2[i][m[i]].double()
        p1, p2 = torch.softmax(l1, 0), torch.softmax(l2, 0)
        out.append(0.5 * ((p1 * (torch.log(p1) - torch.log(p2))).sum() + (p2 * (torch.log(p2) - torch.log(p1))).sum()))
    return torch.stack(out)


def w2_rdrop():
    P("\n== W2 R-Drop: symkl_feed, RDrop.combine, statistics")
    b, X, prand = make_batch()
    g = torch.Generator().manual_seed(5)
    lg1, lg2 = torch.randn(b["y"].shape, generator=g) * 2, torch.randn(b["y"].shape, generator=g) * 2
    m = b["cmask"]
    kl = C8.symkl_feed(lg1, lg2, m)
    ref = ref_symkl(lg1, lg2, m)
    kl_m = C8.symkl_feed(lg1, lg2.masked_fill(~m, 123.0), m)
    check("symkl_feed == float64 per-feed reference (max |diff| < 1e-5) and ignores masked logits",
          (kl.double() - ref).abs().max().item() < 1e-5 and torch.equal(kl, kl_m) and (kl >= 0).all(),
          f"max diff {(kl.double() - ref).abs().max().item():.2e}; KL {[round(x, 4) for x in kl.tolist()]}")
    check("symkl_feed of identical logits is exactly 0", torch.equal(C8.symkl_feed(lg1, lg1.clone(), m), torch.zeros(len(m))))
    a = make_args(rdrop_cd=1.0)
    # eval mode: no dropout -> the two passes coincide
    model = FakeModel(0.3).eval()
    lg = model(b, X, use_pos=True)
    loss1, main1, _ = C8.main_aux_loss(a, model, b, lg, 0.4, prand, None, None)
    rd = C8.RDrop(1.0, a.max_feeds, torch.device("cpu"))
    cl = types.SimpleNamespace(main=main1)
    out = rd.combine(model, a, b, X, lg, loss1, 0.4, prand, None, None, cl)
    check("combine in eval mode (no dropout): returns exactly loss1, KL statistics 0, CLIPLOG main unchanged in value",
          torch.equal(out, loss1) and float(rd.kl_sum) == 0.0 and rd.n_steps == 1 and torch.equal(cl.main, main1)
          and float(rd.n_feeds) == float((b["y"].sum(1) > 0).sum()))
    s = rd.summary(12)
    check("summary reports the mean (0.000000) and resets", "symKL 0.000000 nats" in s and rd.n_steps == 0 and float(rd.kl_sum) == 0.0, s)
    # train mode (dropout): seeded independent recomputation
    model = FakeModel(0.3).train()
    torch.manual_seed(11)
    lg = model(b, X, use_pos=True)
    loss1, main1, _ = C8.main_aux_loss(a, model, b, lg, 0.4, prand, None, None)
    torch.manual_seed(99)
    lg2 = model(b, X, use_pos=True)
    loss2, main2, _ = C8.main_aux_loss(a, model, b, lg2, 0.4, prand, None, None)
    valid = (b["y"].sum(1) > 0).float()
    klv = C8.symkl_feed(lg, lg2, m) * valid
    kl_term = klv.sum() / a.max_feeds
    expected = 0.5 * (loss1 + loss2) + 1.0 * kl_term
    exp_main = 0.5 * (main1 + main2) + 1.0 * kl_term
    torch.manual_seed(11)
    lg = model(b, X, use_pos=True)
    loss1b, main1b, _ = C8.main_aux_loss(a, model, b, lg, 0.4, prand, None, None)
    torch.manual_seed(99)
    rd = C8.RDrop(1.0, a.max_feeds, torch.device("cpu"))
    cl = types.SimpleNamespace(main=main1b)
    out = rd.combine(model, a, b, X, lg, loss1b, 0.4, prand, None, None, cl)
    check("combine in train mode (seeded): == 0.5 (L1 + L2) + alpha sum valid symKL / max_feeds recomputed independently (bitwise); "
          "KL > 0; CLIPLOG main == 0.5 (CE1 + CE2) + alpha KL; statistics accumulate",
          torch.equal(out, expected) and kl_term.item() > 0 and torch.equal(cl.main, exp_main)
          and abs(float(rd.kl_sum) - klv.sum().item()) < 1e-6 and float(rd.n_feeds) == valid.sum().item(),
          f"loss1 {loss1b.item():.5f} loss2 {loss2.item():.5f} KL term {kl_term.item():.6f} mean per valid feed {klv.sum().item() / valid.sum().item():.5f}")
    out.backward()
    check("the combined loss is differentiable (gradients flow to the parameters)",
          all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
    # alpha scaling
    rd2 = C8.RDrop(2.5, a.max_feeds, torch.device("cpu"))
    torch.manual_seed(99)
    out2 = rd2.combine(model, a, b, X, lg.detach(), loss1b.detach(), 0.4, prand, None, None, None)
    check("alpha scales the KL term only (alpha 2.5 vs 1.0)", abs((out2 - out).item() - 1.5 * kl_term.item()) < 1e-6)


def w3_sam():
    P("\n== W3 SAM: eps norm, exact restore, g2 == gradient at w + eps, validity mark")
    a = make_args(sam_cd=0.05)
    b, X, prand = make_batch()
    lines = []
    for rho in (0.01, 1e-9):  # 0.01: ||eps||/||w|| ~0.006 on the tiny fake model (||w|| 1.76); 0.05 would be INVALID there
        model = FakeModel(0.0).eval()
        for p in model.parameters():
            p.grad = None
        lg = model(b, X, use_pos=True)
        loss1, _, _ = C8.main_aux_loss(a, model, b, lg, 0.4, prand, None, None)
        loss1.backward()
        g1 = [p.grad.detach().clone() for p in model.parameters()]
        w = [p.detach().clone() for p in model.parameters()]
        gnorm = torch.sqrt(sum((g ** 2).sum() for g in g1))
        # independent reference: deep copy at w + eps
        ref = FakeModel(0.0).eval()  # independent copy at w (deepcopy trips over the non-leaf aux_out attribute)
        ref.load_state_dict(model.state_dict())
        with torch.no_grad():
            for p, g in zip(ref.parameters(), g1):
                p.add_(g * (rho / (gnorm + 1e-12)))
        lg_r = ref(b, X, use_pos=True)
        loss_r, _, _ = C8.main_aux_loss(a, ref, b, lg_r, 0.4, prand, None, None)
        loss_r.backward()
        g2_ref = [p.grad.detach().clone() for p in ref.parameters()]
        sam = C8.SAM(rho, model, lines.append)
        sam.second_pass(model, a, b, X, loss1, 0.4, prand, None, None, step=0)
        g2 = [p.grad.detach().clone() for p in model.parameters()]
        restored = all(torch.equal(p.detach(), w_) for p, w_ in zip(model.parameters(), w))
        same_g2 = all(torch.equal(x, y) for x, y in zip(g2, g2_ref))
        wnorm = torch.sqrt(sum((x ** 2).sum() for x in w)).item()
        if rho == 0.01:
            check("rho 0.01: parameters restored bitwise; ||eps|| == rho (1e-5 rel); ratio == rho / ||w||; .grad == the gradient at w + eps "
                  "of an independent deep copy (bitwise); logged line VALID; loss statistics accumulated",
                  restored and abs(sam.eps_norm - rho) < 1e-5 * rho and abs(sam.ratio - rho / wnorm) < 1e-6 and same_g2 and sam.valid
                  and len(lines) == 1 and "VALID (<= 0.02)" in lines[0] and sam.n == 1 and abs(float(sam.loss2_sum) - loss_r.item()) < 1e-6,
                  f"eps {sam.eps_norm:.6f} ratio {sam.ratio:.6f} ||g1|| {sam.g1_norm:.4f} | {lines[0][:90]}")
            s = sam.summary(12)
            check("summary reports the ratio / VALID and the two losses, then resets", "VALID" in s and f"{sam.ratio:.6f}" in s and sam.n == 0, s)
            g_diff = max((x - y).abs().max().item() for x, y in zip(g2, g1))
            check("g2 differs from g1 at rho 0.01 (the second pass is at a different point)", g_diff > 1e-8, f"max |g2 - g1| {g_diff:.3e}")
        else:
            close = all(torch.allclose(x, y, rtol=1e-4, atol=1e-7) for x, y in zip(g2, g1))
            check("rho 1e-9: restored bitwise and g2 == g1 to 1e-4 relative", restored and close and same_g2)
    # INVALID mark
    model = FakeModel(0.0).eval()
    lg = model(b, X, use_pos=True)
    loss1, _, _ = C8.main_aux_loss(a, model, b, lg, 0.4, prand, None, None)
    loss1.backward()
    w = [p.detach().clone() for p in model.parameters()]
    wnorm = torch.sqrt(sum((x ** 2).sum() for x in w)).item()
    lines = []
    sam = C8.SAM(0.05 * wnorm, model, lines.append)
    sam.second_pass(model, a, b, X, loss1, 0.4, prand, None, None, step=3)
    check("an absurd rho (0.05 ||w||, ratio 0.05 > 0.02) is marked INVALID in the log line and sam.valid False; parameters still restored",
          not sam.valid and "INVALID" in lines[0] and all(torch.equal(p.detach(), w_) for p, w_ in zip(model.parameters(), w)), lines[0][:120])


def w4_setup():
    P("\n== W4 setup guards")
    b, X, prand = make_batch()
    model = FakeModel(0.1)
    logs = []

    def refuses(fn, **kw):
        try:
            fn(make_args(**kw), model, kw.pop("_t", 0), 100, logs.append)
            return False
        except AssertionError:
            return True
    bad_r = dict(zero=refuses(C8.setup_rdrop, rdrop_cd=0.0), both=refuses(C8.setup_rdrop, rdrop_cd=1.0, sam_cd=0.05),
                 bce=refuses(C8.setup_rdrop, rdrop_cd=1.0, loss="bce"), intent=refuses(C8.setup_rdrop, rdrop_cd=1.0, intent=0.1),
                 valcd=refuses(C8.setup_rdrop, rdrop_cd=1.0, val_mix=0.01), tcool=refuses(C8.setup_rdrop, rdrop_cd=1.0, _t=100))
    bad_s = dict(zero=refuses(C8.setup_sam, sam_cd=0.0), both=refuses(C8.setup_sam, sam_cd=0.05, rdrop_cd=1.0),
                 bce=refuses(C8.setup_sam, sam_cd=0.05, loss="bce"), intent=refuses(C8.setup_sam, sam_cd=0.05, intent=0.1),
                 valcd=refuses(C8.setup_sam, sam_cd=0.05, val_mix=0.01), tcool=refuses(C8.setup_sam, sam_cd=0.05, _t=100))
    check("setup_rdrop refuses alpha 0, both flags, --loss bce, --intent, --val_mix, t_cool >= total", all(bad_r.values()), bad_r)
    check("setup_sam refuses rho 0, both flags, --loss bce, --intent, --val_mix, t_cool >= total", all(bad_s.values()), bad_s)
    logs.clear()
    rd = C8.setup_rdrop(make_args(rdrop_cd=1.0), model, 0, 100, logs.append)
    sm = C8.setup_sam(make_args(sam_cd=0.05), model, 0, 100, logs.append)
    check("positive calls return RDrop(alpha 1.0) / SAM(rho 0.05) and log one line each",
          isinstance(rd, C8.RDrop) and rd.alpha == 1.0 and isinstance(sm, C8.SAM) and sm.rho == 0.05 and len(logs) == 2
          and logs[0].startswith("wave-7 RDROP:") and logs[1].startswith("wave-7 SAM:"), [ln[:80] for ln in logs])


# ================================================================================================= GPU part
E2E = ("--init {init} --epochs 0.0025 --lr 2e-3 --warmup 0 --sched wsd --cooldown_frac 1.0 --muon --aux_pos 0.5 "
       "--aux_mode listwise --aux_tau 10 --aux_anneal 0.1 --cooldown_mix 0.2 --loss_norm feed --holdout 30000 "
       "--select_on holdout --eval_every 12 --seed 811 --no_test --cliplog 6")
NAMES = ["w5_tw7_pre_off", "w5_tw7_new_off", "w5_tw7_rdrop", "w5_tw7_off_d0", "w5_tw7_rdrop_d0", "w5_tw7_sam_d0"]
EXTRA = {"w5_tw7_pre_off": [], "w5_tw7_new_off": [], "w5_tw7_rdrop": ["--rdrop_cd", "1.0", "--save_last", "--save_holdout"],
         "w5_tw7_off_d0": ["--drop_cd", "0"], "w5_tw7_rdrop_d0": ["--drop_cd", "0", "--rdrop_cd", "1.0"],
         "w5_tw7_sam_d0": ["--drop_cd", "0", "--sam_cd", "0.05", "--save_last", "--save_holdout"]}


def outputs(v):
    fs = [f"{WD}/logs/{v}.log", f"{WD}/logs/{v}.json", f"{WD}/logs/{v}_cliplog.npz"]
    for x in (v, f"{v}@last"):
        fs += [f"{WD}/ckpt/{x}.pt", f"{WD}/preds/nn_{x}_trainonly_val.npy", f"{WD}/preds/nn_{x}_trainonly_test.npy",
               f"{WD}/preds/holdout/nn_{x}_holdout.npy"]
    return fs


def clean(v):
    for f in outputs(v):
        if os.path.exists(f):
            os.remove(f)


def gpu_ok(gpu):
    try:
        n = len([x for x in subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid", "--format=csv,noheader"],
                                           capture_output=True, text=True).stdout.splitlines() if x.strip()])
        mem = subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True).stdout.strip().split(",")
        free = (int(mem[1]) - int(mem[0])) / 1024
    except Exception as e:  # noqa: BLE001
        return False, f"nvidia-smi failed: {e}"
    return n < 4 and free >= 60, f"GPU {gpu}: {n} compute processes, {free:.0f} GB free"


def run_e2e(code_dir, v, gpu):
    rec = f"{SCR}/rec_{v}.npz"
    cmd = [PY, f"{code_dir}/tests/e2e_w7.py", f"{code_dir}/train.py", rec, "--variant", v] + E2E.format(init=INIT).split() + EXTRA[v]
    env = {**os.environ, "OMP_NUM_THREADS": str(NT), "CUDA_VISIBLE_DEVICES": str(gpu)}
    t0 = time.time()
    while True:
        ok, msg = gpu_ok(gpu)
        if ok:
            break
        P(f"  waiting for {msg}")
        assert time.time() - t0 < 1800, f"GPU {gpu} did not qualify within 30 min"
        time.sleep(30)
    P(f"  {time.strftime('%H:%M:%S')} start {v} on GPU {gpu} ({msg})")
    out = open(f"{SCR}/{v}.stdout", "w")
    pr = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, cwd=code_dir, env=env)
    rc = pr.wait()
    P(f"  ran {v} from {code_dir} ({time.time() - t0:.0f}s, rc {rc}, GPU {gpu})")
    assert rc == 0, open(f"{SCR}/{v}.stdout").read()[-3000:]
    return dict(np.load(rec, allow_pickle=True)), open(f"{WD}/logs/{v}.log").read()


def metric_lines(log):
    return [re.sub(r" [\d.]+ min", "", ln) for ln in log.splitlines() if ln.startswith("step ") or ln.startswith("BEST")]


def first_loss(log):
    ln = [x for x in log.splitlines() if x.startswith("step ")][0]
    return re.search(r"loss ([\d.]+)", ln).group(1)


def per_forward(rec):
    sh, out, k = rec["f_shape"], [], 0
    for b_, m_ in sh:
        out.append(rec["lg_model"][k:k + b_ * m_].reshape(b_, m_))
        k += b_ * m_
    return out


def gathers(rec):
    off = rec["g_off"]
    return [rec["g_fidx"][off[i]:off[i + 1]] for i in range(len(off) - 1)]


def ck_equal(a, b):
    ca = torch.load(a, map_location="cpu", weights_only=False)["state_dict"]
    cb = torch.load(b, map_location="cpu", weights_only=False)["state_dict"]
    return ca.keys() == cb.keys() and all(torch.equal(ca[k], cb[k]) for k in ca)


def ck_maxdiff(a, b):
    ca = torch.load(a, map_location="cpu", weights_only=False)["state_dict"]
    cb = torch.load(b, map_location="cpu", weights_only=False)["state_dict"]
    return max((ca[k].float() - cb[k].float()).abs().max().item() for k in ca if ca[k].dtype.is_floating_point)


def kl_lines(log):
    return [float(re.search(r"symKL ([\d.]+) nats", ln).group(1)) for ln in log.splitlines() if ln.startswith("wave-7 RDROP step")]


def steps_of(rec):
    """forwards grouped by step: list of lists of forward indices."""
    out = {}
    for i, s in enumerate(rec["f_step"]):
        out.setdefault(int(s), []).append(i)
    return [out[k] for k in sorted(out)]


def g_e2e():
    P("\n== G e2e (deterministic wrapper; 24 CD steps from v17_pre2s_s101@77500, cooldown_mix 0.2, evals at 12 / 24, cliplog 6)")
    gpus = (os.environ.get("W7_GPUS") or _CVD).split()
    for v in NAMES:
        clean(v)
    pre = f"{SCR}/w5_pre_w7_code"  # previous train.py with every other module of this checkout
    shutil.rmtree(pre, ignore_errors=True)
    os.makedirs(f"{pre}/tests")
    for f in os.listdir(W5):
        if f.endswith(".py"):
            shutil.copy(f"{W5}/{f}", f"{pre}/{f}")
    shutil.copy(f"{HERE}/e2e_w7.py", f"{pre}/tests/e2e_w7.py")
    shutil.copy(f"{REF}/train.py.txt", f"{pre}/train.py")
    code = {v: (pre if v == "w5_tw7_pre_off" else W5) for v in NAMES}
    res, err = {}, {}
    queues = {g: [v for i, v in enumerate(NAMES) if i % len(gpus) == k] for k, g in enumerate(gpus)}

    def worker(g):
        for v in queues[g]:
            try:
                res[v] = run_e2e(code[v], v, g)
            except Exception as e:  # noqa: BLE001
                err[v] = repr(e)[-2000:]
                P(f"  ERROR {v}: {err[v][-300:]}")
    ths = [threading.Thread(target=worker, args=(g,)) for g in gpus]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    try:
        assert not err, err
        (ra, la), (rb, lb) = res["w5_tw7_pre_off"], res["w5_tw7_new_off"]
        (rr, lr), (r0, l0), (rd, ld), (rs, ls) = res["w5_tw7_rdrop"], res["w5_tw7_off_d0"], res["w5_tw7_rdrop_d0"], res["w5_tw7_sam_d0"]
        ga, gb = gathers(ra), gathers(rb)
        # ---- G1
        check("G1 flags OFF, previous vs new train.py: identical training-gather sequence (24 steps), one train forward per step, "
              "train-mode logits and loss-input logits",
              len(ga) == len(gb) == 24 and all(np.array_equal(x, y) for x, y in zip(ga, gb)) and len(ra["f_step"]) == len(rb["f_step"]) == 24
              and np.array_equal(ra["lg_model"], rb["lg_model"]) and np.array_equal(ra["lg_loss"], rb["lg_loss"]) and int(rb["n_loss"]) == 24)
        check("G1 identical logged metric lines; bit-identical checkpoint and val preds",
              metric_lines(la) == metric_lines(lb) and len(metric_lines(lb)) >= 3
              and ck_equal(f"{WD}/ckpt/{NAMES[0]}.pt", f"{WD}/ckpt/{NAMES[1]}.pt")
              and np.array_equal(np.load(f"{WD}/preds/nn_{NAMES[0]}_trainonly_val.npy"), np.load(f"{WD}/preds/nn_{NAMES[1]}_trainonly_val.npy")),
              metric_lines(lb)[-1] if metric_lines(lb) else "")
        check("G1 no wave-7 log line in either flags-off run", "wave-7" not in la and "wave-7" not in lb)
        RES["e2e_g1"] = dict(off_lines=metric_lines(lb))
        # ---- G2 rdrop with dropout 0 vs off with dropout 0
        fd, f0 = per_forward(rd), per_forward(r0)
        st_d = steps_of(rd)
        two = len(st_d) == 24 and all(len(s) == 2 for s in st_d)
        same_pass = two and all(np.array_equal(fd[s[0]], fd[s[1]]) for s in st_d)
        eq_off = [np.array_equal(fd[s[0]], f0[i]) for i, s in enumerate(st_d)] if two and len(f0) == 24 else [False, False]
        first_div = next((i for i, e in enumerate(eq_off) if not e), None)
        same_as_off = eq_off[0] and eq_off[1]  # steps 0 and 1 bitwise == flags-off (=> the step-0 gradient was identical); later steps may
        # diverge through the fp residual of the KL gradient (algebraically 0 at p1 == p2, ~1e-7) amplified by Muon's orthogonalisation
        kls = kl_lines(ld)
        gn_rel = (np.abs(rd["o_gnorm"] - r0["o_gnorm"]) / np.maximum(np.abs(r0["o_gnorm"]), 1e-12)) if len(rd["o_gnorm"]) == len(r0["o_gnorm"]) else np.array([np.inf] * 4)
        md = ck_maxdiff(f"{WD}/ckpt/{NAMES[4]}.pt", f"{WD}/ckpt/{NAMES[3]}.pt")
        check("G2 --rdrop_cd 1.0 --drop_cd 0 vs --drop_cd 0: identical gathers; 2 train forwards per step with bitwise-identical logits "
              "(pass 1 == pass 2); params unchanged within a step; 2 RDROP lines with symKL 0.000000; pass-1 logits == flags-off logits "
              "bitwise at steps 0 and 1 and gradient norms at the 4 optimiser entries of those steps bitwise identical (the two-pass loss "
              "reduces exactly to the one-pass loss); first-eval loss identical (later divergence = fp residual of the KL gradient, info)",
              all(np.array_equal(x, y) for x, y in zip(gathers(rd), gathers(r0))) and same_pass and same_as_off and rd["f_same"].all()
              and len(kls) == 2 and all(k == 0.0 for k in kls) and first_loss(ld) == first_loss(l0) and len(gn_rel) == 48 and gn_rel[:4].max() == 0.0,
              f"KL lines {kls}; first step whose logits differ from flags-off {first_div}; max rel grad-norm diff over 24 steps {gn_rel.max():.2e}; "
              f"metric lines equal {metric_lines(ld) == metric_lines(l0)}; checkpoint max |diff| {md:.2e}")
        RES["e2e_g2"] = dict(kl=kls, first_div=first_div, gn_rel_max=float(gn_rel.max()), ck_maxdiff=md, lines_equal=metric_lines(ld) == metric_lines(l0))
        # ---- G3 rdrop with dropout 0.1 vs flags off
        fr = per_forward(rr)
        st_r = steps_of(rr)
        two_r = len(st_r) == 24 and all(len(s) == 2 for s in st_r)
        differ = two_r and all(not np.array_equal(fr[s[0]], fr[s[1]]) for s in st_r)
        klr = kl_lines(lr)
        outs_ok = all(os.path.exists(f) for f in (f"{WD}/ckpt/{NAMES[2]}@last.pt", f"{WD}/preds/nn_{NAMES[2]}@last_trainonly_val.npy",
                                                   f"{WD}/preds/holdout/nn_{NAMES[2]}_holdout.npy", f"{WD}/preds/holdout/nn_{NAMES[2]}@last_holdout.npy"))
        check("G3 --rdrop_cd 1.0 (dropout 0.1) vs flags OFF: identical gathers; 48 forwards (2 per step) whose two passes differ at every "
              "step; 48 loss-input logit sets; logged mean symKL in [0.001, 0.05] nats at both evals; params unchanged within a step; "
              "checkpoint differs; @last / holdout files written",
              all(np.array_equal(x, y) for x, y in zip(gathers(rr), gathers(rb))) and differ and int(rr["n_loss"]) == 48 and rr["f_same"].all()
              and len(klr) == 2 and all(0.001 <= k <= 0.05 for k in klr) and not ck_equal(f"{WD}/ckpt/{NAMES[2]}.pt", f"{WD}/ckpt/{NAMES[1]}.pt")
              and outs_ok and "wave-7 RDROP:" in lr,
              f"KL lines {klr}; first-eval loss rdrop {first_loss(lr)} vs off {first_loss(lb)}")
        RES["e2e_g3"] = dict(kl=klr, lines=metric_lines(lr))
        # ---- G4 sam with dropout 0 vs off with dropout 0
        st_s = steps_of(rs)
        two_s = len(st_s) == 24 and all(len(s) == 2 for s in st_s)
        eps2 = np.array([rs["f_eps"][s[1]] for s in st_s]) if two_s else np.zeros(0)
        same2 = np.array([rs["f_same"][s[1]] for s in st_s]) if two_s else np.ones(1, bool)
        first_line = [ln for ln in ls.splitlines() if ln.startswith("wave-7 SAM first cooldown step")]
        ratio = float(re.search(r"\|\|eps\|\|/\|\|w\|\| ([\d.]+)", first_line[0]).group(1)) if first_line else float("nan")
        gn_diff = (np.abs(rs["o_gnorm"] - r0["o_gnorm"]) / np.maximum(np.abs(r0["o_gnorm"]), 1e-12)) if len(rs["o_gnorm"]) == len(r0["o_gnorm"]) else np.zeros(1)
        check("G4 --sam_cd 0.05 --drop_cd 0 vs --drop_cd 0: identical gathers; 2 forwards per step; ||params - w|| == 0.05 (1e-3 rel) at "
              "every second forward (and != w); parameters bitwise w at the FIRST optimiser entry of every step (restored; the Muon entry "
              "follows AdamW's own update); first-step "
              "line VALID with ratio < 0.02; 2 SAM summary lines; first-eval loss identical (loss at w); gradient norms at the "
              "optimiser differ from flags-off (g2 != g1); checkpoint differs; @last / holdout files written",
              all(np.array_equal(x, y) for x, y in zip(gathers(rs), gathers(r0))) and two_s and len(eps2) == 24
              and np.all(np.abs(eps2 - 0.05) < 0.05 * 1e-3) and not same2.any() and len(rs["o_same"]) == 48 and all(rs["o_same"][0::2])
              and len(first_line) == 1 and "VALID (<= 0.02)" in first_line[0] and ratio < 0.02
              and sum(ln.startswith("wave-7 SAM step") for ln in ls.splitlines()) == 2 and first_loss(ls) == first_loss(l0)
              and gn_diff.max() > 1e-4 and not ck_equal(f"{WD}/ckpt/{NAMES[5]}.pt", f"{WD}/ckpt/{NAMES[3]}.pt")
              and os.path.exists(f"{WD}/ckpt/{NAMES[5]}@last.pt") and os.path.exists(f"{WD}/preds/holdout/nn_{NAMES[5]}@last_holdout.npy"),
              f"eps2 min/max {eps2.min() if len(eps2) else float('nan'):.5f}/{eps2.max() if len(eps2) else float('nan'):.5f}; ratio {ratio:.6f}; "
              f"restored at first entries {int(sum(rs['o_same'][0::2]))}/{len(rs['o_same'][0::2])}; max rel grad-norm diff vs off {gn_diff.max():.2e}; {first_line[0][:140] if first_line else 'no first-step line'}")
        RES["e2e_g4"] = dict(ratio=ratio, eps2=eps2.tolist(), lines=metric_lines(ls), first_line=first_line[0] if first_line else "")
        # control: off_d0 has one forward per step and no wave-7 line
        check("control: --drop_cd 0 flags OFF has 24 forwards and no wave-7 line", len(r0["f_step"]) == 24 and "wave-7" not in l0)
    finally:
        for v in NAMES:
            clean(v)
        shutil.rmtree(pre, ignore_errors=True)


def main():
    t_all = time.time()
    P(f"== nn_x/w5 --rdrop_cd / --sam_cd (wave 7) tests, part {PART} ({time.ctime()}, CPU threads {NT})")
    if PART == "cpu":
        w0_code()
        w1_replica()
        w2_rdrop()
        w3_sam()
        w4_setup()
    else:
        g_e2e()
    RES["failed"] = FAIL
    P(f"\n== {len(RES['checks']) - len(FAIL)}/{len(RES['checks'])} checks passed; failed: {FAIL}; {time.time() - t_all:.0f}s")
    json.dump(RES, open(f"{WD}/logs/test_w7_{PART}.json", "w"), indent=1, default=str)
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())
