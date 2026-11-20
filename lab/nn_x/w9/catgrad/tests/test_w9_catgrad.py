"""Tests of the WAVE-9 arm CATGRAD (nn_x/w9/catgrad = a copy of the nn_x/w5 code + --catw grad; scratchpad w9/catgrad.json with
the two skeptics' fixes: LSE-pooled detached category head, NO train.py loss change, exact L_c + L_w identity, gradient routing).

  cpu part:  cd nn_x/w9/catgrad && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 .venv/bin/python tests/test_w9_catgrad.py cpu
  gpu part:  cd nn_x/w9/catgrad && CUDA_VISIBLE_DEVICES=<0|1> OMP_NUM_THREADS=12 .venv/bin/python tests/test_w9_catgrad.py gpu

CPU part
  C0 code. (a) lane.sh DRY: A:base:23 == nn_x/w5/lane.sh A:base:23 except the variant name (w9_base_s23: A_BASE + HYG exactly);
     A:catgrad differs from A:base ONLY by catw '' -> 'grad'; E:catgrad:951 == w5 E:base:951 except variant + catw; the lane refuses
     some GPUs; bash -n. (b) AST proof train.py: minus the --catw add_argument, the `if a.catw:` block and the `catw=` keyword of the
     Ranker call (exact set), with the w9_ variant assert replaced by the w5_ one == frozen nn_x/w5/train.py (tests/ref_w5, md5 at
     copy time). (c) AST proof model.py: minus the `catw=None` parameter, `self.catw = catw`, the `if catw:` block of __init__, the
     method `catgrad` and the `if self.catw:` block of forward (exact set) == frozen nn_x/w5/model.py. (d) common.py == nn_x/w5
     up to string constants (output directory); data / cd_arms* / finetune / holdout_eval / stale_pred / blend / mini_split
     byte-identical to nn_x/w5. (e) --catw OFF by default; the flag-off guards are False; the base / E:base commands have catw ''.
     (f) call sites: Ranker.catgrad called once, in forward, behind `if self.catw`; cg_* read only in __init__ / catgrad.
  C1 model (real MiniSplit train batches, the F7 architecture = cfg of nn/ckpt/v17_pre2s_s101@77500.pt, its trained weights):
     paired init (every baseline parameter bit-identical with and without the flag at the same seed; cfg differs only by catw);
     at init g_k == log n_k and h == the flag-off logit (bitwise); exact factorisation P(i) = softmax_k(g) * softmax_within(h);
     loss identity listwise_loss(composite, feed) == L_c + L_w (brute force) to 1e-5 on batches with multi-positive feeds,
     singleton categories and padding; gradient routing: d CE(composite) / d trunk == d L_w / d trunk, d / d head == d L_c / d head
     (to 1e-5 rel), L_c has no trunk gradient, L_w no head gradient, and for every (feed, category) sum_j dL/dh_j == 0; with the
     head perturbed (non-trivial g) the same identities; train mode (dropout, K9 aux) forward / backward finite.
  C2 CPU end-to-end (tests/e2e_cpu.py; miniature split; 2 epochs, cooldown_mix 0.3, evals every 10 steps, cliplog, save_last,
     save_holdout with --holdout 30000 = every random-like mini feed; ONE CPU thread per run: multi-threaded CPU training is not
     bitwise reproducible run-to-run (checked: the same code twice differs from the 3rd step at 2 or 12 threads, identical at 1)): flags OFF (this copy) vs the frozen nn_x/w5 code (same modules, common.py output path redirected): identical
     training-batch sequence, train-mode logits, loss-input logits, logged lines, bit-identical checkpoint, val and holdout preds.
     --catw grad: same batch sequence; for EVERY training batch the logits into the loss == g.gather(ccat) + h - LSE_cat(h)
     recomputed in numpy from the recorded g / h (1e-4); checkpoint cfg has catw 'grad' and cg_* parameters; Ranker(**cfg) reload
     reproduces the saved val preds; @last / holdout files written; wave-9 log line; checkpoint differs from flags OFF.
GPU part
  G1 plumbing smoke of the flag ON through lane.sh (W5_SMOKE=1 A:catgrad:23 -> w9_smoke_catgrad_s23: 20K train feeds, 1 epoch):
     run completes, json / log / selected + @last val and holdout preds written, finite, val NDCG > 0.30, ckpt cfg catw. Outputs
     deleted afterwards.
Writes nn_x/w9/catgrad/logs/test_w9_catgrad_<part>.log (+ .json).
"""
import argparse
import ast
import collections
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import warnings

PART = sys.argv[1] if len(sys.argv) > 1 else "cpu"
assert PART in ("cpu", "gpu"), "usage: test_w9_catgrad.py cpu|gpu"
_CVD = os.environ.get("CUDA_VISIBLE_DEVICES", "x")
if PART == "cpu":
    assert _CVD == "", "cpu part: run with CUDA_VISIBLE_DEVICES=''"
else:
    assert _CVD in ("0", "1"), "gpu part: CUDA_VISIBLE_DEVICES=<0|1> (the CATGRAD agent's GPUs)"
NT = min(12, int(os.environ.get("OMP_NUM_THREADS", "12")))
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(NT)
warnings.filterwarnings("ignore", message=".*CUDA is not available.*")
warnings.filterwarnings("ignore", message=".*device_type of 'cuda'.*")
HERE = os.path.dirname(os.path.abspath(__file__))
W9 = os.path.dirname(HERE)
ROOT = "<repo>"
W5 = f"{ROOT}/nn_x/w5"
REF = f"{HERE}/ref_w5"
PY = f"{ROOT}/.venv/bin/python"
INIT = f"{ROOT}/nn/ckpt/v17_pre2s_s101@77500.pt"
SCR = os.environ.get("T9_SCRATCH", "/tmp/scratchpad/t9")
os.makedirs(SCR, exist_ok=True)
os.makedirs(f"{W9}/logs", exist_ok=True)
sys.path.insert(0, HERE)
sys.path.insert(1, W9)
LOG = open(f"{W9}/logs/test_w9_catgrad_{PART}.log", "w")
RES = {"checks": []}
FAIL = []
MODULES = ("data.py", "cd_arms.py", "cd_arms4.py", "cd_arms5.py", "cd_arms6.py", "cd_arms7.py", "cd_arms8.py", "finetune.py",
           "holdout_eval.py", "stale_pred.py", "blend.py")
E2E = ("--no_pos --aux_pos 0.5 --aux_mode listwise --aux_tau 10 --sim_bias --user0 --domain --holdout 30000 --select_on holdout "
       "--loss_norm feed --xt cvnd_pre2s --xt_drop 0 --muon --lr 1e-3 --sched wsd --cooldown_frac 0.5 --cooldown_mix 0.3 "
       "--aux_anneal 0.1 --retr_m 16 --attn_gate --epochs 2 --eval_every 10 --no_test --save_last --save_holdout "
       "--select_min_step 10 --cliplog 5 --warmup 2 --cand_budget 4096 --max_feeds 8 --seed 23")


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


def strip_doc(tree):
    if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant):
        tree.body = tree.body[1:]
    return tree


def load_parse(path):
    import cd_arms as CA
    tree = ast.parse(open(path).read())
    fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "parse"][0]
    ns = {"argparse": argparse, "POOL_P_DEFAULT": CA.POOL_P_DEFAULT}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "train_parse", "exec"), ns)

    def parse(argv):
        sys.argv = ["train.py"] + argv
        return ns["parse"]()
    return parse


def dry_cmds(script, gpu, jobs, rx, env=None):
    out = subprocess.run(["bash", script, str(gpu), jobs], capture_output=True, text=True,
                         env={**os.environ, "DRY": "1", **(env or {})})
    cmds = {}
    for line in out.stdout.splitlines():
        if " train.py " in line:
            m = re.search(rx, line)
            cmds[m.group(0)] = line.split(" train.py ", 1)[1].split()
    return out.returncode, cmds


def diff_ns(a, b, skip=("variant",)):
    return {k: (a[k], b[k]) for k in a if k not in skip and a[k] != b[k]}


# ================================================================================================= AST strippers
class StripTrain(ast.NodeTransformer):
    """Removes exactly the wave-9 statements of train.py."""
    ASSERT_NEW = 'assert a.variant.startswith(("w5_", "w9_")), "nn_x/w9/catgrad copy: variant names must start with w5_ or w9_"'
    ASSERT_OLD = 'assert a.variant.startswith("w5_"), "nn_x/w5 copy: variant names must start with w5_"'
    EXPECT = collections.Counter({"add_argument --catw": 1, "if a.catw": 1, "Ranker(catw=)": 1})

    def __init__(self):
        self.removed = collections.Counter()

    def visit_Expr(self, n):
        v = n.value
        if (isinstance(v, ast.Call) and isinstance(v.func, ast.Attribute) and v.func.attr == "add_argument" and v.args
                and isinstance(v.args[0], ast.Constant) and v.args[0].value == "--catw"):
            self.removed["add_argument --catw"] += 1
            return None
        return self.generic_visit(n)

    def visit_If(self, n):
        if ast.unparse(n.test) == "a.catw":
            self.removed["if a.catw"] += 1
            return None
        return self.generic_visit(n)

    def visit_Call(self, n):
        if isinstance(n.func, ast.Name) and n.func.id == "Ranker":
            kw = [k for k in n.keywords if k.arg == "catw"]
            if kw:
                self.removed["Ranker(catw=)"] += len(kw)
                n.keywords = [k for k in n.keywords if k.arg != "catw"]
        return self.generic_visit(n)


class StripModel(ast.NodeTransformer):
    """Removes exactly the wave-9 statements of model.py (all inside class Ranker)."""
    EXPECT = collections.Counter({"def catgrad": 1, "__init__ catw=None": 1, "self.catw = catw": 1, "if catw": 1, "if self.catw": 1})

    def __init__(self):
        self.removed = collections.Counter()

    def visit_ClassDef(self, n):
        if n.name == "Ranker":
            keep = []
            for x in n.body:
                if isinstance(x, ast.FunctionDef) and x.name == "catgrad":
                    self.removed["def catgrad"] += 1
                    continue
                if isinstance(x, ast.FunctionDef) and x.name == "__init__" and x.args.args and x.args.args[-1].arg == "catw" \
                        and isinstance(x.args.defaults[-1], ast.Constant) and x.args.defaults[-1].value is None:
                    x.args.args.pop()
                    x.args.defaults.pop()
                    self.removed["__init__ catw=None"] += 1
                keep.append(x)
            n.body = keep
        return self.generic_visit(n)

    def visit_Assign(self, n):
        if ast.unparse(n) == "self.catw = catw":
            self.removed["self.catw = catw"] += 1
            return None
        return self.generic_visit(n)

    def visit_If(self, n):
        t = ast.unparse(n.test)
        if t in ("catw", "self.catw"):
            self.removed[f"if {t}"] += 1
            return None
        return self.generic_visit(n)


class NormStr(ast.NodeTransformer):
    def visit_Constant(self, n):
        return ast.Constant(value="S") if isinstance(n.value, str) else n


# ================================================================================================= CPU part
def c0_code(parse):
    P("\n== C0 code: lane DRY, AST flag-off proofs (train.py, model.py), module copies, defaults, call sites")
    rec = dict(line.split()[::-1] for line in open(f"{HERE}/w5_source_md5.txt"))
    for f in ("train.py", "model.py", "common.py", "lane.sh"):
        check(f"frozen reference tests/ref_w5/{f}.txt == nn_x/w5/{f} at copy time (md5 recorded)",
              md5(f"{REF}/{f}.txt") == rec[f"nn_x/w5/{f}"], md5(f"{REF}/{f}.txt"))
        P(f"  info: nn_x/w5/{f} now {'UNCHANGED' if md5(f'{W5}/{f}') == rec[f'nn_x/w5/{f}'] else 'CHANGED'} since the copy")
    # (a) lane
    rc9, c9 = dry_cmds(f"{W9}/lane.sh", 0, "A:base:23 A:catgrad:23 A:base:24 A:catgrad:24 E:catgrad:951 E:catgrad:952", r"w[59]_\w+")
    rc5, c5 = dry_cmds(f"{W5}/lane.sh", 0, "A:base:23 A:base:24 E:base:951 E:base:952", r"w5_\w+")
    check("lane.sh DRY: 6 w9 commands (4 stage-A, 2 stage-E) and 4 w5 reference commands", rc9 == 0 and rc5 == 0 and len(c9) == 6
          and len(c5) == 4, (sorted(c9), sorted(c5)))
    n9 = {k: vars(parse(v)) for k, v in c9.items()}
    n5 = {k: vars(parse(v)) for k, v in c5.items()}
    for s in (23, 24):
        d = diff_ns(n5[f"w5_base_s{s}"], n9[f"w9_base_s{s}"], skip=())
        check(f"A:base:{s}: w9_base_s{s} == nn_x/w5 lane A:base:{s} (A_BASE + HYG) except the variant name",
              d == {"variant": (f"w5_base_s{s}", f"w9_base_s{s}")} and n9[f"w9_base_s{s}"]["catw"] == "", d)
        d = diff_ns(n9[f"w9_base_s{s}"], n9[f"w9_catgrad_s{s}"])
        check(f"A:catgrad:{s} differs from A:base:{s} ONLY by catw '' -> 'grad'", d == {"catw": ("", "grad")}, d)
    for s in (951, 952):
        d = diff_ns(n5[f"w5_e10_base_s{s}"], n9[f"w5_e10_catgrad_s{s}"])
        check(f"E:catgrad:{s} == nn_x/w5 lane E:base:{s} (E_BASE + HYG) + catw grad; variant w5_e10_catgrad_s{s}",
              d == {"catw": ("", "grad")} and n9[f"w5_e10_catgrad_s{s}"]["variant"] == f"w5_e10_catgrad_s{s}", d)
    RES["lane_cmds"] = {k: " ".join(v) for k, v in c9.items()}
    refused = [subprocess.run(["bash", f"{W9}/lane.sh", str(g), "A:base:23"], capture_output=True, text=True,
                              env={**os.environ, "DRY": "1"}).returncode != 0 for g in range(2, 8)]
    check("lane.sh refuses some GPUs (also in DRY mode)", all(refused), refused)
    check("bash -n lane.sh", subprocess.run(["bash", "-n", f"{W9}/lane.sh"], capture_output=True).returncode == 0)
    sm = dry_cmds(f"{W9}/lane.sh", 1, "A:catgrad:23", r"w9_\w+", env={"W5_SMOKE": "1"})[1]
    check("W5_SMOKE=1 names the stage-A smoke variant w9_smoke_catgrad_s23 (20K feeds, 1 epoch, evals every 25)",
          list(sm) == ["w9_smoke_catgrad_s23"] and vars(parse(sm["w9_smoke_catgrad_s23"]))["train_feeds"] == 20000
          and vars(parse(sm["w9_smoke_catgrad_s23"]))["catw"] == "grad", list(sm))
    # (b) train.py AST proof
    new_src = open(f"{W9}/train.py").read()
    ref_src = open(f"{REF}/train.py.txt").read()
    assert new_src.count(StripTrain.ASSERT_NEW) == 1 and ref_src.count(StripTrain.ASSERT_OLD) == 1
    st = StripTrain()
    new_t = strip_doc(st.visit(ast.parse(new_src.replace(StripTrain.ASSERT_NEW, StripTrain.ASSERT_OLD))))
    ident = ast.dump(new_t) == ast.dump(strip_doc(ast.parse(ref_src)))
    check("AST: train.py minus the wave-9 statements (exact set) == frozen nn_x/w5/train.py (flag-off path unchanged)",
          ident and st.removed == StripTrain.EXPECT, dict(st.removed))
    # (c) model.py AST proof
    sm_ = StripModel()
    mt = strip_doc(sm_.visit(ast.parse(open(f"{W9}/model.py").read())))
    ident_m = ast.dump(mt) == ast.dump(strip_doc(ast.parse(open(f"{REF}/model.py.txt").read())))
    check("AST: model.py minus the wave-9 statements (exact set) == frozen nn_x/w5/model.py", ident_m and sm_.removed == StripModel.EXPECT,
          dict(sm_.removed))
    # (d) module copies
    eq_c = ast.dump(NormStr().visit(ast.parse(open(f"{W9}/common.py").read()))) == ast.dump(NormStr().visit(ast.parse(open(f"{REF}/common.py.txt").read())))
    check("common.py == nn_x/w5/common.py up to string constants (output directory nn_x/w9/catgrad)", eq_c)
    same = {f: md5(f"{W9}/{f}") == rec[f"nn_x/w5/{f}"] for f in MODULES}
    same["tests/mini_split.py"] = md5(f"{HERE}/mini_split.py") == rec["nn_x/w5/tests/mini_split.py"]
    check("data / cd_arms* / finetune / holdout_eval / stale_pred / blend / tests/mini_split byte-identical to nn_x/w5", all(same.values()),
          {k: v for k, v in same.items() if not v})
    # (e) defaults / guards
    a_def = parse(["--variant", "w9_x"])
    check("parse(): --catw OFF by default ('')", a_def.catw == "" and not bool(a_def.catw))
    check("guards `a.catw` False at the default; Ranker gets catw=None (a.catw or None)", not bool(a_def.catw) and (a_def.catw or None) is None)
    bad = subprocess.run([PY, "-c", "import sys; sys.argv=['t','--variant','w9_x','--catw','mean']; sys.path.insert(0, %r); "
                          "import ast, argparse\nsrc=open('%s/train.py').read()\nt=ast.parse(src)\nfn=[n for n in t.body if isinstance(n, ast.FunctionDef) "
                          "and n.name=='parse'][0]\nns={'argparse': argparse, 'POOL_P_DEFAULT': 0.5}\nexec(compile(ast.Module(body=[fn], type_ignores=[]), 'p', 'exec'), ns)\nns['parse']()" % (W9, W9)],
                         capture_output=True, text=True)
    check("parse(): --catw accepts only '' / 'grad' (argparse choices)", bad.returncode != 0 and "invalid choice" in bad.stderr, bad.stderr[-80:].strip())
    # (f) call sites
    mtree = ast.parse(open(f"{W9}/model.py").read())
    parents = {ch: p_ for p_ in ast.walk(mtree) for ch in ast.iter_child_nodes(p_)}

    def chain(n):
        c, x = [], n
        while x in parents:
            x = parents[x]
            if isinstance(x, ast.If):
                c.append("if " + ast.unparse(x.test))
            if isinstance(x, ast.FunctionDef):
                c.append("def " + x.name)
        return c
    calls = [chain(n) for n in ast.walk(mtree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "catgrad"]
    check("model.py: Ranker.catgrad called once, in forward, behind `if self.catw`", calls == [["if self.catw", "def forward"]], calls)
    cg_funcs = sorted({chain(n)[-1] for n in ast.walk(mtree) if isinstance(n, ast.Attribute) and n.attr.startswith("cg_")})
    check("model.py: cg_* attributes used only in __init__ and catgrad", cg_funcs == ["def __init__", "def catgrad"], cg_funcs)
    ttree = ast.parse(new_src)
    tp = {ch: p_ for p_ in ast.walk(ttree) for ch in ast.iter_child_nodes(p_)}
    uses = sorted({ast.unparse(tp[n]).split("\n")[0][:60] for n in ast.walk(ttree) if isinstance(n, ast.Attribute) and n.attr == "catw"})
    check("train.py: a.catw read only by its guard, the log line / asserts inside it and the Ranker keyword", len(uses) <= 6, uses)
    others = sorted(f for f in os.listdir(W9) if f.endswith(".py") and f not in ("train.py", "model.py") and "catw" in open(f"{W9}/{f}").read())
    check("no other module of the copy mentions catw (finetune / holdout_eval / stale_pred read the composite through Ranker(**cfg))",
          others == [], others)


def _groups(b):
    """(feed, category) membership of valid candidates: list over feeds of {cat: [m indices]}."""
    cm, cc = b["cmask"].cpu().numpy(), b["ccat"].cpu().numpy()
    out = []
    for i in range(cm.shape[0]):
        d = collections.defaultdict(list)
        for m_ in np.flatnonzero(cm[i]):
            d[int(cc[i, m_])].append(int(m_))
        out.append(d)
    return out


def brute_losses(g, h, y, groups, denom):
    """L_w, L_c (feed-normalised: sum over feeds / denom) from the category logits g [B,K] and item logits h [B,M] (torch)."""
    Lw = h.new_zeros(())
    Lc = h.new_zeros(())
    for i, d in enumerate(groups):
        ys = sum(float(y[i, m_]) for ms in d.values() for m_ in ms)
        if ys <= 0:
            continue
        cats = sorted(d)
        gl = torch.stack([g[i, k] for k in cats])
        logp_c = torch.log_softmax(gl, 0)
        lw = h.new_zeros(())
        lc = h.new_zeros(())
        for j, k in enumerate(cats):
            ms = d[k]
            hk = torch.stack([h[i, m_] for m_ in ms])
            lse = torch.logsumexp(hk, 0)
            yk = torch.stack([y[i, m_] for m_ in ms])
            lw = lw + (yk * (hk - lse)).sum()
            lc = lc + float(yk.sum()) * logp_c[j]
        Lw = Lw - lw / ys
        Lc = Lc - lc / ys
    return Lw / denom, Lc / denom


def c1_model():
    P("\n== C1 model: paired init, size-prior init, factorisation, loss identity, gradient routing (CPU, real MiniSplit batches)")
    import model as Mo
    from mini_split import MiniSplit, pick_train_feeds
    ck = torch.load(INIT, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    ms = MiniSplit("train", pick_train_feeds(12, seed=0), xt_names=cfg["xt_names"])
    P(f"  info: mini train split {ms.n} feeds, {ms.n_cand} candidates; architecture = {INIT} cfg (xt {len(cfg['xt_names'])}, n_dense {cfg['n_dense']})")
    torch.manual_seed(7)
    m0 = Mo.Ranker(**cfg)
    torch.manual_seed(7)
    m1 = Mo.Ranker(**{**cfg, "catw": "grad"})
    sd0, sd1 = m0.state_dict(), m1.state_dict()
    extra = sorted(set(sd1) - set(sd0))
    check("paired init: every baseline parameter bit-identical with / without --catw at the same seed; extra keys = cg_* only",
          set(sd0) <= set(sd1) and all(torch.equal(sd0[k], sd1[k]) for k in sd0) and all(k.startswith("cg_") for k in extra) and len(extra) == 11,
          extra)
    check("cfg: flag-off cfg == nn_x/w5 cfg (no catw key); flag-on cfg == cfg + {catw: 'grad'}", "catw" not in m0.cfg and m1.cfg == {**m0.cfg, "catw": "grad"})
    zero = all(float(sd1[k].abs().max()) == 0 for k in ("cg_q.3.weight", "cg_q.3.bias", "cg_wide.weight", "cg_wide.bias", "cg_cat.weight",
                                                        "cg_cnt.weight", "cg_cnt.bias"))
    check("zero-init of the head's output layers (cg_q.3, cg_wide, cg_cat, cg_cnt)", zero)
    for m_ in (m0, m1):
        m_.set_norm(ck["state_dict"]["f_mean"], ck["state_dict"]["f_std"])
        m_.load_state_dict(ck["state_dict"], strict=m_ is m0)
        m_.eval()
    rng = np.random.default_rng(0)
    bs = ms.batches(np.arange(ms.n), 3072, 6, shuffle=True, rng=rng)
    denom = 512.0
    n_single = n_pad = n_multi = 0
    worst = dict(fac=0.0, h=0.0, g0=0.0, loss=0.0, g_trunk=0.0, g_head=0.0, g_cross=0.0, gsum=0.0)
    nb = 0
    for pert in (False, True):
        if pert:  # non-trivial category head: random weights in the output layers (g no longer the size prior)
            torch.manual_seed(11)
            with torch.no_grad():
                m1.cg_q[3].weight.normal_(0, 0.3)
                m1.cg_wide.weight.normal_(0, 0.05)
                m1.cg_cat.weight.normal_(0, 0.5)
                m1.cg_cnt.weight.fill_(0.7)
        for fb in bs[:8]:
            b = ms.gather(fb, with_pos=True)
            y, cm = b["y"], b["cmask"]
            groups = _groups(b)
            n_single += sum(len(ms_) == 1 for d in groups for ms_ in d.values())
            n_pad += int((~cm).sum())
            n_multi += int((y.sum(1) > 1).sum())
            l0 = m0(b).detach()  # grad-enabled like m1 (same CPU kernel paths; no_grad may pick another SDPA kernel)
            X = m1.dense(b)
            l1 = m1(b, X)
            g, h = m1.cg_g, m1.cg_h
            O = torch.nn.functional.one_hot(b["ccat"], Mo.N_CAT).bool() & cm[..., None]
            n = O.sum(1).float()
            pres = n > 0
            if not pert:
                worst["h"] = max(worst["h"], float((h - l0).abs().max()))
                worst["g0"] = max(worst["g0"], float((g[pres] - n[pres].log()).abs().max()))
            Pf = torch.softmax(l1.masked_fill(~cm, -1e4), 1)
            Pc = torch.softmax(g.masked_fill(~pres, -1e4), 1)
            Pw = torch.softmax(torch.where(O, h[..., None], torch.tensor(-1e4)), 1).gather(2, b["ccat"][..., None]).squeeze(-1)
            worst["fac"] = max(worst["fac"], float((Pf - Pc.gather(1, b["ccat"]) * Pw)[cm].abs().max()))
            loss = Mo.listwise_loss(l1, y, cm, norm="feed", denom=denom)
            Lw, Lc = brute_losses(g, h, y, groups, denom)
            worst["loss"] = max(worst["loss"], float(abs(loss - (Lw + Lc))) / max(1.0, float(loss)))
            trunk = [p for nm, p in m1.named_parameters() if not nm.startswith("cg_")]
            head = [p for nm, p in m1.named_parameters() if nm.startswith("cg_")]
            gc_t, gc_h, gc_hh = torch.autograd.grad(loss, trunk + head + [h], retain_graph=True, allow_unused=True)[:len(trunk)], \
                torch.autograd.grad(loss, head, retain_graph=True, allow_unused=True), torch.autograd.grad(loss, [h], retain_graph=True)[0]
            gw_t = torch.autograd.grad(Lw, trunk, retain_graph=True, allow_unused=True)
            gw_h = torch.autograd.grad(Lw, head, retain_graph=True, allow_unused=True)
            gl_t = torch.autograd.grad(Lc, trunk, retain_graph=True, allow_unused=True)
            gl_h = torch.autograd.grad(Lc, head, retain_graph=True, allow_unused=True)
            z = lambda t: torch.zeros(()) if t is None else t  # noqa: E731
            rel = lambda a_, b_: float(max((z(x) - z(y_)).abs().max() / max(1.0, float(z(x).abs().max())) for x, y_ in zip(a_, b_)))  # noqa: E731
            worst["g_trunk"] = max(worst["g_trunk"], rel(gc_t, gw_t))
            worst["g_head"] = max(worst["g_head"], rel(gc_h, gl_h))
            worst["g_cross"] = max(worst["g_cross"], max(float(z(x).abs().max()) for x in list(gl_t) + list(gw_h)))
            gs = (gc_hh[..., None] * O).sum(1)  # [B,K] sum over the category's candidates of dL/dh
            worst["gsum"] = max(worst["gsum"], float(gs.abs().max()))
            nb += 1
    P(f"  info: {nb} batches (8 unperturbed + 8 perturbed head); singleton (feed, category) groups {n_single}, padded cells {n_pad}, "
      f"multi-positive feeds {n_multi}; worst {worst}")
    check("batches cover singleton categories, padding and multi-positive feeds", n_single > 0 and n_pad > 0 and n_multi > 0)
    check("flag ON at init: h == the flag-off logit (bitwise) and g_k == log n_k (size prior) for every present category",
          worst["h"] == 0.0 and worst["g0"] < 1e-6, (worst["h"], worst["g0"]))
    check("exact factorisation: softmax(composite)_i == softmax_k(g)_{k(i)} * softmax_{within k(i)}(h)_i (<= 1e-5, fp32)", worst["fac"] < 1e-5, worst["fac"])
    check("loss identity: listwise_loss(composite, norm feed) == L_c + L_w (brute force, <= 1e-5 rel)", worst["loss"] < 1e-5, worst["loss"])
    check("gradient routing: d CE(composite)/d trunk == d L_w/d trunk and d/d head == d L_c/d head (<= 1e-5 rel)",
          worst["g_trunk"] < 1e-5 and worst["g_head"] < 1e-5, (worst["g_trunk"], worst["g_head"]))
    check("no cross gradient: L_c -> trunk and L_w -> head are exactly zero / unused", worst["g_cross"] == 0.0, worst["g_cross"])
    check("for every (feed, category): sum_j dL/dh_j == 0 (the trunk receives no category-choice gradient; <= 1e-6)", worst["gsum"] < 1e-6, worst["gsum"])
    # train mode
    m1.train()
    b = ms.gather(bs[0], with_pos=True)
    X = m1.dense(b)
    lg = m1(b, X, use_pos=True)
    loss = Mo.listwise_loss(lg, b["y"], b["cmask"], norm="feed", denom=denom)
    tgt = torch.softmax((-b["pos"].float() / 10.0).masked_fill(~b["cmask"], -1e4), 1) * b["cmask"]
    lp = torch.log_softmax(m1.aux_out.masked_fill(~b["cmask"], -1e4), 1)
    loss = loss + 0.5 * (-(tgt * lp).sum(1)).mean()
    loss.backward()
    fin = all(torch.isfinite(p.grad).all() for p in m1.parameters() if p.grad is not None)
    ng = sum(p.grad is not None and float(p.grad.abs().max()) > 0 for nm, p in m1.named_parameters() if nm.startswith("cg_"))
    check("train mode (dropout, K9 aux): forward / backward finite; the category head receives gradient", bool(torch.isfinite(lg).all()) and fin and ng >= 5,
          f"non-zero cg_ grads {ng}")
    m1.eval()
    m1.zero_grad(set_to_none=True)


def outputs(base, v):
    fs = [f"{base}/logs/{v}.log", f"{base}/logs/{v}.json", f"{base}/logs/{v}_cliplog.npz", f"{base}/logs/{v}.stdout",
          f"{base}/logs/{v}_holdout.stdout"]
    for x in (v, f"{v}@last"):
        fs += [f"{base}/ckpt/{x}.pt", f"{base}/preds/nn_{x}_trainonly_val.npy", f"{base}/preds/nn_{x}_trainonly_test.npy",
               f"{base}/preds/holdout/nn_{x}_holdout.npy"]
    return fs


def clean(base, v):
    for f in outputs(base, v):
        if os.path.exists(f):
            os.remove(f)
    d = f"{base}/logs/queue/claim_{v}"
    if os.path.isdir(d):
        os.rmdir(d)


def e2e_run(code_dir, v, extra):
    rec = f"{SCR}/rec_{v}.npz"
    cmd = [PY, f"{HERE}/e2e_cpu.py", code_dir, rec, "--variant", v] + E2E.split() + extra
    t0 = time.time()
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": ""}  # 1 thread: bitwise-reproducible CPU runs
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=code_dir, env=env)
    open(f"{SCR}/{v}.stdout", "w").write(r.stdout + "\n" + r.stderr)
    P(f"  ran {v} from {code_dir} ({time.time() - t0:.0f}s, rc {r.returncode})")
    assert r.returncode == 0, r.stderr[-3000:]
    return dict(np.load(rec, allow_pickle=True))


def metric_lines(log):
    return [re.sub(r" [\d.]+ min", "", ln) for ln in log.splitlines() if ln.startswith("step ") or ln.startswith("BEST")]


def batches_of(rec, key, shapes="shapes"):
    sh, out, k = rec[shapes], [], 0
    for b_, m_ in sh:
        out.append(rec[key][k:k + b_ * m_].reshape(b_, m_))
        k += b_ * m_
    return out


def ck_equal(a, b):
    ca = torch.load(a, map_location="cpu", weights_only=False)["state_dict"]
    cb = torch.load(b, map_location="cpu", weights_only=False)["state_dict"]
    return ca.keys() == cb.keys() and all(torch.equal(ca[k], cb[k]) for k in ca)


def code_copy(src, dst, out, line):
    """Copy the .py modules of `src` to `dst` with common.py's output path redirected to `out` (string change only; same AST
    shape). Returns True iff every other module is byte-identical to `src` and common.py differs only by string constants."""
    shutil.rmtree(dst, ignore_errors=True)
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(dst)
    for f in os.listdir(src):
        if f.endswith(".py"):
            shutil.copy(f"{src}/{f}", f"{dst}/{f}")
    c = open(f"{dst}/common.py").read()
    assert c.count(line) == 1, line
    open(f"{dst}/common.py", "w").write(c.replace(line, f'W5 = ROOT / "{out}" / "."'))
    same = all(md5(f"{dst}/{f}") == md5(f"{src}/{f}") for f in os.listdir(dst) if f != "common.py")
    eq = ast.dump(NormStr().visit(ast.parse(open(f"{dst}/common.py").read()))) == ast.dump(NormStr().visit(ast.parse(open(f"{src}/common.py").read())))
    return same and eq


def c2_e2e():
    P("\n== C2 CPU end-to-end (e2e_cpu.py, miniature split): flags OFF vs frozen nn_x/w5 code; --catw grad composite identity")
    ref_dir, ref_out = f"{SCR}/ref_w5_code", f"{SCR}/ref_w5_out"
    new_dir, new_out = f"{SCR}/new_w9_code", f"{SCR}/new_w9_out"
    ok_r = code_copy(W5, ref_dir, ref_out, 'W5 = ROOT / "nn_x" / "w5"')
    ok_n = code_copy(W9, new_dir, new_out, 'W5 = ROOT / "nn_x" / "w9/catgrad"')
    check("code copies for the e2e (outputs redirected to scratch so the real preds/holdout meta is never touched): modules byte-identical to "
          "nn_x/w5 / nn_x/w9/catgrad, common.py differs only in the output-path string", ok_r and ok_n, (ok_r, ok_n))
    names = ["w5_t9e2e_ref_off", "w5_t9e2e_new_off", "w5_t9e2e_new_on"]
    ra = e2e_run(ref_dir, names[0], [])
    rb = e2e_run(new_dir, names[1], [])
    rc_ = e2e_run(new_dir, names[2], ["--catw", "grad"])
    la, lb, lc = (open(f"{ref_out}/logs/{names[0]}.log").read(), open(f"{new_out}/logs/{names[1]}.log").read(), open(f"{new_out}/logs/{names[2]}.log").read())
    nb = len(ra["off"]) - 1
    check("flags OFF vs frozen w5: identical training-batch sequence (incl. cooldown pool draws)",
          np.array_equal(ra["fidx"], rb["fidx"]) and np.array_equal(ra["off"], rb["off"]) and np.array_equal(ra["ci"], rb["ci"]) and nb >= 20,
          f"{nb} training batches, {int(rb['n_eval'])} holdout evals, {int(rb['n_predict'])} predicts")
    check("flags OFF vs frozen w5: identical train-mode model logits and loss-input logits (bitwise)",
          np.array_equal(ra["lg_model"], rb["lg_model"]) and np.array_equal(ra["lg_loss"], rb["lg_loss"]) and np.array_equal(rb["lg_loss"], rb["lg_model"])
          and int(rb["n_loss"]) == int(rb["n_model"]) == nb)
    ma, mb = metric_lines(la), metric_lines(lb)
    check("flags OFF vs frozen w5: identical logged loss / val / holdout / BEST lines", ma == mb and len(ma) >= 3, mb[-1] if mb else "")
    same_files = (ck_equal(f"{ref_out}/ckpt/{names[0]}.pt", f"{new_out}/ckpt/{names[1]}.pt") and ck_equal(f"{ref_out}/ckpt/{names[0]}@last.pt", f"{new_out}/ckpt/{names[1]}@last.pt")
                  and all(np.array_equal(np.load(f"{ref_out}/{d_}nn_{names[0]}{s_}"), np.load(f"{new_out}/{d_}nn_{names[1]}{s_}"))
                          for d_, s_ in (("preds/", "_trainonly_val.npy"), ("preds/", "@last_trainonly_val.npy"), ("preds/holdout/", "_holdout.npy"),
                                         ("preds/holdout/", "@last_holdout.npy"))))
    check("flags OFF vs frozen w5: bit-identical selected + @last checkpoints, val preds and holdout preds", same_files)
    ja, jb = json.load(open(f"{ref_out}/logs/{names[0]}.json")), json.load(open(f"{new_out}/logs/{names[1]}.json"))
    d_args = {k: (ja["args"].get(k), jb["args"].get(k)) for k in set(ja["args"]) | set(jb["args"]) if ja["args"].get(k) != jb["args"].get(k)}
    check("flags OFF: stored args == frozen w5 args + {catw: ''}; same best_step / val_ndcg; no wave-9 log line",
          d_args == {"variant": (names[0], names[1]), "catw": (None, "")} and ja["best_step"] == jb["best_step"] and ja["val_ndcg"] == jb["val_ndcg"]
          and "wave-9" not in la and "wave-9" not in lb, d_args)
    # flag ON
    check("--catw grad: same training-batch sequence as flags OFF; one loss-input logit set per train forward",
          np.array_equal(rc_["fidx"], rb["fidx"]) and np.array_equal(rc_["ci"], rb["ci"]) and int(rc_["n_loss"]) == int(rc_["n_model"]) == nb
          and len(rc_["g_shapes"]) == nb)
    lgm, lgl, G, H, CC, CM = (batches_of(rc_, "lg_model"), batches_of(rc_, "lg_loss"), batches_of(rc_, "cg_g", "g_shapes"), batches_of(rc_, "cg_h"),
                              batches_of(rc_, "ccat"), batches_of(rc_, "cmask"))
    lgb = batches_of(rb, "lg_model")
    worst, n_cells, diff_off = 0.0, 0, 0
    for i in range(nb):
        g, h, cc, cm = G[i].astype(np.float64), H[i].astype(np.float64), CC[i], CM[i]
        comp = np.zeros_like(h)
        for r_ in range(h.shape[0]):
            for k in np.unique(cc[r_][cm[r_]]):
                sel = cm[r_] & (cc[r_] == k)
                hk = h[r_, sel]
                lse = np.log(np.exp(hk - hk.max()).sum()) + hk.max()
                comp[r_, sel] = g[r_, k] + hk - lse
        worst = max(worst, float(np.abs((lgl[i] - comp)[cm]).max()))
        n_cells += int(cm.sum())
        diff_off += int(not np.array_equal(lgm[i], lgb[i]))
    check("--catw grad: for EVERY training batch the logits into the loss == g[cat] + h - LSE_cat(h) recomputed from the recorded g / h "
          "(<= 1e-4 on every valid candidate) and == the model output", worst < 1e-4 and all(np.array_equal(lgl[i], lgm[i]) for i in range(nb)),
          f"max |diff| {worst:.2e} over {n_cells} cells")
    check("--catw grad changes training: train-mode logits differ from flags OFF on every batch; checkpoint differs",
          diff_off == nb and not ck_equal(f"{new_out}/ckpt/{names[2]}.pt", f"{new_out}/ckpt/{names[1]}.pt"), f"{diff_off}/{nb} batches differ")
    ck = torch.load(f"{new_out}/ckpt/{names[2]}.pt", map_location="cpu", weights_only=False)
    jc = json.load(open(f"{new_out}/logs/{names[2]}.json"))
    check("--catw grad: checkpoint cfg has catw 'grad' (otherwise == the flags-off cfg), 11 cg_* parameters, args catw 'grad', wave-9 log line, "
          "@last / holdout files written",
          ck["cfg"].get("catw") == "grad" and {k: v for k, v in ck["cfg"].items() if k != "catw"} == torch.load(f"{new_out}/ckpt/{names[1]}.pt", map_location="cpu", weights_only=False)["cfg"]
          and sum(k.startswith("cg_") for k in ck["state_dict"]) == 11 and jc["args"]["catw"] == "grad" and "wave-9 CATGRAD: --catw grad" in lc
          and all(os.path.exists(f"{new_out}/{f}") for f in (f"ckpt/{names[2]}@last.pt", f"preds/nn_{names[2]}@last_trainonly_val.npy",
                                                             f"preds/holdout/nn_{names[2]}_holdout.npy", f"preds/holdout/nn_{names[2]}@last_holdout.npy")),
          f"best_step {jc['best_step']} val {jc['val_ndcg']:.5f}")
    # reload through Ranker(**cfg) (the finetune.py / holdout_eval.py / stale_pred.py path) and re-predict the mini val
    r = subprocess.run([PY, "-c", f"""
import os, sys, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, {HERE!r}); sys.path.insert(1, {W9!r})
import numpy as np, torch
torch.set_num_threads({NT})
from mini_split import MiniSplit
import model as Mo
ck = torch.load({new_out!r} + '/ckpt/{names[2]}.pt', map_location='cpu', weights_only=False)
m = Mo.Ranker(**ck['cfg']); m.load_state_dict(ck['state_dict']); m.eval()
import polars as pl
f = pl.read_parquet('<repo>/data/flat/val/feeds.parquet', columns=['M', 'H'])
M, H = f['M'].to_numpy(), f['H'].to_numpy()
rng = np.random.default_rng(1)
groups = [np.where(M < 50)[0], np.where((M >= 50) & (M <= 200))[0], np.where(M > 200)[0], np.where(H == 0)[0]]
feeds = np.unique(np.concatenate([rng.choice(g_, min(10, len(g_)), replace=False) for g_ in groups]))
va = MiniSplit('val', feeds, xt_names=ck['cfg']['xt_names'])
out = torch.zeros(va.n_cand)
with torch.no_grad():
    for fb in va.batches(np.arange(va.n), 65536, 1024, shuffle=False):
        b = va.gather(fb); va.scatter(out, b, m(b, use_pos=False))
saved = np.load({new_out!r} + '/preds/nn_{names[2]}_trainonly_val.npy')
print('RELOAD', out.numpy().shape == saved.shape, float(np.abs(out.numpy() - saved).max()))
"""], capture_output=True, text=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": str(NT)})
    m_ = re.search(r"RELOAD (True|False) ([\d.e+-]+)", r.stdout)
    check("--catw grad: Ranker(**cfg) + load_state_dict reload (the finetune / holdout_eval / stale_pred path) reproduces the saved val preds (<= 1e-5)",
          r.returncode == 0 and m_ is not None and m_.group(1) == "True" and float(m_.group(2)) < 1e-5, (r.stdout.strip()[-80:] or r.stderr[-300:]))
    RES["e2e"] = dict(off_lines=mb, on_lines=metric_lines(lc), on_wave9=[ln for ln in lc.splitlines() if "wave-9" in ln])
    for d_ in (ref_dir, ref_out, new_dir, new_out):
        shutil.rmtree(d_, ignore_errors=True)


# ================================================================================================= GPU part
def g1_smoke():
    gpu = _CVD
    v = "w9_smoke_catgrad_s23"
    P(f"\n== G1 GPU plumbing smoke: W5_SMOKE=1 lane.sh {gpu} A:catgrad:23 -> {v} (20K train feeds, 1 epoch, evals every 25 steps)")
    clean(W9, v)
    t0 = time.time()
    r = subprocess.run(["bash", f"{W9}/lane.sh", gpu, "A:catgrad:23"], capture_output=True, text=True, cwd=W9,
                       env={**os.environ, "W5_SMOKE": "1", "OMP_NUM_THREADS": str(NT), "WAIT_MAX_H": "1"})
    P(f"  lane rc {r.returncode} ({time.time() - t0:.0f}s); tail: " + " | ".join(r.stdout.strip().splitlines()[-3:]))
    ok_files = all(os.path.exists(f"{W9}/{f}") for f in (f"logs/{v}.json", f"logs/{v}.log", f"ckpt/{v}.pt", f"ckpt/{v}@last.pt",
                                                        f"preds/nn_{v}_trainonly_val.npy", f"preds/nn_{v}@last_trainonly_val.npy",
                                                        f"preds/holdout/nn_{v}_holdout.npy", f"preds/holdout/nn_{v}@last_holdout.npy"))
    check("smoke run completed through the lane; json / log / selected + @last checkpoints, val and holdout preds written", r.returncode == 0 and ok_files)
    if ok_files:
        j = json.load(open(f"{W9}/logs/{v}.json"))
        pv = np.load(f"{W9}/preds/nn_{v}_trainonly_val.npy")
        ph = np.load(f"{W9}/preds/holdout/nn_{v}_holdout.npy")
        ck = torch.load(f"{W9}/ckpt/{v}.pt", map_location="cpu", weights_only=False)
        lg = open(f"{W9}/logs/{v}.log").read()
        hist = j["history"]
        check("smoke: val preds finite with within-feed variation, holdout preds finite, val NDCG > 0.30 after 1 mini epoch, history has holdout evals",
              np.isfinite(pv).all() and np.isfinite(ph).all() and pv.std() > 0 and j["val_ndcg"] > 0.30 and len(hist) >= 2
              and all(np.isfinite(x["holdout_ndcg"]) for x in hist), f"val {j['val_ndcg']:.5f} best_step {j['best_step']} minutes {j['train_minutes']:.1f} evals {len(hist)}")
        check("smoke: checkpoint cfg catw 'grad', args catw 'grad', 11 cg_* parameters (non-zero after training), wave-9 log line",
              ck["cfg"].get("catw") == "grad" and j["args"]["catw"] == "grad" and sum(k.startswith("cg_") for k in ck["state_dict"]) == 11
              and float(ck["state_dict"]["cg_q.3.weight"].abs().max()) > 0 and "wave-9 CATGRAD: --catw grad" in lg)
        RES["smoke"] = dict(val=j["val_ndcg"], best_step=j["best_step"], minutes=j["train_minutes"], lines=metric_lines(lg)[-3:])
        P(f"  info: cg_cnt weight {float(ck['state_dict']['cg_cnt.weight']):.4f} bias {float(ck['state_dict']['cg_cnt.bias']):.4f}; "
          f"|cg_cat| max {float(ck['state_dict']['cg_cat.weight'].abs().max()):.4f}")
    clean(W9, v)


def main():
    t_all = time.time()
    P(f"== nn_x/w9/catgrad WAVE-9 CATGRAD tests, part {PART} ({time.ctime()}, CPU threads {NT})")
    if PART == "cpu":
        c0_code(load_parse(f"{W9}/train.py"))
        c1_model()
        c2_e2e()
    else:
        g1_smoke()
    RES["failed"] = FAIL
    P(f"\n== {len(RES['checks']) - len(FAIL)}/{len(RES['checks'])} checks passed; failed: {FAIL}; {time.time() - t_all:.0f}s")
    json.dump(RES, open(f"{W9}/logs/test_w9_catgrad_{PART}.json", "w"), indent=1, default=str)
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())
