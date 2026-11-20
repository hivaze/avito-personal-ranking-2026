"""Unit + end-to-end tests of the nn_x/abl ABLATION-SCREEN harness (ledger nn_x/abl/ledger/ledger.md section 4):
train.py --xt_exclude / --xt_rel_extra (helpers abl_arms.py), abl_lane.sh / launch_abl.sh / wait_abl_readout.sh, the
flags-off bit-identity of nn_x/abl with nn_x/w5. Written 2026-11-13 ~00:15 UTC.

  cpu part:  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 .venv/bin/python nn_x/abl/tests/test_abl.py cpu
  gpu part:  CUDA_VISIBLE_DEVICES=<5|7> OMP_NUM_THREADS=12 .venv/bin/python nn_x/abl/tests/test_abl.py gpu   (ONE GPU,
             sequential subprocesses, 12 CPU threads)

CPU part
  C0 code. (a) abl_lane.sh DRY commands parse with train.py's own parser: abl_base_s<seed> == nn_x/w5 lane.sh A:base:<seed>
     (A_BASE + HYG) except the variant name, the two abl flags at their defaults; every arm differs from the base ONLY by
     its declared flag change (removals included); abl_lane.sh refuses GPUs N <ids>; scripts pass bash -n; launch
     DRY lists the 26 jobs and 6 lanes. (b) AST proof: train.py minus the abl statements (the abl_arms import, the two
     add_argument lines, the three `if a.xt_exclude ...` guards) with the variant-prefix assert swapped == the frozen
     nn_x/w5/train.py (tests/ref_train_w5.py.txt, md5 recorded at copy time in tests/w5_source_md5.txt). (c) both flags
     default to ''; every guard is False at the defaults and for the base command. (d) call sites: xt_exclude_names /
     xt_rel_extend each called once, behind their guard, in the no-init branch of main(); no other module imports
     abl_arms; common / model / data contain no abl code. (e) module copies: model / data(AST) / cd_arms* / holdout_eval /
     blend / stale_pred / tests/mini_split byte-identical to nn_x/w5 (md5 recorded), common.py AST-equal up to strings.
  C1 helpers on the real preset cvnd_pre2s (34 columns): xdrop_pop -> 26 names, xt_rel = the 3 pre2 columns only,
     xt_idcols None, n_dense = N_DENSE + 2 + 26 + 6 - 3; xdrop_ctxnd -> 24 names, xt_rel 5, idcols 8; xrel_more ->
     xt_rel 9 (XT_REL's 5 + the 4 extras, in order), n_dense + 8; error cases (unknown / duplicate / overlap / empty).
  C2 dense features (MiniSplit, real train feeds, CPU): dense_features with the xdrop_pop / xdrop_ctxnd inputs == the
     full-preset features with the dropped raw columns and their rank/z channels removed (exact); with xrel_more == the
     full features + [pct-rank, z] of the 4 extra columns (exact, == _pctrank / _zscore of the raw column).
  C3 every arm's model builds on the CPU from its parsed command and runs one train-mode forward on a real MiniSplit
     batch (no_set, blocks 3, retr 32 / uniq / tcnt, no retrieval, no sim_bias, no domain, no aux head).
GPU part
  G1 end-to-end CD (deterministic wrapper tests/e2e_abl.py; the w5 pattern: --init v17_pre2s_s101@77500, 39 cooldown
     steps, cooldown_mix 0.2, evals at 20 / 39): frozen w5 train.py vs abl train.py at flags off: identical batch
     sequence, train-mode logits, loss-input logits, logged lines, bit-identical checkpoint and val preds; abl flags
     with --init are refused before any data loads.
  G2 end-to-end FROM SCRATCH (the xt code path; smoke size: A_BASE + HYG, --train_feeds 20000 --epochs 1 --eval_every
     25 --select_min_step 25 --cliplog 10): frozen w5 vs abl at flags off: identical batches, logits, logged lines,
     bit-identical selected + @last checkpoints, val and holdout preds. Arms ON: xdrop_pop (cfg xt_names / xt_rel /
     xt_idcols / xt_relonly / n_dense as C1; a different checkpoint; stale_pred.py reproduces the inputs from the cfg)
     and xrel_more (cfg xt_rel 9, n_dense + 8; stale_pred.py runs).
     Test runs are named w5_ablt_* / abl_t_*; all their outputs (under nn_x/abl only) are deleted afterwards.
Second wave (2026-11-13 ~04:00 UTC; ABLATION_PLAN_1113 section 3.3; arms retr_cat / xdrop_ctxhn_hs, helper abl_retr.py):
  C0 extended: 15 arms x 2 seeds = 30 lane commands; the AST strip removes exactly 2 imports (abl_arms, abl_retr),
     3 add_argument (--xt_exclude, --xt_rel_extra, --retr_cat) and 4 guards; abl_lane.sh accepts GPU N, refuses <ids>;
     setup_retr_cat called once behind `if a.retr_cat` in the no-init branch; abl_retr imported only by train.py (and the
     replay-only f_replay.py, never on the training path).
  C4 retr_cat on real MiniSplit batches (CPU): the arm adds only retr_cat.* parameters and every baseline parameter keeps
     its init; flag-on at init is BIT-IDENTICAL to flag-off in eval mode AND in train mode (logits + aux head; the module's
     dropout uses a dedicated generator, so the global RNG stream is untouched); the module's output is exactly 0 at init;
     the selection keeps exactly the same-category valid items (every valid slot is same-category, #valid = min(m,
     #same-category items), 0 for cmiss); with a random Z the output is bitwise invariant to the embeddings / types /
     sources of other-category events, exactly 0 for candidates without a same-category item, and equals (allclose) the
     parent HistRetrieval run per candidate with the history restricted to its same-category items; the checkpoint cfg
     carries retr_cat 16, Ranker(**cfg) refuses it, abl_retr.load_ranker reproduces the logits; Muon picks the same four
     matrices as for the base retrieval.
  G2 extended: arm abl_t_fs_rcat (--retr_cat 16): same batch sequence as flags off, FIRST-batch train-mode logits and
     loss-input logits bit-identical to flags off (exact no-op at init), later ones differ (Z learns), cfg retr_cat 16 and
     retr_cat.* keys in the checkpoint, setup log line present.
pop1 arm (2026-11-13 ~06:10 UTC; ledger section 7; train.py --xt_add, helper abl_arms.xt_add_names; feature xp_imp_c50):
  C0 extended: 17 arms x 2 seeds = 34 lane commands, pop1 == base + --xt_exclude <the 8 popularity columns of xdrop_pop>
     + --xt_add xp_imp_c50 and nothing else; the AST strip removes exactly 3 imports (abl_arms x 2, abl_retr), 4 add_argument
     (+ --xt_add) and 5 guards (the --init guard now `... or a.xt_add`, + `if a.xt_add`); --xt_add defaults to '' and its
     guard is False at the defaults / for the base command; xt_add_names called once behind `if a.xt_add` in the no-init
     branch, placed right after the `if a.xt_exclude` statement and right before the `if xt_names:` miss check.
  C5 (CPU) xt_add: flag-off identity (layout with add '' == without); pop1 layout = xdrop_pop layout + xp_imp_c50 appended
     LAST, n_dense + 1, xt_rel / xt_idcols / xt_relonly indices unchanged (also on the full preset); error cases (no preset,
     duplicate, already present, seq_cv / it_hist prefix, XT_REL / XT_RELONLY reserved name); dense_features on real
     MiniSplit batches == the xdrop_pop features with exactly one column inserted at the end of the raw xt block, equal to
     the attach_xt transform of xp_imp_c50 (fp16); xp_imp_c50 is a hard link of data/feats in nn/feat_cache for all 3 splits;
     checkpoint cfg round trip (torch.save / load -> Ranker(**cfg) -> bit-identical logits; cfg carries the pop1 layout).
  G1 extended: --xt_add with --init refused. G2 extended: arm abl_t_fs_pop1 ON (cfg == the C5 layout, same batch sequence,
     different checkpoint, setup log line, holdout preds written, stale_pred.py runs on it from the cfg).
Writes nn_x/abl/logs/test_abl_<part>.log (+ .json).
"""
import argparse
import ast
import collections
import copy
import hashlib
import json
import os
import re
import subprocess
import sys
import time

PART = sys.argv[1] if len(sys.argv) > 1 else "cpu"
assert PART in ("cpu", "gpu"), "usage: test_abl.py cpu|gpu"
_CVD = os.environ.get("CUDA_VISIBLE_DEVICES", "x")
if PART == "cpu":
    assert _CVD == "", "cpu part: run with CUDA_VISIBLE_DEVICES=''"
else:
    assert _CVD in ("5", "7", "3"), "gpu part: CUDA_VISIBLE_DEVICES=<one of the ablation GPUs N 7 3>"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
NT = min(12, int(os.environ.get("OMP_NUM_THREADS", "12")))
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(NT)
HERE = os.path.dirname(os.path.abspath(__file__))
ABL = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(1, ABL)
import warnings  # noqa: E402

warnings.filterwarnings("ignore", message=".*CUDA is not available.*")
warnings.filterwarnings("ignore", message=".*device_type of 'cuda'.*")
import abl_arms as AA  # noqa: E402
import abl_retr as AR  # noqa: E402
import cd_arms as CA  # noqa: E402
import model as MO  # noqa: E402
from mini_split import MiniSplit, pick_train_feeds  # noqa: E402

ROOT = "<repo>"
W5 = f"{ROOT}/nn_x/w5"
PY = f"{ROOT}/.venv/bin/python"
INIT = f"{ROOT}/nn/ckpt/v17_pre2s_s101@77500.pt"
TRAIN_PY = f"{ABL}/train.py"
REF_TRAIN = f"{HERE}/ref_train_w5.py.txt"
LANE, LAUNCH, WAITER = f"{ABL}/abl_lane.sh", f"{ABL}/launch_abl.sh", f"{ABL}/wait_abl_readout.sh"
ABL_FLAGS = ("--xt_exclude", "--xt_rel_extra", "--retr_cat", "--xt_add", "--xt_add_relonly", "--kv64")  # + ENCODER-DOSE pre2noid / kv64 (ledger section 10)
ABL_MODULES = ("abl_arms", "abl_retr", "encdose.kv_probe.kv_adapter", "encdose.kv_probe.kv_data")
OFF = dict(xt_exclude="", xt_rel_extra="", retr_cat=0, xt_add="", xt_add_relonly="", kv64=0)
XADD = "xp_imp_c50"  # pop1 exposure input (ledger section 7)
POP = "seq_cv_invd20,seq_cv_pop,seq_cv_norm,it_hist_all_luo,it_hist_et1_luo,it_hist_et2_luo,it_hist_et3_luo,it_hist_et4_luo"
CTXND = ("ctx_nd_hcent,ctx_nd_knn_f1,ctx_nd_knn_h,ctx_nd_knn_h_dev,ctx_nd_knn_u,ctx_nd_knn_u_dev,ctx_nd_max,ctx_nd_top5h,"
         "ctx_nd_top5u,ctx_nd_ucent")
XREL = "seq_cv_pop,it_hist_et1_luo,ctx_nd_knn_h,ctx_nd_ucent"
CTXHNHS = "ctx_hn_top1w,ctx_hn_rrw,ctx_hn_bestrk,ctx_hn_margin,ctx_hn_soft05,ctx_hn_e1_rrw,ctx_hs_mutual,ctx_hs_k128,ctx_hs_e1,ctx_hs_t10"
PRE2 = "pre2_fe_z,pre2_new_z,pre2_fe_s,pre2_new_s,pre2_hmax32,has:pre2_fe"
PRE2N = "pre2n_fe_z,pre2n_new_z,pre2n_fe_s,pre2n_new_s,pre2n_hmax32,has:pre2n_fe"  # ENCODER-DOSE pre2noid (ledger section 10)
PRE2N_RO = "pre2n_fe_s,pre2n_new_s,pre2n_hmax32"
ARM_DIFF = {  # arm -> exact namespace diff vs the base (base value, arm value)
    "no_aux": {"aux_pos": (0.5, 0.0)},
    "no_retr": {"retr_m": (16, 0), "attn_gate": (True, False)},
    "retr32": {"retr_m": (16, 32)},
    "retr32u": {"retr_m": (16, 32), "retr_uniq": (False, True), "retr_tcnt": (False, True)},
    "xdrop_pop": {"xt_exclude": ("", POP)},
    "xdrop_ctxnd": {"xt_exclude": ("", CTXND)},
    "no_sim": {"sim_bias": (1, 0)},
    "no_domain": {"domain": (True, False)},
    "xrel_more": {"xt_rel_extra": ("", XREL)},
    "lnorm_batch": {"loss_norm": ("feed", "batch")},
    "blocks3": {"blocks": (2, 3)},
    "no_set": {"no_set": (False, True)},
    "retr_cat": {"retr_cat": (0, 16)},  # second wave
    "xdrop_ctxhn_hs": {"xt_exclude": ("", CTXHNHS)},  # second wave
    "xdrop_pre2": {"xt_exclude": ("", PRE2)},  # second wave (coordinator request)
    "pop1": {"xt_exclude": ("", POP), "xt_add": ("", XADD)},  # pop1 (ledger section 7)
    "pre2noid": {"xt_exclude": ("", PRE2), "xt_add": ("", PRE2N), "xt_add_relonly": ("", PRE2N_RO)},  # ENCODER-DOSE (ledger section 10)
    "kv64": {"kv64": (0, 1)},  # ENCODER-DOSE (C) (ledger section 10.C)
}
ARMS = ["base"] + list(ARM_DIFF)
SCR = os.environ.get("TABL_SCRATCH", "/tmp/scratchpad/tabl")
os.makedirs(SCR, exist_ok=True)
os.makedirs(f"{ABL}/logs", exist_ok=True)
LOG = open(f"{ABL}/logs/test_abl_{PART}.log", "w")
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


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def load_parse(path):
    tree = ast.parse(open(path).read())
    fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "parse"][0]
    ns = {"argparse": argparse, "POOL_P_DEFAULT": CA.POOL_P_DEFAULT}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "train_parse", "exec"), ns)

    def parse(argv):
        sys.argv = ["train.py"] + argv
        return ns["parse"]()
    return parse


def presets(path):
    """XT_PRESETS / XT_REL / XT_RELONLY exactly as train.py defines them (module-level assignments exec'd)."""
    tree = ast.parse(open(path).read())
    body = [n for n in tree.body if isinstance(n, ast.Assign)
            and any(isinstance(t, (ast.Name, ast.Subscript)) and ast.unparse(t).startswith("XT_") for t in n.targets)]
    ns = {}
    exec(compile(ast.Module(body=body, type_ignores=[]), "train_presets", "exec"), ns)
    return ns["XT_PRESETS"], ns["XT_REL"], ns["XT_RELONLY"]


def xt_layout(preset, exclude="", rel_extra="", add="", add_relonly=""):
    """(xt_names, xt_rel, xt_idcols, xt_relonly, n_dense) exactly as train.py derives them (helpers of abl_arms)."""
    XP, XR, XRO = presets(TRAIN_PY)
    names = list(XP[preset])
    if exclude:
        names = AA.xt_exclude_names(names, exclude, lambda *a: None)
    if add:  # pop1: train.py's hook (after --xt_exclude, before the miss check / the XT_REL lookups)
        names = AA.xt_add_names(names, add, lambda *a: None, reserved=XR.get(preset, []) + XRO.get(preset, []))
    xt_rel = [names.index(n) for n in XR.get(preset, []) if n in names] or None
    if rel_extra:
        xt_rel = AA.xt_rel_extend(xt_rel, names, rel_extra, lambda *a: None)
    idcols = [i for i, n in enumerate(names) if n.startswith(("seq_cv", "it_hist"))] or None
    relonly = [names.index(n) for n in XRO.get(preset, []) if n in names] or None
    if add_relonly:  # ENCODER-DOSE pre2noid: train.py's hook right after the XT_RELONLY lookup (ledger section 10)
        xt_rel, relonly = AA.xt_add_relonly_idx(xt_rel, relonly, names, add, add_relonly, lambda *a: None)
    n_dense = MO.N_DENSE + 2 + len(names) + 2 * len(xt_rel or []) - len(relonly or [])
    return names, xt_rel, idcols, relonly, n_dense


def dense_cols(names, xt_rel, relonly):
    """Column labels of the xt part of dense_features: raw columns (minus relonly) then [pct, z] per xt_rel index."""
    raw = [f"raw:{n}" for i, n in enumerate(names) if i not in set(relonly or [])]
    rel = [f"{k}:{names[j]}" for j in (xt_rel or []) for k in ("pct", "z")]
    return raw + rel


# ================================================================================================= CPU part
_norm = lambda s: ast.unparse(ast.parse(s, mode="eval").body)  # noqa: E731
ABL_IF = collections.Counter({_norm("a.xt_exclude or a.xt_rel_extra or a.retr_cat or a.xt_add or a.xt_add_relonly or a.kv64"): 1, _norm("a.xt_exclude"): 1,
                              _norm("a.xt_rel_extra"): 1, _norm("a.retr_cat"): 1, _norm("a.xt_add"): 1, _norm("a.xt_add_relonly"): 1, _norm("a.kv64"): 2})
REF_ASSERT = ('assert a.variant.startswith("w5_"), "nn_x/w5 copy: variant names must start with w5_"',
              'assert a.variant.startswith("abl_"), "nn_x/abl copy: variant names must start with abl_"')


class StripAbl(ast.NodeTransformer):
    def __init__(self):
        self.ifs, self.other = collections.Counter(), []

    def visit_ImportFrom(self, n):
        if n.module in ABL_MODULES:
            self.other.append(f"from {n.module} import {[a.name for a in n.names]}")
            return None
        return n

    def visit_Expr(self, n):
        v = n.value
        if (isinstance(v, ast.Call) and isinstance(v.func, ast.Attribute) and v.func.attr == "add_argument"
                and v.args and isinstance(v.args[0], ast.Constant) and v.args[0].value in ABL_FLAGS):
            self.other.append(f"add_argument {v.args[0].value}")
            return None
        return self.generic_visit(n)

    def visit_If(self, n):
        t = ast.unparse(n.test)
        if t in ABL_IF:
            self.ifs[t] += 1
            return None
        return self.generic_visit(n)


def strip_doc(tree):
    if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant):
        tree.body = tree.body[1:]
    return tree


def dry_cmds(script, gpu, jobs, rx):
    out = subprocess.run(["bash", script, str(gpu), jobs], capture_output=True, text=True, env={**os.environ, "DRY": "1"})
    cmds = {}
    for line in out.stdout.splitlines():
        if " train.py " in line:
            m = re.search(rx, line)
            cmds[m.group(0)] = line.split(" train.py ", 1)[1].split()
    return out.returncode, cmds


def diff_ns(a, b, skip=("variant",)):
    return {k: (a[k], b[k]) for k in a if k not in skip and a[k] != b[k]}


def c0_code(parse):
    P("\n== C0 code: lane / launch commands, AST flag-off proof, guards, call sites, module copies")
    jobs = " ".join(f"{a}:{s}" for a in ARMS for s in (23, 24))
    rc, cmds = dry_cmds(LANE, 5, jobs, r"abl_\w+_s2[34]")
    check("abl_lane.sh DRY: one train.py command per job (19 arms incl. base x 2 seeds = 38)", rc == 0 and len(cmds) == 38,
          len(cmds))
    ns = {k: vars(parse(v)) for k, v in cmds.items()}
    rc5, c5 = dry_cmds(f"{W5}/lane.sh", 0, "A:base:23 A:base:24", r"w5_base_s2[34]")
    for s in (23, 24):
        base = ns[f"abl_base_s{s}"]
        ref = vars(parse(c5[f"w5_base_s{s}"] + []))
        d = diff_ns(ref, base)
        check(f"abl_base_s{s} == nn_x/w5 lane.sh A:base:{s} (A_BASE + HYG) except the variant name; abl flags at defaults",
              rc5 == 0 and d == {} and base["seed"] == s and base["cooldown_mix"] == 0.02 and base["epochs"] == 2.0
              and base["init"] == "" and {k: base[k] for k in OFF} == OFF, d)
        for arm, want in ARM_DIFF.items():
            d = diff_ns(base, ns[f"abl_{arm}_s{s}"])
            check(f"abl_{arm}_s{s} differs from the base ONLY by {want}", d == want, d if d != want else "")
    RES["lane_cmds"] = {k: " ".join(v) for k, v in cmds.items()}
    refused = [subprocess.run(["bash", LANE, str(g), "base:23"], capture_output=True, text=True,
                              env={**os.environ, "DRY": "1"}).returncode != 0 for g in (0, 1, 2, 4, 6)]
    check("abl_lane.sh refuses GPUs N <ids> (also in DRY mode)", all(refused), refused)
    rc3, c3_ = dry_cmds(LANE, 3, "base:23", r"abl_\w+_s23")
    check("abl_lane.sh accepts GPU N (second wave: ALLOWED default '5 7 3')", rc3 == 0 and len(c3_) == 1
          and "ALLOWED=${ABL_GPUS:-\"5 7 3\"}" in open(LANE).read())
    bad = subprocess.run(["bash", LANE, "5", "nosucharm:23"], capture_output=True, text=True, env={**os.environ, "DRY": "1"})
    check("abl_lane.sh refuses an unknown arm / a seed other than 23 / 24", bad.returncode != 0 and subprocess.run(
        ["bash", LANE, "5", "base:25"], capture_output=True, env={**os.environ, "DRY": "1"}).returncode != 0)
    syn = [subprocess.run(["bash", "-n", f], capture_output=True).returncode == 0 for f in (LANE, LAUNCH, WAITER)]
    check("bash -n: abl_lane.sh, launch_abl.sh, wait_abl_readout.sh", all(syn), syn)
    out = subprocess.run(["bash", LAUNCH], capture_output=True, text=True, env={**os.environ, "DRY": "1"})
    jl = re.search(r"jobs: (.*)", out.stdout)
    jobs_l = jl.group(1).split() if jl else []
    check("launch_abl.sh DRY: 26 jobs in rank order (base first), 3 lanes on each of GPUs N 7, every command printed",
          out.returncode == 0 and len(jobs_l) == 26 and jobs_l[:4] == ["base:23", "base:24", "no_aux:23", "no_aux:24"]
          and out.stdout.count("lane GPU") == 6 and out.stdout.count(" train.py ") == 26, f"{len(jobs_l)} jobs")

    # (b) AST proof
    rec = dict(line.split()[::-1] for line in open(f"{HERE}/w5_source_md5.txt"))
    check("frozen reference == nn_x/w5/train.py at copy time (md5 in tests/w5_source_md5.txt)",
          md5(REF_TRAIN) == rec["nn_x/w5/train.py"], md5(REF_TRAIN))
    P(f"  info: nn_x/w5/train.py now {'UNCHANGED' if md5(f'{W5}/train.py') == rec['nn_x/w5/train.py'] else 'CHANGED'} "
      f"since the copy")
    ref_src = open(REF_TRAIN).read()
    assert ref_src.count(REF_ASSERT[0]) == 1
    ref_src = ref_src.replace(*REF_ASSERT)
    new_src = open(TRAIN_PY).read()
    st = StripAbl()
    new_t = strip_doc(st.visit(ast.parse(new_src)))
    ident = ast.dump(new_t) == ast.dump(strip_doc(ast.parse(ref_src)))
    check("AST: train.py minus the abl statements (exact set: 6 imports, 6 add_argument, 7 guards) == frozen nn_x/w5/train.py "
          "(flag-off path unchanged)", ident and st.ifs == ABL_IF and len(st.other) == 12
          and sum(x.startswith("add_argument") for x in st.other) == 6
          and sorted(x for x in st.other if x.startswith("from")) == ["from abl_arms import ['xt_add_names']",
                                                                       "from abl_arms import ['xt_add_relonly_idx']",
                                                                       "from abl_arms import ['xt_exclude_names', 'xt_rel_extend']",
                                                                       "from abl_retr import ['setup_retr_cat']",
                                                                       "from encdose.kv_probe.kv_adapter import ['setup_kv64']",
                                                                       "from encdose.kv_probe.kv_data import ['attach_kv64']"],
          f"ifs {dict(st.ifs)}; other {st.other}")
    # (c) guards
    a_def = parse(["--variant", "abl_x"])
    check("parse(): every abl flag OFF ('' / 0) by default (incl. --xt_add '', --xt_add_relonly '')", {k: getattr(a_def, k) for k in OFF} == OFF)
    vals = {g: bool(eval(g, {}, dict(a=a_def))) for g in ABL_IF}
    check("every abl guard is False at the defaults", not any(vals.values()), vals)
    aa = argparse.Namespace(**ns["abl_base_s23"])
    vals = {g: bool(eval(g, {}, dict(a=aa))) for g in ABL_IF}
    check("abl_base_s23 command: every abl guard False", not any(vals.values()), vals)
    for arm, g_ in (("xdrop_pop", "a.xt_exclude"), ("xrel_more", "a.xt_rel_extra"), ("retr_cat", "a.retr_cat"),
                    ("xdrop_ctxhn_hs", "a.xt_exclude")):
        aa = argparse.Namespace(**ns[f"abl_{arm}_s23"])
        on = {g: bool(eval(g, {}, dict(a=aa))) for g in ABL_IF}
        check(f"abl_{arm}_s23 command: exactly its guard (and the --init guard) True", sum(on.values()) == 2
              and on[_norm(g_)], on)
    aa = argparse.Namespace(**ns["abl_pop1_s23"])
    on = {g: bool(eval(g, {}, dict(a=aa))) for g in ABL_IF}
    check("abl_pop1_s23 command: exactly the xt_exclude + xt_add guards (and the --init guard) True", sum(on.values()) == 3
          and on["a.xt_exclude"] and on["a.xt_add"], on)
    # (d) call sites
    tree = ast.parse(new_src)
    parents = {ch: p_ for p_ in ast.walk(tree) for ch in ast.iter_child_nodes(p_)}

    def chain(n):
        c, x = [], n
        while x in parents:
            x = parents[x]
            if isinstance(x, ast.If):
                c.append(("else " if x.orelse and any(n_ is x_ or n_ in ast.walk(x_) for x_ in x.orelse for n_ in [n]) else "if ")
                         + ast.unparse(x.test))
            if isinstance(x, ast.For):
                c.append("for " + ast.unparse(x.target))
            if isinstance(x, ast.FunctionDef):
                c.append("def " + x.name)
        return c
    calls = {f: [chain(n) for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == f]
             for f in ("xt_exclude_names", "xt_rel_extend", "setup_retr_cat", "xt_add_names", "xt_add_relonly_idx")}
    check("xt_exclude_names: one call behind `if a.xt_exclude` inside the no-init branch of main()",
          calls["xt_exclude_names"] == [["if a.xt_exclude", "else a.init", "def main"]], calls["xt_exclude_names"])
    check("xt_rel_extend: one call behind `if a.xt_rel_extra` inside the no-init branch of main()",
          calls["xt_rel_extend"] == [["if a.xt_rel_extra", "else a.init", "def main"]], calls["xt_rel_extend"])
    check("setup_retr_cat: one call behind `if a.retr_cat` inside the no-init branch of main()",
          calls["setup_retr_cat"] == [["if a.retr_cat", "else a.init", "def main"]], calls["setup_retr_cat"])
    check("xt_add_names: one call behind `if a.xt_add` inside the no-init branch of main()",
          calls["xt_add_names"] == [["if a.xt_add", "else a.init", "def main"]], calls["xt_add_names"])
    check("xt_add_relonly_idx: one call behind `if a.xt_add_relonly` inside the no-init branch of main() (ENCODER-DOSE, ledger 10)",
          calls["xt_add_relonly_idx"] == [["if a.xt_add_relonly", "else a.init", "def main"]], calls["xt_add_relonly_idx"])
    mainf = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"][0]
    init_if = [n for n in mainf.body if isinstance(n, ast.If) and ast.unparse(n.test) == "a.init"
               and any(isinstance(x, ast.Assign) and "XT_PRESETS" in ast.unparse(x) for x in n.orelse)][0]
    tests_ = [ast.unparse(x.test) if isinstance(x, ast.If) else None for x in init_if.orelse]
    i_ex = tests_.index("a.xt_exclude")
    check("the `if a.xt_add` hook sits right after `if a.xt_exclude` and right before the `if xt_names:` miss check",
          tests_[i_ex + 1:i_ex + 3] == ["a.xt_add", "xt_names"], tests_[i_ex:i_ex + 3])
    users = sorted(f for f in os.listdir(ABL) if f.endswith(".py") and f not in ("abl_arms.py", "train.py")
                   and re.search(r"\babl_arms\b", open(f"{ABL}/{f}").read()))
    check("no other nn_x/abl module imports abl_arms (predict / eval / holdout_eval / stale_pred untouched)", users == [], users)
    users = sorted(f for f in os.listdir(ABL) if f.endswith(".py") and f not in ("abl_retr.py", "train.py")
                   and re.search(r"\babl_retr\b", open(f"{ABL}/{f}").read()))
    check("abl_retr imported only by train.py and the replay-only f_replay.py (never on the training path)",
          users == ["f_replay.py"], users)
    cm = open(f"{ABL}/common.py").read() + open(f"{ABL}/model.py").read() + open(f"{ABL}/data.py").read()
    check("common.py / model.py / data.py contain no xt_exclude / xt_rel_extra / retr_cat / xt_add code",
          not re.search(r"xt_exclude|xt_rel_extra|retr_cat|xt_add", cm))
    # (e) module copies
    same = {f: md5(f"{ABL}/{f}") == rec[f"nn_x/w5/{f}"] for f in (
        "model.py", "cd_arms.py", "cd_arms4.py", "cd_arms5.py", "cd_arms6.py", "cd_arms7.py", "blend.py", "holdout_eval.py",
        "stale_pred.py", "tests/mini_split.py")}
    check("model / cd_arms* / blend / holdout_eval / stale_pred / tests/mini_split byte-identical to the nn_x/w5 copies",
          all(same.values()), same)
    check("tests/e2e_abl.py byte-identical to nn_x/w5/tests/e2e_w5.py", md5(f"{HERE}/e2e_abl.py") == rec["nn_x/w5/tests/e2e_w5.py"])

    class Norm(ast.NodeTransformer):
        def visit_Constant(self, n):
            return ast.Constant(value="S") if isinstance(n.value, str) else n
    eq = {}
    for f in ("data.py", "common.py"):
        eq[f] = (ast.dump(Norm().visit(ast.parse(open(f"{ABL}/{f}").read())))
                 == ast.dump(Norm().visit(ast.parse(open(f"{W5}/{f}").read())))) and md5(f"{W5}/{f}") == rec[f"nn_x/w5/{f}"]
    check("data / common == nn_x/w5 up to string constants (paths, messages only)", all(eq.values()), eq)
    ok_dirs = ("W5 = ROOT / \"nn_x\" / \"abl\"" in open(f"{ABL}/common.py").read()
               and "FEAT_CACHE = ROOT / \"nn_x\" / \"abl\" / \"feat_cache\"" in open(f"{ABL}/data.py").read()
               and os.path.realpath(f"{ABL}/feat_cache") == os.path.realpath(f"{ROOT}/nn/feat_cache"))
    check("outputs go to nn_x/abl/{ckpt,logs,preds}; feat_cache symlink -> nn/feat_cache (read-only)", ok_dirs)


def c1_helpers():
    P("\n== C1 abl_arms helpers on the real preset cvnd_pre2s")
    XP, XR, XRO = presets(TRAIN_PY)
    full = xt_layout("cvnd_pre2s")
    check("cvnd_pre2s: 34 columns, XT_REL 5 (seq_cv_invd20, it_hist_all_luo + 3 pre2), relonly 3, idcols 8, n_dense "
          f"{MO.N_DENSE + 2 + 34 + 10 - 3}", len(full[0]) == 34 and len(full[1]) == 5 and len(full[3]) == 3
          and len(full[2]) == 8 and full[4] == MO.N_DENSE + 2 + 34 + 10 - 3, (len(full[0]), full[1], full[2], full[3], full[4]))
    ex = xt_layout("cvnd_pre2s", exclude=POP)
    check("xdrop_pop: 26 names (no seq_cv / it_hist), xt_rel = the 3 pre2 columns only, idcols None, n_dense = N_DENSE+2+26+6-3",
          len(ex[0]) == 26 and not any(n.startswith(("seq_cv", "it_hist")) for n in ex[0])
          and [ex[0][j] for j in ex[1]] == ["pre2_fe_s", "pre2_new_s", "pre2_hmax32"] and ex[2] is None
          and ex[4] == MO.N_DENSE + 2 + 26 + 6 - 3 and ex[3] == ex[1], (len(ex[0]), ex[1], ex[2], ex[3], ex[4]))
    ex2 = xt_layout("cvnd_pre2s", exclude=CTXND)
    check("xdrop_ctxnd: 24 names (no ctx_nd), xt_rel 5 (re-indexed), idcols 8, n_dense = N_DENSE+2+24+10-3",
          len(ex2[0]) == 24 and not any(n.startswith("ctx_nd") for n in ex2[0]) and len(ex2[1]) == 5
          and [ex2[0][j] for j in ex2[1]] == [full[0][j] for j in full[1]] and ex2[2] == list(range(8))
          and ex2[4] == MO.N_DENSE + 2 + 24 + 10 - 3, (len(ex2[0]), ex2[1], ex2[4]))
    ex3 = xt_layout("cvnd_pre2s", exclude=PRE2)
    check("xdrop_pre2: 28 names (== the cvnd preset, every pre2* / has:pre2_fe column gone), xt_rel 2 (seq_cv_invd20, it_hist_all_luo), "
          "relonly None, idcols 8, n_dense = N_DENSE+2+28+4",
          ex3[0] == XP["cvnd"] and not any("pre2" in n for n in ex3[0]) and [ex3[0][j] for j in ex3[1]] == ["seq_cv_invd20", "it_hist_all_luo"]
          and ex3[3] is None and ex3[2] == list(range(8)) and ex3[4] == MO.N_DENSE + 2 + 28 + 4, (len(ex3[0]), ex3[1], ex3[3], ex3[4]))
    rx = xt_layout("cvnd_pre2s", rel_extra=XREL)
    check("xrel_more: 34 names, xt_rel 9 = XT_REL's 5 + the 4 extras in order, n_dense + 8",
          rx[0] == full[0] and rx[1][:5] == full[1] and [rx[0][j] for j in rx[1][5:]] == XREL.split(",")
          and rx[4] == full[4] + 8 and rx[2] == full[2] and rx[3] == full[3], (rx[1], rx[4]))
    errs = {}
    for nm, fn in (("unknown", lambda: AA.xt_exclude_names(full[0], "nosuch", lambda *a: None)),
                   ("dup", lambda: AA.xt_exclude_names(full[0], "seq_cv_pop,seq_cv_pop", lambda *a: None)),
                   ("empty", lambda: AA.xt_exclude_names(["a"], "a", lambda *a: None)),
                   ("rel_unknown", lambda: AA.xt_rel_extend(full[1], full[0], "nosuch", lambda *a: None)),
                   ("rel_overlap", lambda: AA.xt_rel_extend(full[1], full[0], "seq_cv_invd20", lambda *a: None)),
                   ("rel_dup", lambda: AA.xt_rel_extend(full[1], full[0], "seq_cv_pop,seq_cv_pop", lambda *a: None))):
        try:
            fn()
            errs[nm] = "no error"
        except AssertionError as e:
            errs[nm] = f"AssertionError: {str(e)[:40]}"
    check("helpers refuse unknown / duplicated / preset-emptying / XT_REL-overlapping names", all(v.startswith("AssertionError") for v in errs.values()), errs)
    RES["layouts"] = dict(full=full, xdrop_pop=ex, xdrop_ctxnd=ex2, xrel_more=rx)


def c2_dense():
    P("\n== C2 dense_features on real MiniSplit train batches: exclude == column removal, rel_extra == appended rank/z (exact)")
    feeds = pick_train_feeds(n_each=24, seed=7)
    full = xt_layout("cvnd_pre2s")
    ms_full = MiniSplit("train", feeds, xt_names=full[0])
    cols_full = dense_cols(full[0], full[1], full[3])
    rng = np.random.default_rng(0)
    fb = rng.choice(ms_full.n, 48, replace=False)
    b = ms_full.gather(fb, with_pos=True)
    X = MO.dense_features(b, full[4], xt_rel=full[1], xt_drop=0.0, train=False, xt_idcols=full[2], xt_relonly=full[3])
    nb = MO.N_DENSE + 2
    check("full preset: dense_features width == n_dense (base + xt layout)", X.shape[-1] == full[4] == nb + len(cols_full), X.shape)
    for arm, excl in (("xdrop_pop", POP), ("xdrop_ctxnd", CTXND)):
        lay = xt_layout("cvnd_pre2s", exclude=excl)
        ms = MiniSplit("train", feeds, xt_names=lay[0])
        b2 = ms.gather(fb, with_pos=True)
        X2 = MO.dense_features(b2, lay[4], xt_rel=lay[1], xt_drop=0.0, train=False, xt_idcols=lay[2], xt_relonly=lay[3])
        cols = dense_cols(lay[0], lay[1], lay[3])
        sel = list(range(nb)) + [nb + cols_full.index(c) for c in cols]
        dropped = [c for c in cols_full if c not in cols]
        same = X2.shape[-1] == lay[4] and torch.equal(X[..., sel], X2)
        check(f"{arm}: features == full features minus {len(dropped)} columns (raw + rank/z of the dropped names), exact",
              same, f"dropped {dropped}")
    lay = xt_layout("cvnd_pre2s", rel_extra=XREL)
    X3 = MO.dense_features(b, lay[4], xt_rel=lay[1], xt_drop=0.0, train=False, xt_idcols=lay[2], xt_relonly=lay[3])
    cmask, mf = b["cmask"], b["cmask"].float()
    cnt = mf.sum(1, keepdim=True)
    extra = []
    for n in XREL.split(","):
        j = full[0].index(n)
        extra += [MO._pctrank(b["xt"][..., j], cmask, cnt), MO._zscore(b["xt"][..., j], mf, cnt)]
    extra = torch.nan_to_num(torch.stack(extra, -1), nan=0.0, posinf=0.0, neginf=0.0)
    check("xrel_more: features == full features + [pct-rank, z] of the 4 extra columns (exact; == _pctrank / _zscore of the raw column)",
          X3.shape[-1] == full[4] + 8 and torch.equal(X3[..., :full[4]], X) and torch.equal(X3[..., full[4]:], extra),
          (X3.shape, float((X3[..., full[4]:] - extra).abs().max())))


def c3_arms_build(parse):
    P("\n== C3 every arm's model builds on the CPU from its command and runs one train-mode forward on a real batch")
    feeds = pick_train_feeds(n_each=16, seed=11)
    full = xt_layout("cvnd_pre2s")
    cache = {}
    rc, cmds = dry_cmds(LANE, 5, " ".join(f"{a}:23" for a in ARMS), r"abl_\w+_s23")
    for arm in ARMS:
        a = parse(cmds[f"abl_{arm}_s23"])
        lay = xt_layout("cvnd_pre2s", exclude=a.xt_exclude, rel_extra=a.xt_rel_extra, add=a.xt_add)
        key = tuple(lay[0])
        if key not in cache:
            cache[key] = MiniSplit("train", feeds, xt_names=lay[0])
        ms = cache[key]
        torch.manual_seed(0)
        try:
            m = MO.Ranker(d=a.d, heads=a.heads, hist_layers=a.hist_layers, blocks=a.blocks, drop=a.drop, use_hist=not a.no_hist,
                          use_set=not a.no_set, use_pos_tower=not a.no_pos, aux_pos=a.aux_pos > 0, ff=a.ff, sim_bias=a.sim_bias,
                          pos_mode=a.pos_mode, n_dense=lay[4], domain=a.domain, intent=a.intent > 0, xt_names=lay[0],
                          id_buckets=a.id_buckets, id_drop=a.id_drop, arch=a.arch, id_hist_only=a.id_hist_only, xt_rel=lay[1],
                          xt_drop=a.xt_drop, xt_idcols=lay[2], xt_relonly=lay[3], attn_gate=a.attn_gate, split_kv=a.muon,
                          retr_m=a.retr_m, attn_drop=a.attn_drop if a.attn_drop >= 0 else None, retr_prod=a.retr_prod,
                          retr_uniq=a.retr_uniq, cretr_m=a.cretr_m, set_sim_bias=a.set_sim_bias, typed_cross=a.typed_cross,
                          retr_tcnt=a.retr_tcnt)
            if a.retr_cat:  # second wave: exactly train.py's guarded call (after the Ranker, before the optimisers)
                m = AR.setup_retr_cat(m, a, lambda *x: None)
            fb = np.arange(min(32, ms.n))
            b = ms.gather(fb, with_pos=True)
            Xd = m.dense(b)
            m.set_norm(Xd[b["cmask"]].mean(0), Xd[b["cmask"]].std(0).clamp_min(1e-3))
            m.train()
            lg = m(b, m.dense(b), use_pos=True)
            aux = getattr(m, "aux_out", None)
            ok = lg.shape == b["cmask"].shape and torch.isfinite(lg[b["cmask"]]).all() and Xd.shape[-1] == lay[4] \
                and ((aux is not None) == (a.aux_pos > 0)) and (hasattr(m, "retr_cat") == bool(a.retr_cat))
            info = f"params {sum(p.numel() for p in m.parameters()) / 1e6:.2f}M, n_dense {lay[4]}, aux head {aux is not None}"
        except Exception as e:  # noqa: BLE001
            ok, info = False, f"{type(e).__name__}: {str(e)[:120]}"
        check(f"{arm}: Ranker builds and one train-mode forward runs (logits finite, dense width n_dense)", ok, info)


def _ranker(a, lay):
    """Ranker exactly as train.py builds it (no-init branch) from a parsed command and an xt layout; seed 0."""
    torch.manual_seed(0)
    return MO.Ranker(d=a.d, heads=a.heads, hist_layers=a.hist_layers, blocks=a.blocks, drop=a.drop, use_hist=not a.no_hist,
                     use_set=not a.no_set, use_pos_tower=not a.no_pos, aux_pos=a.aux_pos > 0, ff=a.ff, sim_bias=a.sim_bias,
                     pos_mode=a.pos_mode, n_dense=lay[4], domain=a.domain, intent=a.intent > 0, xt_names=lay[0],
                     id_buckets=a.id_buckets, id_drop=a.id_drop, arch=a.arch, id_hist_only=a.id_hist_only, xt_rel=lay[1],
                     xt_drop=a.xt_drop, xt_idcols=lay[2], xt_relonly=lay[3], attn_gate=a.attn_gate, split_kv=a.muon,
                     retr_m=a.retr_m, attn_drop=a.attn_drop if a.attn_drop >= 0 else None, retr_prod=a.retr_prod,
                     retr_uniq=a.retr_uniq, cretr_m=a.cretr_m, set_sim_bias=a.set_sim_bias, typed_cross=a.typed_cross,
                     retr_tcnt=a.retr_tcnt)


def c4_retr_cat(parse):
    P("\n== C4 retr_cat (abl_retr.py) on real MiniSplit batches: init identity (eval + train), exact-zero output, same-category "
      "selection, invariance to other-category events, per-candidate reference, checkpoint round trip")
    feeds = pick_train_feeds(n_each=24, seed=5)
    lay = xt_layout("cvnd_pre2s")
    ms = MiniSplit("train", feeds, xt_names=lay[0])
    rc, cmds = dry_cmds(LANE, 5, "base:23 retr_cat:23", r"abl_\w+_s23")
    a0, a1 = parse(cmds["abl_base_s23"]), parse(cmds["abl_retr_cat_s23"])
    m0, m1 = _ranker(a0, lay), _ranker(a1, lay)
    m1 = AR.setup_retr_cat(m1, a1, lambda *x: None)
    sd0, sd1 = m0.state_dict(), m1.state_dict()
    extra = sorted(k for k in sd1 if k not in sd0)
    check("retr_cat adds only retr_cat.* parameters (same names as the base retr) and every baseline parameter keeps exactly its init",
          extra and all(k.startswith("retr_cat.") for k in extra) and set(sd0) <= set(sd1)
          and all(torch.equal(sd0[k], sd1[k]) for k in sd0)
          and sorted(k[len("retr_cat."):] for k in extra) == sorted(k[len("retr."):] for k in sd0 if k.startswith("retr.")),
          f"{len(extra)} new tensors")
    check("Z zero-init (weight + bias); cfg['retr_cat'] == 16 on the arm, absent on the base; m == 16",
          float(m1.retr_cat.Z.weight.abs().max()) == 0.0 and float(m1.retr_cat.Z.bias.abs().max()) == 0.0
          and m1.cfg.get("retr_cat") == 16 and "retr_cat" not in m0.cfg and m1.retr_cat.m == 16)
    muon = sorted(n for n, mod in m1.named_modules() if isinstance(mod, torch.nn.Linear) and min(mod.weight.shape) > 1
                  and n.startswith("retr_cat"))
    check("Muon rule picks the same four matrices as for the base retrieval (wk, t1, t2, Z)",
          muon == ["retr_cat.Z", "retr_cat.t1", "retr_cat.t2", "retr_cat.wk"], muon)
    rng = np.random.default_rng(3)
    fb = rng.choice(ms.n, 48, replace=False)
    b = ms.gather(fb, with_pos=True)
    X = m0.dense(b)
    mean, std = X[b["cmask"]].mean(0), X[b["cmask"]].std(0).clamp_min(1e-3)
    m0.set_norm(mean, std)
    m1.set_norm(mean, std)
    m0.eval()
    m1.eval()
    l0, l1 = m0(b, m0.dense(b)), m1(b, m1.dense(b))
    check("eval mode: flag-on at init bit-identical to flag-off (logits)", torch.equal(l0, l1), float((l0 - l1).abs().max()))
    m0.train()
    m1.train()
    torch.manual_seed(77)
    l0 = m0(b, m0.dense(b), use_pos=True)
    x0 = m0.aux_out.clone()
    torch.manual_seed(77)
    l1 = m1(b, m1.dense(b), use_pos=True)
    x1 = m1.aux_out.clone()
    check("train mode: flag-on at init bit-identical to flag-off (logits + aux head: the module's dropout never touches the "
          "global RNG stream)", torch.equal(l0, l1) and torch.equal(x0, x1), float((l0 - l1).abs().max()))
    # the module alone
    ce, he, hv, het, hes, cmiss = b["ce"], b["he"], b["hemb_ok"], b["het"], b["hes"], b["cmiss"]
    S = torch.einsum("bmd,bkd->bmk", ce, he) * hv[:, None, :]
    same = AR.same_cat_mask(b, hv)
    st = torch.get_rng_state()
    out = m1.retr_cat(ce, he, hv, het, hes, S, cmiss, same)
    check("module output exactly 0 at init in train mode; global CPU RNG state unchanged by the call",
          float(out.abs().max()) == 0.0 and out.shape == (*ce.shape[:2], m1.cfg["d"]) and torch.equal(st, torch.get_rng_state()))
    dp = m1.retr_cat.dp
    z = dp(torch.ones(400000))
    share0 = float((z == 0).float().mean())
    check("GenDropout: nn.Dropout semantics (drop share ~p, kept values 1/(1-p)); successive calls differ (the generator advances)",
          abs(share0 - 0.1) < 0.005 and torch.equal(z[z != 0], torch.full_like(z[z != 0], 1 / 0.9))
          and not torch.equal(dp(torch.ones(1000)), dp(torch.ones(1000))), f"drop share {share0:.4f}")
    # selection: replicate the inherited forward's top-m / valid on the masked scores
    K, mm = he.shape[1], min(m1.retr_cat.m, he.shape[1])
    top, idx = S.masked_fill(~same, -2.0).float().masked_fill(~hv[:, None, :], -2.0).topk(mm, -1)
    valid = (top > -1.5) & ~cmiss[..., None]
    n_same = same.sum(-1)
    check("selection: every valid slot is a same-category valid event; #valid slots == min(m, #same-category events), 0 for cmiss",
          bool((same.gather(2, idx) | ~valid).all())
          and torch.equal(valid.sum(-1), torch.where(cmiss, torch.zeros_like(n_same), n_same.clamp_max(mm))),
          f"candidates {int(cmiss.numel())}, with same-category items {int((n_same > 0).sum())}, capped at m {int((n_same > mm).sum())}")
    # random Z: invariance to other-category events, zero where none, per-candidate reference
    with torch.no_grad():
        m1.retr_cat.Z.weight.normal_(0, 0.1)
        m1.retr_cat.Z.bias.normal_(0, 0.1)
    m1.eval()
    o1 = m1.retr_cat(ce, he, hv, het, hes, S, cmiss, same)
    other = ~same.any(1)  # [B,K] events that share a category with NO candidate of the feed (incl. padded events)
    g = torch.Generator().manual_seed(9)
    he2 = torch.where(other[..., None], torch.nn.functional.normalize(torch.randn(he.shape, generator=g), dim=-1), he)
    het2 = torch.where(other, torch.randint(1, MO.N_ET + 1, het.shape, generator=g), het)
    hes2 = torch.where(other, torch.randint(0, MO.N_ES, hes.shape, generator=g), hes)
    S2 = torch.einsum("bmd,bkd->bmk", ce, he2) * hv[:, None, :]
    o2 = m1.retr_cat(ce, he2, hv, het2, hes2, S2, cmiss, same)
    check("random Z: output bitwise invariant to the embeddings / types / sources of other-category history events",
          torch.equal(o1, o2) and bool(other.any()), f"{int(other.sum())} other-category events perturbed")
    has = same.any(-1) & ~cmiss
    check("random Z: output exactly 0 for candidates without a same-category event (and cmiss), non-zero for every other candidate",
          float(o1[~has].abs().max()) == 0.0 and bool((o1[has].abs().sum(-1) > 0).all()) and bool((~has).any()) and bool(has.any()),
          f"{int((~has).sum())} without, {int(has.sum())} with")
    ref_ok, nref, worst = True, 0, 0.0
    for bi, ci in zip(*torch.where(has)):
        if nref >= 40:
            break
        nref += 1
        r = MO.HistRetrieval.forward(m1.retr_cat, ce[bi:bi + 1, ci:ci + 1], he[bi:bi + 1], same[bi, ci][None], het[bi:bi + 1],
                                     hes[bi:bi + 1], S[bi:bi + 1, ci:ci + 1], cmiss[bi:bi + 1, ci:ci + 1])[0, 0]
        worst = max(worst, float((r - o1[bi, ci]).abs().max()))
        ref_ok = ref_ok and torch.allclose(r, o1[bi, ci], atol=1e-5, rtol=1e-4)
    check(f"random Z: == the parent HistRetrieval run per candidate with the history RESTRICTED to its same-category events "
          f"({nref} candidates, allclose)", ref_ok and nref > 0, f"max |d| {worst:.2e}")
    # checkpoint round trip
    ck = {"state_dict": {k: v.clone() for k, v in m1.state_dict().items()}, "cfg": dict(m1.cfg)}
    try:
        MO.Ranker(**ck["cfg"])
        refused = False
    except TypeError:
        refused = True
    m2 = AR.load_ranker(ck, "cpu")
    m2.eval()
    l1, l2 = m1(b, m1.dense(b)), m2(b, m2.dense(b))
    check("checkpoint: Ranker(**cfg) refuses cfg['retr_cat'] (loud); abl_retr.load_ranker rebuilds + reloads, logits bit-identical",
          refused and torch.equal(l1, l2) and hasattr(m2, "retr_cat"), float((l1 - l2).abs().max()))
    try:
        AR.setup_retr_cat(_ranker(a0, lay), argparse.Namespace(**{**vars(a1), "retr_m": 0}), lambda *x: None)
        bad = "no error"
    except AssertionError as e:
        bad = f"AssertionError: {str(e)[:50]}"
    check("setup_retr_cat refuses a run without the base retrieval (--retr_m 0)", bad.startswith("AssertionError"), bad)


def c5_xt_add(parse):
    P("\n== C5 --xt_add (pop1): flag-off identity, layout, errors, feature files, dense features on real batches, checkpoint round trip")
    XP, XR, XRO = presets(TRAIN_PY)
    full, xp = xt_layout("cvnd_pre2s"), xt_layout("cvnd_pre2s", exclude=POP)
    check("flag off: layout with --xt_add '' == the layout without (full preset and xdrop_pop)",
          xt_layout("cvnd_pre2s", add="") == full and xt_layout("cvnd_pre2s", exclude=POP, add="") == xp)
    p1 = xt_layout("cvnd_pre2s", exclude=POP, add=XADD)
    check(f"pop1: xt_names = the 26 xdrop_pop names + {XADD} appended LAST; xt_rel / xt_idcols / xt_relonly unchanged; "
          f"n_dense = xdrop_pop + 1 = N_DENSE+2+27+6-3 = {MO.N_DENSE + 2 + 27 + 6 - 3}",
          p1[0] == xp[0] + [XADD] and p1[1] == xp[1] and p1[2] == xp[2] and p1[3] == xp[3] and p1[4] == xp[4] + 1
          == MO.N_DENSE + 2 + 27 + 6 - 3, (len(p1[0]), p1[1], p1[2], p1[3], p1[4]))
    fa = xt_layout("cvnd_pre2s", add=XADD)
    check(f"full preset + --xt_add: 35 names, {XADD} last, xt_rel / xt_idcols / xt_relonly indices unchanged, n_dense + 1",
          fa[0] == full[0] + [XADD] and fa[1] == full[1] and fa[2] == full[2] and fa[3] == full[3] and fa[4] == full[4] + 1,
          (len(fa[0]), fa[4]))
    rsv = XR["cvnd_pre2s"] + XRO["cvnd_pre2s"]
    errs = {}
    for nm, fn in (("no_preset", lambda: AA.xt_add_names(None, XADD, lambda *a: None)),
                   ("empty", lambda: AA.xt_add_names(xp[0], ",", lambda *a: None)),
                   ("dup_spec", lambda: AA.xt_add_names(xp[0], f"{XADD},{XADD}", lambda *a: None)),
                   ("present", lambda: AA.xt_add_names(full[0], "ctx_nd_max", lambda *a: None)),
                   ("idcol_prefix", lambda: AA.xt_add_names(xp[0], "it_hist_all_luo", lambda *a: None)),
                   ("reserved", lambda: AA.xt_add_names(AA.xt_exclude_names(full[0], "pre2_fe_s", lambda *a: None), "pre2_fe_s",
                                                        lambda *a: None, reserved=rsv))):
        try:
            fn()
            errs[nm] = "no error"
        except AssertionError as e:
            errs[nm] = f"AssertionError: {str(e)[:40]}"
    check("xt_add_names refuses: no preset, empty / duplicated names, a name already present, seq_cv / it_hist prefix, an "
          "XT_REL / XT_RELONLY name of the preset", all(v.startswith("AssertionError") for v in errs.values()), errs)
    # feature files: new, hard-linked into nn/feat_cache, == log1p(min(tte_it_imp, 50)) float32
    ok_f, info = True, {}
    for sp in ("train", "val", "test"):
        a_, b_ = f"{ROOT}/data/feats/{sp}/{XADD}.npy", f"{ROOT}/nn/feat_cache/{sp}/{XADD}.npy"
        same_ino = os.stat(a_).st_ino == os.stat(b_).st_ino and os.stat(a_).st_nlink == 2
        x = np.load(b_, mmap_mode="r")
        if sp == "train":
            x, ref = np.asarray(x[::997]), np.load(f"{ROOT}/data/feats/{sp}/tte_it_imp.npy", mmap_mode="r")[::997]
        else:
            x, ref = np.asarray(x), np.load(f"{ROOT}/data/feats/{sp}/tte_it_imp.npy")
        eq = x.dtype == np.float32 and np.array_equal(x, np.log1p(np.minimum(np.asarray(ref), 50)).astype(np.float32))
        ok_f = ok_f and same_ino and eq and os.path.exists(f"{ABL}/feat_cache/{sp}/{XADD}.npy")
        info[sp] = (same_ino, eq, float(x.max()))
    check(f"{XADD}: data/feats file hard-linked into nn/feat_cache (same inode, nlink 2; visible through nn_x/abl/feat_cache) "
          "and == log1p(min(tte_it_imp, 50)) float32 for train (every 997th row) / val / test", ok_f, info)
    # dense features on real MiniSplit train batches
    feeds = pick_train_feeds(n_each=24, seed=7)
    ms_xp, ms_p1 = MiniSplit("train", feeds, xt_names=xp[0]), MiniSplit("train", feeds, xt_names=p1[0])
    f = __import__("polars").read_parquet(f"{ROOT}/data/flat/train/feeds.parquet")
    fs = np.sort(np.asarray(feeds))
    crow = np.concatenate([np.arange(o, o + m) for o, m in zip(f["cand_off"].to_numpy().astype(np.int64)[fs],
                                                                f["M"].to_numpy().astype(np.int64)[fs])])
    tte = np.load(f"{ROOT}/data/feats/train/tte_it_imp.npy", mmap_mode="r")
    want = torch.from_numpy(np.clip(np.log1p(np.minimum(np.asarray(tte[crow]), 50)), -10, 10).astype(np.float32)).half()
    check(f"MiniSplit xt: the pop1 columns == the xdrop_pop columns + one last column == fp16(clip(log1p(min(tte_it_imp, 50)), "
          f"+-10)) on {len(crow)} real train candidates (attach_xt default branch; nothing clipped)",
          torch.equal(ms_p1.xt[:, :-1], ms_xp.xt) and torch.equal(ms_p1.xt[:, -1], want) and float(ms_p1.xt[:, -1].max()) < 3.94,
          float(ms_p1.xt[:, -1].float().max()))
    rng = np.random.default_rng(0)
    fb = rng.choice(ms_xp.n, 48, replace=False)
    b0, b1 = ms_xp.gather(fb, with_pos=True), ms_p1.gather(fb, with_pos=True)
    X0 = MO.dense_features(b0, xp[4], xt_rel=xp[1], xt_drop=0.0, train=False, xt_idcols=xp[2], xt_relonly=xp[3])
    X1 = MO.dense_features(b1, p1[4], xt_rel=p1[1], xt_drop=0.0, train=False, xt_idcols=p1[2], xt_relonly=p1[3])
    nb = MO.N_DENSE + 2
    c0_, c1_ = dense_cols(xp[0], xp[1], xp[3]), dense_cols(p1[0], p1[1], p1[3])
    j = c1_.index(f"raw:{XADD}")
    nraw = sum(c.startswith("raw:") for c in c0_)
    check("dense_features: pop1 == xdrop_pop with exactly ONE column inserted at the end of the raw xt block (before the rank / z "
          "channels), equal to the batch's last xt column; width n_dense",
          j == nraw and c1_ == c0_[:nraw] + [f"raw:{XADD}"] + c0_[nraw:] and X1.shape[-1] == p1[4] and X0.shape[-1] == xp[4]
          and torch.equal(torch.cat([X1[..., :nb + j], X1[..., nb + j + 1:]], -1), X0)
          and torch.equal(X1[..., nb + j], b1["xt"][..., -1].float()), (X0.shape, X1.shape, j))
    # checkpoint cfg round trip
    rc, cmds = dry_cmds(LANE, 5, "pop1:23", r"abl_\w+_s23")
    a = parse(cmds["abl_pop1_s23"])
    lay = xt_layout("cvnd_pre2s", exclude=a.xt_exclude, rel_extra=a.xt_rel_extra, add=a.xt_add)
    m = _ranker(a, lay)
    Xd = m.dense(b1)
    m.set_norm(Xd[b1["cmask"]].mean(0), Xd[b1["cmask"]].std(0).clamp_min(1e-3))
    m.eval()
    l1 = m(b1, m.dense(b1))
    import io
    buf = io.BytesIO()
    torch.save({"cfg": dict(m.cfg), "state_dict": m.state_dict()}, buf)
    buf.seek(0)
    ck = torch.load(buf, map_location="cpu", weights_only=False)
    m2 = MO.Ranker(**ck["cfg"])
    m2.load_state_dict(ck["state_dict"])
    m2.eval()
    l2 = m2(b1, m2.dense(b1))
    cfg = ck["cfg"]
    check("checkpoint round trip (torch.save / load -> Ranker(**cfg) -> load_state_dict): logits bit-identical; cfg carries the pop1 "
          f"layout (xt_names ending in {XADD}, xt_rel / xt_idcols / xt_relonly of xdrop_pop, n_dense {p1[4]})",
          lay == p1 and torch.equal(l1, l2) and list(cfg["xt_names"]) == p1[0] and (cfg.get("xt_rel") or None) == p1[1]
          and (cfg.get("xt_idcols") or None) == p1[2] and (cfg.get("xt_relonly") or None) == p1[3] and cfg["n_dense"] == p1[4],
          (float((l1 - l2).abs().max()), cfg["n_dense"]))
    RES["layouts"]["pop1"] = p1


# ================================================================================================= GPU part
E2E_CD = ("--init {init} --epochs 0.004 --lr 2e-3 --warmup 0 --sched wsd --cooldown_frac 1.0 --muon --aux_pos 0.5 "
          "--aux_mode listwise --aux_tau 10 --aux_anneal 0.1 --cooldown_mix 0.2 --loss_norm feed --holdout 30000 "
          "--select_on holdout --eval_every 20 --seed 811 --no_test")
E2E_FS = ("--no_pos --aux_pos 0.5 --aux_mode listwise --aux_tau 10 --sim_bias --user0 --domain --holdout 30000 --select_on "
          "holdout --loss_norm feed --xt cvnd_pre2s --xt_drop 0 --muon --lr 1e-3 --sched wsd --cooldown_frac 0.2 "
          "--cooldown_mix 0.02 --aux_anneal 0.1 --retr_m 16 --attn_gate --no_test --save_last --save_holdout --cliplog 10 "
          "--seed 23 --train_feeds 20000 --epochs 1 --eval_every 25 --select_min_step 25")


def outputs(v):
    fs = [f"{ABL}/logs/{v}.log", f"{ABL}/logs/{v}.json", f"{ABL}/logs/{v}_cliplog.npz", f"{ABL}/logs/{v}_holdout.stdout"]
    for x in (v, f"{v}@last"):
        fs += [f"{ABL}/ckpt/{x}.pt", f"{ABL}/preds/nn_{x}_trainonly_val.npy", f"{ABL}/preds/nn_{x}_trainonly_test.npy",
               f"{ABL}/preds/holdout/nn_{x}_holdout.npy", f"{ABL}/preds/stale/nn_{x}_staleval.npy"]
    return fs


def clean(v):
    for f in outputs(v):
        if os.path.exists(f):
            os.remove(f)


def e2e_run(train_file, v, args, extra):
    rec = f"{SCR}/rec_{v}.npz"
    cmd = [PY, f"{HERE}/e2e_abl.py", train_file, rec, "--variant", v] + args.split() + extra
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ABL, env={**os.environ, "OMP_NUM_THREADS": str(NT)})
    open(f"{SCR}/{v}.stdout", "w").write(r.stdout + "\n" + r.stderr)
    P(f"  ran {v} ({time.time() - t0:.0f}s, rc {r.returncode})")
    assert r.returncode == 0, r.stderr[-3000:]
    return dict(np.load(rec, allow_pickle=True)), open(f"{ABL}/logs/{v}.log").read()


def metric_lines(log):
    return [re.sub(r" [\d.]+ min", "", ln) for ln in log.splitlines() if ln.startswith("step ") or ln.startswith("BEST")]


def ck_equal(a, b):
    ca = torch.load(a, map_location="cpu", weights_only=False)["state_dict"]
    cb = torch.load(b, map_location="cpu", weights_only=False)["state_dict"]
    return ca.keys() == cb.keys() and all(torch.equal(ca[k], cb[k]) for k in ca)


def same_run(ra, la, rb, lb, va, vb, what, holdout=False):
    same_b = np.array_equal(ra["fidx"], rb["fidx"]) and np.array_equal(ra["off"], rb["off"])
    nb = len(ra["off"]) - 1
    check(f"{what}: identical training-batch sequence ({nb} batches, incl. pool steps)", same_b and nb > 0, nb)
    check(f"{what}: identical train-mode model logits and loss-input logits",
          np.array_equal(ra["lg_model"], rb["lg_model"]) and np.array_equal(ra["lg_loss"], rb["lg_loss"]))
    ma, mb = metric_lines(la), metric_lines(lb)
    check(f"{what}: identical logged loss / val / holdout / BEST lines", ma == mb and len(ma) >= 2, mb[-1] if mb else "")
    ok = ck_equal(f"{ABL}/ckpt/{va}.pt", f"{ABL}/ckpt/{vb}.pt") and np.array_equal(
        np.load(f"{ABL}/preds/nn_{va}_trainonly_val.npy"), np.load(f"{ABL}/preds/nn_{vb}_trainonly_val.npy"))
    if holdout:
        ok = ok and ck_equal(f"{ABL}/ckpt/{va}@last.pt", f"{ABL}/ckpt/{vb}@last.pt") and all(np.array_equal(
            np.load(f"{ABL}/preds/holdout/nn_{va}{s}_holdout.npy"), np.load(f"{ABL}/preds/holdout/nn_{vb}{s}_holdout.npy"))
            for s in ("", "@last")) and np.array_equal(np.load(f"{ABL}/preds/nn_{va}@last_trainonly_val.npy"),
                                                       np.load(f"{ABL}/preds/nn_{vb}@last_trainonly_val.npy"))
    check(f"{what}: bit-identical checkpoint and val preds" + (" (selected + @last; holdout preds too)" if holdout else ""), ok)
    return mb


def g1_cd():
    P("\n== G1 end-to-end CD (w5 pattern: --init, 39 cooldown steps, evals at 20 / 39): frozen w5 vs abl at flags off")
    names = ["w5_ablt_cd_ref", "abl_t_cd_new"]
    for v in names:
        clean(v)
    ref_copy = f"{SCR}/train_w5_frozen.py"
    open(ref_copy, "w").write(open(REF_TRAIN).read())
    args = E2E_CD.format(init=INIT)
    ra, la = e2e_run(ref_copy, names[0], args, [])
    rb, lb = e2e_run(TRAIN_PY, names[1], args, [])
    mb = same_run(ra, la, rb, lb, names[0], names[1], "CD flags OFF vs frozen w5")
    RES["cd_lines"] = mb
    r = subprocess.run([PY, "train.py", "--variant", "abl_t_cd_bad"] + args.split() + ["--xt_exclude", POP],
                       capture_output=True, text=True, cwd=ABL, env={**os.environ, "OMP_NUM_THREADS": str(NT)}, timeout=600)
    check("abl flags with --init are refused (assert before any data loads)", r.returncode != 0
          and "no effect with --init" in r.stderr and not os.path.exists(f"{ABL}/logs/abl_t_cd_bad.json"), r.stderr[-120:].strip())
    for f in (f"{ABL}/logs/abl_t_cd_bad.log",):
        if os.path.exists(f):
            os.remove(f)
    r = subprocess.run([PY, "train.py", "--variant", "abl_t_cd_bad2"] + args.split() + ["--xt_add", XADD],
                       capture_output=True, text=True, cwd=ABL, env={**os.environ, "OMP_NUM_THREADS": str(NT)}, timeout=600)
    check("--xt_add with --init is refused (assert before any data loads)", r.returncode != 0
          and "no effect with --init" in r.stderr and not os.path.exists(f"{ABL}/logs/abl_t_cd_bad2.json"), r.stderr[-120:].strip())
    for f in (f"{ABL}/logs/abl_t_cd_bad2.log",):
        if os.path.exists(f):
            os.remove(f)
    for v in names:
        clean(v)


def g2_scratch():
    P("\n== G2 end-to-end FROM SCRATCH (smoke size, A_BASE + HYG, 20K train feeds, 1 epoch): flags off identity; arms ON")
    names = ["w5_ablt_fs_ref", "abl_t_fs_new", "abl_t_fs_xpop", "abl_t_fs_xrel", "abl_t_fs_rcat", "abl_t_fs_pop1"]
    for v in names:
        clean(v)
    ref_copy = f"{SCR}/train_w5_frozen.py"
    open(ref_copy, "w").write(open(REF_TRAIN).read())
    ra, la = e2e_run(ref_copy, names[0], E2E_FS, [])
    rb, lb = e2e_run(TRAIN_PY, names[1], E2E_FS, [])
    mb = same_run(ra, la, rb, lb, names[0], names[1], "from-scratch flags OFF vs frozen w5", holdout=True)
    RES["fs_lines"] = mb
    jj = json.load(open(f"{ABL}/logs/{names[1]}.json"))
    check("from-scratch run: selection at step >= 25 only; @last checkpoint + holdout preds written",
          jj["best_step"] >= 25 and os.path.exists(f"{ABL}/preds/holdout/nn_{names[1]}@last_holdout.npy"), jj["best_step"])
    # arms ON
    exp = {"abl_t_fs_xpop": (xt_layout("cvnd_pre2s", exclude=POP), ["--xt_exclude", POP]),
           "abl_t_fs_xrel": (xt_layout("cvnd_pre2s", rel_extra=XREL), ["--xt_rel_extra", XREL])}
    for v, (lay, extra) in exp.items():
        rc_, lc = e2e_run(TRAIN_PY, v, E2E_FS, extra)
        ck = torch.load(f"{ABL}/ckpt/{v}.pt", map_location="cpu", weights_only=False)
        cfg = ck["cfg"]
        got = (cfg["xt_names"], cfg.get("xt_rel"), cfg.get("xt_idcols"), cfg.get("xt_relonly"), cfg["n_dense"])
        check(f"{v}: checkpoint cfg (xt_names, xt_rel, xt_idcols, xt_relonly, n_dense) == the C1 layout",
              list(got[0]) == lay[0] and (got[1] or None) == lay[1] and (got[2] or None) == lay[2]
              and (got[3] or None) == lay[3] and got[4] == lay[4], (len(got[0]), got[1], got[2], got[3], got[4]))
        check(f"{v}: same training-batch sequence as flags OFF (the data order is shared); checkpoint differs (changes training)",
              np.array_equal(rc_["fidx"], rb["fidx"]) and not ck_equal(f"{ABL}/ckpt/{v}.pt", f"{ABL}/ckpt/{names[1]}.pt"))
        check(f"{v}: setup log line of the abl flag present",
              ("abl xt_exclude:" in lc) if "xpop" in v else ("abl xt_rel_extra:" in lc),
              [ln for ln in lc.splitlines() if ln.startswith("abl ")][:1])
    # second wave: retr_cat ON (zero-init second retrieval): exact no-op at init, then a different trajectory
    v = "abl_t_fs_rcat"
    rc_, lc = e2e_run(TRAIN_PY, v, E2E_FS, ["--retr_cat", "16"])
    ck = torch.load(f"{ABL}/ckpt/{v}.pt", map_location="cpu", weights_only=False)
    rk = sorted(k for k in ck["state_dict"] if k.startswith("retr_cat."))
    check(f"{v}: checkpoint cfg retr_cat 16 and retr_cat.* tensors (same names as retr.*); xt layout == the base layout",
          ck["cfg"].get("retr_cat") == 16 and rk and [k[9:] for k in rk] == sorted(k[5:] for k in ck["state_dict"] if k.startswith("retr."))
          and list(ck["cfg"]["xt_names"]) == xt_layout("cvnd_pre2s")[0] and ck["cfg"]["n_dense"] == xt_layout("cvnd_pre2s")[4],
          f"{len(rk)} retr_cat tensors")
    n0 = int(np.prod(rb["shapes"][0]))
    check(f"{v}: same training-batch sequence as flags OFF; FIRST-batch train-mode logits and loss-input logits bit-identical "
          "to flags OFF (exact no-op at init); later batches differ (Z learns); checkpoint differs",
          np.array_equal(rc_["fidx"], rb["fidx"]) and np.array_equal(rc_["shapes"], rb["shapes"])
          and np.array_equal(rc_["lg_model"][:n0], rb["lg_model"][:n0]) and np.array_equal(rc_["lg_loss"][:n0], rb["lg_loss"][:n0])
          and not np.array_equal(rc_["lg_model"], rb["lg_model"]) and not ck_equal(f"{ABL}/ckpt/{v}.pt", f"{ABL}/ckpt/{names[1]}.pt"),
          f"first batch {n0} logits; max |d| later {float(np.abs(rc_['lg_model'] - rb['lg_model']).max()):.3e}")
    check(f"{v}: setup log line present; params line larger than flags off",
          "abl retr_cat:" in lc and re.search(r"params: ([\d.]+)M", lc).group(1) > re.search(r"params: ([\d.]+)M", lb).group(1),
          [ln for ln in lc.splitlines() if ln.startswith("abl ")][:1])
    # pop1 arm ON (--xt_exclude <8 popularity columns> --xt_add xp_imp_c50): layout, data order, a different trajectory
    v = "abl_t_fs_pop1"
    lay = xt_layout("cvnd_pre2s", exclude=POP, add=XADD)
    rc_, lc = e2e_run(TRAIN_PY, v, E2E_FS, ["--xt_exclude", POP, "--xt_add", XADD])
    ck = torch.load(f"{ABL}/ckpt/{v}.pt", map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    got = (cfg["xt_names"], cfg.get("xt_rel"), cfg.get("xt_idcols"), cfg.get("xt_relonly"), cfg["n_dense"])
    check(f"{v}: checkpoint cfg (xt_names ending in {XADD}, xt_rel, xt_idcols, xt_relonly, n_dense) == the C5 pop1 layout",
          list(got[0]) == lay[0] and got[0][-1] == XADD and (got[1] or None) == lay[1] and (got[2] or None) == lay[2]
          and (got[3] or None) == lay[3] and got[4] == lay[4], (len(got[0]), got[1], got[2], got[3], got[4]))
    check(f"{v}: same training-batch sequence as flags OFF; checkpoint differs; selected + @last holdout preds written",
          np.array_equal(rc_["fidx"], rb["fidx"]) and not ck_equal(f"{ABL}/ckpt/{v}.pt", f"{ABL}/ckpt/{names[1]}.pt")
          and all(os.path.exists(f"{ABL}/preds/holdout/nn_{v}{s_}_holdout.npy") for s_ in ("", "@last")))
    check(f"{v}: setup log lines of both flags present (xt_exclude first, then xt_add); train-stage features line lists 27 columns",
          "abl xt_exclude:" in lc and "abl xt_add:" in lc and lc.index("abl xt_exclude:") < lc.index("abl xt_add:")
          and f"train-stage extra features (27):" in lc, [ln for ln in lc.splitlines() if ln.startswith("abl ")])
    # stale_pred.py reproduces the inputs from the cfg (xt_names filtered / xt_rel extended / xt_add appended) and writes stale-val preds
    r = subprocess.run([PY, "stale_pred.py", "abl_t_fs_xpop", "abl_t_fs_xrel@last", "abl_t_fs_pop1@last"], capture_output=True,
                       text=True, cwd=ABL, env={**os.environ, "OMP_NUM_THREADS": str(NT)}, timeout=1800)
    outs = [os.path.exists(f"{ABL}/preds/stale/nn_{x}_staleval.npy") for x in ("abl_t_fs_xpop", "abl_t_fs_xrel@last")]
    used = re.findall(r"overridden inputs used (\[.*?\])", r.stdout)
    check("stale_pred.py runs on the xdrop_pop and xrel_more checkpoints (inputs from the cfg): both stale-val files written; "
          "xdrop_pop uses only the pre2 overrides", r.returncode == 0 and all(outs) and used
          and "seq_cv_pop" not in used[0] and "pre2_fe_s" in used[0], (used, r.stderr[-200:].strip() if r.returncode else ""))
    check("stale_pred.py runs on the pop1 checkpoint (27 inputs from the cfg incl. xp_imp_c50): stale-val file written; only the "
          "pre2 overrides used (xp_imp_c50 has no private-like counterfactual: it stays at its val value)",
          r.returncode == 0 and os.path.exists(f"{ABL}/preds/stale/nn_abl_t_fs_pop1@last_staleval.npy") and len(used) == 3
          and used[2] == used[0], used[2:] if len(used) > 2 else used)
    for v in names:
        clean(v)
    for f in (f"{ABL}/preds/stale/stale_pred.log",):
        if os.path.exists(f):
            os.remove(f)


def main():
    t_all = time.time()
    P(f"== nn_x/abl ablation-screen tests, part {PART} ({time.ctime()}, CPU threads {NT})")
    if PART == "cpu":
        parse = load_parse(TRAIN_PY)
        c0_code(parse)
        c1_helpers()
        c2_dense()
        c3_arms_build(parse)
        c4_retr_cat(parse)
        c5_xt_add(parse)
    else:
        g1_cd()
        g2_scratch()
    RES["failed"] = FAIL
    P(f"\n== {len(RES['checks']) - len(FAIL)}/{len(RES['checks'])} checks passed; failed: {FAIL}; {time.time() - t_all:.0f}s")
    json.dump(RES, open(f"{ABL}/logs/test_abl_{PART}.json", "w"), indent=1, default=str)
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())
