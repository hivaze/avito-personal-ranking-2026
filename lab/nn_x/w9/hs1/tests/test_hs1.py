"""Tests of the nn_x/w9/hs1 WAVE-9 arm HS1 (feed-surface history view, cd_arms9.py; scratchpad w9/hs1.json + skeptic fixes).

  cpu part:  cd <repo> && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 .venv/bin/python nn_x/w9/hs1/tests/test_hs1.py cpu
  gpu part:  CUDA_VISIBLE_DEVICES=N OMP_NUM_THREADS=12 .venv/bin/python nn_x/w9/hs1/tests/test_hs1.py gpu   (GPU N or 5 only)

CPU part
  C0 code. (a) md5: data / cd_arms / cd_arms4-8 / blend / tests/mini_split / tests/e2e_w5 byte-identical to nn_x/w5 (md5s recorded
     at copy time in tests/w5_source_md5.txt, and nn_x/w5 unchanged since). (b) common.py AST-equal to nn_x/w5 up to string constants
     (the output path). (c) AST proof train.py: minus the HS1 statements (exact set: the cd_arms9 import, the two add_argument
     lines, the three `if a.hist_src or a.hist_et [...]` blocks, the two Ranker(...) keywords) and with the variant assert
     normalised == nn_x/w5/train.py. (d) model.py minus the two Ranker.__init__ keywords and `if hist_src or hist_et` == nn_x/w5.
     (e) holdout_eval / stale_pred minus the import + `views = {}` and with view_for(X, ...) -> X == nn_x/w5; finetune minus the
     import + the two guarded blocks == nn_x/w5. (f) lane.sh DRY: A:base == nn_x/w5 A:base except the w9_ variant prefix;
     A:hs1 / E:hs1 == base + --hist_src 1 only; A:hs1b / E:hs1b == base + --hist_et <ids>,9 only; E:base identical; some GPUs,6,7
     refused. (g) parse defaults: hist_src '' hist_et '' -> every guard False, view_spec None. (h) the nn_x/abl 2-epoch base pair
     abl_base_s23 / s24 == A_BASE + HYG (stored args; the only differences are flags absent on one side at their off values).
  C1 view (MiniSplit, real train + val feeds, CPU): history_view(keep_src=(1,)) == the numpy-filtered arrays (same order, kept
     events all source 1, H = per-feed s1 count, hist_off = cumsum, n_hist), shared arrays are the SAME objects, the original
     is untouched; gather(view) == gather of a split rebuilt from the numpy-filtered arrays (every tensor) and its candidate
     tensors == the original's; dead per-type columns: types 2 3 5 6 8 9 10 carry no source, so the s1 view holds types 1 / 4 / 7
     only; --hist_et view and the combined view; stats == direct recomputation; view_for identity / cache / xt sync;
     check_init_view.
  C2 empty-slice feeds (val feeds whose s1 slice is empty, with and without an empty full history) go through dense_features,
     retrieval and the full forward (eval and train mode) with finite outputs and the empty-history feature values; full-split
     empty-slice shares on train / val / test (the skeptic's 2.87 / 2.97 / 2.37%) recomputed from the flat files.
  C3 end-to-end on the CPU mini split (tests/e2e_cpu.py, ONE CPU thread: bit-reproducible): ref_off (nn_x/w5/train.py) vs new_off (this train.py): identical
     batch sequence (feeds, candidate rows, H, sources), model logits, loss-input logits, logged step lines, bit-identical
     checkpoints, val / holdout / @last preds. new_on (--hist_src 1, with test preds): completes; every gathered history
     event of every training batch has source 1; view log lines for train / val / test; ckpt args + cfg carry hist_src;
     preds (val, test, holdout, @last) exist; CPU holdout parity: view_for(ck args) + predict == the saved holdout preds;
     on != off (negative control).
GPU part (one of GPUs N; sequential subprocesses, 12 CPU threads)
  G1 flag-on smoke: lane.sh W5_SMOKE A:hs1:23 (20K train feeds, 1 epoch, evals every 25 steps) with test preds -> completes,
     view lines, outputs, holdout_meta == nn_x/w5's.
  G2 holdout parity: holdout_eval.py on the smoke checkpoint (selected and @last) == train.py's saved holdout preds; negative
     control: the same checkpoint scored on the FULL history differs.
  G3 stale_pred.py base check on the smoke checkpoint reproduces the stored train-only val preds (the view is applied).
  G4 finetune.py smoke (2 folds, 1 epoch, n2c_pre_v3 extras) on the smoke checkpoint: view lines for val / test, outputs.
Writes nn_x/w9/hs1/logs/test_hs1_<part>.log (+ .json).
"""
import ast
import collections
import copy
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time

PART = sys.argv[1] if len(sys.argv) > 1 else "cpu"
assert PART in ("cpu", "gpu"), "usage: test_hs1.py cpu|gpu"
_CVD = os.environ.get("CUDA_VISIBLE_DEVICES", "x")
if PART == "cpu":
    assert _CVD == "", "cpu part: run with CUDA_VISIBLE_DEVICES=''"
else:
    assert _CVD in ("4", "5"), "gpu part: CUDA_VISIBLE_DEVICES=<4 or 5, the wave-9 HS1 GPUs>"
NT = min(12, int(os.environ.get("OMP_NUM_THREADS", "12")))
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(NT)
HERE = os.path.dirname(os.path.abspath(__file__))
W9 = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(1, W9)
import warnings  # noqa: E402

warnings.filterwarnings("ignore", message=".*CUDA is not available.*")
warnings.filterwarnings("ignore", message=".*device_type of 'cuda'.*")
import cd_arms9 as C9  # noqa: E402
import data as D  # noqa: E402
from mini_split import MiniSplit, pick_train_feeds  # noqa: E402

ROOT = "<repo>"
W5 = f"{ROOT}/nn_x/w5"
PY = f"{ROOT}/.venv/bin/python"
SCR = os.environ.get("HS1_SCRATCH", "/tmp/scratchpad/hs1_tests")
os.makedirs(SCR, exist_ok=True)
os.makedirs(f"{W9}/logs", exist_ok=True)
LOG = open(f"{W9}/logs/test_hs1_{PART}.log", "w")
RES = {"checks": []}
FAIL = []
HS1_FLAGS = ("--hist_src", "--hist_et")
XT = ["seq_cv_invd20", "seq_cv_pop", "seq_cv_norm", "it_hist_all_luo", "it_hist_et1_luo", "it_hist_et2_luo", "it_hist_et3_luo",
      "it_hist_et4_luo", "ctx_nd_hcent", "ctx_nd_knn_f1", "ctx_nd_knn_h", "ctx_nd_knn_h_dev", "ctx_nd_knn_u", "ctx_nd_knn_u_dev",
      "ctx_nd_max", "ctx_nd_top5h", "ctx_nd_top5u", "ctx_nd_ucent", "ctx_hn_top1w", "ctx_hn_rrw", "ctx_hn_bestrk", "ctx_hn_margin",
      "ctx_hn_soft05", "ctx_hn_e1_rrw", "ctx_hs_mutual", "ctx_hs_k128", "ctx_hs_e1", "ctx_hs_t10", "pre2_fe_z", "pre2_new_z",
      "pre2_fe_s", "pre2_new_s", "pre2_hmax32", "has:pre2_fe"]


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
    import argparse
    import cd_arms as CA
    tree = ast.parse(open(path).read())
    fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "parse"][0]
    ns = {"argparse": argparse, "POOL_P_DEFAULT": CA.POOL_P_DEFAULT}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "train_parse", "exec"), ns)

    def parse(argv):
        sys.argv = ["train.py"] + argv
        return ns["parse"]()
    return parse


# ================================================================================================= AST strippers
class StripHS1(ast.NodeTransformer):
    """Removes the HS1 statements of one file; counts them (the tests assert the exact set)."""

    def __init__(self, ifs=(), rm_call=None, rm_kw=()):
        self.ifs, self.rm_call, self.rm_kw = set(ifs), rm_call, set(rm_kw)
        self.removed = collections.Counter()

    def visit_ImportFrom(self, n):
        if n.module == "cd_arms9":
            self.removed["import cd_arms9"] += 1
            return None
        return n

    def visit_Expr(self, n):
        v = n.value
        if (isinstance(v, ast.Call) and isinstance(v.func, ast.Attribute) and v.func.attr == "add_argument" and v.args
                and isinstance(v.args[0], ast.Constant) and v.args[0].value in HS1_FLAGS):
            self.removed[f"add_argument {v.args[0].value}"] += 1
            return None
        return self.generic_visit(n)

    def visit_Assign(self, n):
        if ast.unparse(n) == "views = {}":
            self.removed["views = {}"] += 1
            return None
        return self.generic_visit(n)

    def visit_If(self, n):
        t = ast.unparse(n.test)
        if t in self.ifs:
            self.removed[f"if {t}"] += 1
            return None
        return self.generic_visit(n)

    def visit_Call(self, n):
        if self.rm_call and isinstance(n.func, ast.Name) and n.func.id == self.rm_call:
            self.removed[f"{self.rm_call}(...) -> arg"] += 1
            return self.visit(n.args[0])
        if self.rm_kw and isinstance(n.func, ast.Name) and n.func.id == "Ranker":
            kept = [k for k in n.keywords if k.arg not in self.rm_kw]
            self.removed["Ranker keyword"] += len(n.keywords) - len(kept)
            n.keywords = kept
        return self.generic_visit(n)

    def visit_FunctionDef(self, n):
        if self.rm_kw and n.name == "__init__":
            a = n.args
            idx = [i for i, x in enumerate(a.args) if x.arg in self.rm_kw]
            if idx:
                n_def = len(a.defaults)
                off = len(a.args) - n_def
                a.defaults = [d for i, d in enumerate(a.defaults) if (i + off) not in idx]
                a.args = [x for i, x in enumerate(a.args) if i not in idx]
                self.removed["__init__ keyword"] += len(idx)
        return self.generic_visit(n)


class NormStr(ast.NodeTransformer):
    def visit_Constant(self, n):
        return ast.Constant(value="S") if isinstance(n.value, str) else n


def ast_eq(a, b):
    return ast.dump(a) == ast.dump(b)


def dry_cmds(script, gpu, jobs, env=None):
    out = subprocess.run(["bash", script, str(gpu), jobs], capture_output=True, text=True,
                         env={**os.environ, "DRY": "1", **(env or {})})
    cmds = {}
    for line in out.stdout.splitlines():
        if " train.py " in line:
            argv = line.split(" train.py ", 1)[1].split()
            cmds[argv[argv.index("--variant") + 1]] = argv
    return out.returncode, cmds


# ================================================================================================= CPU part
def c0_code():
    P("\n== C0 code: md5 identity, AST flag-off proofs, lane.sh DRY, parse defaults, abl base pair args")
    rec = dict(line.split()[::-1] for line in open(f"{HERE}/w5_source_md5.txt"))
    for f in ("data.py", "cd_arms.py", "cd_arms4.py", "cd_arms5.py", "cd_arms6.py", "cd_arms7.py", "cd_arms8.py", "blend.py",
              "tests/mini_split.py", "tests/e2e_w5.py"):
        check(f"{f} byte-identical to nn_x/w5 (copy-time md5; nn_x/w5 file unchanged since)",
              md5(f"{W9}/{f}") == rec[f] == md5(f"{W5}/{f}"), rec[f])
    w5_now = {f: md5(f"{W5}/{f}") == rec[f] for f in ("train.py", "model.py", "common.py", "finetune.py", "holdout_eval.py",
                                                       "stale_pred.py", "lane.sh")}
    check("nn_x/w5 reference files unchanged since the copy (train / model / common / finetune / holdout_eval / stale_pred / lane)",
          all(w5_now.values()), w5_now)
    # (b) common.py up to strings
    check("common.py AST-equal to nn_x/w5 up to string constants (output path nn_x/w9/hs1)",
          ast_eq(NormStr().visit(ast.parse(open(f"{W9}/common.py").read())), NormStr().visit(ast.parse(open(f"{W5}/common.py").read()))))
    # (c) train.py
    REF_ASSERT = ('assert a.variant.startswith("w5_"), "nn_x/w5 copy: variant names must start with w5_"',
                  'assert a.variant.startswith(("w5_", "w9_")), "nn_x/w9/hs1 copy: variant names must start with w5_ or w9_"')
    ref_src, new_src = open(f"{W5}/train.py").read(), open(f"{W9}/train.py").read()
    check("train.py: the variant-prefix assert is the one normalised statement (w5_ -> w5_ | w9_)",
          ref_src.count(REF_ASSERT[0]) == 1 and new_src.count(REF_ASSERT[1]) == 1)
    ref_src = ref_src.replace(*REF_ASSERT)
    st = StripHS1(ifs=("a.hist_src or a.hist_et", "a.hist_src or a.hist_et or ck0['cfg'].get('hist_src') or ck0['cfg'].get('hist_et')"),
                  rm_kw=("hist_src", "hist_et"))
    new_t = strip_doc(st.visit(ast.parse(new_src)))
    want = collections.Counter({"import cd_arms9": 1, "add_argument --hist_src": 1, "add_argument --hist_et": 1,
                                "if a.hist_src or a.hist_et": 2,
                                "if a.hist_src or a.hist_et or ck0['cfg'].get('hist_src') or ck0['cfg'].get('hist_et')": 1,
                                "Ranker keyword": 2})
    check("AST: train.py minus the HS1 statements (exact set) == nn_x/w5/train.py (flag-off path unchanged)",
          ast_eq(new_t, strip_doc(ast.parse(ref_src))) and st.removed == want, dict(st.removed))
    # (d) model.py
    st = StripHS1(ifs=("hist_src or hist_et",), rm_kw=("hist_src", "hist_et"))
    new_t = strip_doc(st.visit(ast.parse(open(f"{W9}/model.py").read())))
    check("AST: model.py minus the two Ranker.__init__ keywords and `if hist_src or hist_et` == nn_x/w5/model.py",
          ast_eq(new_t, strip_doc(ast.parse(open(f"{W5}/model.py").read())))
          and st.removed == collections.Counter({"__init__ keyword": 2, "if hist_src or hist_et": 1}), dict(st.removed))
    # (e) holdout_eval / stale_pred / finetune
    for f, kw in (("holdout_eval.py", dict(rm_call="view_for")), ("stale_pred.py", dict(rm_call="view_for")),
                  ("finetune.py", dict(ifs=("ck['args'].get('hist_src') or ck['args'].get('hist_et')",)))):
        st = StripHS1(**kw)
        new_t = strip_doc(st.visit(ast.parse(open(f"{W9}/{f}").read())))
        ref_t = strip_doc(ast.parse(open(f"{W5}/{f}").read()))
        want = (collections.Counter({"import cd_arms9": 1, "views = {}": 1, "view_for(...) -> arg": 1 if f == "holdout_eval.py" else 2})
                if "rm_call" in kw else collections.Counter({"import cd_arms9": 1,
                                                             "if ck['args'].get('hist_src') or ck['args'].get('hist_et')": 2}))
        check(f"AST: {f} minus the HS1 statements (exact set) == nn_x/w5/{f}", ast_eq(new_t, ref_t) and st.removed == want,
              dict(st.removed))
    # (f) lane.sh
    parse = load_parse(f"{W9}/train.py")
    rc, c9 = dry_cmds(f"{W9}/lane.sh", 4, "A:base:23 A:base:24 A:hs1:23 A:hs1:24 A:hs1b:23 A:hs1b:24 E:base:951 E:hs1:951 E:hs1:952 "
                      "E:hs1b:951")
    rc5, c5 = dry_cmds(f"{W5}/lane.sh", 4, "A:base:23 A:base:24 E:base:951", env={"W5_GPUS": "4"})
    check("lane.sh DRY: 10 commands; nn_x/w5 lane.sh DRY: 3 commands", rc == 0 and rc5 == 0 and len(c9) == 10 and len(c5) == 3,
          (rc, sorted(c9), rc5, sorted(c5)))
    ns9 = {k: vars(parse(v)) for k, v in c9.items()}
    ns5 = {k: vars(parse(v)) for k, v in c5.items()}

    def diff(a, b, skip=("variant",)):
        return {k: (a.get(k), b.get(k)) for k in set(a) | set(b) if k not in skip and a.get(k) != b.get(k)}
    for s in (23, 24):
        check(f"A:base:{s}: w9_base_s{s} == nn_x/w5 w5_base_s{s} except the variant name",
              diff(ns5[f"w5_base_s{s}"], ns9[f"w9_base_s{s}"]) == {} and ns9[f"w9_base_s{s}"]["variant"] == f"w9_base_s{s}")
        check(f"A:hs1:{s} differs from A:base ONLY by hist_src '1'",
              diff(ns9[f"w9_base_s{s}"], ns9[f"w9_hs1_s{s}"]) == {"hist_src": ("", "1")}, diff(ns9[f"w9_base_s{s}"], ns9[f"w9_hs1_s{s}"]))
        check(f"A:hs1b:{s} differs from A:base ONLY by hist_et '2,3,4,7,9'",
              diff(ns9[f"w9_base_s{s}"], ns9[f"w9_hs1b_s{s}"]) == {"hist_et": ("", "2,3,4,7,9")})
    check("E:base:951 identical to nn_x/w5 E:base:951 (same variant name w5_e10_base_s951)",
          diff(ns5["w5_e10_base_s951"], ns9["w5_e10_base_s951"], skip=()) == {})
    for s in (951, 952):
        check(f"E:hs1:{s} == E_BASE + HYG + seed + hist_src '1' only",
              diff(ns9["w5_e10_base_s951"], ns9[f"w5_e10_hs1_s{s}"], skip=("variant", "seed")) == {"hist_src": ("", "1")}
              and ns9[f"w5_e10_hs1_s{s}"]["seed"] == s and ns9[f"w5_e10_hs1_s{s}"]["variant"] == f"w5_e10_hs1_s{s}")
    check("E:hs1b:951 == E_BASE + HYG + seed + hist_et only",
          diff(ns9["w5_e10_base_s951"], ns9["w5_e10_hs1b_s951"]) == {"hist_et": ("", "2,3,4,7,9")})
    refused = [subprocess.run(["bash", f"{W9}/lane.sh", str(g), "A:hs1:23"], capture_output=True, text=True,
                              env={**os.environ, "DRY": "1"}).returncode != 0 for g in (0, 1, 2, 3, 6, 7)]
    check("lane.sh refuses GPUs N <ids> (DRY mode too)", all(refused), refused)
    RES["lane_cmds"] = {k: " ".join(v) for k, v in c9.items()}
    # (g) defaults
    a_def = vars(parse(["--variant", "w9_x"]))
    check("parse(): hist_src '' and hist_et '' by default; guard False; view_spec None",
          a_def["hist_src"] == "" and a_def["hist_et"] == "" and not (a_def["hist_src"] or a_def["hist_et"]) and C9.view_spec(a_def) is None)
    check("view_spec: '1' -> ((1,), ()); '2,3,4,7,9' -> ((), (2,3,4,7,9)); both; bad codes refused",
          C9.view_spec({"hist_src": "1"}) == ((1,), ()) and C9.view_spec({"hist_et": "2,3,4,7,9"}) == ((), (2, 3, 4, 7, 9))
          and C9.view_spec({"hist_src": "1", "hist_et": "1,4"}) == ((1,), (1, 4)) and _raises(lambda: C9.view_spec({"hist_et": "11"})))
    # (h) abl base pair
    for s in (23, 24):
        j = json.load(open(f"{ROOT}/nn_x/abl/logs/abl_base_s{s}.json"))
        d = diff(ns9[f"w9_base_s{s}"], j["args"])
        only_off = all((v[0] in ("", 0.0, 0, None) and v[1] in ("", 0.0, 0, None)) for v in d.values())
        check(f"nn_x/abl abl_base_s{s} stored args == A_BASE + HYG (seed {s}); differences only flags absent on one side at off values",
              only_off and j["args"]["seed"] == s and j["args"]["epochs"] == 2.0 and j["args"]["select_min_step"] == 12000, d)
        have = all(os.path.exists(f"{ROOT}/nn_x/abl/preds/{p}") for p in
                   (f"nn_abl_base_s{s}_trainonly_val.npy", f"nn_abl_base_s{s}@last_trainonly_val.npy",
                    f"holdout/nn_abl_base_s{s}_holdout.npy", f"holdout/nn_abl_base_s{s}@last_holdout.npy"))
        check(f"abl_base_s{s}: selected + @last val and holdout preds exist (the 2-epoch screen references)", have)
    meta_ok = np.array_equal(np.load(f"{ROOT}/nn_x/abl/preds/holdout/holdout_meta.npy"), np.load(f"{W5}/preds/holdout/holdout_meta.npy"))
    check("nn_x/abl holdout_meta == nn_x/w5 holdout_meta (same fixed 30K holdout)", meta_ok)


def _raises(fn):
    try:
        fn()
    except AssertionError:
        return True
    return False


def ref_view(ms, keep_src=(), keep_et=()):
    """numpy reference: filtered + compacted copy of a MiniSplit (independent of cd_arms9)."""
    es, et = ms.h_es.numpy(), ms.h_et.numpy()
    keep = np.ones(len(es), bool)
    if keep_src:
        keep &= np.isin(es, keep_src)
    if keep_et:
        keep &= np.isin(et, keep_et)
    H = ms.H.numpy().astype(np.int64)
    fid = np.repeat(np.arange(ms.n), H)
    H2 = np.bincount(fid[keep], minlength=ms.n)
    r = copy.copy(ms)
    r.h_emb, r.h_cat, r.h_et, r.h_es = ms.h_emb[keep], ms.h_cat[keep], ms.h_et[keep], ms.h_es[keep]
    r.H = torch.from_numpy(H2.astype(ms.H.numpy().dtype))
    r.H_np = H2
    r.hist_off = torch.from_numpy((np.cumsum(H2) - H2).astype(ms.hist_off.numpy().dtype))
    r.n_hist = int(H2.sum())
    if r.n_hist == 0:
        r.h_emb, r.h_cat, r.h_et, r.h_es, r.n_hist = ms.h_emb[:1], ms.h_cat[:1], ms.h_et[:1], ms.h_es[:1], 1
    return r, keep


def c1_view():
    P("\n== C1 history_view on real MiniSplit train / val feeds (CPU)")
    t0 = time.time()
    feeds = pick_train_feeds(48, 0)
    ms_tr = MiniSplit("train", feeds, xt_names=XT[:3])
    fv = pl.read_parquet(f"{ROOT}/data/flat/val/feeds.parquet", columns=["H"])
    Hv = fv["H"].to_numpy().astype(np.int64)
    rng = np.random.default_rng(3)
    vfeeds = np.unique(np.concatenate([rng.choice(len(Hv), 200, replace=False), np.where(Hv == 0)[0][:10]]))
    ms_va = MiniSplit("val", vfeeds, xt_names=XT[:3])
    for ms, nm in ((ms_tr, "train"), (ms_va, "val")):
        H0, es0, n0 = ms.H.clone(), ms.h_es.clone(), ms.n_hist
        v, st = C9.history_view(ms, keep_src=(1,))
        r, keep = ref_view(ms, keep_src=(1,))
        arrays = all(torch.equal(getattr(v, k), getattr(r, k)) for k in ("h_emb", "h_cat", "h_et", "h_es", "H", "hist_off"))
        check(f"{nm}: view arrays == numpy-filtered compacted arrays (order preserved), H == per-feed s1 count, hist_off == cumsum, "
              f"n_hist", arrays and np.array_equal(v.H_np, r.H_np) and v.n_hist == r.n_hist == int(keep.sum()),
              f"events {n0} -> {v.n_hist}, feeds {ms.n}")
        check(f"{nm}: every kept event has source 1; kept types subset of {{1, 4, 7}} (types 2 3 5 6 8 9 10 carry no source: dead "
              f"per-type columns)", bool((v.h_es[:v.n_hist] == 1).all()) and set(v.h_et[:v.n_hist].tolist()) <= {1, 4, 7}
              and all(st["kept_by_type"][t] == 0 for t in (2, 3, 5, 6, 8, 9, 10)), f"kept by type {st['kept_by_type']}")
        shared = all(getattr(v, k) is getattr(ms, k) for k in ("c_emb", "c_cat", "c_f", "table", "user_emb", "M", "cand_off", "xt",
                                                              "user0", "c_tgt", "M_np", "cand_off_np"))
        check(f"{nm}: candidate / table / xt / user arrays are the SAME objects; the original split is untouched",
              shared and torch.equal(ms.H, H0) and torch.equal(ms.h_es, es0) and ms.n_hist == n0 and v is not ms)
        # stats
        H2 = r.H_np
        want = dict(events_before=n0, events_after=int(H2.sum()), feeds_empty_before=int((H0.numpy() == 0).sum()),
                    feeds_empty_after=int((H2 == 0).sum()), feeds_emptied=int(((H2 == 0) & (H0.numpy() > 0)).sum()),
                    feeds_changed=int((H2 != H0.numpy()).sum()))
        check(f"{nm}: stats == direct recomputation", all(st[k] == want[k] for k in want), {k: st[k] for k in want})
        # gather equality on a batch that includes emptied feeds
        emptied = np.where((H2 == 0) & (H0.numpy() > 0))[0]
        fb = np.unique(np.concatenate([np.arange(min(ms.n, 64)), emptied[:16]]))
        bv, br, b0 = v.gather(fb, with_pos=True), r.gather(fb, with_pos=True), ms.gather(fb, with_pos=True)
        hk = ("he_raw", "hemb_ok", "hcat", "het", "hes", "hmask", "H", "hidx")
        ck_ = ("ce_raw", "cmiss", "ccat", "cf", "cmask", "ci", "M", "ue_raw", "user0", "xt") + (("y", "pos") if nm == "train" else ("y",))
        check(f"{nm}: gather(view) == gather(numpy-filtered split) on every history tensor; candidate tensors == the original's",
              all(torch.equal(bv[k], br[k]) for k in hk) and all(torch.equal(bv[k], b0[k]) for k in ck_ if k in b0)
              and bool((bv["hes"][bv["hmask"]] == 1).all()) and bool((bv["H"][torch.from_numpy(np.isin(fb, emptied))] == 0).all())
              and bool((~bv["hmask"][torch.from_numpy(np.isin(fb, emptied))]).all()), f"batch {len(fb)} feeds, {len(emptied[:16])} emptied")
        # et view and combined view
        v2, st2 = C9.history_view(ms, keep_et=(2, 3, 4, 7, 9))
        r2, keep2 = ref_view(ms, keep_et=(2, 3, 4, 7, 9))
        v3, st3 = C9.history_view(ms, keep_src=(1,), keep_et=(1, 4))
        r3, keep3 = ref_view(ms, keep_src=(1,), keep_et=(1, 4))
        check(f"{nm}: --hist_et 2,3,4,7,9 view (model-5 filter) == numpy reference; kept types subset; combined src 1 & et 1,4 == "
              f"reference", all(torch.equal(getattr(v2, k), getattr(r2, k)) for k in ("h_emb", "h_et", "h_es", "H", "hist_off"))
              and set(v2.h_et[:v2.n_hist].tolist()) <= {2, 3, 4, 7, 9}
              and all(torch.equal(getattr(v3, k), getattr(r3, k)) for k in ("h_emb", "h_et", "h_es", "H", "hist_off"))
              and set(v3.h_et[:v3.n_hist].tolist()) <= {1, 4} and bool((v3.h_es[:v3.n_hist] == 1).all()),
              f"et view kept {st2['kept_share']:.4f} (empty after {st2['feeds_empty_after_share']:.4f}); combined kept {st3['kept_share']:.4f}")
    # view_for
    cache = {}
    same = C9.view_for(ms_va, {"hist_src": "", "hist_et": ""}, cache) is ms_va and cache == {}
    v = C9.view_for(ms_va, {"hist_src": "1"}, cache)
    ms_va.xt = ms_va.xt * 2
    v_again = C9.view_for(ms_va, {"hist_src": "1"}, cache)
    check("view_for: identity (same object) without a stored view; cached view per spec; xt re-synced from the split on every call",
          same and v is v_again and v.xt is ms_va.xt and len(cache) == 1 and bool((v.h_es[:v.n_hist] == 1).all()))
    check("check_init_view: same spec passes, a different spec (or none) is refused",
          not _raises(lambda: C9.check_init_view({"hist_src": "1", "hist_et": ""}, {"hist_src": "1", "hist_et": ""}))
          and _raises(lambda: C9.check_init_view({"hist_src": "", "hist_et": ""}, {"hist_src": "1", "hist_et": ""}))
          and _raises(lambda: C9.check_init_view({"hist_src": "1", "hist_et": ""}, {})))
    P(f"  C1 {time.time() - t0:.0f}s")


def c2_empty():
    P("\n== C2 empty-slice feeds through dense_features / retrieval / forward; full-split empty-slice shares")
    import model as Mo
    t0 = time.time()
    fv = pl.read_parquet(f"{ROOT}/data/flat/val/feeds.parquet", columns=["H"])
    H = fv["H"].to_numpy().astype(np.int64)
    es = np.load(f"{ROOT}/data/flat/val/hist_event_source.npy")
    s1 = np.bincount(np.repeat(np.arange(len(H)), H)[es == 1], minlength=len(H))
    rng = np.random.default_rng(5)
    emp = rng.choice(np.where((s1 == 0) & (H > 0))[0], 40, replace=False)
    zero = rng.choice(np.where(H == 0)[0], 8, replace=False)
    norm = rng.choice(np.where(s1 > 0)[0], 40, replace=False)
    feeds = np.unique(np.concatenate([emp, zero, norm]))
    ms = MiniSplit("val", feeds, xt_names=XT)
    v, st = C9.history_view(ms, keep_src=(1,))
    xt_rel = [XT.index(n) for n in ("seq_cv_invd20", "it_hist_all_luo", "pre2_fe_s", "pre2_new_s", "pre2_hmax32")]
    xt_relonly = [XT.index(n) for n in ("pre2_fe_s", "pre2_new_s", "pre2_hmax32")]
    xt_idcols = [i for i, n in enumerate(XT) if n.startswith(("seq_cv", "it_hist"))]
    n_dense = Mo.N_DENSE + 2 + len(XT) + 2 * len(xt_rel) - len(xt_relonly)
    torch.manual_seed(0)
    m = Mo.Ranker(d=128, heads=4, hist_layers=1, blocks=2, drop=0.1, use_pos_tower=False, aux_pos=True, ff=2, sim_bias=1,
                  domain=True, xt_names=XT, n_dense=n_dense, xt_rel=xt_rel, xt_drop=0.0, xt_idcols=xt_idcols, xt_relonly=xt_relonly,
                  attn_gate=True, split_kv=True, retr_m=16, hist_src="1")
    m.set_norm(torch.zeros(n_dense), torch.ones(n_dense))
    check("Ranker(hist_src='1'): cfg carries hist_src / hist_et; nn_x/w5 model.py refuses such a cfg (TypeError = loud failure)",
          m.cfg.get("hist_src") == "1" and m.cfg.get("hist_et") == "" and _type_error(lambda: _w5_ranker(m.cfg)))
    m0 = Mo.Ranker(d=128, heads=4, hist_layers=1, blocks=2, drop=0.1, use_pos_tower=False, aux_pos=True, ff=2, sim_bias=1,
                   domain=True, xt_names=XT, n_dense=n_dense, xt_rel=xt_rel, xt_drop=0.0, xt_idcols=xt_idcols, xt_relonly=xt_relonly,
                   attn_gate=True, split_kv=True, retr_m=16)
    check("Ranker without the keywords: cfg has no hist_src / hist_et key (flag-off checkpoints keep the nn_x/w5 cfg)",
          "hist_src" not in m0.cfg and "hist_et" not in m0.cfg)
    fb = np.arange(ms.n)
    b = v.gather(fb)
    pos = np.isin(feeds, emp)
    X = Mo.dense_features(b, n_dense, xt_rel=xt_rel, xt_idcols=xt_idcols, xt_relonly=xt_relonly)
    cm = b["cmask"]
    # dense feature columns: 15 cand f stats, cmiss, log norm, cos_u, dot_u, then smax smax8 smax32 smean lse top3 (idx 19..24)
    sm = X[..., 19:25]
    ok_emp = bool(torch.isfinite(X[cm]).all()) and bool((b["H"][pos] == 0).all()) and bool((~b["hmask"][pos]).all()) \
        and bool((sm[pos][cm[pos]][:, [0, 1, 2, 4, 5]] == -1.5).all()) and bool((sm[pos][cm[pos]][:, 3] == 0).all())
    check("dense_features on the view: finite everywhere; emptied feeds have H 0, no history mask, smax / smax8 / smax32 / lse / top3 "
          "= -1.5 and smean 0 (the empty-history values of the base)", ok_emp,
          f"{int(pos.sum())} emptied feeds, {int(np.isin(feeds, zero).sum())} empty before, {int(np.isin(feeds, norm).sum())} with s1 events")
    m.eval()
    with torch.no_grad():
        lg_e = m(b, use_pos=False)
    m.train()
    lg_t = m(b, use_pos=False)
    ok_f = bool(torch.isfinite(lg_e[cm]).all()) and bool(torch.isfinite(lg_t[cm]).all()) and bool(torch.isfinite(m.aux_out[cm]).all())
    # retrieval alone on the emptied feeds: all mass on the sink -> output == Z(0) == bias-only (Z zero-init -> 0)
    with torch.no_grad():
        ce, he, hv = b["ce"], b["he"], b["hemb_ok"]
        S = torch.einsum("bmd,bkd->bmk", ce, he) * hv[:, None, :]
        r = m.retr(ce, he, hv, b["het"], b["hes"], S, b["cmiss"])
    check("full forward (eval + train mode, aux head) and the TabR retrieval run on the emptied feeds with finite outputs; retrieval "
          "output on emptied feeds == the zero-init Z bias (all mass on the sink)",
          ok_f and bool(torch.isfinite(r).all()) and bool((r[pos].abs() < 1e-6).all()), f"logits {lg_e.shape}")
    # full-split shares
    shares = {}
    for sp in ("train", "val", "test"):
        f = pl.read_parquet(f"{ROOT}/data/flat/{sp}/feeds.parquet", columns=["H"])
        Hs = f["H"].to_numpy().astype(np.int64)
        e = np.load(f"{ROOT}/data/flat/{sp}/hist_event_source.npy")
        fid = np.repeat(np.arange(len(Hs), dtype=np.int32), Hs)
        c1 = np.bincount(fid[e == 1], minlength=len(Hs))
        shares[sp] = dict(feeds=int(len(Hs)), s1_event_share=float((e == 1).mean()), empty_before=float((Hs == 0).mean()),
                          empty_after=float((c1 == 0).mean()), emptied=float(((c1 == 0) & (Hs > 0)).mean()))
        del fid, e, c1
    RES["empty_slice_shares"] = shares
    P(f"  info: full-split s1 view: " + "; ".join(f"{sp}: s1 share {d['s1_event_share']:.4f}, empty before {d['empty_before']:.4f}, "
                                                f"after {d['empty_after']:.4f}, emptied {d['emptied']:.4f}" for sp, d in shares.items()))
    check("empty-slice share after the s1 view: val 2.97% / test 2.37% exactly as the skeptic (+-0.05 pt); train within 0.2 pt of the "
          "skeptic's sampled 2.87% (exact full-train value reported above)",
          abs(shares["train"]["empty_after"] - 0.0287) < 0.002 and abs(shares["val"]["empty_after"] - 0.0297) < 0.0005
          and abs(shares["test"]["empty_after"] - 0.0237) < 0.0005)
    P(f"  C2 {time.time() - t0:.0f}s")


def _type_error(fn):
    try:
        fn()
    except TypeError:
        return True
    return False


def _w5_ranker(cfg):
    import importlib.util
    spec = importlib.util.spec_from_file_location("w5_model", f"{W5}/model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Ranker(**cfg)


E2E = ("--no_pos --aux_pos 0.5 --aux_mode listwise --aux_tau 10 --sim_bias --user0 --domain --holdout 30000 --select_on holdout "
       "--loss_norm feed --xt cvnd_pre2s --xt_drop 0 --muon --lr 1e-3 --sched wsd --cooldown_frac 0.3 --cooldown_mix 0.3 "
       "--aux_anneal 0.1 --retr_m 16 --attn_gate --epochs 2 --eval_every 10 --warmup 5 --save_last --save_holdout "
       "--select_min_step 8 --cliplog 4 --seed 23")


def run_e2e(train_file, v, extra):
    out = f"{SCR}/{v}"
    shutil.rmtree(out, ignore_errors=True)
    rec = f"{out}_rec.npz"
    cmd = [PY, f"{HERE}/e2e_cpu.py", train_file, rec, out, "--variant", v] + E2E.split() + extra
    t0 = time.time()
    # ONE CPU thread: with 12 threads the CPU backward / optimiser is not bit-reproducible run to run (nn_x/w5/train.py differs from
    # itself from batch ~8); single-threaded runs are bit-identical (checked twice on the reference file, ~30 s per run)
    r = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"})
    open(f"{out}.stdout", "w").write(r.stdout + "\n--- stderr ---\n" + r.stderr)
    return dict(rc=r.returncode, out=out, rec=rec, sec=time.time() - t0, stdout=r.stdout)


def step_lines(path):
    return [re.sub(r"w[59]_t9e2e_\w+", "V", re.sub(r" \d+\.\d min", " T min", ln)) for ln in open(path).read().splitlines()
            if ln.startswith(("step ", "BEST step", "wave-5 LAST", "wave-5 saved", "holdout:", "p_random:", "train feeds", "cooldown mix"))]


def c3_e2e():
    P("\n== C3 end-to-end on the CPU mini split (tests/e2e_cpu.py): ref_off (nn_x/w5) vs new_off; new_on smoke")
    runs = {"ref_off": run_e2e(f"{W5}/train.py", "w5_t9e2e_ref_off", ["--no_test"]),
            "new_off": run_e2e(f"{W9}/train.py", "w5_t9e2e_new_off", ["--no_test"]),
            "new_on": run_e2e(f"{W9}/train.py", "w9_t9e2e_new_on", ["--hist_src", "1"])}
    for k, r in runs.items():
        check(f"e2e {k} completed (rc 0, {r['sec']:.0f}s)", r["rc"] == 0, r["out"] + ".stdout" if r["rc"] else "")
    if any(r["rc"] for r in runs.values()):
        return
    ro, no = runs["ref_off"], runs["new_off"]
    a, b = np.load(ro["rec"]), np.load(no["rec"])
    keys = ("fidx", "off", "shapes", "ci", "H", "hes", "lg_model", "lg_loss", "n_model", "n_loss", "n_eval", "n_predict")
    same = {k: np.array_equal(a[k], b[k]) for k in keys}
    check("new_off == ref_off: identical batch sequence (feeds, rows, H, sources), model logits, loss-input logits, eval / predict counts",
          all(same.values()), f"{dict(same)}; {int(a['n_model'])} training batches, {int(a['n_eval'])} evals")
    vo, vn = "w5_t9e2e_ref_off", "w5_t9e2e_new_off"
    la, lb = step_lines(f"{ro['out']}/logs/{vo}.log"), step_lines(f"{no['out']}/logs/{vn}.log")
    check("new_off == ref_off: logged step / BEST / holdout lines identical (times masked)", la == lb and len(la) > 5, f"{len(la)} lines")
    eq_ck = {}
    for suf in ("", "@last"):
        ca = torch.load(f"{ro['out']}/ckpt/{vo}{suf}.pt", map_location="cpu", weights_only=False)
        cb = torch.load(f"{no['out']}/ckpt/{vn}{suf}.pt", map_location="cpu", weights_only=False)
        eq_ck[suf or "selected"] = (ca["state_dict"].keys() == cb["state_dict"].keys()
                                    and all(torch.equal(ca["state_dict"][k], cb["state_dict"][k]) for k in ca["state_dict"])
                                    and ca["cfg"] == cb["cfg"]
                                    and {k: v for k, v in ca["args"].items() if k != "variant"}
                                    == {k: v for k, v in cb["args"].items() if k not in ("variant", "hist_src", "hist_et")})
    check("new_off == ref_off: bit-identical selected and @last checkpoints (state_dict, cfg, args up to the new flags)", all(eq_ck.values()), eq_ck)
    eq_p = {}
    for f_ in (f"nn_{{v}}_trainonly_val.npy", f"nn_{{v}}@last_trainonly_val.npy", f"holdout/nn_{{v}}_holdout.npy", f"holdout/nn_{{v}}@last_holdout.npy"):
        pa, pb = np.load(f"{ro['out']}/preds/" + f_.format(v=vo)), np.load(f"{no['out']}/preds/" + f_.format(v=vn))
        eq_p[f_.format(v="")] = np.array_equal(pa, pb) and pa.dtype == pb.dtype
    check("new_off == ref_off: bit-identical val / holdout preds (selected and @last)", all(eq_p.values()), eq_p)
    # new_on
    on = runs["new_on"]
    v_on = "w9_t9e2e_new_on"
    c = np.load(on["rec"])
    log = open(f"{on['out']}/logs/{v_on}.log").read()
    hes_ok = bool((c["hes"][c["hes"] >= 0] == 1).all()) and int((c["hes"] >= 0).sum()) > 0
    nview = {sp: log.count(f"wave-9 HS1 history view on {sp}:") for sp in ("train", "val", "test")}
    check("new_on: every gathered history event of every training batch has source 1; H never above the off run's H per feed",
          hes_ok and bool((c["H"] <= _H_by_feed(a, c)).all()), f"{int((c['hes'] >= 0).sum())} gathered events")
    check("new_on: one view log line each for train / val / test", nview == {"train": 1, "val": 1, "test": 1}, nview)
    ck = torch.load(f"{on['out']}/ckpt/{v_on}.pt", map_location="cpu", weights_only=False)
    js = json.load(open(f"{on['out']}/logs/{v_on}.json"))
    check("new_on: checkpoint args + cfg and the json args carry hist_src '1'",
          ck["args"]["hist_src"] == "1" and ck["cfg"].get("hist_src") == "1" and js["args"]["hist_src"] == "1")
    files = [f"nn_{v_on}_trainonly_val.npy", f"nn_{v_on}_trainonly_test.npy", f"nn_{v_on}@last_trainonly_val.npy",
             f"nn_{v_on}@last_trainonly_test.npy", f"holdout/nn_{v_on}_holdout.npy", f"holdout/nn_{v_on}@last_holdout.npy"]
    ex = {f_: os.path.exists(f"{on['out']}/preds/{f_}") for f_ in files}
    check("new_on: val / test / holdout preds of the selected and @last checkpoints written", all(ex.values()), ex)
    # CPU holdout parity: rebuild the mini train split, view_for from the checkpoint args, predict == saved
    import model as Mo
    import common_cpu as Cc  # noqa: F401  (predict without the one-GPU guard: see below)
    ms = MiniSplit("train", c["mini_train"], xt_names=None)
    ms.attach_xt = lambda names: setattr(ms, "xt", torch.stack([_xt(ms, n) for n in names], 1))
    ms.attach_xt(ck["cfg"]["xt_names"])
    pr = np.nan_to_num(np.load(f"{ROOT}/eda/labels_position/out/train_prandom.npz")["p_random"].astype(np.float32), nan=0.0)[c["mini_train"]]
    pr[ms.M_np < 50] = 0.0
    prt = torch.from_numpy(pr)
    ms.dom = torch.stack([1 - prt, prt, torch.zeros_like(prt)], 1)
    cand = np.where((pr > 0.9) & (ms.M_np >= 50))[0]
    hold = np.sort(np.random.default_rng(12345).choice(cand, min(30000, len(cand)), replace=False))
    rows = np.concatenate([np.arange(o, o + m) for o, m in zip(ms.cand_off_np[hold], ms.M_np[hold])])
    m = Mo.Ranker(**ck["cfg"])
    m.load_state_dict(ck["state_dict"])
    torch.set_num_threads(1)  # the e2e ran single-threaded; 12-thread CPU reductions differ by ~2e-7
    sv = Cc.predict(m, C9.view_for(ms, ck["args"], {}), hold)[rows].astype(np.float32)
    sf = Cc.predict(m, ms, hold)[rows].astype(np.float32)
    torch.set_num_threads(NT)
    saved = np.load(f"{on['out']}/preds/holdout/nn_{v_on}_holdout.npy")
    check("new_on CPU holdout parity: view_for(checkpoint args) + predict == train.py's saved holdout preds (bitwise); the same "
          "checkpoint on the FULL history differs (negative control)",
          np.array_equal(sv, saved) and not np.array_equal(sf, saved), f"corr(view, full) {np.corrcoef(sv, sf)[0, 1]:.4f}")
    hv_on, hv_off = np.load(f"{on['out']}/preds/holdout/nn_{v_on}_holdout.npy"), np.load(f"{no['out']}/preds/holdout/nn_{vn}_holdout.npy")
    check("new_on != new_off holdout preds (the view changes the model)", not np.array_equal(hv_on, hv_off),
          f"corr {np.corrcoef(hv_on, hv_off)[0, 1]:.4f}")
    RES["e2e_seconds"] = {k: r["sec"] for k, r in runs.items()}


def _xt(ms, n):
    from mini_split import _xt_col
    f = pl.read_parquet(f"{ROOT}/data/flat/train/feeds.parquet", columns=["M", "cand_off"])
    M = f["M"].to_numpy().astype(np.int64)[ms.src_feeds]
    coff = f["cand_off"].to_numpy().astype(np.int64)[ms.src_feeds]
    crow = np.concatenate([np.arange(o, o + m) for o, m in zip(coff, M)])
    return _xt_col("train", n, crow)


def _H_by_feed(a, c):
    """H of the off run per gathered feed, aligned with the on run's batches (same feed sequence: same seed / batching)."""
    assert np.array_equal(a["fidx"], c["fidx"]), "on / off runs must gather the same feed sequence"
    return a["H"]


# ================================================================================================= GPU part
def g_smoke():
    P("\n== G1 flag-on smoke on the GPU (lane.sh W5_SMOKE A:hs1:23 with test preds)")
    rc, cmds = dry_cmds(f"{W9}/lane.sh", _CVD, "A:hs1:23", env={"W5_SMOKE": "1"})
    v = "w9_smoke_hs1_s23"
    assert rc == 0 and v in cmds, cmds
    argv = [x for x in cmds[v] if x != "--no_test"]
    for f_ in (f"{W9}/logs/{v}.json", f"{W9}/logs/{v}.log"):
        if os.path.exists(f_):
            os.remove(f_)
    t0 = time.time()
    r = subprocess.run([PY, f"{W9}/train.py"] + argv, capture_output=True, text=True, cwd=W9,
                       env={**os.environ, "OMP_NUM_THREADS": str(NT), "MKL_NUM_THREADS": str(NT), "OPENBLAS_NUM_THREADS": str(NT),
                            "POLARS_MAX_THREADS": str(NT)})
    open(f"{W9}/logs/{v}.stdout", "w").write(r.stdout + "\n--- stderr ---\n" + r.stderr)
    check(f"smoke {v} completed (rc 0, {(time.time() - t0) / 60:.1f} min)", r.returncode == 0, f"{W9}/logs/{v}.stdout" if r.returncode else "")
    if r.returncode:
        return None
    log = open(f"{W9}/logs/{v}.log").read()
    nview = {sp: log.count(f"wave-9 HS1 history view on {sp}:") for sp in ("train", "val", "test")}
    check("smoke: one view log line each for train / val / test", nview == {"train": 1, "val": 1, "test": 1}, nview)
    m = re.search(r"history view on train: .*?empty feeds before \d+ \((\d\.\d+)\) -> after \d+ \((\d\.\d+)\), emptied by the view \d+ \((\d\.\d+)\)", log)
    P("  info: " + "\n  info: ".join(ln for ln in log.splitlines() if "wave-9 HS1" in ln))
    check("smoke: train view stats (empty after 2.96% +-0.05 pt on the full train split; the skeptic's sampled 2.87%)", m is not None and abs(float(m.group(2)) - 0.0296) < 0.0005,
          m.groups() if m else None)
    files = [f"nn_{v}_trainonly_val.npy", f"nn_{v}_trainonly_test.npy", f"nn_{v}@last_trainonly_val.npy", f"nn_{v}@last_trainonly_test.npy",
             f"holdout/nn_{v}_holdout.npy", f"holdout/nn_{v}@last_holdout.npy"]
    ex = {f_: os.path.exists(f"{W9}/preds/{f_}") for f_ in files}
    check("smoke: val / test / holdout preds (selected + @last) and both checkpoints written",
          all(ex.values()) and os.path.exists(f"{W9}/ckpt/{v}.pt") and os.path.exists(f"{W9}/ckpt/{v}@last.pt"), ex)
    check("smoke: holdout_meta / holdout_y == nn_x/w5's (same fixed 30K holdout)",
          np.array_equal(np.load(f"{W9}/preds/holdout/holdout_meta.npy"), np.load(f"{W5}/preds/holdout/holdout_meta.npy"))
          and np.array_equal(np.load(f"{W9}/preds/holdout/holdout_y.npy"), np.load(f"{W5}/preds/holdout/holdout_y.npy")))
    ck = torch.load(f"{W9}/ckpt/{v}.pt", map_location="cpu", weights_only=False)
    check("smoke: checkpoint args + cfg carry hist_src '1'", ck["args"]["hist_src"] == "1" and ck["cfg"].get("hist_src") == "1")
    return v


def g_holdout(v):
    P("\n== G2 holdout parity: holdout_eval.py == train.py's saved holdout preds (selected + @last); negative control")
    saved = {}
    for suf in ("", "@last"):
        f_ = f"{W9}/preds/holdout/nn_{v}{suf}_holdout.npy"
        saved[suf] = np.load(f_)
        os.replace(f_, f_.replace("_holdout.npy", "_holdout_trainpy.npy"))
    t0 = time.time()
    r = subprocess.run([PY, f"{W9}/holdout_eval.py", v, f"{v}@last"], capture_output=True, text=True, cwd=W9,
                       env={**os.environ, "OMP_NUM_THREADS": str(NT)})
    open(f"{W9}/logs/{v}_holdout_eval.stdout", "w").write(r.stdout + "\n--- stderr ---\n" + r.stderr)
    check(f"holdout_eval.py ran on the smoke checkpoints (rc 0, {(time.time() - t0) / 60:.1f} min)", r.returncode == 0,
          r.stderr[-800:] if r.returncode else "")
    if r.returncode:
        return
    for suf in ("", "@last"):
        he = np.load(f"{W9}/preds/holdout/nn_{v}{suf}_holdout.npy")
        d = np.abs(he - saved[suf]).max()
        check(f"holdout_eval.py{suf or ' (selected)'} == train.py's saved holdout preds", np.array_equal(he, saved[suf]),
              f"max|d| {d:.2e} corr {np.corrcoef(he, saved[suf])[0, 1]:.7f}")
    # negative control on the GPU: the selected checkpoint scored on the FULL history
    code = f"""
import sys, numpy as np, torch
sys.path.insert(0, '{W9}')
from common import CKPT, ROOT, predict
from data import SplitData
from model import Ranker
dev = torch.device('cuda:0')
tr = SplitData('train', dev)
pr = np.nan_to_num(np.load(ROOT / 'eda/labels_position/out/train_prandom.npz')['p_random'].astype(np.float32), nan=0.0)
pr[tr.M_np < 50] = 0.0
cand = np.where((pr > 0.9) & (tr.M_np >= 50))[0]
hold = np.sort(np.random.default_rng(12345).choice(cand, 30000, replace=False))
rows = np.concatenate([np.arange(o, o + m) for o, m in zip(tr.cand_off_np[hold], tr.M_np[hold])])
ck = torch.load(CKPT / '{v}.pt', map_location=dev, weights_only=False)
tr.attach_xt(ck['cfg']['xt_names'])
prt = torch.from_numpy(pr).to(dev)
tr.dom = torch.stack([1 - prt, prt, torch.zeros_like(prt)], 1)
m = Ranker(**ck['cfg']).to(dev); m.load_state_dict(ck['state_dict'])
s = predict(m, tr, hold)[rows].astype(np.float32)
np.save('{SCR}/{v}_holdout_fullhist.npy', s)
"""
    r = subprocess.run([PY, "-c", code], capture_output=True, text=True, cwd=W9, env={**os.environ, "OMP_NUM_THREADS": str(NT)})
    if r.returncode:
        check("negative control (full-history scoring) ran", False, r.stderr[-800:])
        return
    sf = np.load(f"{SCR}/{v}_holdout_fullhist.npy")
    cr = np.corrcoef(sf, saved[""])[0, 1]
    check("negative control: the smoke checkpoint scored on the FULL history differs from the view preds (the parity test is informative)",
          not np.array_equal(sf, saved[""]) and cr < 0.9999, f"corr {cr:.5f}")


def g_stale(v):
    P("\n== G3 stale_pred.py base check on the smoke checkpoint (reproduces the stored train-only val preds through the view)")
    out = f"{W9}/preds/stale/nn_{v}_staleval.npy"
    if os.path.exists(out):
        os.remove(out)
    t0 = time.time()
    r = subprocess.run([PY, f"{W9}/stale_pred.py", v], capture_output=True, text=True, cwd=W9, env={**os.environ, "OMP_NUM_THREADS": str(NT)})
    open(f"{W9}/logs/{v}_stale_pred.stdout", "w").write(r.stdout + "\n--- stderr ---\n" + r.stderr)
    m = re.search(r"base check vs stored: corr ([\d.]+) max\|d\| ([\d.e+-]+)", r.stdout)
    check(f"stale_pred.py ran (rc 0, {(time.time() - t0) / 60:.1f} min) and its base check reproduces the stored val preds (corr >= 0.999999, max|d| < 1e-3)",
          r.returncode == 0 and m is not None and float(m.group(1)) >= 0.999999 and float(m.group(2)) < 1e-3,
          m.groups() if m else r.stderr[-600:])
    P("  info: " + (r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""))


def g_finetune(v):
    P("\n== G4 finetune.py smoke on the smoke checkpoint (2 folds, 1 epoch, n2c_pre_v3 extras)")
    out = f"{v}_ft"
    for f_ in (f"{W9}/logs/{out}_ft.log", f"{W9}/logs/{out}_ft.json"):  # the Logger appends: start from a fresh log
        if os.path.exists(f_):
            os.remove(f_)
    t0 = time.time()
    r = subprocess.run([PY, f"{W9}/finetune.py", "--variant", v, "--out", out, "--epochs", "1", "--folds", "2", "--lr", "5e-5",
                        "--extra", "n2c_pre_v3", "--u0_singleton", "--loss_norm", "feed"], capture_output=True, text=True, cwd=W9,
                       env={**os.environ, "OMP_NUM_THREADS": str(NT)})
    open(f"{W9}/logs/{out}.stdout", "w").write(r.stdout + "\n--- stderr ---\n" + r.stderr)
    check(f"finetune.py ran (rc 0, {(time.time() - t0) / 60:.1f} min)", r.returncode == 0, r.stderr[-800:] if r.returncode else "")
    if r.returncode:
        return
    log = open(f"{W9}/logs/{out}_ft.log").read()
    nview = {sp: log.count(f"wave-9 HS1 history view on {sp}:") for sp in ("val", "test")}
    ex = all(os.path.exists(f"{W9}/preds/{f_}") for f_ in (f"nn_{out}_val_oof.npy", f"nn_{out}_test.npy"))
    check("finetune: view log lines for val and test; OOF / test preds written", nview == {"val": 1, "test": 1} and ex, nview)
    P("  info: " + "; ".join(ln for ln in log.splitlines() if ln.startswith(("OOF", "best epoch", "snapshot"))))


# ================================================================================================= main
if __name__ == "__main__":
    t0 = time.time()
    P(f"test_hs1.py {PART} {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} torch {torch.__version__} threads {NT}")
    if PART == "cpu":
        # a CPU 'common' (predict / eval without the one-GPU import guard) for the C3 parity check
        src = open(f"{W9}/common.py").read().replace('if os.environ.get("CUDA_VISIBLE_DEVICES", "") not in', 'if False and os.environ.get("CUDA_VISIBLE_DEVICES", "") not in')
        open(f"{HERE}/common_cpu.py", "w").write(src)
        c0_code()
        c1_view()
        c2_empty()
        c3_e2e()
    else:
        v = g_smoke()
        if v:
            g_holdout(v)
            g_stale(v)
            g_finetune(v)
    P(f"\n{'ALL PASSED' if not FAIL else 'FAILED: ' + '; '.join(FAIL)} ({len(RES['checks'])} checks, {(time.time() - t0) / 60:.1f} min)")
    RES["failed"] = FAIL
    json.dump(RES, open(f"{W9}/logs/test_hs1_{PART}.json", "w"), indent=1, default=str)
    sys.exit(1 if FAIL else 0)
