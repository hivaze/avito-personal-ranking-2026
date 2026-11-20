"""Unit tests for the r6 WAVE-4 cooldown-data arms (RESEARCH_SWEEP_1112 P5 hcap / P4 rec33 / P3 rho70): train.py --hcap,
--rec_q/--rec_days, --rho_file/--rho_keep (helpers nn_x/r6/cd_arms4.py), nn_x/r6/rho_score.py, nn_x/r6/cd_lane.sh arms.

  cpu part (non-NN checks only; data plumbing, numpy):
    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 .venv/bin/python nn_x/r6/tests/test_cdw4.py cpu
  gpu part (NN checks + end-to-end train.py parity runs; ONE GPU, sequential subprocesses, <= 12 CPU threads):
    CUDA_VISIBLE_DEVICES=<g> OMP_NUM_THREADS=12 .venv/bin/python nn_x/r6/tests/test_cdw4.py gpu
  TRAIN_PY=<path> / CD_LANE=<path> test staged copies (defaults nn_x/r6/train.py, nn_x/r6/cd_lane.sh).

CPU part
  C0 code: (a) cd_lane.sh DRY commands parse with train.py's own parser; m02 leaves every wave-4 flag OFF; hcap / rec33 /
     rho70 differ from m02 ONLY by their own flags; rho70 points to rho/excess_r<root>.npz of its own root.
     (b) AST proof of the flag-off path: train.py minus the wave-4 statements (each an `if` whose test is False at the
     defaults, or the `tr_cap, cool_stream = None, None` init) with the two wave-4 conditional expressions replaced by
     their else-branch `tr` is IDENTICAL to the pre-wave-4 train.py (tests/ref_train_pre_cdw4.py.txt).
     (c) every wave-4 guard / conditional test is False for m02 (tr_cap = cool_stream = None).
     (d) call sites: setup_wave4 once, behind its guard; cool_stream.next once, in the training loop, on non-pool
     cooldown steps only; tr_cap only in the training-batch gather (cooldown steps) and the selection-holdout eval; no
     val / test / predict path and no other nn_x/r6 module except rho_score.py imports cd_arms4.
     (e) parse(): wave-4 defaults OFF; parse() needs no new global names (test_r6.py T12 execs it alone).
  C1 hcap, FULL train split (CPU torch on the 583.6M stored events): kept per-feed per-type count == min(count, cap) for
     every feed and type; uncapped types untouched; dropped share / changed-feed share (vs hcap_caps.txt sample).
  C2 hcap, MiniSplit with heavy feeds: gather on the cap view == gather on a split whose stored histories were capped by
     an independent numpy loop (every tensor of the batch identical); caps respected in every gathered batch; the
     newest events kept; no-op caps (>= 256) give a gather identical to the original split.
  C3 rec: recent_feeds on the real timestamps = Sep 17-19 MSK (boundary 00:00 MSK Sep 17), 0 holdout feeds, every feed
     of the window and none before it; loop simulation of 19,670 CD steps (train.py's m02 mix RNG, seed 811 -> 812):
     pool steps identical with and without the stream, recent share of non-pool steps ~0.33 (within 3 SE), recent
     batches hold only recent feeds, the other non-pool steps return the regular batch object itself.
  C4 rho: synthetic excess file on the real production feeds: kept = exactly the top 70% by excess (kept min >= dropped
     max), every pool-eligible feed kept, 0 holdout feeds; the file / snapshot / keep-range guards fire; loop
     simulation: every non-pool batch is drawn from the stream, dropped feeds never appear, ~2.8 passes.
GPU part
  G1 hcap no-op caps: same seed -> identical train-mode logits, loss (listwise feed + K9 aux) and every gradient vs the
     original split (F7 @77.5k weights, real pool + regular batches); real caps change the loss (active), finite grads.
  G2 rho_score's per-feed CE == model.listwise_loss of each feed alone, / log M (bf16 tolerance).
  G3 end-to-end parity: the pre-wave-4 train.py and the new train.py, flags OFF, m02-like CD runs (39 steps,
     cooldown_mix 0.2 so pool steps occur, eval at 20 and 39) under a deterministic wrapper that records every training
     batch: identical batch sequences, identical logged loss / val / holdout lines, bit-identical saved checkpoint.
     Flags ON (hcap + rec_q 0.33 + synthetic rho file in one run): every training batch and every selection-holdout eval
     read the capped view, every recorded batch respects the caps, stream sources as logged, recent / rho batches only
     from their feed sets. Test runs are named r6_t4e2e_*; their ckpt/preds/logs are deleted afterwards.
Writes nn_x/r6/logs/test_cdw4_<part>.log (+ .json).
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

PART = sys.argv[1] if len(sys.argv) > 1 else "cpu"
assert PART in ("cpu", "gpu"), "usage: test_cdw4.py cpu|gpu"
_CVD = os.environ.get("CUDA_VISIBLE_DEVICES", "x")
if PART == "cpu":
    assert _CVD == "", "cpu part: run with CUDA_VISIBLE_DEVICES=''"
else:
    assert _CVD in tuple(str(i) for i in range(8)), "gpu part: run with CUDA_VISIBLE_DEVICES=<one GPU id>"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
NT = min(12, int(os.environ.get("OMP_NUM_THREADS", "12")))
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
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
R6 = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(1, R6)
import warnings  # noqa: E402

warnings.filterwarnings("ignore", message=".*CUDA is not available.*")
warnings.filterwarnings("ignore", message=".*device_type of 'cuda'.*")
import cd_arms as CA  # noqa: E402
import cd_arms4 as C4  # noqa: E402
import data as D  # noqa: E402
from mini_split import MiniSplit, pick_train_feeds  # noqa: E402

ROOT = "<repo>"
PY = f"{ROOT}/.venv/bin/python"
INIT = f"{ROOT}/nn/ckpt/v17_pre2s_s101@77500.pt"
TRAIN_PY = os.path.abspath(os.environ.get("TRAIN_PY", f"{R6}/train.py"))
CD_LANE = os.path.abspath(os.environ.get("CD_LANE", f"{R6}/cd_lane.sh"))
REF_TRAIN = f"{HERE}/ref_train_pre_cdw4.py.txt"
CAPS = C4.parse_caps(C4.HCAP_F7)
W4_FLAGS = ("--hcap", "--rec_q", "--rec_days", "--rho_file", "--rho_keep")
OFF4 = dict(hcap="", rec_q=0.0, rec_days=3, rho_file="", rho_keep=0.7)
ARM4 = {"hcap": dict(hcap=C4.HCAP_F7), "rec33": dict(rec_q=0.33), "rho70": dict(rho_file=f"{R6}/rho/excess_r{{r}}.npz")}
SCR = os.environ.get("T4_SCRATCH", "/tmp/scratchpad/t4")
os.makedirs(SCR, exist_ok=True)
LOG = open(f"{R6}/logs/test_cdw4_{PART}.log", "w")
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


def load_parse(path):
    tree = ast.parse(open(path).read())
    fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "parse"][0]
    ns = {"argparse": argparse, "POOL_P_DEFAULT": CA.POOL_P_DEFAULT}  # exactly what test_r6.py T12 provides
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "train_parse", "exec"), ns)

    def parse(argv):
        sys.argv = ["train.py"] + argv
        return ns["parse"]()
    return parse


def train_meta():
    """Real train arrays exactly as train.py builds them: M, timestamps, p_random (M<50 -> 0), holdout, feeds, pool."""
    f = pl.read_parquet(f"{ROOT}/data/flat/train/feeds.parquet", columns=["M", "H", "hist_off", "timestamp"])
    M = f["M"].to_numpy().astype(np.int64)
    pr = np.load(f"{ROOT}/eda/labels_position/out/train_prandom.npz")
    p = np.nan_to_num(pr["p_random"].astype(np.float32), nan=0.0)
    p[M < 50] = 0.0
    cand = np.where((p > 0.9) & (M >= 50))[0]
    hold = np.sort(np.random.default_rng(12345).choice(cand, 30000, replace=False))
    feeds = np.setdiff1d(np.arange(len(M)), hold)
    return dict(M=M, H=f["H"].to_numpy().astype(np.int64), hist_off=f["hist_off"].to_numpy().astype(np.int64),
                ts=f["timestamp"].to_numpy(), p=p, hold=hold, feeds=feeds)


class FakeTr:
    """Only what CoolStream / the loop simulation need: M_np and SplitData.batches (the real bucketing)."""
    batches = D.SplitData.batches

    def __init__(self, M):
        self.M_np = M


def synth_rho(meta, path, seed=0, snap=INIT):
    prod = C4.production_feeds(meta["feeds"], meta["p"])
    exc = np.random.default_rng(seed).normal(size=len(prod))
    np.savez(path, fidx=prod, L_snap=exc + 1, L_ref=np.ones(len(prod)), excess=exc, M=meta["M"][prod],
             npos=np.ones(len(prod)), ntop3=np.zeros(len(prod)), snap=np.array(snap), ref=np.array("synthetic"))
    return prod, exc


# ================================================================================================= CPU part
_norm = lambda s: ast.unparse(ast.parse(s, mode="eval").body)  # noqa: E731  (the form ast.unparse prints)
G4_IF = {_norm(s) for s in ("a.hcap or a.rec_q > 0 or a.rho_file",
                            "cool_stream is not None and not is_pool and step >= t_cool", "cool_stream is not None")}
G4_IFEXP = {_norm(s) for s in ("tr_cap is not None and step >= t_cool", "tr_cap is not None")}


class StripWave4(ast.NodeTransformer):
    def __init__(self):
        self.removed_ifs, self.removed_other, self.ifexps = [], [], []

    def visit_ImportFrom(self, n):
        if n.module == "cd_arms4":
            self.removed_other.append("import cd_arms4")
            return None
        return n

    def visit_Expr(self, n):
        v = n.value
        if (isinstance(v, ast.Call) and isinstance(v.func, ast.Attribute) and v.func.attr == "add_argument"
                and v.args and isinstance(v.args[0], ast.Constant) and v.args[0].value in W4_FLAGS):
            self.removed_other.append(f"add_argument {v.args[0].value}")
            return None
        return self.generic_visit(n)

    def visit_Assign(self, n):
        if ast.unparse(n) == "tr_cap, cool_stream = (None, None)":
            self.removed_other.append(ast.unparse(n))
            return None
        return self.generic_visit(n)

    def visit_If(self, n):
        t = ast.unparse(n.test)
        if t in G4_IF:
            self.removed_ifs.append(t)
            return None
        return self.generic_visit(n)

    def visit_IfExp(self, n):
        t = ast.unparse(n.test)
        if t in G4_IFEXP:
            self.ifexps.append((t, ast.unparse(n.orelse)))
            return self.visit(n.orelse)
        return self.generic_visit(n)


def strip_doc(tree):
    if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant):
        tree.body = tree.body[1:]
    return tree


def c0_code(parse):
    P("\n== C0 code: cd_lane.sh commands, AST flag-off proof, guards, call sites, parser")
    jobs = " ".join(f"{a}:{r}" for r in (101, 104) for a in ("m02", "c0", "hcap", "rec33", "rho70"))
    out = subprocess.run(["bash", CD_LANE, "0", jobs], capture_output=True, text=True, env={**os.environ, "DRY": "1"})
    cmds = {}
    for line in out.stdout.splitlines():
        if " train.py " in line:
            mt = re.search(r"r6_cd_(\w+)_r(\d+) ", line)
            cmds[(mt.group(1), int(mt.group(2)))] = line.split(" train.py ", 1)[1].split()
    check("cd_lane.sh DRY: one train.py command per arm and root", out.returncode == 0 and len(cmds) == 10,
          sorted(cmds))
    ns = {k: vars(parse(v)) for k, v in cmds.items()}
    for r in (101, 104):
        m02 = ns[("m02", r)]
        check(f"r{r} m02: every wave-4 flag OFF (cooldown_mix 0.02, seed 811, init s{r}@77500)",
              {k: m02[k] for k in OFF4} == OFF4 and m02["cooldown_mix"] == 0.02 and m02["seed"] == 811
              and m02["init"].endswith(f"v17_pre2s_s{r}@77500.pt"), {k: m02[k] for k in OFF4})
        check(f"r{r} c0: every wave-4 flag OFF", {k: ns[("c0", r)][k] for k in OFF4} == OFF4)
        for arm, fl in ARM4.items():
            want = {k: (OFF4[k], v.format(r=r) if isinstance(v, str) else v) for k, v in fl.items()}
            diff = {k: (m02[k], v) for k, v in ns[(arm, r)].items() if k != "variant" and v != m02[k]}
            check(f"r{r} {arm}: differs from m02 only by {list(fl)}", diff == want, diff)
    RES["cd_lane_cmds"] = {f"{k[0]}:{k[1]}": " ".join(v) for k, v in cmds.items()}

    new_src, old_src = open(TRAIN_PY).read(), open(REF_TRAIN).read()
    st = StripWave4()
    new_t = strip_doc(st.visit(ast.parse(new_src)))
    ident = ast.dump(new_t) == ast.dump(strip_doc(ast.parse(old_src)))
    check("AST: train.py minus the wave-4 statements == pre-wave-4 train.py (flag-off path unchanged)",
          ident and sorted(st.removed_ifs) == sorted(G4_IF) and sorted(t for t, _ in st.ifexps) == sorted(G4_IFEXP)
          and all(e == "tr" for _, e in st.ifexps) and len(st.removed_other) == 1 + len(W4_FLAGS) + 1,
          f"removed ifs {st.removed_ifs}; ifexp -> else {st.ifexps}; other {st.removed_other}")
    a0 = argparse.Namespace(**ns[("m02", 101)])
    env = dict(a=a0, tr_cap=None, cool_stream=None, step=5, t_cool=0)
    vals = {g: bool(eval(g, {}, {**env, "is_pool": ip})) for g in G4_IF | G4_IFEXP for ip in (False,)}
    vals.update({g + " [pool step]": bool(eval(g, {}, {**env, "is_pool": True})) for g in G4_IF | G4_IFEXP})
    check("every wave-4 guard / conditional test is False for m02 (any step, pool or not)", not any(vals.values()), vals)
    for arm in ARM4:
        aa = argparse.Namespace(**ns[(arm, 101)])
        check(f"{arm}: the setup guard is True", bool(eval("a.hcap or a.rec_q > 0 or a.rho_file", {}, dict(a=aa))))
    tree = ast.parse(new_src)
    parents = {ch: p_ for p_ in ast.walk(tree) for ch in ast.iter_child_nodes(p_)}

    def chain(n):
        c, x = [], n
        while x in parents:
            x = parents[x]
            if isinstance(x, ast.If):
                c.append("if " + ast.unparse(x.test))
            if isinstance(x, ast.For):
                c.append("for " + ast.unparse(x.target))
            if isinstance(x, ast.FunctionDef):
                c.append("def " + x.name)
        return c
    sites = {"setup_wave4": [], "next": [], "tr_cap": [], "cool_stream": []}
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "setup_wave4":
            sites["setup_wave4"].append(chain(n))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "next" \
                and ast.unparse(n.func.value) == "cool_stream":
            sites["next"].append(chain(n))
        if isinstance(n, ast.Name) and n.id in ("tr_cap", "cool_stream"):
            p_ = parents[n]
            while not isinstance(p_, (ast.stmt,)):
                p_ = parents[p_]
            sites[n.id].append(ast.unparse(p_).split("\n")[0][:100])
    check("setup_wave4: one call, behind `if a.hcap or a.rec_q > 0 or a.rho_file`, in main()",
          sites["setup_wave4"] == [["if a.hcap or a.rec_q > 0 or a.rho_file", "def main"]], sites["setup_wave4"])
    check("cool_stream.next: one call, in the training loop, on non-pool cooldown steps only",
          sites["next"] == [["if " + _norm("cool_stream is not None and not is_pool and step >= t_cool"),
                             "for (step, fb)", "def main"]], sites["next"])
    trc = sorted(set(sites["tr_cap"]))
    check("tr_cap used only by: init, setup, training-batch gather (cooldown), selection-holdout eval",
          len(trc) == 4 and any(s.startswith("b = (tr_cap if tr_cap is not None and step >= t_cool else tr).gather")
                                for s in trc)
          and any(s.startswith("ndh = eval_subset(model, tr_cap if tr_cap is not None else tr, hold") for s in trc), trc)
    users = sorted(f for f in os.listdir(R6) if f.endswith(".py") and f not in ("cd_arms4.py", "train.py")
                   and re.search(r"\bcd_arms4\b", open(f"{R6}/{f}").read()))
    check("no other nn_x/r6 module imports cd_arms4 except rho_score.py (predict / holdout_eval / finetune untouched)",
          users == ["rho_score.py"], users)
    a_def = parse(["--variant", "r6_x"])
    check("parse(): wave-4 defaults OFF; parse() runs with only argparse + POOL_P_DEFAULT (test_r6 T12)",
          {k: getattr(a_def, k) for k in OFF4} == OFF4, {k: getattr(a_def, k) for k in OFF4})


def c1_full_caps(meta):
    P("\n== C1 hcap on the FULL train split (CPU torch, 583.6M stored events)")
    t0 = time.time()
    et = torch.from_numpy(np.load(f"{ROOT}/data/flat/train/hist_event_type.npy"))
    H, off = torch.from_numpy(meta["H"]), torch.from_numpy(meta["hist_off"])
    drop, feed_ev = C4.cap_drop_mask(et, off, H, CAPS)
    key = feed_ev.long() * 11 + et.long()
    before = torch.bincount(key, minlength=len(H) * 11).view(len(H), 11)
    after = torch.bincount(key[~drop], minlength=len(H) * 11).view(len(H), 11)
    del key
    want = before.clone()
    for t, c in CAPS.items():
        want[:, t] = before[:, t].clamp_max(c)
    unc = [t for t in range(1, 11) if t not in CAPS]
    check("every feed and type: kept count == min(stored count, cap); uncapped types (2,3,5,7,9) untouched",
          torch.equal(after, want) and torch.equal(after[:, unc], before[:, unc]))
    ds, fc = float(drop.float().mean()), float((before - after).sum(1).gt(0).float().mean())
    check("dropped share ~0.096 and changed-feed share ~0.317 (hcap_caps.txt, 200K-feed sample)",
          abs(ds - 0.0957) < 0.003 and abs(fc - 0.3171) < 0.005, f"dropped {ds:.4f}, feeds changed {fc:.4f}, "
          f"{time.time() - t0:.0f}s")
    RES["hcap_full"] = dict(dropped_share=ds, feeds_changed=fc, mean_H_after=float((before - (before - after)).sum(1)
                                                                                     .float().mean()))
    del et, drop, feed_ev, before, after


def numpy_capped_split(ms, caps):
    """Independent reference: a MiniSplit copy whose stored histories are capped by a plain numpy loop."""
    et = ms.h_et.numpy()
    H, off = ms.H.numpy(), ms.hist_off.numpy()
    keep = np.ones(len(et), bool)
    for i in range(ms.n):
        cnt = {}
        for j in range(off[i], off[i] + H[i]):
            t = int(et[j])
            cnt[t] = cnt.get(t, 0) + 1
            if t in caps and cnt[t] > caps[t]:
                keep[j] = False
    v = copy.copy(ms)
    k = torch.from_numpy(keep)
    v.h_emb, v.h_cat, v.h_et, v.h_es = ms.h_emb[k], ms.h_cat[k], ms.h_et[k], ms.h_es[k]
    H2 = np.array([keep[off[i]:off[i] + H[i]].sum() for i in range(ms.n)], dtype=np.int64)
    v.H, v.H_np = torch.from_numpy(H2), H2
    v.hist_off = torch.from_numpy(np.concatenate([[0], np.cumsum(H2)[:-1]]).astype(np.int64))
    v.n_hist = int(H2.sum())
    return v


def heavy_feeds(meta, n_each=24, seed=1):
    """Train feeds over the caps: type-1 > 150, type-4 > 117, type-6 > 41, type-10 > 40 (from a 300K-feed sample)."""
    rng = np.random.default_rng(seed)
    smp = np.sort(rng.choice(len(meta["M"]), 300000, replace=False))
    et = np.load(f"{ROOT}/data/flat/train/hist_event_type.npy", mmap_mode="r")
    H, off = meta["H"][smp], meta["hist_off"][smp]
    rows = np.concatenate([np.arange(o, o + h) for o, h in zip(off, H)])
    fid = np.repeat(np.arange(len(smp)), H)
    C = np.bincount(fid * 11 + np.asarray(et[rows]).astype(np.int64), minlength=len(smp) * 11).reshape(len(smp), 11)
    out = []
    for t, thr in ((1, 150), (4, 117), (6, 41), (10, 40), (8, 42)):
        w = np.where(C[:, t] > thr)[0]
        out.append(smp[rng.choice(w, min(n_each, len(w)), replace=False)])
    return np.unique(np.concatenate(out))


def type_counts(b):
    het, hm = b["het"], b["hmask"]
    return torch.stack([((het == t) & hm).sum(1) for t in range(11)], 1)


def c2_mini_caps(meta):
    P("\n== C2 hcap on a MiniSplit with heavy feeds: cap view == independently capped split")
    feeds = np.union1d(pick_train_feeds(n_each=24, seed=3), heavy_feeds(meta))
    ms = MiniSplit("train", feeds, xt_names=None)
    v, st = C4.cap_history_view(ms, CAPS)
    ref = numpy_capped_split(ms, CAPS)
    check("cap view arrays == numpy-capped arrays (h_emb, h_cat, h_et, h_es, H, hist_off, n_hist)",
          all(torch.equal(getattr(v, k), getattr(ref, k)) for k in ("h_emb", "h_cat", "h_et", "h_es", "H", "hist_off"))
          and v.n_hist == ref.n_hist and np.array_equal(v.H_np, ref.H_np), st)
    rng = np.random.default_rng(0)
    bad, nb, maxc = [], 0, {}
    for fb in ms.batches(np.arange(ms.n), 8192, 64, shuffle=True, rng=rng):
        b1, b2, b0 = v.gather(fb, with_pos=True), ref.gather(fb, with_pos=True), ms.gather(fb, with_pos=True)
        bad += [k for k in b1 if not torch.equal(b1[k], b2[k])]
        tc, tc0 = type_counts(b1), type_counts(b0)
        for t, c in CAPS.items():
            maxc[t] = max(maxc.get(t, 0), int(tc[:, t].max()))
        nb += 1
        # newest kept: for every capped type the kept events are the first min(n, cap) of that type in stored order
        if nb == 1:
            for i in range(len(fb)):
                for t, c in CAPS.items():
                    e0 = b0["hidx"][i][b0["hmask"][i] & (b0["het"][i] == t)][:c]
                    e1 = b1["hidx"][i][b1["hmask"][i] & (b1["het"][i] == t)]
                    if not torch.equal(e0, e1):
                        bad.append(f"newest feed {i} type {t}")
    check(f"every gathered batch ({nb}) identical between the cap view and the numpy-capped split", not bad,
          sorted(set(bad))[:10])
    check("caps respected in every gathered batch (max per-feed count per capped type <= cap)",
          all(maxc[t] <= c for t, c in CAPS.items()) and maxc[1] == 110, maxc)
    n_heavy = int((type_counts(ms.gather(np.arange(ms.n)))[:, 1] > 110).sum())
    check("the fixture is over the caps before capping (feeds with > 110 views)", n_heavy >= 20, n_heavy)
    vn, _ = C4.cap_history_view(ms, {t: 256 for t in CAPS})
    fb = np.arange(ms.n)[:64]
    b0, bn = ms.gather(fb, with_pos=True), vn.gather(fb, with_pos=True)
    check("no-op caps (256): gather identical to the original split", all(torch.equal(b0[k], bn[k]) for k in b0))


def loop_sim(meta, stream, steps=19670, mix=0.02, seed=811):
    """train.py's batch selection for a CD run (cooldown_frac 1.0): regular epoch batches from feeds (rng seed), m02
    pool steps from the mix RNG (seed + 1, which also shuffles the pool stream first), then the wave-4 stream."""
    tr = FakeTr(meta["M"])
    rng = np.random.default_rng(seed)
    regular = tr.batches(meta["feeds"], 32768, 512, shuffle=True, rng=rng)
    pool = meta["feeds"][meta["p"][meta["feeds"]] > 0.5]
    mix_rng = np.random.default_rng(seed + 1)
    expl, ei = tr.batches(pool, 32768, 512, shuffle=True, rng=mix_rng), 0
    out = []
    for step in range(steps):
        fb = regular[step % len(regular)]
        is_pool = False
        if mix_rng.random() < mix:
            if ei >= len(expl):
                expl, ei = tr.batches(pool, 32768, 512, shuffle=True, rng=mix_rng), 0
            fb, ei, is_pool = expl[ei], ei + 1, True
        fb0 = fb
        if stream is not None and not is_pool:
            fb = stream.next(fb)
        out.append((is_pool, fb, fb is fb0))
    return out


def c3_rec(meta):
    P("\n== C3 rec33: recent window on the real timestamps + loop simulation")
    rec, bnd, dates = C4.recent_feeds(meta["feeds"], meta["ts"], 3)
    check("last 3 train days = Sep 17-19 (MSK), boundary = 2026-09-17 00:00 MSK (1789592400)",
          dates == ["2026-09-17", "2026-09-18", "2026-09-19"] and bnd == 1789592400, (dates, bnd))
    ts = meta["ts"]
    inwin = meta["feeds"][ts[meta["feeds"]] >= bnd]
    check("recent = every training feed at/after the boundary, none before; 0 holdout feeds",
          np.array_equal(rec, inwin) and (ts[rec] >= bnd).all() and not np.isin(rec, meta["hold"]).any(),
          f"{len(rec)} feeds ({len(rec) / len(meta['feeds']):.4f}); of which Sep 20 spill-over "
          f"{int((ts[rec] >= bnd + 3 * 86400).sum())}")
    base = loop_sim(meta, None)
    s = C4.CoolStream(FakeTr(meta["M"]), 32768, 512, np.random.default_rng(813), rec=rec, rec_q=0.33)
    sim = loop_sim(meta, s)
    check("pool steps identical with and without the rec stream (m02 mix RNG untouched)",
          [x[0] for x in base] == [x[0] for x in sim], f"pool steps {sum(x[0] for x in sim)} / {len(sim)}")
    npool = [x for x in sim if not x[0]]
    share = s.n["recent"] / len(npool)
    se = np.sqrt(0.33 * 0.67 / len(npool))
    check("recent share of non-pool cooldown batches ~0.33 (within 3 SE)", abs(share - 0.33) < 3 * se,
          f"{share:.4f} (SE {se:.4f}); {s.summary()}")
    recset = set(rec.tolist())
    rb = [x[1] for x in npool if not x[2]]
    check("recent batches hold only recent feeds; the other non-pool steps return the regular batch object itself",
          all(set(np.asarray(fb).tolist()) <= recset for fb in rb) and len(rb) == s.n["recent"]
          and sum(x[2] for x in npool) == s.n["regular"])
    isrec = np.zeros(len(meta["M"]), bool)
    isrec[rec] = True
    fr = sum(len(fb) for fb in rb) + sum(int(isrec[x[1]].sum()) for x in npool if x[2])
    tot = sum(len(x[1]) for x in npool)
    RES["rec"] = dict(n_recent=len(rec), share_batches=share, eff_recent_feed_share_nonpool=fr / tot)
    P(f"  info: effective share of recent FEEDS in non-pool cooldown batches {fr / tot:.3f} (natural "
      f"{len(rec) / len(meta['feeds']):.3f})")


def c4_rho(meta):
    P("\n== C4 rho70: stream from a synthetic excess file on the real production feeds + loop simulation")
    path = f"{SCR}/synth_excess.npz"
    prod, exc = synth_rho(meta, path)
    stream, st = C4.rho_stream_feeds(meta["feeds"], meta["p"], meta["hold"], path, 0.7, INIT)
    kept = np.intersect1d(stream, prod)
    dropped = np.setdiff1d(prod, kept)
    pos = np.searchsorted(prod, kept)
    posd = np.searchsorted(prod, dropped)
    pool = meta["feeds"][meta["p"][meta["feeds"]] > 0.5]
    check("kept = round(0.7 x production feeds); exactly the top 70% by excess (kept min >= dropped max)",
          len(kept) == int(round(0.7 * len(prod))) and exc[pos].min() >= exc[posd].max(), st)
    check("every pool-eligible feed in the stream; stream = kept + pool; 0 holdout feeds",
          np.isin(pool, stream).all() and len(stream) == len(kept) + len(pool) and not np.isin(stream, meta["hold"]).any())
    bad = f"{SCR}/synth_bad.npz"
    z = dict(np.load(path))
    z["fidx"] = z["fidx"][1:]
    z["excess"] = z["excess"][1:]
    np.savez(bad, **z)
    fired = []
    for kw in (dict(path=bad, keep=0.7, init=INIT), dict(path=path, keep=0.7, init=INIT.replace("s101", "s102")),
               dict(path=path, keep=0.4, init=INIT)):
        try:
            C4.rho_stream_feeds(meta["feeds"], meta["p"], meta["hold"], kw["path"], kw["keep"], kw["init"])
            fired.append(False)
        except AssertionError:
            fired.append(True)
    check("guards fire: file on other feeds, file scored with another root's snapshot, keep < 0.5", all(fired), fired)
    s = C4.CoolStream(FakeTr(meta["M"]), 32768, 512, np.random.default_rng(813), rho=stream)
    sim = loop_sim(meta, s)
    npool = [x for x in sim if not x[0]]
    seen = np.unique(np.concatenate([x[1] for x in npool]))
    check("every non-pool cooldown batch is drawn from the rho stream; dropped production feeds never appear",
          s.n["rho"] == len(npool) and s.n["regular"] == 0 and np.isin(seen, stream).all()
          and not np.isin(dropped, seen).any(), s.summary())
    cov = np.isin(kept, seen).mean()
    check("~2.8 passes over the kept stream (3 shuffles started), >= 99.9% of kept production feeds seen",
          s.st["rho"][2] == 3 and cov > 0.999, f"passes {s.st['rho'][2]}, coverage {cov:.5f}")
    RES["rho_synth"] = st


# ================================================================================================= GPU part
def fresh(b):
    return {k: (v.clone() if torch.is_tensor(v) else copy.deepcopy(v)) for k, v in b.items()}


def clone(m):
    for mod in m.modules():
        for k in ("aux_out", "intent_logp"):
            mod.__dict__.pop(k, None)
    return copy.deepcopy(m)


def train_loss(model, b, prand_b, aux_w=0.5, tau=10.0, denom=512):
    import model as M6
    X = model.dense(b)
    lg = model(b, X, use_pos=True)
    y, m = b["y"], b["cmask"]
    loss = M6.listwise_loss(lg, y, m, None, norm="feed", denom=denom)
    tgt = torch.softmax((-b["pos"].float() / tau).masked_fill(~m, -1e4), 1) * m
    lp = torch.log_softmax(model.aux_out.masked_fill(~m, -1e4), 1)
    fw_aux = 1 - prand_b
    return loss + aux_w * (-(tgt * lp).sum(1) * fw_aux).sum() / fw_aux.sum().clamp_min(1e-6), lg


def g1_g2(meta):
    import model as M6
    P("\n== G1 hcap no-op caps: loss / gradient parity with the original split; G2 per-feed CE")
    ck = torch.load(INIT, map_location="cpu", weights_only=False)
    base = M6.Ranker(**ck["cfg"])
    base.load_state_dict(ck["state_dict"])
    base.to(DEV)
    feeds = np.union1d(pick_train_feeds(n_each=24, seed=3), heavy_feeds(meta))
    ms = MiniSplit("train", feeds, xt_names=ck["cfg"]["xt_names"]).to(DEV)
    vn, _ = C4.cap_history_view(ms, {t: 256 for t in CAPS})
    vc, _ = C4.cap_history_view(ms, CAPS)
    prn = ms.prand.cpu().numpy()
    idx = {"pool": np.where((prn > 0.5) & (ms.M_np >= 50))[0][:48],
           "regular": np.where((prn < 0.1) & (ms.M_np >= 50))[0][:48]}
    for nm, ix in idx.items():
        outs = []
        for src in (ms, vn, vc):
            mdl = clone(base)
            mdl.train()
            torch.manual_seed(77)
            b = src.gather(ix, with_pos=True)
            lo, lg = train_loss(mdl, b, ms.prand[b["fidx"]])
            lo.backward()
            outs.append((lo.detach(), lg.detach(), [p.grad.clone() if p.grad is not None else None
                                                    for p in mdl.parameters()]))
        g_ok = all((g1 is None and g2 is None) or (g1 is not None and g2 is not None and torch.equal(g1, g2))
                   for g1, g2 in zip(outs[0][2], outs[1][2]))
        check(f"no-op caps vs original split, {nm} batch, same seed: train logits, loss and all grads identical",
              torch.equal(outs[0][0], outs[1][0]) and torch.equal(outs[0][1], outs[1][1]) and g_ok,
              f"loss {outs[0][0].item():.6f} vs {outs[1][0].item():.6f}")
        fin = all(g is None or bool(torch.isfinite(g).all()) for g in outs[2][2])
        check(f"real caps, {nm} batch: loss changes (arm active), finite loss and grads",
              outs[2][0].item() != outs[0][0].item() and fin and np.isfinite(outs[2][0].item()),
              f"loss capped {outs[2][0].item():.6f} vs stored {outs[0][0].item():.6f}")
    base.eval()
    ids = np.arange(ms.n)
    L = C4.per_feed_ce(base, ms, ids)
    ref = []
    for i in ids:
        b = ms.gather(np.array([i]))
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            lg = base(b, use_pos=False)
        ref.append(M6.listwise_loss(lg.float(), b["y"], b["cmask"], None, norm="batch").item()
                   / np.log(max(ms.M_np[i], 2)))
    ref = np.array(ref)
    d = np.abs(L - ref)
    check("per_feed_ce == listwise_loss of the feed alone / log M (bf16 batch-shape tolerance 5e-4)",
          not np.isnan(L).any() and d.max() < 5e-4, f"max|d| {d.max():.2e} over {len(ids)} feeds, mean L {L.mean():.4f}")


E2E_COMMON = ("--init {init} --epochs 0.004 --lr 2e-3 --warmup 0 --sched wsd --cooldown_frac 1.0 --muon --aux_pos 0.5 "
              "--aux_mode listwise --aux_tau 10 --aux_anneal 0.1 --cooldown_mix 0.2 --loss_norm feed --holdout 30000 "
              "--select_on holdout --eval_every 20 --seed 811 --no_test")


def e2e_run(train_file, variant, extra, rec_path):
    args = E2E_COMMON.format(init=INIT).split() + extra
    cmd = [PY, f"{HERE}/e2e_cdw4.py", train_file, rec_path, "--variant", variant] + args
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=R6, env={**os.environ, "OMP_NUM_THREADS": str(NT)})
    open(f"{SCR}/{variant}.stdout", "w").write(r.stdout + "\n" + r.stderr)
    P(f"  ran {variant} ({time.time() - t0:.0f}s, rc {r.returncode})")
    assert r.returncode == 0, r.stderr[-3000:]
    return np.load(rec_path, allow_pickle=True), open(f"{R6}/logs/{variant}.log").read()


def metric_lines(log):
    return [re.sub(r" [\d.]+ min", "", ln) for ln in log.splitlines() if ln.startswith("step ") or ln.startswith("BEST")]


def g3_e2e(meta):
    P("\n== G3 end-to-end train.py runs (deterministic wrapper; 39 CD steps, cooldown_mix 0.2, evals at 20 and 39)")
    names = ["r6_t4e2e_ref_off", "r6_t4e2e_new_off", "r6_t4e2e_new_on"]
    for v in names:
        for f in (f"{R6}/logs/{v}.log", f"{R6}/logs/{v}.json", f"{R6}/ckpt/{v}.pt", f"{R6}/preds/nn_{v}_trainonly_val.npy"):
            if os.path.exists(f):
                os.remove(f)
    ref_copy = f"{SCR}/train_pre_cdw4.py"
    open(ref_copy, "w").write(open(REF_TRAIN).read())
    ra, la = e2e_run(ref_copy, names[0], [], f"{SCR}/rec_ref_off.npz")
    rb, lb = e2e_run(TRAIN_PY, names[1], [], f"{SCR}/rec_new_off.npz")
    same_b = (len(ra["off"]) == len(rb["off"]) and np.array_equal(ra["fidx"], rb["fidx"])
              and np.array_equal(ra["off"], rb["off"]))
    check("flags OFF: identical training-batch sequence (feed ids of every step, incl. pool steps)", same_b,
          f"{len(ra['off']) - 1} training batches")
    check("flags OFF: identical per-batch history type counts / H (same histories gathered)",
          np.array_equal(ra["tcnt"], rb["tcnt"]) and not rb["capped"].any())
    ma, mb = metric_lines(la), metric_lines(lb)
    check("flags OFF: identical logged loss / val_ndcg / holdout_ndcg / BEST lines", ma == mb and len(ma) >= 3,
          ma[-1] if ma else "")
    ca = torch.load(f"{R6}/ckpt/{names[0]}.pt", map_location="cpu", weights_only=False)["state_dict"]
    cb = torch.load(f"{R6}/ckpt/{names[1]}.pt", map_location="cpu", weights_only=False)["state_dict"]
    mx = max(float((ca[k].float() - cb[k].float()).abs().max()) for k in ca)
    check("flags OFF: saved checkpoint bit-identical (every tensor)", all(torch.equal(ca[k], cb[k]) for k in ca),
          f"max|d| {mx:.2e}")
    va, vb = np.load(f"{R6}/preds/nn_{names[0]}_trainonly_val.npy"), np.load(f"{R6}/preds/nn_{names[1]}_trainonly_val.npy")
    check("flags OFF: val predictions bit-identical", np.array_equal(va, vb))
    # flags ON: hcap + rec + rho (synthetic file of this run's production feeds, scored 'with' s101)
    path = f"{SCR}/synth_excess.npz"
    synth_rho(meta, path)
    rc, lc = e2e_run(TRAIN_PY, names[2], ["--hcap", C4.HCAP_F7, "--rec_q", "0.33", "--rho_file", path],
                     f"{SCR}/rec_new_on.npz")
    nb = len(rc["off"]) - 1
    check("flags ON: every training batch gathered from the capped view", bool(rc["capped"].all()) and nb == 39,
          f"{nb} batches")
    tc = rc["tcnt"]
    check("flags ON: caps respected in every training batch (max per-feed count of each capped type <= cap)",
          all(int(tc[:, t].max()) <= c for t, c in CAPS.items()), {t: int(tc[:, t].max()) for t in CAPS})
    check("flags ON: every selection-holdout eval read the capped view; val predict did not",
          len(rc["eval_capped"]) == 2 and bool(rc["eval_capped"].all()) and not rc["predict_capped"].any(),
          f"evals {rc['eval_capped'].tolist()}, val predicts capped {rc['predict_capped'].tolist()}")
    rec, _, _ = C4.recent_feeds(meta["feeds"], meta["ts"], 3)
    stream, _ = C4.rho_stream_feeds(meta["feeds"], meta["p"], meta["hold"], path, 0.7, INIT)
    src = rc["src"]
    ok, f_off = True, rc["off"]
    for i in range(nb):
        fb = rc["fidx"][f_off[i]:f_off[i + 1]]
        if src[i] == "recent":
            ok &= bool(np.isin(fb, rec).all())
        elif src[i] == "rho":
            ok &= bool(np.isin(fb, stream).all())
    n = {k: int((src == k).sum()) for k in ("pool", "recent", "rho", "regular")}
    check("flags ON: recent batches only recent feeds, rho batches only stream feeds, no plain regular batch",
          ok and n["regular"] == 0 and n["pool"] + n["recent"] + n["rho"] == nb, n)
    check("flags ON: setup log lines present (hcap / rec / rho) and stream summary logged at each eval",
          "wave-4 hcap: caps" in lc and "wave-4 rec: q 0.33" in lc and "wave-4 rho:" in lc
          and lc.count("wave-4 stream (non-pool cooldown batches)") == 2)
    RES["e2e"] = dict(off_lines=mb, on_lines=metric_lines(lc), on_sources=n)
    for v in names:
        for f in (f"{R6}/logs/{v}.log", f"{R6}/logs/{v}.json", f"{R6}/ckpt/{v}.pt", f"{R6}/preds/nn_{v}_trainonly_val.npy"):
            if os.path.exists(f):
                os.remove(f)


def main():
    t_all = time.time()
    P(f"== r6 wave-4 tests, part {PART} ({time.ctime()}, device {DEV}, CPU threads {NT})")
    P(f"   train.py: {TRAIN_PY}   cd_lane: {CD_LANE}   caps: {CAPS}")
    meta = train_meta()
    if PART == "cpu":
        c0_code(load_parse(TRAIN_PY))
        c3_rec(meta)
        c4_rho(meta)
        c2_mini_caps(meta)
        c1_full_caps(meta)
    else:
        g1_g2(meta)
        g3_e2e(meta)
    RES["failed"] = FAIL
    P(f"\n== {len(RES['checks']) - len(FAIL)}/{len(RES['checks'])} checks passed; failed: {FAIL}; "
      f"{time.time() - t_all:.0f}s")
    json.dump(RES, open(f"{R6}/logs/test_cdw4_{PART}.json", "w"), indent=1, default=str)
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())
