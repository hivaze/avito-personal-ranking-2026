"""CPU tests of the gap1117 loss options of nn_x/w5/train.py (--wcl / --ccl / --fsw [--fsw_norm], helper nn_x/w5/cd_arms9.py) and of the
lane.sh arms K/L: wcl05 wcl10 ccl05 fsw25 fsw50 fsw25t fsw50t. Frozen pre-change copies: tests/ref_pre_gap1117/{train.py,lane.sh}.txt
(md5.txt). Written 2026-11-17.

  cd <repo> && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 .venv/bin/python lab/nn_x/w12/gap1117/tests/test_gap1117.py

T1 identity CE_items = CE_cat + sum_c t_c CE_within_c, through cd_arms9.category_split (the code path train.py runs), on random padded
   batches with ties (rounded scores, all-equal feeds), single-candidate categories, all-singleton feeds, one-category feeds, M = 1 feeds,
   feeds without positives, zero-target categories, garbage on padding positions; targets = clicks / max(sum, 1) and the KD mixture.
   float32 tolerance 1e-5, float64 1e-10; each part also == a float64 brute-force loop over the categories of every feed; gradcheck
   (float64) and grad(CE_cat + within) == grad(main CE).
T2 reductions: GapLoss.add_terms == loss + wcl * sum_f w_f within_f / max_feeds + ccl * sum_f w_f CE_cat_f / max_feeds (loss_norm feed;
   and the weighted mean for batch), w_f = fw * [feed has a positive]; wcl = ccl = 1 adds exactly one more main loss; the loss object is
   returned unchanged when wcl = ccl = 0; diagnostics accumulate / reset.
T3 fsw: feed_weight == M^a / mean_batch(M^a) (batch) or M^a / mean_train(M^a) (train), times fw; returns fw itself when --fsw 0;
   listwise_loss with that weight == brute force.
T4 default unchanged (static): train.py minus the gap1117 statements == the frozen pre-change train.py (AST, docstrings stripped); the
   exact removed set is asserted; parse(): every previous default unchanged, new ones wcl 0, ccl 0, fsw 0, fsw_norm batch; the setup guard
   is False at the defaults and in every K/L DRY command of the previous arms; `fw` never reaches the K9 aux statement; the two loop
   statements sit at the top level of the training loop (every batch, pool batches included).
T5 lane.sh: DRY output of every previous K/L arm (seeds 951 / 958) byte-identical to the frozen lane.sh; each new arm == K/L:ref + exactly
   its flags; kw03 / e4 / lr4 still parse; every command parses with train.py's own parser; refusal of an unknown arm unchanged.
T6 names: for every job of the T1117 jobs files, lane.sh's --variant == fill.sh vname() == arm_read8.py rc lambda (w5_<k|l>cd_<arm>_s<seed>).
Writes tests/out/test_gap1117_cpu.log.
"""
import argparse
import ast
import json
import os
import re
import subprocess
import sys
import types

assert os.environ.get("CUDA_VISIBLE_DEVICES", "x") == "", "run with CUDA_VISIBLE_DEVICES=''"
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(min(12, int(os.environ.get("OMP_NUM_THREADS", "12"))))
HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, "ref_pre_gap1117")
R = "<repo>"
W5 = f"{R}/lab/nn_x/w5"
KD = f"{R}/lab/nn_x/w12/kd"
sys.path.insert(0, W5)
import cd_arms9 as C9  # noqa: E402
from model import listwise_loss  # noqa: E402

OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)
LOG = open(os.path.join(OUT, "test_gap1117_cpu.log"), "w")
RES = {}


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + "\n")
    LOG.flush()


def check(name, ok, info=""):
    RES[name] = bool(ok)
    say(f"[{'PASS' if ok else 'FAIL'}] {name} {info}")


# ------------------------------------------------------------------ random padded batches
def make_batch(seed, B=48, Mx=64, dtype=torch.float32, kd=False):
    g = np.random.default_rng(seed)
    M = g.integers(1, Mx + 1, B)
    M[0], M[1], M[2], M[3] = 1, Mx, 2, 7
    m = np.arange(Mx)[None, :] < M[:, None]
    cat = g.integers(2, 9, (B, Mx))  # 7 categories: many repeats -> multi-candidate categories
    cat[4] = np.arange(Mx) + 2  # all-singleton feed
    cat[5] = 3  # one-category feed
    cat[6, : M[6]] = g.integers(2, 60, M[6])  # mostly singletons
    lg = g.normal(0, 2, (B, Mx))
    lg[7] = 0.25  # all-equal scores (ties everywhere)
    lg[8:20] = np.round(lg[8:20] * 2) / 2  # many ties
    lg[~m] = g.normal(30, 10, (~m).sum())  # garbage on padding (must be masked)
    y = (g.random((B, Mx)) < 0.08).astype(np.float64)
    y[9] = 0  # feed without positives
    y[10, :] = 0
    y[10, 0] = 1
    y[11, :3] = 1
    y = y * m
    for f in range(12, B, 5):  # make sure most feeds have a positive
        if y[f].sum() == 0:
            y[f, g.integers(0, M[f])] = 1
    T = lambda x: torch.tensor(x, dtype=dtype)  # noqa: E731
    lg_t, y_t, m_t, cat_t = T(lg), T(y), torch.tensor(m), torch.tensor(cat, dtype=torch.long)
    if kd:  # KD mixture target (train.py statement)
        tl = T(g.normal(0, 1.5, (B, Mx)))
        qt = torch.softmax(tl.masked_fill(~m_t, -1e4), 1) * m_t
        y_t = 0.5 * y_t / y_t.sum(1, keepdim=True).clamp_min(1) + 0.5 * qt * (y_t.sum(1, keepdim=True) > 0)
    return dict(lg=lg_t, y=y_t, m=m_t, ccat=cat_t, M=torch.tensor(M), M_np=M)


def main_ce(lg, y, m):  # per-feed main CE exactly as model.listwise_loss computes l
    logp = torch.log_softmax(lg.masked_fill(~m, -1e4), dim=1)
    ys = y.sum(1)
    return -(y * logp).sum(1) / ys.clamp_min(1)


def brute(lg, t, m, cat):  # float64 loop: (CE_cat, sum_c t_c CE_within_c) per feed, categories with >= 2 cands and t_c > 0 only
    lg, t, m, cat = lg.double().numpy(), t.double().numpy(), m.numpy(), cat.numpy()
    cc, cw = np.zeros(len(lg)), np.zeros(len(lg))
    for f in range(len(lg)):
        idx = np.flatnonzero(m[f])
        s, tt, c = lg[f, idx], t[f, idx], cat[f, idx]
        lse = np.logaddexp.reduce(s)
        for k in np.unique(c):
            j = c == k
            Sc, tc = np.logaddexp.reduce(s[j]), tt[j].sum()
            cc[f] -= tc * (Sc - lse)
            if j.sum() >= 2 and tc > 0:
                p = s[j] - Sc
                cw[f] += tc * -(tt[j] / tc * p).sum()
    return cc, cw


def t1():
    worst32, worst64, worstb = 0.0, 0.0, 0.0
    for seed in range(12):
        for kd in (False, True):
            for dt, tol in ((torch.float32, 1e-5), (torch.float64, 1e-10)):
                b = make_batch(seed, dtype=dt, kd=kd, B=48 if seed % 2 else 31, Mx=64 if seed % 3 else 112)
                ys = b["y"].sum(1)
                t = b["y"] / ys.clamp_min(1)[:, None]
                cc, cw = C9.category_split(b["lg"], t, b["m"], b["ccat"])
                ref = main_ce(b["lg"], b["y"], b["m"])
                d = float((cc + cw - ref).abs().max())
                if dt == torch.float32:
                    worst32 = max(worst32, d)
                else:
                    worst64 = max(worst64, d)
                assert d <= tol, (seed, kd, dt, d)
                bc, bw = brute(b["lg"], t, b["m"], b["ccat"])
                db = max(float(np.abs(cc.double().numpy() - bc).max()), float(np.abs(cw.double().numpy() - bw).max()))
                worstb = max(worstb, db if dt == torch.float64 else 0.0)
                assert db <= (1e-4 if dt == torch.float32 else 1e-10), (seed, kd, dt, db)
                # single-candidate categories contribute exactly 0 to the within part; padding never matters
                assert float(cw[4].abs()) == 0.0, float(cw[4])
    check("T1a identity CE_items == CE_cat + within (float32 tol 1e-5, float64 tol 1e-10; 48 batches)", True,
          f"max |diff| float32 {worst32:.2e}, float64 {worst64:.2e}; vs float64 brute-force loop {worstb:.2e}")
    # padding garbage changes nothing
    b = make_batch(3, dtype=torch.float64)
    t = b["y"] / b["y"].sum(1).clamp_min(1)[:, None]
    c1, w1 = C9.category_split(b["lg"], t, b["m"], b["ccat"])
    lg2, cat2 = b["lg"].clone(), b["ccat"].clone()
    lg2[~b["m"]] = -77.0
    cat2[~b["m"]] = 5
    c2, w2 = C9.category_split(lg2, t, b["m"], cat2)
    check("T1b padding scores / categories have no effect", torch.equal(c1, c2) and torch.equal(w1, w2))
    # gradients
    b = make_batch(5, dtype=torch.float64, B=14, Mx=16)
    t = b["y"] / b["y"].sum(1).clamp_min(1)[:, None]
    f = lambda x: C9.category_split(x, t, b["m"], b["ccat"])  # noqa: E731
    gc = torch.autograd.gradcheck(f, (b["lg"].clone().requires_grad_(True),), eps=1e-6, atol=1e-7)
    x1 = b["lg"].clone().requires_grad_(True)
    cc, cw = C9.category_split(x1, t, b["m"], b["ccat"])
    (cc + cw).sum().backward()
    x2 = b["lg"].clone().requires_grad_(True)
    main_ce(x2, b["y"], b["m"]).sum().backward()
    dg = float((x1.grad - x2.grad).abs().max())
    check("T1c gradcheck (float64) and grad(CE_cat + within) == grad(main CE)", gc and dg < 1e-12, f"max |grad diff| {dg:.2e}")
    # gradient of within never reaches single-candidate categories' scores (their within part is identically 0)
    x3 = b["lg"].clone().requires_grad_(True)
    _, cw = C9.category_split(x3, t, b["m"], b["ccat"])
    cw.sum().backward()
    single = torch.zeros_like(b["m"])
    for r in range(b["m"].shape[0]):
        cs = b["ccat"][r][b["m"][r]]
        for j in range(int(b["m"][r].sum())):
            single[r, j] = int((cs == cs[j]).sum()) == 1
    check("T1d within-part gradient is 0 on single-candidate categories and on padding",
          float(x3.grad[single].abs().max() if single.any() else 0) == 0.0 and float(x3.grad[~b["m"]].abs().max()) == 0.0,
          f"({int(single.sum())} single-candidate positions)")


def args_ns(**k):
    d = dict(wcl=0.0, ccl=0.0, fsw=0.0, fsw_norm="batch", loss_norm="feed", max_feeds=512, loss="softmax", rdrop_cd=0.0, sam_cd=0.0)
    d.update(k)
    return types.SimpleNamespace(**d)


def t2():
    ok, worst = True, 0.0
    for norm in ("feed", "batch"):
        for wcl, ccl in ((0.5, 0.0), (1.0, 0.0), (0.0, 0.5), (0.3, 0.7)):
            for use_fw in (False, True):
                b = make_batch(11, dtype=torch.float64, kd=True)
                b["y"][9] = 0  # one feed without positives (excluded like in the main loss)
                fw = torch.rand(b["y"].shape[0], dtype=torch.float64, generator=torch.Generator().manual_seed(1)) + 0.5 if use_fw else None
                g = C9.GapLoss(args_ns(wcl=wcl, ccl=ccl, loss_norm=norm), b["M_np"])
                base = torch.tensor(1.2345, dtype=torch.float64)
                out = g.add_terms(base, b["lg"], b["y"], b["m"], dict(ccat=b["ccat"], M=b["M"]), fw)
                ys = b["y"].sum(1)
                t = b["y"] / ys.clamp_min(1)[:, None]
                bc, bw = brute(b["lg"], t, b["m"], b["ccat"])
                w = (np.ones(len(ys)) if fw is None else fw.numpy()) * (ys.numpy() > 0)
                den = 512 if norm == "feed" else w.sum()
                exp = 1.2345 + wcl * (w * bw).sum() / den + ccl * (w * bc).sum() / den
                d = abs(float(out) - exp)
                worst = max(worst, d)
                ok &= d < 1e-10
    check("T2a add_terms == loss + wcl * reduce(within) + ccl * reduce(CE_cat) (feed / batch norm, fw on / off, KD target)", ok,
          f"max |diff| {worst:.2e}")
    b = make_batch(12, dtype=torch.float64, kd=True)
    for norm in ("feed", "batch"):
        g = C9.GapLoss(args_ns(wcl=1.0, ccl=1.0, loss_norm=norm), b["M_np"])
        main = listwise_loss(b["lg"], b["y"], b["m"], None, norm=norm, denom=512)
        out = g.add_terms(main, b["lg"], b["y"], b["m"], dict(ccat=b["ccat"], M=b["M"]), None)
        check(f"T2b wcl = ccl = 1 adds exactly one more main loss ({norm} norm)", abs(float(out) - 2 * float(main)) < 1e-10,
              f"{float(out):.12f} vs 2 x {float(main):.12f}")
    g = C9.GapLoss(args_ns(fsw=0.5), b["M_np"])
    base = torch.tensor(0.5, dtype=torch.float64)
    fw = g.feed_weight(dict(M=b["M"]), None)
    out = g.add_terms(base, b["lg"], b["y"], b["m"], dict(ccat=b["ccat"], M=b["M"]), fw)
    s = g.summary(100)
    check("T2c wcl = ccl = 0 (fsw only): the loss object is returned unchanged, decomposition + fsw stats logged",
          out is base and "per-feed CE_cat" in s and "fsw weight" in s and g.acc is None, s)
    say("   example summary line:", s)


def t3():
    b = make_batch(2, dtype=torch.float32)
    trainM = np.random.default_rng(0).integers(5, 301, 100000)
    ok = True
    for a_ in (0.25, 0.5, -0.3):
        g = C9.GapLoss(args_ns(fsw=a_, fsw_norm="batch"), trainM)
        w = g.feed_weight(dict(M=b["M"]), None)
        e = b["M_np"].astype(np.float64) ** a_
        ok &= np.allclose(w.numpy(), e / e.mean(), rtol=1e-6) and abs(float(w.mean()) - 1) < 1e-6
        fw = torch.rand(len(e), generator=torch.Generator().manual_seed(2))
        ok &= np.allclose(g.feed_weight(dict(M=b["M"]), fw).numpy(), fw.numpy() * e / e.mean(), rtol=1e-6)
        g2 = C9.GapLoss(args_ns(fsw=a_, fsw_norm="train"), trainM)
        w2 = g2.feed_weight(dict(M=b["M"]), None)
        ok &= np.allclose(w2.numpy(), e / np.mean(trainM.astype(np.float64) ** a_), rtol=1e-6)
    g0 = C9.GapLoss(args_ns(), trainM)
    sentinel = object()
    ok &= g0.feed_weight(dict(M=b["M"]), sentinel) is sentinel
    check("T3a feed_weight: M^a / batch mean, M^a / training mean, x fw; identity object when --fsw 0", ok)
    g = C9.GapLoss(args_ns(fsw=0.5), trainM)
    w = g.feed_weight(dict(M=b["M"]), None).double()
    lw = listwise_loss(b["lg"].double(), b["y"].double(), b["m"], w, norm="feed", denom=512)
    l = main_ce(b["lg"].double(), b["y"].double(), b["m"])
    v = (b["y"].sum(1) > 0).double()
    check("T3b main loss with the fsw weight == sum_f w_f [valid] CE_f / max_feeds", abs(float(lw) - float((l * w * v).sum() / 512)) < 1e-12)


# ------------------------------------------------------------------ T4 static default-off proof
GAP_IFS = ("a.wcl or a.ccl or a.fsw", "gap is not None")


class StripGap(ast.NodeTransformer):
    def __init__(self):
        self.removed = []

    def visit_ImportFrom(self, n):
        if n.module == "cd_arms9":
            self.removed.append("import cd_arms9")
            return None
        return n

    def visit_Expr(self, n):
        v = n.value
        if (isinstance(v, ast.Call) and isinstance(v.func, ast.Attribute) and v.func.attr == "add_argument" and v.args
                and isinstance(v.args[0], ast.Constant) and v.args[0].value in ("--wcl", "--ccl", "--fsw", "--fsw_norm")):
            self.removed.append(f"add_argument {v.args[0].value}")
            return None
        return self.generic_visit(n)

    def visit_Assign(self, n):
        if len(n.targets) == 1 and getattr(n.targets[0], "id", None) == "gap" and isinstance(n.value, ast.Constant) and n.value.value is None:
            self.removed.append("gap = None")
            return None
        return self.generic_visit(n)

    def visit_If(self, n):
        t = ast.unparse(n.test)
        if t in GAP_IFS:
            self.removed.append(f"if {t}: {ast.unparse(n.body[0])[:60]}")
            return None
        return self.generic_visit(n)


def nodoc(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)) and node.body and isinstance(node.body[0], ast.Expr) \
                and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
            node.body = node.body[1:] or [ast.Pass()]
    return tree


def parse_fn(path):
    """train.py's parse() exec'd alone (no common.py import, which needs a GPU id)."""
    tree = ast.parse(open(path).read())
    fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "parse"][0]
    import cd_arms
    ns = {"argparse": argparse, "POOL_P_DEFAULT": cd_arms.POOL_P_DEFAULT}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), path, "exec"), ns)

    def run(argv):
        old = sys.argv
        sys.argv = ["train.py"] + argv
        try:
            return ns["parse"]()
        finally:
            sys.argv = old
    return run


def t4():
    new_src, old_src = open(f"{W5}/train.py").read(), open(f"{REF}/train.py.txt").read()
    md5 = dict(line.split()[::-1] for line in open(f"{REF}/md5.txt") if line.strip())
    import hashlib
    check("T4a frozen copy intact", hashlib.md5(old_src.encode()).hexdigest() == md5["train.py.txt"])
    st = StripGap()
    stripped = nodoc(st.visit(ast.parse(new_src)))
    exp = ["import cd_arms9", "add_argument --wcl", "add_argument --ccl", "add_argument --fsw", "add_argument --fsw_norm", "gap = None",
           "if a.wcl or a.ccl or a.fsw", "if gap is not None", "if gap is not None", "if gap is not None"]
    got = sorted(r.split(":")[0] for r in st.removed)
    check("T4b removed set == the gap1117 statements", got == sorted(exp), str(st.removed))
    same = ast.dump(stripped) == ast.dump(nodoc(ast.parse(old_src)))
    check("T4c train.py minus gap1117 statements == pre-change train.py (AST, docstrings stripped)", same)
    # 'gap' appears nowhere else
    names = [n for n in ast.walk(ast.parse(new_src)) if isinstance(n, ast.Name) and n.id == "gap"]
    check("T4d `gap` is referenced only by the removed statements", len(names) == 1 + 1 + 3 + 3,
          f"{len(names)} Name nodes (gap = None, gap = setup_gap(...), 3 guards, 3 uses)")
    # parse defaults
    pn, po = parse_fn(f"{W5}/train.py"), parse_fn(f"{REF}/train.py.txt")
    dn, do = vars(pn(["--variant", "w5_x"])), vars(po(["--variant", "w5_x"]))
    extra = {k: dn[k] for k in dn if k not in do}
    check("T4e parse(): previous defaults unchanged; new defaults wcl 0, ccl 0, fsw 0, fsw_norm batch",
          all(dn[k] == do[k] for k in do) and extra == dict(wcl=0.0, ccl=0.0, fsw=0.0, fsw_norm="batch"), str(extra))
    # loop placement: the fsw weight right after `fw = ...`, add_terms right after the main-loss If, both top level in the step loop
    tree = ast.parse(new_src)
    main_fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"][0]
    loop = [n for n in ast.walk(main_fn) if isinstance(n, ast.For) and ast.unparse(n.target) == "(step, fb)"][0]
    body = [ast.unparse(s) for s in loop.body]
    i_fw = [i for i, s in enumerate(body) if s.startswith("fw = None if prand is None")][0]
    i_ls = [i for i, s in enumerate(body) if s.startswith("if a.loss == 'softmax'")][0]
    okp = body[i_fw + 1].startswith("if gap is not None:\n    fw = gap.feed_weight(b, fw)") and \
        body[i_ls + 1].startswith("if gap is not None:\n    loss = gap.add_terms(loss, lg, y, m, b, fw)")
    aux = [s for s in loop.body if ast.unparse(s).startswith("if a.aux_pos > 0")][0]
    no_fw = not any(isinstance(n, ast.Name) and n.id == "fw" for n in ast.walk(aux))
    check("T4f loop: fsw weight right after `fw = ...`, add_terms right after the main loss, both top-level in the step loop "
          "(every batch incl. pool); the K9 aux statement never reads fw", okp and no_fw)


# ------------------------------------------------------------------ T5 / T6 lane + names
def dry(lane, jobs):
    r = subprocess.run(["bash", lane, "0", " ".join(jobs)], env=dict(os.environ, DRY="1"), capture_output=True, text=True)
    return r.returncode, r.stdout, r.stderr


NEW_ARMS = {"wcl05": "--wcl 0.5", "wcl10": "--wcl 1.0", "ccl05": "--ccl 0.5", "fsw25": "--fsw 0.25", "fsw50": "--fsw 0.5",
            "fsw25t": "--fsw 0.25 --fsw_norm train", "fsw50t": "--fsw 0.5 --fsw_norm train"}


def t5():
    old_lane = f"{REF}/lane.sh.txt"
    src = open(old_lane).read()
    arms = sorted({a for st, a in re.findall(r"\b([KL]):([A-Za-z0-9]+)\b", src.split("arm_flags()")[1].split("check_gpu()")[0]) if st == "K"})
    jobs = [f"{st}:{a}:{x}" for a in arms for st in "KL" for x in (951, 958)]
    r_old, r_new = dry(old_lane, jobs), dry(f"{W5}/lane.sh", jobs)
    check(f"T5a DRY of the {len(jobs)} previous K/L jobs ({len(arms)} arms x K/L x 951/958) byte-identical to the frozen lane.sh",
          r_old == r_new and r_new[0] == 0 and r_new[1].count("\n") == len(jobs), f"rc {r_new[0]}")
    pn, po = parse_fn(f"{W5}/train.py"), parse_fn(f"{REF}/train.py.txt")
    def argv(line):
        return line.split("train.py ", 1)[1].split()
    ok_parse = all(vars(po(argv(l))) == {k: v for k, v in vars(pn(argv(l))).items() if k in vars(po(argv(l)))} for l in r_new[1].splitlines())
    guard_off = all(not (lambda a: a.wcl or a.ccl or a.fsw)(pn(argv(l))) for l in r_new[1].splitlines())
    check("T5b every previous K/L command parses identically with the old and the new parser; gap guard False in all of them",
          ok_parse and guard_off)
    lines = {}
    for a, fl in list(NEW_ARMS.items()) + [("ref", ""), ("kw03", "--kd_w 0.3"), ("e4", "--epochs 4"), ("lr4", "--lr 4e-3")]:
        for st in "KL":
            for x in (951, 958):
                rc, out, err = dry(f"{W5}/lane.sh", [f"{st}:{a}:{x}"])
                assert rc == 0 and out.count("\n") == 1, (a, st, x, rc, err)
                lines[(a, st, x)] = out.strip()
    ok = True
    for (a, st, x), l in lines.items():
        ref = lines[("ref", st, x)]
        fl = NEW_ARMS.get(a, {"kw03": "--kd_w 0.3", "e4": "--epochs 4", "lr4": "--lr 4e-3", "ref": ""}[a] if a in ("kw03", "e4", "lr4", "ref") else None)
        exp = ref.replace(f"_ref_s{x}", f"_{a}_s{x}") + (f" {fl}" if fl else "")
        ok &= l == exp
        if a != "ref":
            ns, nr = vars(pn(argv(l))), vars(pn(argv(ref)))
            diff = {k: ns[k] for k in ns if ns[k] != nr[k] and k != "variant"}
            want = {"wcl05": dict(wcl=0.5), "wcl10": dict(wcl=1.0), "ccl05": dict(ccl=0.5), "fsw25": dict(fsw=0.25),
                    "fsw50": dict(fsw=0.5), "fsw25t": dict(fsw=0.25, fsw_norm="train"), "fsw50t": dict(fsw=0.5, fsw_norm="train"),
                    "kw03": dict(kd_w=0.3), "e4": dict(epochs=4.0), "lr4": dict(lr=4e-3)}[a]
            ok &= diff == want
            ok &= ns["variant"] == f"w5_{st.lower()}cd_{a}_s{x}"
    check("T5c new arms (+ kw03 / e4 / lr4) == K/L:ref command + exactly their flags; parsed args differ from ref only there", ok)
    say("   example:", lines[("wcl05", "K", 951)])
    say("   example:", lines[("fsw50t", "L", 958)])
    rc, out, err = dry(f"{W5}/lane.sh", ["K:nosucharm:951"])
    check("T5d unknown arm still refused", rc != 0 and "unknown arm" in err)


def t6():
    src = open(f"{KD}/arm_read8.py").read()
    m_ = re.search(r"rc = (lambda a, st, x: .*?)\n", src)
    rc = eval(m_.group(1))
    vname_src = [l for l in open(f"{KD}/fill.sh").read().splitlines() if l.startswith("vname()")][0]
    files = [f for f in (f"{KD}/logs/fill_jobs_T1117.txt", f"{KD}/logs/fill_jobs_T1117_fswt.txt") if os.path.exists(f)]
    ok, n = True, 0
    for fpath in files:
        jobs = open(fpath).read().split()
        for j in jobs:
            st, a, x = j.split(":")
            fv = subprocess.run(["bash", "-c", f"{vname_src}\nvname {j}"], capture_output=True, text=True).stdout.strip()
            rv = rc(a, st.lower(), int(x))
            rc_, out, _ = dry(f"{W5}/lane.sh", [j])
            lv = re.search(r"--variant (\S+)", out).group(1)
            ok &= rc_ == 0 and fv == rv == lv == f"w5_{st.lower()}cd_{a}_s{x}"
            n += 1
        say(f"   {fpath}: {len(jobs)} jobs, {len(set(jobs))} distinct")
    check(f"T6 names: lane --variant == fill.sh vname == arm_read8 rc lambda for all {n} jobs of {len(files)} jobs files", ok and n > 0)


if __name__ == "__main__":
    for f in (t1, t2, t3, t4, t5, t6):
        try:
            f()
        except Exception as e:  # noqa: BLE001
            import traceback
            check(f.__name__ + " raised", False, traceback.format_exc())
    say(f"SUMMARY: {sum(RES.values())}/{len(RES)} PASS")
    json.dump(RES, open(os.path.join(OUT, "test_gap1117_cpu.json"), "w"), indent=1)
    sys.exit(0 if all(RES.values()) else 1)
