"""GPU tests of the gap1117 loss options (train.py --wcl / --ccl / --fsw, nn_x/w5/cd_arms9.py). GPU N only.

  cd <repo> && CUDA_VISIBLE_DEVICES=N OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 POLARS_MAX_THREADS=12 \
      NUMBA_NUM_THREADS=12 .venv/bin/python lab/nn_x/w12/gap1117/tests/test_gap1117_gpu.py [g3] [g1] [g2]

Every train.py run goes through tests/e2e_gap1117.py, which redirects ALL outputs to tests/out/runs/<name>/ (nothing is written into
nn_x/w5). Each launch is gated on nvidia-smi: GPU N must have <= 3 compute processes and >= 60 GB free (polled every 60 s, 1 h max).
G3 micro-benchmark: category_split forward + backward on the re-cool's batch shapes (B x Mx = 512 x 16 ... 108 x 304), vs the main
   listwise loss; float32 identity on those shapes.
G1 default unchanged (deterministic kernels, E2E_DET=1): the frozen pre-change train.py vs the new train.py, flags OFF, both with the
   K:ref:951 re-cool command (init w5_e10_kd05w192_s951@77500, KD teacher_mean11) shortened to --train_feeds 20000 --epochs 1, cooldown
   mix 0.3 (many pool batches), final eval only, --no_test: identical training gathers, bitwise-identical per-step loss values, loss-
   input logits and optimiser-entry gradient norms, identical log lines (timings removed), bitwise-identical selected / @last
   checkpoints (state_dict), val and holdout predictions.
G2 smoke of each arm (normal kernels): K:<arm>:951 for ref, wcl05, wcl10, ccl05, fsw25, fsw50, fsw25t, fsw50t with --train_feeds 60000
   --epochs 1 --eval_every 100 --select_min_step 100 --cliplog 10 --no_test: finite losses, gap1117 lines logged at every eval with
   finite numbers, add_terms called on every step (pool batches included), same training gathers as ref; step-time overhead vs ref
   (paired over the same steps, eval intervals and the first 5 steps excluded).
Writes tests/out/test_gap1117_gpu.log (+ .json).
"""
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0", "GPU N only: CUDA_VISIBLE_DEVICES=N"
HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
RUNS = OUT / "runs"
RUNS.mkdir(parents=True, exist_ok=True)
R = "<repo>"
W5 = f"{R}/lab/nn_x/w5"
PY = f"{R}/.venv/bin/python"
WRAP = str(HERE / "e2e_gap1117.py")
LOGF = open(OUT / "test_gap1117_gpu.log", "a")
RES = {}
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES="N", OMP_NUM_THREADS="12", MKL_NUM_THREADS="12", OPENBLAS_NUM_THREADS="12",
           POLARS_MAX_THREADS="12", NUMBA_NUM_THREADS="12")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOGF.write(s + "\n")
    LOGF.flush()


def check(name, ok, info=""):
    RES[name] = bool(ok)
    say(f"[{'PASS' if ok else 'FAIL'}] {name} {info}")


def gpu_ok():
    n = subprocess.run(["nvidia-smi", "-i", "0", "--query-compute-apps=pid", "--format=csv,noheader"], capture_output=True,
                       text=True).stdout
    n = len([x for x in n.splitlines() if x.strip()])
    free = int(subprocess.run(["nvidia-smi", "-i", "0", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True).stdout.strip())
    return n, free


def wait_gpu():
    t0 = time.time()
    while True:
        n, free = gpu_ok()
        if n <= 3 and free >= 61440:
            say(f"{time.strftime('%FT%TZ', time.gmtime())} GPU N: {n} compute processes, {free} MiB free -> launch")
            return
        assert time.time() - t0 < 3600, "GPU N never qualified within 1 h"
        say(f"GPU N: {n} processes, {free} MiB free: waiting")
        time.sleep(60)


def lane_args(job):
    r = subprocess.run(["bash", f"{W5}/lane.sh", "0", job], env=dict(ENV, DRY="1"), capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout.strip().split("train.py ", 1)[1].split()


def set_arg(args, flag, val):
    """Replace (or append) `flag val` in an argv list (argparse: last occurrence wins; we keep one)."""
    out, i = [], 0
    while i < len(args):
        if args[i] == flag:
            i += 2
            continue
        out.append(args[i])
        i += 1
    return out + [flag, str(val)] if val is not None else out + [flag]


def launch(train_file, name, args, det):
    d = RUNS / name
    if d.exists():
        shutil.rmtree(d)  # only our own scratch run directories
    d.mkdir(parents=True)
    wait_gpu()
    env = dict(ENV, E2E_DET="1" if det else "0")
    f = open(d / "stdout.txt", "w")
    p = subprocess.Popen([PY, WRAP, train_file, str(d), str(d / "rec.npz")] + args, cwd=W5, env=env, stdout=f, stderr=subprocess.STDOUT)
    say(f"   started {name} pid {p.pid}")
    return p, d


def wait_loaded(p, d):  # train.py logs "steps/epoch" once its data are loaded (lane.sh rule: the next launch only after that)
    while p.poll() is None and "steps/epoch" not in (d / "stdout.txt").read_text():
        time.sleep(10)


import numpy as np  # noqa: E402
import torch  # noqa: E402


# ------------------------------------------------------------------ G3 micro-benchmark
def g3():
    sys.path.insert(0, W5)
    import cd_arms9 as C9
    from model import listwise_loss
    dev = torch.device("cuda:0")
    g = torch.Generator(device=dev).manual_seed(0)
    rows = []
    worst = 0.0
    for B, Mx in ((512, 16), (512, 64), (256, 128), (160, 208), (108, 304)):
        lg = torch.randn(B, Mx, device=dev, generator=g) * 2
        M = torch.randint(max(1, Mx - 15), Mx + 1, (B,), device=dev, generator=g)
        m = torch.arange(Mx, device=dev)[None] < M[:, None]
        cat = torch.randint(2, 54, (B, Mx), device=dev, generator=g)
        y = ((torch.rand(B, Mx, device=dev, generator=g) < 0.03) & m).float()
        y[:, 0] = 1
        t = y / y.sum(1, keepdim=True)
        x = lg.clone().requires_grad_(True)
        cc, cw = C9.category_split(x, t, m, cat)
        ref = -(t * torch.log_softmax(x.masked_fill(~m, -1e4), 1)).sum(1)
        worst = max(worst, float((cc + cw - ref).abs().max()))

        def run_split():
            x.grad = None
            cc, cw = C9.category_split(x, t, m, cat)
            ((cc + cw).sum() / 512).backward()

        def run_main():
            x.grad = None
            listwise_loss(x, y, m, None, norm="feed", denom=512).backward()
        res = []
        for fn in (run_split, run_main):
            for _ in range(5):
                fn()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(50):
                fn()
            torch.cuda.synchronize()
            res.append((time.perf_counter() - t0) / 50 * 1e3)
        rows.append(f"B {B} x Mx {Mx}: category_split fwd+bwd {res[0]:.3f} ms, main listwise loss fwd+bwd {res[1]:.3f} ms")
    for r in rows:
        say("   " + r)
    check("G3 micro-benchmark ran; float32 identity on the re-cool shapes (tol 1e-5)", worst <= 1e-5, f"max |diff| {worst:.2e}")


# ------------------------------------------------------------------ G1 default unchanged
def strip_log(path):
    out = []
    for l in Path(path).read_text().splitlines():
        if l.startswith("==== "):
            continue  # args line (time + new parser keys)
        l = re.sub(r"\d+\.\d+ min", "<t> min", l)
        l = re.sub(r"train time [\d.]+ min", "train time <t>", l)
        out.append(l)
    return out


def g1():
    ref_py = RUNS / "code_ref" / "train.py"  # frozen pre-change train.py as a .py file (runpy)
    ref_py.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(HERE / "ref_pre_gap1117" / "train.py.txt", ref_py)
    base = lane_args("K:ref:951")
    base = set_arg(base, "--variant", "w5_g1117_off")
    for fl, v in (("--train_feeds", 20000), ("--epochs", 1), ("--eval_every", 1000), ("--select_min_step", 10), ("--cliplog", 10),
                  ("--cooldown_mix", 0.3), ("--no_test", None)):
        base = set_arg(base, fl, v)
    say("G1 args:", " ".join(base))
    p1, d1 = launch(str(ref_py), "g1_old", base, det=True)
    wait_loaded(p1, d1)
    p2, d2 = launch(f"{W5}/train.py", "g1_new", base, det=True)
    r1, r2 = p1.wait(), p2.wait()
    check("G1a both deterministic runs finished", r1 == 0 and r2 == 0, f"rc {r1} {r2}")
    if r1 or r2:
        return
    a, b = np.load(d1 / "rec.npz"), np.load(d2 / "rec.npz")
    n = int(len(a["loss"]))
    check("G1b identical training gathers", np.array_equal(a["g_fidx"], b["g_fidx"]) and np.array_equal(a["g_off"], b["g_off"]),
          f"({len(a['g_off']) - 1} steps)")
    check("G1c per-step loss values bitwise identical", np.array_equal(a["loss"], b["loss"]) and n > 0,
          f"({n} steps, loss[0] {a['loss'][0]:.6f}, loss[-1] {a['loss'][-1]:.6f})")
    check("G1d loss-input logits bitwise identical", np.array_equal(a["lg_loss"], b["lg_loss"]), f"({a['lg_loss'].size} values)")
    check("G1e optimiser-entry gradient norms bitwise identical", np.array_equal(a["gnorm"], b["gnorm"]))
    la, lb = strip_log(d1 / "logs/w5_g1117_off.log"), strip_log(d2 / "logs/w5_g1117_off.log")
    check("G1f log lines identical (args line and timings removed)", la == lb, f"({len(la)} lines)")
    ok = True
    for ck in ("w5_g1117_off.pt", "w5_g1117_off@last.pt"):
        sa = torch.load(d1 / "ckpt" / ck, map_location="cpu", weights_only=False)["state_dict"]
        sb = torch.load(d2 / "ckpt" / ck, map_location="cpu", weights_only=False)["state_dict"]
        ok &= sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)
    check("G1g selected and @last checkpoints bitwise identical (state_dict)", ok)
    pa = sorted(p.relative_to(d1) for p in (d1 / "preds").rglob("*.npy"))
    pb = sorted(p.relative_to(d2) for p in (d2 / "preds").rglob("*.npy"))
    ok = pa == pb and len(pa) > 0 and all(np.array_equal(np.load(d1 / p), np.load(d2 / p)) for p in pa)
    check("G1h val / holdout prediction files bitwise identical", ok, f"({len(pa)} files: {[str(p) for p in pa]})")
    check("G1i flags off: add_terms never called", int(b["add_n"]) == 0)


# ------------------------------------------------------------------ G2 smoke + overhead
ARMS = ["ref", "wcl05", "wcl10", "ccl05", "fsw25", "fsw50", "fsw25t", "fsw50t"]


def g2():
    recs = {}
    for arm in ARMS:
        args = lane_args(f"K:{arm}:951")
        args = set_arg(args, "--variant", f"w5_g1117_smoke_{arm}")
        for fl, v in (("--train_feeds", 60000), ("--epochs", 1), ("--eval_every", 100), ("--select_min_step", 100), ("--cliplog", 10),
                      ("--no_test", None)):
            args = set_arg(args, fl, v)
        if arm == "ref":
            say("G2 args (ref):", " ".join(args))
        p, d = launch(f"{W5}/train.py", f"g2_{arm}", args, det=False)
        rc = p.wait()
        check(f"G2 {arm}: run finished", rc == 0, f"rc {rc}")
        if rc:
            say((d / "stdout.txt").read_text()[-3000:])
            continue
        r = np.load(d / "rec.npz")
        recs[arm] = r
        log = (d / "logs" / f"w5_g1117_smoke_{arm}.log").read_text()
        losses = [float(x) for x in re.findall(r" loss (\S+) val_ndcg", log)]
        ok = np.isfinite(r["loss"]).all() and len(losses) > 0 and np.isfinite(losses).all() and np.isfinite(r["gnorm"]).all()
        gl = [l for l in log.splitlines() if l.startswith("gap1117 step")]
        nums = [float(x) for l in gl for x in re.findall(r"-?\d+\.\d+", l)]
        steps = len(r["g_off"]) - 1
        evl = re.findall(r"step (\d+)/(\d+) loss (\S+) val_ndcg (\S+) holdout_ndcg (\S+)", log)
        if arm == "ref":
            check(f"G2 {arm}: finite losses, no gap1117 line, add_terms never called", ok and not gl and int(r["add_n"]) == 0,
                  f"({steps} steps; evals {evl})")
        else:
            check(f"G2 {arm}: finite losses; gap1117 line at every eval with finite numbers; add_terms on every step (pool incl.)",
                  ok and len(gl) == len(evl) and np.isfinite(nums).all() and int(r["add_n"]) == steps and bool(r["add_finite"]),
                  f"({steps} steps, add_terms calls {int(r['add_n'])}, mean added term {np.mean(r['add_val']):.4f}; evals {evl})")
            for l in gl:
                say("     " + l)
        for l in log.splitlines():
            if l.startswith("gap1117:") or "wave-5 LAST" in l or l.startswith("BEST"):
                say("     " + l)
    if "ref" not in recs:
        return
    ref = recs["ref"]
    rt, re_ = ref["step_time"], ref["step_eval"]
    say("   step-time overhead vs ref (paired over identical steps; eval intervals and the first 5 steps excluded):")
    for arm, r in recs.items():
        same = np.array_equal(r["g_fidx"], ref["g_fidx"]) and np.array_equal(r["g_off"], ref["g_off"])
        n = min(len(rt), len(r["step_time"]))
        keep = ~(re_[:n] | r["step_eval"][:n])
        keep[:5] = False
        a_, b_ = r["step_time"][:n][keep], rt[:n][keep]
        say(f"     {arm:7s}: same gathers as ref {same}; {keep.sum()} steps; median step {np.median(a_) * 1e3:.1f} ms (ref "
            f"{np.median(b_) * 1e3:.1f} ms); sum ratio {a_.sum() / b_.sum() - 1:+.2%}; median per-step ratio {np.median(a_ / b_) - 1:+.2%}")
        if arm != "ref":
            RES[f"G2 {arm}: same training gathers as ref"] = bool(same)


if __name__ == "__main__":
    parts = sys.argv[1:] or ["g3", "g1", "g2"]
    say(f"==== {time.strftime('%FT%TZ', time.gmtime())} test_gap1117_gpu.py {parts}")
    for p_ in parts:
        try:
            {"g1": g1, "g2": g2, "g3": g3}[p_]()
        except Exception:  # noqa: BLE001
            import traceback
            check(f"{p_} raised", False, traceback.format_exc())
    say(f"SUMMARY: {sum(RES.values())}/{len(RES)} PASS")
    prev = {}
    jp = OUT / "test_gap1117_gpu.json"
    if jp.exists():
        prev = json.load(open(jp))
    prev.update(RES)
    json.dump(prev, open(jp, "w"), indent=1)
    sys.exit(0 if all(RES.values()) else 1)
