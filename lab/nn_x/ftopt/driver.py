"""Optuna (TPE) driver for the val fine-tune stage of m02 trunks (protocol: nn_x/ftopt/plan.txt).

Each trial fine-tunes the 4 set-A members with one finetune.py config (via ftrun.py, val OOF only), builds the
4-member pool 0.5 z(mean z(ft)) + 0.5 z(mean z(to)) and returns its mean per-feed NDCG gain on TUNE feeds over the
same pool with the production n2cpv3 fine-tune. CONFIRM feeds never enter the objective.

Control files in nn_x/ftopt: PAUSE (launch no new jobs), STOP (ask no new trials; exit when in-flight trials end).
GPU policy: GPUs N and 7 only; <= 2 own jobs per GPU; launch only if the GPU has < 4 compute processes in total
(other processes + own jobs) and >= 40 GB free.
"""
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import optuna

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ftlib import FT_DIR, ROOT, SET_A, Val, code_of, load_base_ft, load_to  # noqa: E402

PY = str(ROOT / ".venv" / "bin" / "python")
GPUS = [<ids>]
MAX_OWN_PER_GPU = 2
MAX_TOTAL_PER_GPU = 4
MIN_FREE_MB = 40 * 1024
N_TRIALS = int(os.environ.get("FTOPT_N_TRIALS", "60"))
MAX_INFLIGHT = 2
OUT = FT_DIR / "out"
LOGD = FT_DIR / "logs"
STUDY_DB = f"sqlite:///{FT_DIR / 'study.db'}"
STUDY = "ftopt_m02_v1"

V3 = ['centrality', 'it_hist_all_luo', 'it_hist_all_luo_rk', 'it_hist_all_luo_zsl', 'it_hist_all_luo_gap',
      *[f'it_hist_et{i}_luo' for i in range(1, 11)], 'seq_cv_invd20', 'seq_cv_invd20_rk', 'seq_cv_pop',
      'seq_max_all', 'seq_top3', 'seq_argmax_hpos', 'ctx_nd_hcent', 'ctx_nd_knn_f1', 'ctx_nd_knn_h',
      'ctx_nd_knn_h_dev', 'ctx_nd_knn_u', 'ctx_nd_knn_u_dev', 'ctx_nd_max', 'ctx_nd_top5h', 'ctx_nd_top5u',
      'ctx_nd_ucent', 'pre_fe', 'pre_fe_z', 'pre_new', 'pre_new_z', 'pre_hmax32_c', 'pre_hmax8_c', 'pre_fe_noid',
      'pre_id_contrib_v2', 'pre_idmax32', 'pre_idmax32_has', 'pre_hmax32_sent']
assert len(V3) == 42
PRESETS = {"v3": "n2c_pre_v3",
           "v3_noctx": ",".join(n for n in V3 if not n.startswith("ctx_nd_") and n != "centrality"),
           "v3_nopre": ",".join(n for n in V3 if not n.startswith("pre_")),
           "none": "none"}

BASE_PARAMS = dict(lr=5e-5, epochs=4, decay="cos", ema="0.99", ema_start=0.5, wd=0.02, drop=0.1,
                   loss_norm="batch", preset="v3", extra_lr_mult=10.0, seeds=1)


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} {msg}"
    print(line, flush=True)
    with open(LOGD / "driver.log", "a") as fh:
        fh.write(line + "\n")


def suggest(tr):
    p = dict(lr=tr.suggest_float("lr", 1e-5, 3e-4, log=True),
             epochs=tr.suggest_int("epochs", 2, 6),
             decay=tr.suggest_categorical("decay", ["const", "cos"]),
             ema=tr.suggest_categorical("ema", ["0", "0.99", "0.995"]))
    if p["ema"] != "0":
        p["ema_start"] = tr.suggest_categorical("ema_start", [0.25, 0.5, 1.0])
    p["wd"] = tr.suggest_categorical("wd", [0.0, 0.005, 0.02, 0.05, 0.1])
    p["drop"] = tr.suggest_categorical("drop", [0.0, 0.1, 0.2])
    p["loss_norm"] = tr.suggest_categorical("loss_norm", ["batch", "feed"])
    p["preset"] = tr.suggest_categorical("preset", ["v3", "v3_noctx", "v3_nopre", "none"])
    if p["preset"] != "none":
        p["extra_lr_mult"] = tr.suggest_categorical("extra_lr_mult", [3.0, 10.0, 30.0])
    p["seeds"] = tr.suggest_int("seeds", 1, 2)
    return p


def ft_args(m, tag, p):
    a = ["--variant", m, "--out", f"{m}_{tag}", "--extra", PRESETS[p["preset"]], "--lr", f"{p['lr']:.6g}",
         "--epochs", str(p["epochs"]), "--decay", p["decay"], "--ema", p["ema"], "--wd", str(p["wd"]),
         "--drop", str(p["drop"]), "--loss_norm", p["loss_norm"], "--seeds", str(p["seeds"]), "--u0_singleton"]
    if p["ema"] != "0":
        a += ["--ema_start", str(p["ema_start"])]
    if p["preset"] != "none":
        a += ["--extra_lr_mult", str(p["extra_lr_mult"])]
    return a


def gpu_state():
    q = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.total", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, timeout=60).stdout
    info = {}
    for ln in q.strip().splitlines():
        i, u, used, tot = [x.strip() for x in ln.split(",")]
        info[int(i)] = dict(uuid=u, free=int(tot) - int(used), pids=[])
    q = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
                       capture_output=True, text=True, timeout=60).stdout
    u2i = {v["uuid"]: k for k, v in info.items()}
    for ln in q.strip().splitlines():
        if not ln.strip():
            continue
        u, pid = [x.strip() for x in ln.split(",")]
        if u in u2i:
            info[u2i[u]]["pids"].append(int(pid))
    return info


class Job:
    def __init__(self, trial_no, m, cmd_args, outdir):
        self.trial_no, self.m, self.cmd_args, self.outdir = trial_no, m, cmd_args, outdir
        self.proc, self.gpu, self.rc, self.t0, self.t1, self.fh = None, None, None, None, None, None


def main():
    LOGD.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    v = Val()
    to_z = v.zmean([load_to(m) for m in SET_A])
    base_ft = [load_base_ft(m) for m in SET_A]
    base_pool = v.pool(base_ft, to_z)
    nd_base = v.nd(base_pool)
    base_mix = {m: v.nd(0.5 * v.z(f) + 0.5 * v.z(load_to(m))) for m, f in zip(SET_A, base_ft)}
    log(f"driver pid {os.getpid()}; baseline set-A pool (n2cpv3) TUNE {nd_base[v.tune].mean():.5f} "
        f"(CONFIRM withheld); N_TRIALS {N_TRIALS}")
    sampler = optuna.samplers.TPESampler(seed=20261112, n_startup_trials=10, multivariate=True, group=True,
                                         constant_liar=True)
    study = optuna.create_study(study_name=STUDY, storage=STUDY_DB, direction="maximize", sampler=sampler,
                                load_if_exists=True)
    if len(study.trials) == 0:
        study.enqueue_trial(BASE_PARAMS, user_attrs={"note": "n2cpv3 re-run (noise reference)"})
    for t in study.trials:  # trials left RUNNING by a killed driver are failed, not resumed
        if t.state == optuna.trial.TrialState.RUNNING:
            study.tell(t.number, state=optuna.trial.TrialState.FAIL)
    inflight = {}  # trial number -> dict(trial, params, jobs, t0)
    queue, running = [], []
    own = {g: 0 for g in GPUS}
    last_gpu_check = 0.0

    def finalize(tn):
        rec = inflight.pop(tn)
        tr, p, jobs = rec["trial"], rec["params"], rec["jobs"]
        bad = [j for j in jobs if j.rc != 0]
        if bad:
            log(f"trial {tn} FAILED ({[(j.m, j.rc) for j in bad]})")
            study.tell(tr, state=optuna.trial.TrialState.FAIL)
            return
        try:
            ft = []
            info = {}
            for j in sorted(jobs, key=lambda j: SET_A.index(j.m)):  # SET_A order (single-member stats below)
                f = j.outdir / f"nn_{j.m}_t{tn}_val_oof.npy"
                ft.append(np.load(f).astype(np.float64))
                js = json.loads((j.outdir / f"{j.m}_t{tn}_ft.json").read_text())
                info[j.m] = dict(oof_by_epoch=[round(x, 5) for x in js["oof_by_epoch"]],
                                 oof_saved=round(js["oof_ndcg_ema_final"] if js.get("oof_ndcg_ema_final") is not None
                                                 else js["oof_ndcg_snapshot"], 5),
                                 min=round((j.t1 - j.t0) / 60, 2))
                for suf in ("test", "test_folds"):  # zero test placeholders (--no_test)
                    tf = j.outdir / f"nn_{j.m}_t{tn}_{suf}.npy"
                    if tf.exists():
                        tf.unlink()
            pool = v.pool(ft, to_z)
            d = v.nd(pool) - nd_base
            obj, se, _, _ = v.stat(d, v.tune)
            ua = dict(tune_delta=obj, tune_se=se, minutes=round((time.time() - rec["t0"]) / 60, 1))
            for w in (0.3, 0.7):
                dw = v.nd(v.pool(ft, to_z, w)) - nd_base
                ua[f"tune_delta_w{w}"] = v.stat(dw, v.tune)[0]
            dft = v.nd(v.zmean(ft)) - v.nd(v.zmean(base_ft))
            ua["tune_delta_ftonly"] = v.stat(dft, v.tune)[0]
            for m, f in zip(SET_A, ft):
                dm = v.nd(0.5 * v.z(f) + 0.5 * v.z(load_to(m))) - base_mix[m]
                ua[f"tune_mix_{m}"] = v.stat(dm, v.tune)[0]
            for k, val in ua.items():
                tr.set_user_attr(k, val)
            tr.set_user_attr("members", info)
            study.tell(tr, obj)
            row = dict(trial=tn, value=obj, params=p, **ua, members=info)
            with open(FT_DIR / "trials.jsonl", "a") as fh:
                fh.write(json.dumps(row) + "\n")
            best = study.best_trial
            log(f"trial {tn} done: TUNE pool delta {obj:+.5f} +- {se:.5f} (w0.3 {ua['tune_delta_w0.3']:+.5f}, "
                f"w0.7 {ua['tune_delta_w0.7']:+.5f}, ft-only {ua['tune_delta_ftonly']:+.5f}) "
                f"{ua['minutes']} min | best #{best.number} {best.value:+.5f} | params {p}")
        except Exception:
            log(f"trial {tn} scoring error: {traceback.format_exc()}")
            study.tell(tr, state=optuna.trial.TrialState.FAIL)

    while True:
        # 1. reap finished jobs
        for j in list(running):
            rc = j.proc.poll()
            if rc is not None:
                j.rc, j.t1 = rc, time.time()
                running.remove(j)
                own[j.gpu] -= 1
                j.fh.close()
                log(f"job t{j.trial_no} {j.m} gpu{j.gpu} pid {j.proc.pid} exit {rc} ({(j.t1 - j.t0) / 60:.1f} min)")
        for tn in [tn for tn, r in inflight.items() if all(j.rc is not None for j in r["jobs"])]:
            finalize(tn)
        stop = (FT_DIR / "STOP").exists()
        # 2. ask new trials
        n_done_or_running = len([t for t in study.trials if t.state != optuna.trial.TrialState.WAITING])
        while not stop and len(inflight) < MAX_INFLIGHT and n_done_or_running < N_TRIALS:
            tr = study.ask()
            p = suggest(tr)  # for an enqueued trial the suggest calls return its fixed params
            tn = tr.number
            od = OUT / f"t{tn}"
            jobs = []
            for m in sorted(SET_A, key=lambda m: 0 if "gru" in m else 1):  # slowest member first
                cmd = [PY, str(FT_DIR / "ftrun.py"), "--code", code_of(m), "--outdir", str(od), "--no_test", "--"] + \
                      ft_args(m, f"t{tn}", p)
                jobs.append(Job(tn, m, cmd, od))
            inflight[tn] = dict(trial=tr, params=p, jobs=jobs, t0=time.time())
            queue.extend(jobs)
            n_done_or_running += 1
            log(f"trial {tn} asked: {p}")
        # 3. launch queued jobs
        if queue and not (FT_DIR / "PAUSE").exists() and time.time() - last_gpu_check > 20:
            last_gpu_check = time.time()
            try:
                st = gpu_state()
            except Exception as e:  # nvidia-smi hiccup: try again next round
                log(f"nvidia-smi error {e}")
                st = None
            if st is not None:
                mine = {j.proc.pid for j in running}
                for g in GPUS:
                    others = [pid for pid in st[g]["pids"] if pid not in mine]
                    if queue and own[g] < MAX_OWN_PER_GPU and len(others) + own[g] < MAX_TOTAL_PER_GPU \
                            and st[g]["free"] >= MIN_FREE_MB:
                        j = queue.pop(0)
                        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), OMP_NUM_THREADS="8", MKL_NUM_THREADS="8",
                                   POLARS_MAX_THREADS="8", PYTHONUNBUFFERED="1")
                        j.outdir.mkdir(parents=True, exist_ok=True)
                        fh = open(j.outdir / f"{j.m}.stdout", "w")
                        j.proc = subprocess.Popen(j.cmd_args, env=env, stdout=fh, stderr=subprocess.STDOUT,
                                                  cwd=str(ROOT))
                        j.gpu, j.t0, j.fh = g, time.time(), fh
                        own[g] += 1
                        running.append(j)
                        log(f"launch t{j.trial_no} {j.m} on gpu{g} pid {j.proc.pid} (others on gpu: {len(others)})")
                        break  # one launch per GPU check, so nvidia-smi sees it before the next one
        if stop and not inflight:
            log("STOP file present and no trials in flight: exiting")
            break
        if not inflight and n_done_or_running >= N_TRIALS:
            log("all trials done: exiting")
            break
        time.sleep(10)


if __name__ == "__main__":
    main()
