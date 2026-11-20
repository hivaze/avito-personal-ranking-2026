"""witness worker: one heavy CPU process (12 threads) that works through a priority task list; two of them may run side by side.

  cd <repo>/nn_x/w12/stale/witness && CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 setsid nohup \
      ../../../../.venv/bin/python wt_worker.py <worker id> <deadline as UTC HH:MM> [tasks file = tasks.txt] > logs/worker_<id>.out 2>&1 < /dev/null &

tasks file: one task per line, highest priority first, re-read before every pick (lines may be appended while workers run):
    lag <member> <views csv>      -> lagged_score.score_member (subset, CPU fp32)
    sv  <member> <stale|base|both> -> stale_val_cpu.score_member (val, CPU fp32)
A task is ready when the member's checkpoint exists AND, for harness / lane runs still in training (w5_kcd_*, w5_lcd_*), when
nn_x/w5/logs/<variant>.json exists (the selected checkpoint file is rewritten during training; the json marks the end of the run).
A task is claimed with an atomic mkdir in cache/locks (stale locks of dead workers are taken over). The worker stops at the deadline
(checked between tasks) or when the file logs/STOP_<id> exists; it never touches a process or a file outside this directory.
"""
from wt_common import *
import torch
torch.set_num_threads(int(NT))
torch.set_num_interop_threads(1)
import lagged_score as LS
import stale_val_cpu as SVC

WID = sys.argv[1]
hh, mm = (int(x) for x in sys.argv[2].split(':'))
TASKS = HERE / (sys.argv[3] if len(sys.argv) > 3 else 'tasks.txt')
LOCKS = CACHE / 'locks'
LOGF = open(LOGS / f'worker_{WID}.log', 'a')
TIMING = LOGS / 'timing.tsv'


def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); LOGF.write(s + '\n'); LOGF.flush()


def past_deadline():
    t = time.gmtime()
    return (t.tm_hour, t.tm_min) >= (hh, mm) and t.tm_hour < 20      # the run starts after 02:00Z and ends the same day


def ready(v):
    if not ckpt_path(v).exists():
        return False
    if v.startswith(('w5_kcd_', 'w5_lcd_')):
        return done_json(v).exists()
    return True


def outputs(kind, v, arg):
    if kind == 'lag':
        return [LS.pred_path(w, v) for w in (list(VIEWS) if arg == 'all' else arg.split(','))]
    return [SVC.sv_path(v, md) for md in (('stale', 'base') if arg == 'both' else (arg,))]


def claim(name):
    d = LOCKS / name
    try:
        os.mkdir(d)
    except FileExistsError:
        pf = d / 'pid'
        try:
            pid = int(pf.read_text().split()[0])
            alive = 'wt_worker.py' in Path(f'/proc/{pid}/cmdline').read_text()
        except Exception:
            alive = time.time() - d.stat().st_mtime < 60          # lock being created right now
        if alive:
            return False
        log(f'{now()} stale lock {name} taken over')
    (d / 'pid').write_text(f'{os.getpid()} {WID} {now()}\n')
    return True


def pick():
    for ln in TASKS.read_text().splitlines():
        ln = ln.split('#')[0].split()
        if len(ln) != 3:
            continue
        kind, v, arg = ln
        if all(p.exists() for p in outputs(kind, v, arg)) or not ready(v):
            continue
        if claim(f'{kind}__{v}__{arg}'):
            return kind, v, arg
    return None


def main():
    log(f'{now()} worker {WID} pid {os.getpid()} start; deadline {hh:02d}:{mm:02d}Z; tasks {TASKS}')
    sub = vs = None
    idle = 0
    while True:
        if past_deadline() or (LOGS / f'STOP_{WID}').exists():
            log(f'{now()} worker {WID} stops (deadline or STOP file)'); break
        t = pick()
        if t is None:
            idle += 1
            if idle % 10 == 1:
                log(f'{now()} worker {WID}: nothing ready, waiting')
            time.sleep(45)
            continue
        idle = 0
        kind, v, arg = t
        try:
            if kind == 'lag':
                if sub is None:
                    sub = LS.SubSplit()
                sec = LS.score_member(sub, v, list(VIEWS) if arg == 'all' else arg.split(','), log)
            else:
                if vs is None:
                    vs = SVC.ValSplit()
                sec = SVC.score_member(vs, v, ('stale', 'base') if arg == 'both' else (arg,), log)
            with open(TIMING, 'a') as fh:
                for k, s in sec.items():
                    fh.write(f'{now()}\t{WID}\t{kind}\t{v}\t{k}\t{s:.1f}\t{os.getloadavg()[0]:.0f}\n')
        except Exception as e:  # noqa: BLE001
            log(f'{now()} FAILED {kind} {v} {arg}: {type(e).__name__}: {e}')
            (LOCKS / f'{kind}__{v}__{arg}' / 'FAILED').write_text(f'{type(e).__name__}: {e}\n')
            with open(LOGS / 'failed.txt', 'a') as fh:
                fh.write(f'{now()} {kind} {v} {arg}: {type(e).__name__}: {e}\n')
            time.sleep(5)


if __name__ == '__main__':
    main()
