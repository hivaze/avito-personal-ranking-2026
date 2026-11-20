"""Real GPU cost of w5 stage-E runs, from logs/RUNNING_w5.txt (START / END with gpu ids) and logs/<v>.json.
CPU only, read-only. GPU-hours of a run = integral over its wall time of 1 / (number of w5 processes on the same GPU).
Other lanes (nn_x/abl, fine-tunes) are not in RUNNING_w5.txt, so the occupancy is a lower bound and GPU-h an upper bound.
  CUDA_VISIBLE_DEVICES="" .venv/bin/python nn_x/w12/recheck/feasibility_value/s1_costs.py"""
import json, re, collections
from datetime import datetime
from pathlib import Path

L = Path('<repo>/nn_x/w5/logs')
from datetime import timezone
ts = lambda s: datetime.strptime(s, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc).timestamp()
start, end = {}, {}
for line in open(L / 'RUNNING_w5.txt'):
    m = re.match(r'(\S+) START pid=(\d+) .*variant=(\S+) gpu=(\d+)', line)
    if m:
        start[m.group(2)] = (ts(m.group(1)), m.group(3), int(m.group(4)))
    m = re.match(r'(\S+) END pid=(\d+) variant=(\S+) rc=(\d+)', line)
    if m:
        end[m.group(2)] = (ts(m.group(1)), int(m.group(4)))
from datetime import timezone
now = datetime.now(timezone.utc).timestamp()
ts = lambda s: datetime.strptime(s, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc).timestamp()
runs = []
for pid, (t0, v, g) in start.items():
    t1, rc = end.get(pid, (now, -1))
    if rc == -1 and now - t0 > 5 * 3600:  # no END line and older than 5 h: a dead / killed job, not an occupant
        continue
    runs.append(dict(pid=pid, v=v, g=g, t0=t0, t1=t1, rc=rc))
bygpu = collections.defaultdict(list)
for r in runs:
    bygpu[r['g']].append(r)
for r in runs:
    ev = sorted({r['t0'], r['t1']} | {x for o in bygpu[r['g']] for x in (o['t0'], o['t1']) if r['t0'] < x < r['t1']})
    gh = 0.0; occ = []
    for a, b in zip(ev[:-1], ev[1:]):
        mid = (a + b) / 2
        n = sum(1 for o in bygpu[r['g']] if o['t0'] <= mid < o['t1'])
        gh += (b - a) / 3600 / max(n, 1); occ.append((b - a, n))
    r['wall_h'] = (r['t1'] - r['t0']) / 3600; r['gpu_h'] = gh
    r['occ'] = sum(d * n for d, n in occ) / max(sum(d for d, n in occ), 1e-9)
    j = L / f"{r['v']}.json"
    if j.exists():
        a = json.load(open(j)); r['d'] = a['args'].get('d'); r['ep'] = a['args'].get('epochs'); r['train_min'] = a.get('train_minutes')
        r['arch'] = a['args'].get('arch'); r['init'] = bool(a['args'].get('init'))
print('today (11-14) finished stage-E 10-epoch runs: wall hours, mean number of w5 processes on the GPU, GPU-hours (upper bound)')
g = collections.defaultdict(list)
for r in runs:
    if r['rc'] == 0 and r.get('ep') and r['v'].startswith('w5_e10_') and r['t0'] > ts('2026-11-14T00:00:00Z'):
        key = (r['d'], r['ep'], round(r['occ']))
        g[key].append(r)
print(f"{'d':>4} {'ep':>4} {'procs/GPU':>9} {'n':>3} {'wall h':>7} {'train min':>9} {'GPU-h/run':>9}")
for k in sorted(g):
    x = g[k]
    print(f"{k[0]:>4} {k[1]:>4} {k[2]:>9} {len(x):>3} {sum(r['wall_h'] for r in x)/len(x):>7.2f} "
          f"{sum(r['train_min'] for r in x)/len(x):>9.1f} {sum(r['gpu_h'] for r in x)/len(x):>9.2f}")
# throughput summary for d192 / 10 epochs
x1 = [r for k, v in g.items() if k[0] == 192 and k[1] == 10 and k[2] <= 1 for r in v]
x4 = [r for k, v in g.items() if k[0] == 192 and k[1] == 10 and k[2] >= 3 for r in v]
if x1 and x4:
    w1 = sum(r['wall_h'] for r in x1) / len(x1); w4 = sum(r['wall_h'] for r in x4) / len(x4)
    o4 = sum(r['occ'] for r in x4) / len(x4)
    print(f"\nd192 x 10 epochs: alone {w1:.2f} h wall = {w1:.2f} GPU-h/run ({len(x1)} runs); shared ({o4:.1f} procs) {w4:.2f} h wall = {w4/o4:.2f} GPU-h/run ({len(x4)} runs)")
    print(f"cluster throughput at 4 procs x the GPUs: {32/w4*24:.0f} d192 runs per day; to the 12-09 deadline (25 days): {32/w4*24*25:.0f} runs = {8*24*25} GPU-h")
