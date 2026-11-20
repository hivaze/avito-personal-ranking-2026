"""print the eval curve of a cb_member_cpu.py run next to the GPU record's curve (nn_x/cb26/logs/e3_xm_yeti.log, val half A).
usage: curve.py <tag>"""
import json, re, sys
import numpy as np
tag = sys.argv[1]
r = json.load(open(f"<repo>/nn_x/w12/cb/infra/out/{tag}.curve.json"))
gpu = {}
for line in open("<repo>/nn_x/cb26/logs/e3_xm_yeti.log"):
    m = re.match(r"^(\d+):\s+test: ([0-9.]+)", line)
    if m: gpu[int(m.group(1)) + 1] = float(m.group(2))  # log index is 0-based -> trees built (index 100 = 101 trees)
it = np.array([x[0] for x in r]); t = np.array([x[1] for x in r]); v = np.array([x[2]["validation:NDCG:type=Base"] for x in r])
print(f"{tag}: {len(r)} iterations, {t[-1] / 60:.1f} min, median s/iter {np.median(np.diff(t)):.2f}, best {v.max():.5f} at {it[v.argmax()]}")
print("trees  cpu_sel_ndcg  gpu_record_halfA  diff   minutes")
for k in sorted(set([1, 25, 50] + list(range(101, 2600, 100)) + [int(it[-1])])):
    if k <= it[-1]:
        g = gpu.get(k); c = v[k - 1]
        print(f"{k:5d}  {c:.5f}       {('%.5f' % g) if g else '   -   '}        {('%+.5f' % (c - g)) if g else ''}  {t[k - 1] / 60:6.1f}")
