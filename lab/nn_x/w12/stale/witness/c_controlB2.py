"""witness control B' (DECLARATION.txt block 3): stale-val on CPU against the stored GPU files, numerics offset cancelled (B'1) and at pool level (B'2).
  cd <repo>/nn_x/w12/stale/witness && WT_THREADS=4 CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 ../../../../.venv/bin/python c_controlB2.py"""
import os
os.environ.setdefault('WT_THREADS', '4')
from wt_common import *
import stale_val_cpu as SVC
import wt_read as WR
OUT = []


def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); OUT.append(s)


wt = WR.Witness(); V = wt.V
ev = lambda x: ndcg_fast(feed_z(V.off, V.M, x), V.y, V.off, V.M)
P(f"=== witness control B' {now()} (val 14,423 feeds; +- = 1 SE paired per feed) ===")
P("B'1: effect of the stale override per member: E_cpu = CPU stale - CPU no-override; E_gpu = stored GPU stale - stored GPU val")
ok1 = True; dd = []
for v in ('w5_cd_hzcd_r101', 'w5_cd_m02ref_r101', 'abl_pop1_s23', 'abl_base_s23'):
    ec = ev(np.load(SVC.sv_path(v, 'stale'))) - ev(np.load(SVC.sv_path(v, 'base'))); eg = ev(np.load(stored_stale(v))) - ev(np.load(stored_val(v)))
    d = ec - eg; ok = abs(d.mean()) <= 2 * se(d); ok1 &= ok; dd.append(d.mean())
    P(f'  {v:22s}: E_cpu {fmt(ec)} | E_gpu {fmt(eg)} | E_cpu - E_gpu {fmt(d)} ({d.mean() / se(d):+.2f} SE) -> {"ok" if ok else "FAIL"}')
m4 = float(np.mean(dd)); ok1 &= abs(m4) <= 0.00015
P(f"  mean of E_cpu - E_gpu over the four members: {m4:+.5f} (bar +-0.00015) -> B'1 {'holds' if ok1 else 'FAILS'}")
P("B'2: 4-pool w5_cd_hzcd_r101-104 vs 4-pool w5_cd_m02ref_r101-104, stale-val delta (the 11-12 closure read)")
A = [f'w5_cd_hzcd_r{r}' for r in (101, 102, 103, 104)]; B = [f'w5_cd_m02ref_r{r}' for r in (101, 102, 103, 104)]
pc = lambda mem, kind: ndcg_fast(np.mean([V.z(kind, v) for v in mem], 0), V.y, V.off, V.M)
dc = pc(A, 'stale') - pc(B, 'stale'); dg = pc(A, 'gstale') - pc(B, 'gstale'); d = dc - dg
ok2 = abs(d.mean()) <= 2 * se(d) and abs(d.mean()) <= 0.0003
P(f'  CPU fp32 {fmt(dc)} | stored GPU bf16 {fmt(dg)} | CPU - GPU {fmt(d)} ({d.mean() / se(d):+.2f} SE) -> {"ok" if ok2 else "FAIL"}')
dv = pc(A, 'val') - pc(B, 'val')
P(f'  for reference, the same pools on val (stored GPU val files): {fmt(dv)}')
P(f"B'2 {'holds' if ok2 else 'FAILS'}")
P(f"CONTROL B': {'VALID' if ok1 and ok2 else 'NOT VALID'}")
(LOGS / 'control_Bprime.txt').write_text('\n'.join(OUT) + '\n')
