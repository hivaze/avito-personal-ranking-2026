"""Interim matched-step read of the running KD arms vs kd05w192 (same seed). Reads logs only."""
import re,os,sys
def rd(v):
    p=f'<repo>/nn_x/w5/logs/{v}.log'
    if not os.path.exists(p): return {}
    out={}
    for l in open(p):
        m=re.search(r'step (\d+)/\d+ .*?val_ndcg ([\d.]+).*?holdout_ndcg ([\d.]+)',l)
        if m: out[int(m.group(1))]=(float(m.group(2)),float(m.group(3)))
    return out
ref={s:rd(f'w5_e10_kd05w192_s{s}') for s in (951,952,953,954)}
for arm,seeds in (('kwsub5',(951,952)),('kwsub2',(951,952)),('kwT07',(951,952)),('kwT15',(951,952)),('kd05w224',(951,952,953,954)),('kdmixw192',(951,952,953,954)),('kd05w192gru',(951,952,953,954))):
    for s in seeds:
        a=rd(f'w5_e10_{arm}_s{s}')
        cs=[k for k in sorted(a) if k in ref[s] and k>=5000]
        if not cs: print(arm,s,'no steps'); continue
        print(f'{arm:12s} s{s} last {cs[-1]:6d} | '+' '.join(f'{k//1000}K:{a[k][0]-ref[s][k][0]:+.4f}/{a[k][1]-ref[s][k][1]:+.4f}' for k in cs[-6:]) + f' | mean(all>=5K) val {sum(a[k][0]-ref[s][k][0] for k in cs)/len(cs):+.5f} hold {sum(a[k][1]-ref[s][k][1] for k in cs)/len(cs):+.5f} n={len(cs)}')
