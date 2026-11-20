"""Interim (plateau) matched-step read of the RUNNING arms vs kd05w192 same seed, from the training logs only (no decision; final read-outs follow the lead's rule)."""
import re, glob, numpy as np
L='<repo>/nn_x/w5/logs/'
def hist(v):
    out={}
    try:
        for ln in open(L+v+'.log'):
            m=re.match(r'step (\d+)/\d+ loss [\d.]+ val_ndcg ([\d.]+) holdout_ndcg ([\d.]+)',ln)
            if m: out[int(m.group(1))]=(float(m.group(2)),float(m.group(3)))
    except FileNotFoundError: pass
    return out
for a in ('kwsub5','kwsub2','kwT07','kwT15','kd05w224','kd05w192gru','kdmixw192'):
    rows=[]
    for s in (951,952,953,954):
        h=hist(f'w5_e10_{a}_s{s}'); r=hist(f'w5_e10_kd05w192_s{s}')
        st=sorted(set(h)&set(r))
        if not st: continue
        st=[x for x in st if x>=10000]
        if not st: continue
        dv=np.mean([h[x][0]-r[x][0] for x in st]); dh=np.mean([h[x][1]-r[x][1] for x in st]); rows.append((s,max(st),len(st),dv,dh))
    print(a,' | '.join(f's{s} to {mx//1000}K (n={n} evals >=10K): val {dv:+.4f} hold {dh:+.4f}' for s,mx,n,dv,dh in rows) or 'no matched evals yet')
