"""M6: training curves from logs/<v>.json (history: step, val_ndcg, holdout_ndcg). Plateau level (mean of evals in the last 30% of the
stable phase), value at the cooldown start, peak in the cooldown, last step; arms averaged over seeds. No preds needed."""
import json, numpy as np, sys
from pathlib import Path
L=Path('<repo>/nn_x/w5/logs')
arms={'base':(951,952,953,954),'kd05':(951,952,953,954),'kd10':(951,952,953,954),'kd05w192':(951,952,953,954),'kd05t25w192':(951,952,953,954),'kd05t25w256':(951,952),
      'kw_drop0':(951,952),'kw_e20':(951,952),'e20':(951,952,953,954),'kw_hl2':(951,952),'kw_ff4':(951,952),'kw_id':(951,952),'kx05':(951,952,953,954),'kd05g2w192':(951,952,953,954)}
print(f'{"arm":14s} n | plateau val/hold (last 30% of stable phase) | at cooldown start | best-in-cooldown (hold-selected step) | last | cooldown gain hold | last - best hold | best step / total')
for a,sd in arms.items():
    R=[]
    for s in sd:
        p=L/f'w5_e10_{a}_s{s}.json'
        if not p.exists(): continue
        j=json.load(open(p)); h=j['history']; tot=h[-1]['step']; ep=j['args']['epochs']; tc=int(round(0.8*tot))
        st=np.array([x['step'] for x in h]); v=np.array([x['val_ndcg'] for x in h]); ho=np.array([x['holdout_ndcg'] for x in h])
        pl=(st>0.56*tot)&(st<=tc); cd=st>tc; ib=np.argmax(np.where(cd,ho,-1))
        i0=np.argmin(np.abs(st-tc))
        R.append((v[pl].mean(),ho[pl].mean(),v[i0],ho[i0],v[ib],ho[ib],v[-1],ho[-1],st[ib],tot))
    if not R: continue
    R=np.array(R); m=R.mean(0)
    print(f'{a:14s} {len(R)} | {m[0]:.5f}/{m[1]:.5f} | {m[2]:.5f}/{m[3]:.5f} | {m[4]:.5f}/{m[5]:.5f} | {m[6]:.5f}/{m[7]:.5f} | {m[5]-m[1]:+.5f} | {m[7]-m[5]:+.5f} | {m[8]:.0f}/{m[9]:.0f}')
