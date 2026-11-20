"""Val / holdout NDCG by training step (logs/*.json history), mean over seeds, arm minus base. CPU only."""
import json, numpy as np
L='<repo>/nn_x/w5/logs/'
def cur(arm,seeds=(951,952,953,954)):
    H=[]
    for s in seeds:
        try: h=json.load(open(L+f'w5_e10_{arm}_s{s}.json'))['history']
        except Exception: continue
        H.append({x['step']:(x['val_ndcg'],x['holdout_ndcg'],x['minutes']) for x in h})
    return H
B=cur('base'); steps=[20000,40000,60000,70000,77500,80000,85000,90000,95000,97500,98350]
def at(H,st,i): return np.mean([h[st][i] for h in H if st in h])
print('base        '+' '.join(f'{st//1000:>3d}K v{at(B,st,0):.4f}/h{at(B,st,1):.4f}' for st in steps))
for a in ('kd03','kd05','kd08','kd10','kd05w192','kd05t25w192','kx05','kd05t25w256','kw_drop0'):
    H=cur(a); 
    if not H: continue
    nb=len(H); Bs=B[:nb]
    print(f'{a:12s}'+' '.join(f'{st//1000:>3d}K {at(H,st,0)-at(Bs,st,0):+.4f}/{at(H,st,1)-at(Bs,st,1):+.4f}' for st in steps)+f' | n {nb} | min {np.mean([max(v[2] for v in h.values()) for h in H]):.0f}')
print('gain plateau(77.5K) -> end(98350): val / hold')
for a in ('base','kd03','kd05','kd08','kd10','kd05w192','kd05t25w192','kx05'):
    H=cur(a); print(f'  {a:12s} {at(H,98350,0)-at(H,77500,0):+.4f} / {at(H,98350,1)-at(H,77500,1):+.4f}   plateau level {at(H,77500,0):.4f}/{at(H,77500,1):.4f}  end {at(H,98350,0):.4f}/{at(H,98350,1):.4f}')
