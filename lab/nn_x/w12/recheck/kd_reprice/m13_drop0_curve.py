import json, numpy as np
from pathlib import Path
L=Path('<repo>/nn_x/w5/logs')
def cur(a,sd=(951,952)):
    H=[json.load(open(L/f'w5_e10_{a}_s{s}.json'))['history'] for s in sd]
    st=[x['step'] for x in H[0]]
    return st, np.mean([[x['val_ndcg'] for x in h] for h in H],0), np.mean([[x['holdout_ndcg'] for x in h] for h in H],0)
st,v0,h0=cur('kd05w192'); _,v1,h1=cur('kw_drop0')
print('step   kd05w192 val/hold | kw_drop0 val/hold | delta val / hold   (seeds 951,952 mean)')
for i,s in enumerate(st):
    if s>=60000 or s in (20000,40000): print(f'{s:6d} {v0[i]:.5f}/{h0[i]:.5f} | {v1[i]:.5f}/{h1[i]:.5f} | {v1[i]-v0[i]:+.5f} / {h1[i]-h0[i]:+.5f}')
m=(np.array(st)>55000)&(np.array(st)<=78680)
print(f'plateau mean (55k-78.7k): delta val {np.mean(v1[m]-v0[m]):+.5f} hold {np.mean(h1[m]-h0[m]):+.5f}')
for a in ('kd05w192','kw_drop0','kd05','base'):
    j=json.load(open(L/f'w5_e10_{a}_s951.json')); print(a,'train minutes',round(j['train_minutes'],1), 'params/args d',j['args']['d'],'drop',j['args']['drop'])
