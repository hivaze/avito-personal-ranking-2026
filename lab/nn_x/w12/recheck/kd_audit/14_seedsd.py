"""Seed sd of member val / holdout for base vs KD arms (how sensitive is a KD student as a test bench), from logs json. No heavy CPU."""
import json, numpy as np
L='<repo>/nn_x/w5/logs/'
for a in ('base','kd05','kd05w192','kd05t25w192','kd08','kx05'):
    v=[];h=[]
    for s in (951,952,953,954):
        j=json.load(open(L+f'w5_e10_{a}_s{s}.json'))['history']; last=j[-1]; v.append(last['val_ndcg']); h.append(last['holdout_ndcg'])
    print(f'{a:12s} @last val mean {np.mean(v):.5f} sd {np.std(v,ddof=1):.5f} | hold mean {np.mean(h):.5f} sd {np.std(h,ddof=1):.5f}')
