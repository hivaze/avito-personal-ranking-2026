"""Paired member delta of arm vs reference arm on the same seeds. Usage: member_vs.py <arm> <ref> <seeds csv>"""
import sys; sys.path.insert(0,'<repo>/src')
from common import *
from metrics import ndcg_per_feed
from pathlib import Path
arm,ref=sys.argv[1],sys.argv[2]; seeds=[int(s) for s in sys.argv[3].split(',')]
W=Path('<repo>/nn_x/w5/preds')
fv,Mv,ov=feed_meta('val'); yv=np.load(FLAT/'val/cand_target.npy')
Mh,oh=np.load(W/'holdout/holdout_meta.npy'); Mh=Mh.astype(np.int64); oh=oh.astype(np.int64); yh=np.load(W/'holdout/holdout_y.npy')
ev=lambda v: ndcg_per_feed(np.load(W/f'nn_{v}_trainonly_val.npy').astype(np.float32),yv,ov,Mv); eh=lambda v: ndcg_per_feed(np.load(W/f'holdout/nn_{v}_holdout.npy').astype(np.float32),yh,oh,Mh)
se=lambda d: d.std()/np.sqrt(len(d)); f=lambda d: f'{d.mean():+.5f}+-{se(d):.5f}'
dv=[];dh=[]
for s in seeds:
    a,b=f'w5_e10_{arm}_s{s}',f'w5_e10_{ref}_s{s}'; va,vb,ha,hb=ev(a),ev(b),eh(a),eh(b); dv.append(va-vb); dh.append(ha-hb)
    print(f' s{s}: val {va.mean():.5f} (ref {vb.mean():.5f}) hold {ha.mean():.5f} (ref {hb.mean():.5f})')
print(f'== {arm} vs {ref}: member delta val {f(np.mean(dv,0))} | holdout {f(np.mean(dh,0))}')
