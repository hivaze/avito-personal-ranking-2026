"""KD pool read-out (train-only level). Usage: pool_read.py <arm> <seeds csv> [suffix '' or '@last']
 member deltas vs w5_e10_base (same seeds); N-member arm pool vs N-member base pool; C3 TO pool (28) + arm vs C3 TO pool; corr structure. val + holdout."""
import sys; sys.path.insert(0,'<repo>/src')
from common import *
from metrics import ndcg_per_feed
from stack_val import feed_rank_z
from pathlib import Path
arm=sys.argv[1]; seeds=[int(s) for s in sys.argv[2].split(',')]; suf=sys.argv[3] if len(sys.argv)>3 else ''
W=Path('<repo>/nn_x/w5/preds'); P=Path('<repo>/preds')
fv,Mv,ov=feed_meta('val'); yv=np.load(FLAT/'val/cand_target.npy')
Mh,oh=np.load(W/'holdout/holdout_meta.npy'); Mh=Mh.astype(np.int64); oh=oh.astype(np.int64); yh=np.load(W/'holdout/holdout_y.npy')
ev=lambda s: ndcg_per_feed(np.asarray(s,np.float32),yv,ov,Mv); eh=lambda s: ndcg_per_feed(np.asarray(s,np.float32),yh,oh,Mh)
zv=lambda x: feed_rank_z(ov,Mv,np.asarray(x,np.float32))[:,1]; zh=lambda x: feed_rank_z(oh,Mh,np.asarray(x,np.float32))[:,1]
se=lambda d: d.std()/np.sqrt(len(d)); f=lambda d: f'{d.mean():+.5f}+-{se(d):.5f}'
ld=lambda v,s='': (zv(np.load(W/f'nn_{v}{s}_trainonly_val.npy')), zh(np.load(W/f'holdout/nn_{v}{s}_holdout.npy')))
A=[ld(f'w5_e10_{arm}_s{s}',suf) for s in seeds]; B=[ld(f'w5_e10_base_s{s}',suf) for s in seeds]
print(f'== {arm}{suf} seeds {seeds}')
for s,a,b in zip(seeds,A,B): print(f' member s{s}: val {ev(a[0]).mean():.5f} (base {ev(b[0]).mean():.5f}) hold {eh(a[1]).mean():.5f} (base {eh(b[1]).mean():.5f})')
dv=np.mean([ev(a[0])-ev(b[0]) for a,b in zip(A,B)],0); dh=np.mean([eh(a[1])-eh(b[1]) for a,b in zip(A,B)],0)
print(f' pooled member delta: val {f(dv)} (M<50 {dv[Mv<50].mean():+.5f}, M>=50 {dv[Mv>=50].mean():+.5f}) | holdout {f(dh)}')
pa=(np.mean([a[0] for a in A],0),np.mean([a[1] for a in A],0)); pb=(np.mean([b[0] for b in B],0),np.mean([b[1] for b in B],0))
print(f' {len(seeds)}-member pool: arm val {ev(pa[0]).mean():.5f} hold {eh(pa[1]).mean():.5f} | base val {ev(pb[0]).mean():.5f} hold {eh(pb[1]).mean():.5f} | delta val {f(ev(pa[0])-ev(pb[0]))} hold {f(eh(pa[1])-eh(pb[1]))}')
if len(seeds)>1:
    c=np.corrcoef([a[0] for a in A]); cb=np.corrcoef([b[0] for b in B]); iu=np.triu_indices(len(seeds),1)
    print(f' val member corr: arm-arm {c[iu].mean():.4f} | base-base {cb[iu].mean():.4f} | same-seed arm-base {np.mean([np.corrcoef(a[0],b[0])[0,1] for a,b in zip(A,B)]):.4f}')
c3=np.load(P/'ens_Q2a8s3_to_val.npy').astype(np.float64); n=28; e0=ev(c3)
for k in (len(seeds),):
    p=(c3*n+np.sum([a[0] for a in A],0))/(n+k); pb_=(c3*n+np.sum([b[0] for b in B],0))/(n+k)
    print(f' C3 TO pool val {e0.mean():.5f}; + arm x{k}: {f(ev(p)-e0)} | + base x{k} (control): {f(ev(pb_)-e0)} | arm-only pool vs C3 TO: {f(ev(pa[0])-e0)}; corr(arm pool, C3 TO) {np.corrcoef(pa[0],c3)[0,1]:.4f}')
    for w in (0.3,0.5):
        q=(1-w)*zv(c3)+w*zv(pa[0]); print(f'   {1-w:.1f} z(C3 TO) + {w:.1f} z(arm pool): {f(ev(q)-e0)}')
