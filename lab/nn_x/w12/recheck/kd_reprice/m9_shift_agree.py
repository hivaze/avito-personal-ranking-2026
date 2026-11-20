"""M9: does student agreement degrade under the train->test shift? Mean pairwise z-corr within a family on holdout (in-period),
val, test-public, test-private (label-free). If students agree equally everywhere there is no 'fidelity gap' for transductive KD to fix."""
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
from lib import *
import itertools
ft_,Mt,ot=feed_meta('test'); ot=ot.astype(np.int64); ts=ft_['timestamp'].to_numpy()
pub=ts<=1790497920; rows=np.repeat(pub,Mt)
zt=lambda x: feed_rank_z(ot,Mt,np.asarray(x,np.float32))[:,1]
def tst(v):
    D=W if v.startswith('w5_') else (AB if v.startswith('abl_') else P)
    return zt(np.load(D/f'nn_{v}_trainonly_test.npy'))
print(f'public feeds {pub.sum()} private {(~pub).sum()}')
for name,vs in (('wide (kd05w192)',fam('kd05w192')),('wide t25',fam('kd05t25w192')),('narrow kd05',fam('kd05')),('hard base',fam('base')),('pop1kd',fam('pop1kd05t25',(951,952,953,954),'abl_e10_'))):
    T=[tst(v) for v in vs]; Z=[ld(v) for v in vs]
    pr=list(itertools.combinations(range(len(vs)),2))
    c=lambda X,m=None: np.mean([np.corrcoef(X[i] if m is None else X[i][m],X[j] if m is None else X[j][m])[0,1] for i,j in pr])
    print(f'{name:16s} within-corr holdout {c([z[1] for z in Z]):.4f} | val {c([z[0] for z in Z]):.4f} (M<50 {c([z[0] for z in Z],np.repeat(Mv<50,Mv)):.4f} M>=50 {c([z[0] for z in Z],np.repeat(Mv>=50,Mv)):.4f}) | public {c(T,rows):.4f} | private {c(T,~rows):.4f} (M>=50 {c(T,(~rows)&np.repeat(Mt>=50,Mt)):.4f})')
# cross: wide vs pop1kd on each split
A=fam('kd05t25w192'); B=fam('pop1kd05t25',(951,952,953,954),'abl_e10_')
TA=[tst(v) for v in A]; TB=[tst(v) for v in B]; ZA=[ld(v) for v in A]; ZB=[ld(v) for v in B]
cx=lambda X,Y,m=None: np.mean([np.corrcoef(x if m is None else x[m],y if m is None else y[m])[0,1] for i,x in enumerate(X) for j,y in enumerate(Y) if i!=j])
print(f'cross wide-t25 x pop1kd: holdout {cx([z[1] for z in ZA],[z[1] for z in ZB]):.4f} | val {cx([z[0] for z in ZA],[z[0] for z in ZB]):.4f} | public {cx(TA,TB,rows):.4f} | private {cx(TA,TB,~rows):.4f}')
