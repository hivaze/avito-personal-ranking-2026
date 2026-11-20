"""M11: structural diversity (G_mix at fixed N=2) and strength cost for the 2-epoch screen families and the remaining 10-epoch arms.
 pureA = pool(A_s1, A_s2); pureX = pool(X_s1, X_s2); mix = mean of pool(A_s1, X_s2), pool(A_s2, X_s1) (different seeds only).
 G_mix = mix - 0.5 (pureA + pureX);  dQ2 = pureX - pureA;  net = mix - pureA = G_mix + dQ2 / 2."""
from lib import *
import json, itertools
def LD(d,v):
    a=d/f'nn_{v}_trainonly_val.npy'; h=d/f'holdout/nn_{v}_holdout.npy'
    if not (a.exists() and h.exists()): return None
    return zv(np.load(a)), zh(np.load(h))
def chk(d):
    m=np.load(d/'holdout/holdout_meta.npy'); return np.array_equal(m[0].astype(np.int64),Mh)
def g2(A,X,label):
    if any(z is None for z in A+X): print(f'{label:30s} MISSING'); return
    res=[]
    for k,e in ((0,ev),(1,eh)):
        pa=e((A[0][k]+A[1][k])/2); px=e((X[0][k]+X[1][k])/2); pm=0.5*(e((A[0][k]+X[1][k])/2)+e((A[1][k]+X[0][k])/2))
        g=pm-0.5*(pa+px); dq=px-pa; net=pm-pa
        cw=lambda Z: np.corrcoef(Z[0][k],Z[1][k])[0,1]; cx=0.5*(np.corrcoef(A[0][k],X[1][k])[0,1]+np.corrcoef(A[1][k],X[0][k])[0,1])
        res.append((g,dq,net,cx/np.sqrt(cw(A)*cw(X))))
    print(f'{label:30s} | G_mix {f(res[0][0])} / {f(res[1][0])} | dQ2 {f(res[0][1])} / {f(res[1][1])} | net {f(res[0][2])} / {f(res[1][2])} | rho_sys {res[0][3]:.4f}/{res[1][3]:.4f}',flush=True)
    out[label]=dict(g=(res[0][0].mean(),se(res[0][0]),res[1][0].mean(),se(res[1][0])),dq=(res[0][1].mean(),res[1][1].mean()),net=(res[0][2].mean(),se(res[0][2]),res[1][2].mean(),se(res[1][2])),rho=(res[0][3],res[1][3]))
out={}
W9=ROOT/'nn_x/w9'
print('2-epoch screen families (seeds 23/24), val / holdout')
for d in (W9/'catgrad/preds',W9/'hs1/preds',W9/'joint/preds',W9/'dw3/preds',AB): assert chk(d), d
ab=[LD(AB,'abl_base_s23'),LD(AB,'abl_base_s24')]
cb=[LD(W9/'catgrad/preds','w9_base_s23'),LD(W9/'catgrad/preds','w9_base_s24')]
db=[LD(W9/'dw3/preds','w9_base_s23'),LD(W9/'dw3/preds','w9_base_s24')]
g2(cb,[LD(W9/'catgrad/preds','w9_catgrad_s23'),LD(W9/'catgrad/preds','w9_catgrad_s24')],'CATGRAD (2ep)')
g2(db,[LD(W9/'dw3/preds','w9_dw3_s23'),LD(W9/'dw3/preds','w9_dw3_s24')],'DW3 (2ep)')
for n in ('hs1','hs1b'): g2(ab,[LD(W9/'hs1/preds',f'w9_{n}_s23'),LD(W9/'hs1/preds',f'w9_{n}_s24')],f'{n.upper()} vs abl_base (2ep)')
for n in ('joint','jointhq0'): g2(ab,[LD(W9/'joint/preds',f'w9_{n}_s23'),LD(W9/'joint/preds',f'w9_{n}_s24')],f'{n.upper()} vs abl_base (2ep)')
for n in ('pop1','pop2','xdrop_pop','xdrop_pre2','xdrop_ctxnd','xdrop_ctxhn_hs','no_retr','no_set','no_sim','no_aux','no_domain','lnorm_batch','time','retr_cat','retr32','retr32u','xrel_more','blocks3','kv64','pre2noid'):
    g2(ab,[LD(AB,f'abl_{n}_s23'),LD(AB,f'abl_{n}_s24')],f'abl_{n} (2ep)')
print('10-epoch arms, seeds 951/952 (n=2)')
b=[ld('w5_e10_base_s951'),ld('w5_e10_base_s952')]; w=[ld('w5_e10_kd05w192_s951'),ld('w5_e10_kd05w192_s952')]
for n in ('kw_id','kw_drop0','kw_e20','kw_hl2','kw_ff4','kd05t25w256','kd05t25w192b3','kx08','kd10'):
    g2(w,[ld(f'w5_e10_{n}_s951'),ld(f'w5_e10_{n}_s952')],f'{n} x kd05w192')
for n,pre in (('pop1','abl_e10_'),('pop2','abl_e10_'),('xdrop_pop','abl_e10_'),('xdrop_pre2','abl_e10_'),('xrel_more','abl_e10_'),('retr32','abl_e10_'),('retr32u','abl_e10_'),('blocks3','abl_e10_'),('latedrop','abl_e10_'),
              ('bs4','w5_e10_'),('bs4lr4','w5_e10_'),('pseudo','w5_e10_'),('trunc','w5_e10_'),('e20','w5_e10_'),('ap1','w5_e10_')):
    g2(b,[ld(f'{pre}{n}_s951'),ld(f'{pre}{n}_s952')],f'{n} x hard base (10ep)')
print('cooldown arms roots 101/102 vs m02 reference (n=2)')
m=[ld('w5_cd_m02ref_r101'),ld('w5_cd_m02ref_r102')]; m6=[ld('r6_cd_m02_r101'),ld('r6_cd_m02_r102')]
for n in ('hzcd','rdrop','sam','m04','fpcd','xp2m10','cbs025'): g2(m,[ld(f'w5_cd_{n}_r101'),ld(f'w5_cd_{n}_r102')],f'cd_{n} x m02ref')
for n in ('d0','d20','rec33','rho70','aug','hcap','m0','c0'): g2(m6,[ld(f'r6_cd_{n}_r101'),ld(f'r6_cd_{n}_r102')],f'r6_cd_{n} x m02')
json.dump(out,open(OUT/'m11_gmix2.json','w'),indent=1,default=float)
