"""Read-out of the CPU fine-tune proxy (11_ftkd_cpu.py): paired per-feed deltas on the held-out folds. Usage: 16_ftkd_read.py <member> <folds csv> <w csv> [seed]"""
import sys, os; sys.path.insert(0,'<repo>/src')
import numpy as np
from sklearn.model_selection import GroupKFold
from common import feed_meta, FLAT
from metrics import ndcg_per_feed
from stack_val import feed_rank_z
member=sys.argv[1]; folds=[int(x) for x in sys.argv[2].split(',')]; WS=[float(x) for x in sys.argv[3].split(',')]; seed=int(sys.argv[4]) if len(sys.argv)>4 else 0
fv,Mv,ov=feed_meta('val'); yv=np.load(FLAT/'val/cand_target.npy').astype(np.float32); ov=ov.astype(np.int64)
uid=fv['user_id'].to_numpy().astype(np.int64); g=np.where(uid==0,-1-np.arange(len(uid)),uid)
P='<repo>/preds/'; to_pool=np.load(P+'ens_W8kd_to_val.npy'); ftfile=np.load(P+f'nn_{member}_n2cpv3s3_val_oof.npy'); tofile=np.load(P+f'nn_{member}_trainonly_val.npy')
zv=lambda x: feed_rank_z(ov,Mv,np.asarray(x,np.float32))[:,1]
se=lambda d: d.std()/np.sqrt(len(d)); f=lambda d: f'{d.mean():+.5f}+-{se(d):.5f}'
ho_all=[]; R={w:[] for w in WS}; ok=[]
for k,(tr_f,ho_f) in enumerate(GroupKFold(n_splits=5).split(np.arange(len(uid)),groups=g)):
    if k not in folds: continue
    fs=[f'out/ftkd_{member}_f{k}_w{w}_s{seed}.npy' for w in WS]
    if not all(os.path.exists(x) for x in fs): print('fold',k,'incomplete'); continue
    ok.append(k); ho_all.append(np.sort(ho_f))
    for w,x in zip(WS,fs): R[w].append(np.load(x))
ho=np.concatenate(ho_all); 
def nd(p):  # p flat over all val cands -> per-feed ndcg on the held-out feeds
    return ndcg_per_feed(p,yv,ov,Mv)[ho]
def comb(w,i): # i = epoch index (0..4) or -1 for EMA final; held-out rows are disjoint across folds, so sum
    return np.sum([r[i] for r in R[w]],0)
print(f'member {member}, folds {ok}, held-out feeds {len(ho)}')
e_to=nd(comb(WS[0],0)); print(f' TO (epoch 0, CPU fp32) {e_to.mean():.5f} | saved TO file {nd(tofile).mean():.5f} | saved GPU FT file (3 seeds) {nd(ftfile).mean():.5f} | W8 TO pool {nd(to_pool).mean():.5f}')
E={}
for w in WS:
    E[w]=nd(comb(w,-1)); print(f' w={w}: by epoch {[round(float(nd(comb(w,i)).mean()),5) for i in range(5)]} | EMA-final {E[w].mean():.5f} | FT-TO {f(E[w]-e_to)}')
zp=zv(to_pool)
for w in WS[1:]:
    print(f' FT-KD(w={w}) - FT-hard: EMA-final {f(E[w]-E[WS[0]])} | per epoch '+' '.join(f(nd(comb(w,i))-nd(comb(WS[0],i))) for i in range(1,5)))
    mh=nd(0.5*zv(comb(WS[0],-1))+0.5*zp); mk=nd(0.5*zv(comb(w,-1))+0.5*zp)
    print(f'   0.5 z(FT member) + 0.5 z(W8 TO pool): hard {mh.mean():.5f} KD {mk.mean():.5f} | KD-hard {f(mk-mh)} | vs TO pool alone: hard {f(mh-nd(to_pool))} KD {f(mk-nd(to_pool))}')
    print(f'   corr on held-out rows: FT-hard vs TO member {np.corrcoef(zv(comb(WS[0],-1))[np.repeat(np.isin(np.arange(len(Mv)),ho),Mv)],zv(tofile)[np.repeat(np.isin(np.arange(len(Mv)),ho),Mv)])[0,1]:.4f} | FT-KD vs TO member {np.corrcoef(zv(comb(w,-1))[np.repeat(np.isin(np.arange(len(Mv)),ho),Mv)],zv(tofile)[np.repeat(np.isin(np.arange(len(Mv)),ho),Mv)])[0,1]:.4f}')
# ---- per-epoch read of the hard-label FT (the saved file is the EMA-final): member level and in the 0.5/0.5 mix with the W8 TO pool
w0=WS[0]; fin_mix=nd(0.5*zv(comb(w0,-1))+0.5*zp); e_fin=E[w0]
print(' hard-label FT by epoch vs its EMA-final (paired): '+' | '.join(f'ep{i} member {f(nd(comb(w0,i))-e_fin)} mix {f(nd(0.5*zv(comb(w0,i))+0.5*zp)-fin_mix)}' for i in range(1,5)))
print(f' mix with EMA-final {fin_mix.mean():.5f}; TO pool alone {nd(to_pool).mean():.5f}; mix with epoch-1 {nd(0.5*zv(comb(w0,1))+0.5*zp).mean():.5f}')
