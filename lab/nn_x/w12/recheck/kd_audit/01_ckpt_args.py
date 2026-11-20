"""Teacher-member checkpoint provenance: holdout / select_on / domain / arch / xt for every member of teacher_mean11/25/w8g1 (CPU only)."""
import torch, sys, glob
R='<repo>'
mem=[f'{R}/nn_x/r6/ckpt/r6_cd_m02_r{r}.pt' for r in range(101,108)]+[f'{R}/nn_x/w5/ckpt/w5_e10_base_s{s}.pt' for s in range(951,955)]
mem+=[f'{R}/nn/ckpt/v23_gru_pre2s_s{s}.pt' for s in range(501,507)]+[f'{R}/nn_x/abl/ckpt/abl_e10_pop1_s{s}.pt' for s in range(951,959)]
mem+=[f'{R}/nn_x/w5/ckpt/w5_e10_kd05w192_s951.pt',f'{R}/nn_x/w5/ckpt/w5_e10_kd05t25w192_s951.pt',f'{R}/nn_x/w5/ckpt/w5_e10_xf0_s951.pt',f'{R}/nn/ckpt/v17_pre2s_s101@77500.pt']
keys=['holdout','select_on','domain','arch','xt','init','seed','epochs','lr','cooldown_mix','aux_pos','aux_anneal','kd_file','kd_w','xfold','d','keep','select_min_step','no_pos','loss_norm','drop','max_hist']
for m in mem:
    try:
        ck=torch.load(m,map_location='cpu',weights_only=False)
    except Exception as e:
        print(m,'ERR',e); continue
    a=ck.get('args',{}); c=ck['cfg']
    print(m.split('/')[-1],{k:(a.get(k) if not isinstance(a.get(k),str) else a.get(k).split('/')[-1]) for k in keys}, 'cfg.d',c.get('d'),'dom',c.get('domain'),'arch',c.get('arch'),'n_xt',len(c.get('xt_names') or []),'extra_cfg',{k:v for k,v in c.items() if k in('attn_drop','retr_prod','retr_uniq','cretr_m')})
