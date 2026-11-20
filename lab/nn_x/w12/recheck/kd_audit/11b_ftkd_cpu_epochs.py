"""CPU proxy of the val fine-tune (ft_w5.sh recipe: n2c_pre_v3 extras, lr 5e-5, cos decay, EMA 0.99 from 0.5 epoch, 4 epochs, u0 singleton groups,
 5-fold GroupKFold by user) for ONE student, with and without a KD term in the fine-tune:
   target = (1-w) y/sum(y) + w softmax(teacher_val),  teacher_val = mean per-feed-centred TRAIN-ONLY logits of the 8 wide students on val (label-free).
 Arms share seed, batch order and dropout stream (paired). fp32 on CPU (the GPU recipe runs bf16 autocast), so only the paired deltas are read.
 Usage: 11_ftkd_cpu.py <member> <folds csv> <w csv> [seed]   -> out/ftkd_<member>_f<k>_w<w>_s<seed>.npy (held-out fold preds per epoch + EMA final)"""
import sys, os, copy, math, time
sys.path.insert(0,'<repo>/nn_x/w5'); sys.path.insert(0,'<repo>/src')
import numpy as np, torch
torch.set_num_threads(12)
from sklearn.model_selection import GroupKFold
from data import SplitData, feat_exists
from model import Ranker, listwise_loss
from metrics import ndcg_per_feed
R='<repo>'; dev=torch.device('cpu')
member=sys.argv[1]; folds=[int(x) for x in sys.argv[2].split(',')]; WS=[float(x) for x in sys.argv[3].split(',')]; seed=int(sys.argv[4]) if len(sys.argv)>4 else 0
os.makedirs('out',exist_ok=True)
# ---- presets copied from nn_x/w5/finetune.py (n2c_pre_v3)
ITEM_PRESET = ["it_cand_cnt_split", "it_cand_cnt_split_rk", "it_cand_cnt_split_z", "it_cand_cnt_split_gap", "it_cand_cnt_train", "it_hist_all", "it_hist_all_rk", "it_hist_all_z", "it_hist_all_gap",
               *[f"it_hist_et{i}" for i in range(1, 11)], "it_tr_coec", "it_tr_coec_rk", "it_tr_coec_z", "it_tr_coec_gap", "it_tr_ctr", "it_tr_pos", "u_tr_ctr", "u_tr_feeds", "hour", "centrality"]
ITEM_SAFE = [n for n in ITEM_PRESET if n not in ("it_cand_cnt_split", "it_cand_cnt_split_gap")]
ITEM_LUO = [n for n in ITEM_SAFE if not n.startswith("it_hist_")] + ["it_hist_all_luo", "it_hist_all_luo_rk", "it_hist_all_luo_z", "it_hist_all_luo_gap", *[f"it_hist_et{i}_luo" for i in range(1, 11)]]
COVIS_PRESET = ["seq_cv_invd20", "seq_cv_invd20_rk", "seq_cv_invd20_z", "seq_cv_w5", "seq_cv_fwd5", "seq_cv_t1", "seq_cv_seqs", "seq_cv_pop", "seq_cv_norm"]
SEQ_PRESET = ["seq_max_all", "seq_top3", "seq_dcos_hl5", "seq_nd90_cnt", "seq_argmax_hpos"]
CTX_PRESET = ["ctx_nd_c80", "ctx_nd_c90", "ctx_nd_c95", "ctx_nd_dupb_f1", "ctx_nd_dupb_h", "ctx_nd_dupb_u", "ctx_nd_hcent", "ctx_nd_knn_f1", "ctx_nd_knn_h", "ctx_nd_knn_h_dev", "ctx_nd_knn_u", "ctx_nd_knn_u_dev",
              "ctx_nd_max", "ctx_nd_mean", "ctx_nd_top5h", "ctx_nd_top5u", "ctx_nd_ucent"]
ITEM_CLEAN = [n for n in ITEM_LUO if not n.startswith("it_cand_cnt_split")]
PRE = ["pre_fe", "pre_fe_z", "pre_new", "pre_new_z", "pre_hmax32", "pre_hmax8", "pre_fe_noid", "pre_id_contrib", "pre_idmax32"]
n2c_pre = ITEM_CLEAN + COVIS_PRESET + SEQ_PRESET + CTX_PRESET + PRE
v2 = [n for n in n2c_pre if n != "hour" and not n.startswith(("it_tr_", "u_tr_", "it_cand_cnt_train"))]
v2 = [("pre_id_contrib_v2" if n == "pre_id_contrib" else n) for n in v2]
PHANTOM = {"seq_cv_invd20_z", "seq_cv_w5", "seq_cv_fwd5", "seq_cv_t1", "seq_cv_seqs", "seq_dcos_hl5", "seq_nd90_cnt", "ctx_nd_c80", "ctx_nd_c90", "ctx_nd_c95", "ctx_nd_dupb_f1", "ctx_nd_dupb_h", "ctx_nd_dupb_u", "ctx_nd_mean"}
REN = {"pre_hmax32": "pre_hmax32_c", "pre_hmax8": "pre_hmax8_c", "it_hist_all_luo_z": "it_hist_all_luo_zsl"}
names = [REN.get(n, n) for n in v2 if n not in PHANTOM and n != "seq_cv_norm"] + ["pre_idmax32_has", "pre_hmax32_sent"]
names = [n for n in names if all(feat_exists(sp, n) for sp in ("val", "test"))]
ck=torch.load(f'{R}/nn_x/w5/ckpt/{member}.pt',map_location='cpu',weights_only=False)
va=SplitData('val',dev,max_hist=ck['args']['max_hist']); va.attach_xt(ck['cfg']['xt_names']); va.attach_extra(names); print('extras',len(names),flush=True)
M=va.M_np; off=va.cand_off_np; fid=np.repeat(np.arange(va.n),M)
tl=[]
for a in ('kd05w192','kd05t25w192'):
    for s in (951,952,953,954):
        x=np.load(f'{R}/nn_x/w5/preds/nn_w5_e10_{a}_s{s}_trainonly_val.npy').astype(np.float64); tl.append(x-np.bincount(fid,weights=x)[fid]/M[fid])
teacher=torch.from_numpy(np.mean(tl,0).astype(np.float32))
def build():
    m=Ranker(**dict(ck['cfg'])); m.add_extra(len(names)); m.load_state_dict(ck['state_dict'],strict=False)
    with torch.no_grad(): m.dom.weight[2]=m.dom.weight[1]
    m.infer_domain=2; return m
@torch.no_grad()
def predict(model,feeds):
    model.eval(); out=torch.zeros(va.n_cand)
    for fb in va.batches(feeds,65536,1024,shuffle=False):
        b=va.gather(fb); va.scatter(out,b,model(b,use_pos=False))
    model.train(); return out.numpy()
g=va.user_id.astype(np.int64); u0=g==0; g=np.where(u0,-1-np.arange(va.n),g)
E,LR,WD,WARM,EMA,EMA0,XM=int(os.environ.get("FT_EPOCHS","4")),5e-5,0.02,30,0.99,0.5,10.0
y_np=va.target_np
def nd(p,ho):
    rows=np.concatenate([np.arange(off[f],off[f]+M[f]) for f in ho]); Ms=M[ho]; o2=np.concatenate([[0],np.cumsum(Ms)[:-1]]); return ndcg_per_feed(p[rows],y_np[rows],o2,Ms)
for k,(tr_f,ho_f) in enumerate(GroupKFold(n_splits=5).split(np.arange(va.n),groups=g)):
    if k not in folds: continue
    ho=np.sort(ho_f)
    for w in WS:
        outp=f'out/ftkd_{member}_f{k}_w{w}_s{seed}_e{E}.npy'
        if os.path.exists(outp): print('exists',outp); continue
        torch.manual_seed(1000*k+seed); rng=np.random.default_rng(1000*k+seed); model=build(); t0=time.time()
        res=[predict(model,ho_f)]
        ex=[p for n,p in model.named_parameters() if n.startswith('extra.')]; base=[p for n,p in model.named_parameters() if not n.startswith('extra.')]
        opt=torch.optim.AdamW([{'params':base,'mult':1.0},{'params':ex,'mult':XM}],lr=LR,weight_decay=WD)
        step=0; S_total=None; ema=None; params=list(model.parameters())
        for ep in range(1,E+1):
            model.train(); jobs=va.batches(tr_f,8192,128,shuffle=True,rng=rng); jobs=[jobs[i] for i in rng.permutation(len(jobs))]
            if S_total is None: S_total=len(jobs)*E; ema0=int(EMA0*len(jobs))
            for fb in jobs:
                lr=LR*min(1.0,(step+1)/WARM)*0.5*(1+math.cos(math.pi*min(step,S_total)/S_total))
                for gp in opt.param_groups: gp['lr']=lr*gp['mult']
                b=va.gather(fb); X=model.dense(b); lg=model(b,X,use_pos=False); m=b['cmask']; y=b['y']
                if w>0:
                    qt=torch.softmax(teacher[b['ci']].masked_fill(~m,-1e4),1)*m
                    y=(1-w)*y/y.sum(1,keepdim=True).clamp_min(1)+w*qt*(y.sum(1,keepdim=True)>0)
                loss=listwise_loss(lg,y,m,norm='batch',denom=128)
                opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
                if step>=ema0:
                    with torch.no_grad():
                        if ema is None:
                            for attr in ('aux_out','intent_logp'):
                                if hasattr(model,attr): delattr(model,attr)
                            ema=copy.deepcopy(model); ema_params=list(ema.parameters())
                        else: torch._foreach_lerp_(ema_params,params,1.0-EMA)
                step+=1
            res.append(predict(model,ho_f)); print(f'  fold {k} w {w} epoch {ep}: held-out NDCG {nd(res[-1],ho).mean():.5f} (epoch0 {nd(res[0],ho).mean():.5f}) steps {step} {(time.time()-t0)/60:.1f} min',flush=True)
        res.append(predict(ema,ho_f)); np.save(outp,np.stack(res).astype(np.float32))
        print(f'fold {k} w {w}: EMA-final {nd(res[-1],ho).mean():.5f} | by epoch {[round(float(nd(r,ho).mean()),5) for r in res[:-1]]} | {(time.time()-t0)/60:.1f} min',flush=True)
