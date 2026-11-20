"""P7: is a STORED full-feed teacher logit a consistent target on a random SUB-feed? (kwsub uses stored logits renormalised
over the kept set; function matching needs the teacher and the student to see the same view.)
CPU inference of one student checkpoint on val feeds: logits on the full feed vs logits recomputed on a random sub-feed
(label-free keep mask, q fixed). Reports within-feed corr on the kept candidates, top-1 agreement and NDCG on the sub-feed
for (a) the recomputed sub-feed logits and (b) the stale full-feed logits. No GPU, 2 threads, reads nn_x/w5 code read-only."""
import sys, os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "POLARS_MAX_THREADS", "NUMBA_NUM_THREADS"): os.environ[k] = "2"
sys.path.insert(0, '<repo>/nn_x/w5')
import numpy as np, torch
torch.set_num_threads(2)
from data import SplitData
from model import Ranker
ck_name = sys.argv[1] if len(sys.argv) > 1 else 'w5_e10_kd05w192_s951'
n_feeds = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
dev = torch.device('cpu')
va = SplitData('val', dev, verbose=False)
ck = torch.load(f'<repo>/nn_x/w5/ckpt/{ck_name}.pt', map_location=dev, weights_only=False); cfg = ck['cfg']
if cfg.get('xt_names'): va.attach_xt(cfg['xt_names'])
m = Ranker(**cfg).to(dev); m.load_state_dict(ck['state_dict']); m.eval()
rng = np.random.default_rng(0); feeds = np.sort(rng.choice(np.where(va.M_np >= 20)[0], n_feeds, replace=False))
torch.manual_seed(0)
def nd(s, y):  # full-list NDCG, binary gain, one feed
    o = np.argsort(-s, kind='stable'); g = y[o]; d = 1 / np.log2(np.arange(2, len(s) + 2)); n = int(y.sum())
    return float((g * d).sum() / d[:n].sum()) if n else np.nan
res = {q: dict(corr=[], top1=[], nd_sub=[], nd_stale=[], nd_full=[]) for q in (0.15, 0.3, 0.5)}
ref_check = []
with torch.no_grad():
    for fb in va.batches(feeds, 16384, 256, shuffle=False):
        b = va.gather(fb); full = m(b, use_pos=False).float(); cm = b["cmask"].clone(); y = b["y"].clone()
        hm0 = b["hmask"].clone(); ho0 = b["hemb_ok"].clone()
        for q in res:  # q = history-event drop probability (the --aug_hist augmentation of cd_arms.augment_pool_batch)
            hk = torch.rand(hm0.shape) >= q
            b["hmask"] = hm0 & hk; b["hemb_ok"] = ho0 & hk
            sub = m(b, use_pos=False).float()
            for i in range(cm.shape[0]):
                k = cm[i].numpy()
                if hm0[i].sum() < 8: continue
                a, s_ = full[i].numpy()[k], sub[i].numpy()[k]; yy = y[i].numpy()[k]
                r = res[q]; r['corr'].append(np.corrcoef(a, s_)[0, 1]); r['top1'].append(float(a.argmax() == s_.argmax()))
                if yy.sum() > 0: r['nd_sub'].append(nd(s_, yy)); r['nd_stale'].append(nd(a, yy))
            b["hmask"] = hm0; b["hemb_ok"] = ho0
        for i in range(cm.shape[0]):
            ref_check.append(nd(full[i].numpy()[cm[i].numpy()], y[i].numpy()[cm[i].numpy()]))
print(f'{ck_name}: {n_feeds} val feeds (M >= 20); full-feed NDCG on these feeds {np.nanmean(ref_check):.5f}')
for q, r in res.items():
    a, s_ = np.array(r['nd_sub']), np.array(r['nd_stale']); d = a - s_
    print(f' history drop {q}: within-feed corr(full-history logit, dropped-history logit) mean {np.mean(r["corr"]):.4f} (p10 {np.quantile(r["corr"], 0.1):.4f}) | top-1 agreement {np.mean(r["top1"]):.3f}'
          f' | NDCG: dropped-history model output {a.mean():.5f} vs full-history logits {s_.mean():.5f}  delta {d.mean():+.5f}+-{d.std()/np.sqrt(len(d)):.5f} (n {len(d)})')
