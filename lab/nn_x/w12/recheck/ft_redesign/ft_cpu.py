"""CPU port of nn_x/w5/finetune.py (n2cpv3 recipe) for PILOT experiments of fine-tune variants. NEW file; imports the
UNCHANGED nn_x/w5/model.py and data.py read-only (never nn_x/w5/common.py, which requires a GPU id). float32 on CPU (the GPU
recipe runs bf16 autocast), so numbers are compared only between CPU runs (baseline arm re-run here) and sanity-checked
against the saved GPU OOF. Writes ONLY into nn_x/w12/recheck/ft_redesign/out/.

Arms (all else = n2cpv3: AdamW lr 5e-5 wd 0.02, warmup 30, cosine, EMA 0.99 from 0.5 epoch, 4 epochs, extra n2c_pre_v3 x10 lr,
GroupKFold(5) by user with user-0 singletons, clip 1.0, softmax loss batch-normalised):
  --kd_file F --kd_w W --kd_T T : soft targets, target = (1-W) y/sum(y) + W softmax(teacher/T) (teacher = label-free train-only
                                   pool logits on val; identical to train.py's KD target); --kd_all also trains feeds without positives
  --l2sp L                       : L2 anchor to the initial (train-only) weights, loss += L/2 * ||theta - theta0||^2 (trunk only)
  --lr / --epochs / --extra none : plain recipe changes
Per-epoch held-out predictions (raw weights) and the final-EMA predictions are saved: out/ftcpu_<variant>_<tag>_oof.npy [E+2, n_cand].
"""
import argparse, copy, math, os, sys, time, json
assert os.environ.get('CUDA_VISIBLE_DEVICES', None) == '', 'CPU only: run with CUDA_VISIBLE_DEVICES=""'
import numpy as np, torch
torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '12')))
sys.path.insert(0, '<repo>/nn_x/w5'); sys.path.insert(0, '<repo>/src')
from sklearn.model_selection import GroupKFold
from data import SplitData, feat_exists
from model import Ranker, listwise_loss
from metrics import ndcg
from pathlib import Path
ROOT = Path('<repo>'); CK = ROOT / 'nn_x/w5/ckpt'; OUT = ROOT / 'nn_x/w12/recheck/ft_redesign/out'; OUT.mkdir(exist_ok=True)
N2C_PRE_V3 = ['centrality', 'it_hist_all_luo', 'it_hist_all_luo_rk', 'it_hist_all_luo_zsl', 'it_hist_all_luo_gap', 'it_hist_et1_luo', 'it_hist_et2_luo', 'it_hist_et3_luo', 'it_hist_et4_luo', 'it_hist_et5_luo', 'it_hist_et6_luo', 'it_hist_et7_luo', 'it_hist_et8_luo', 'it_hist_et9_luo', 'it_hist_et10_luo', 'seq_cv_invd20', 'seq_cv_invd20_rk', 'seq_cv_pop', 'seq_max_all', 'seq_top3', 'seq_argmax_hpos', 'ctx_nd_hcent', 'ctx_nd_knn_f1', 'ctx_nd_knn_h', 'ctx_nd_knn_h_dev', 'ctx_nd_knn_u', 'ctx_nd_knn_u_dev', 'ctx_nd_max', 'ctx_nd_top5h', 'ctx_nd_top5u', 'ctx_nd_ucent', 'pre_fe', 'pre_fe_z', 'pre_new', 'pre_new_z', 'pre_hmax32_c', 'pre_hmax8_c', 'pre_fe_noid', 'pre_id_contrib_v2', 'pre_idmax32', 'pre_idmax32_has', 'pre_hmax32_sent']  # = the logged list of every n2cpv3 run

def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--variant', required=True); p.add_argument('--tag', required=True)
    p.add_argument('--epochs', type=int, default=4); p.add_argument('--lr', type=float, default=5e-5); p.add_argument('--wd', type=float, default=0.02)
    p.add_argument('--warmup', type=int, default=30); p.add_argument('--cand_budget', type=int, default=8192); p.add_argument('--max_feeds', type=int, default=128)
    p.add_argument('--folds', type=int, default=5); p.add_argument('--run_folds', default='0,1,2,3,4'); p.add_argument('--seeds', type=int, default=1)
    p.add_argument('--extra', default='n2c_pre_v3'); p.add_argument('--extra_lr_mult', type=float, default=10.0)
    p.add_argument('--ema', type=float, default=0.99); p.add_argument('--ema_start', type=float, default=0.5)
    p.add_argument('--kd_file', default=''); p.add_argument('--kd_w', type=float, default=0.5); p.add_argument('--kd_T', type=float, default=1.0)
    p.add_argument('--kd_all', action='store_true', help='soft target also on feeds without positives (pure teacher target there)')
    p.add_argument('--l2sp', type=float, default=0.0)
    p.add_argument('--bench', type=int, default=0, help='>0: time this many training steps of fold 0 and exit')
    return p.parse_args()

@torch.no_grad()
def predict(model, data, feed_ids, cand_budget=32768, max_feeds=512):
    model.eval(); out = torch.zeros(data.n_cand, dtype=torch.float32)
    for fb in data.batches(feed_ids, cand_budget, max_feeds, shuffle=False):
        b = data.gather(fb); lg = model(b, use_pos=False); data.scatter(out, b, lg)
    model.train(); return out.numpy()

def ev(scores, data, feed_ids):
    M, off = data.M_np, data.cand_off_np; feed_ids = np.sort(feed_ids)
    rows = np.concatenate([np.arange(off[f], off[f] + M[f]) for f in feed_ids]); Ms = M[feed_ids]
    return ndcg(scores[rows], data.target_np[rows], np.concatenate([[0], np.cumsum(Ms)[:-1]]), Ms)

def build_model(ck, n_extra):
    cfg = dict(ck['cfg']); m = Ranker(**cfg)
    if n_extra: m.add_extra(n_extra)
    m.load_state_dict(ck['state_dict'], strict=not n_extra)
    assert not m.id_buckets
    if m.domain:
        with torch.no_grad(): m.dom.weight[2] = m.dom.weight[1]
        m.infer_domain = 2
    return m

def main():
    a = parse(); dev = torch.device('cpu'); t0 = time.time()
    logf = open(OUT / f'ftcpu_{a.variant}_{a.tag}.log', 'a')
    def log(*x):
        s = ' '.join(str(v) for v in x); print(s, flush=True); logf.write(s + '\n'); logf.flush()
    log(f'==== {time.ctime()} args: {vars(a)} threads {torch.get_num_threads()}')
    ck = torch.load(CK / f'{a.variant}.pt', map_location=dev, weights_only=False)
    va = SplitData('val', dev, max_hist=ck['args']['max_hist'])
    if ck['cfg'].get('xt_names'): va.attach_xt(ck['cfg']['xt_names'])
    names = N2C_PRE_V3 if a.extra == 'n2c_pre_v3' else []
    assert all(feat_exists('val', n) for n in names)
    if names: va.attach_extra(names)
    kd = None
    if a.kd_file:
        kd = torch.from_numpy(np.load(a.kd_file).astype(np.float32)); assert len(kd) == va.n_cand
        log(f'KD-in-FT: teacher {a.kd_file} w {a.kd_w} T {a.kd_T} all {a.kd_all}; teacher logit sd {float(kd.std()):.3f}')
    E = a.epochs; oof = np.zeros((E + 2, va.n_cand), np.float32)
    fold_groups = va.user_id.astype(np.int64); u0 = fold_groups == 0
    fold_groups = np.where(u0, -1 - np.arange(va.n), fold_groups)
    run_folds = [int(x) for x in a.run_folds.split(',')]; done = np.zeros(va.n, bool)
    for k, (tr_f, ho_f) in enumerate(GroupKFold(n_splits=a.folds).split(np.arange(va.n), groups=fold_groups)):
        if k not in run_folds: continue
        curve = None
        for s in range(a.seeds):
            torch.manual_seed(1000 * k + s); rng = np.random.default_rng(1000 * k + s)
            model = build_model(ck, len(names))
            if s == 0: oof[0] += predict(model, va, ho_f)
            ex = [p for n, p in model.named_parameters() if n.startswith('extra.')]; base = [p for n, p in model.named_parameters() if not n.startswith('extra.')]
            groups = [{'params': base, 'mult': 1.0}] + ([{'params': ex, 'mult': a.extra_lr_mult}] if ex else [])
            groups = [dict(g, params=[p for p in g['params'] if p.requires_grad]) for g in groups]
            opt = torch.optim.AdamW(groups, lr=a.lr, weight_decay=a.wd)
            theta0 = [p.detach().clone() for p in base] if a.l2sp > 0 else None
            step = 0; S_total = ema = ema_params = ema_step0 = None; params = list(model.parameters()); ts = time.time()
            for ep in range(1, E + 1):
                model.train()
                jobs = va.batches(tr_f, a.cand_budget, a.max_feeds, shuffle=True, rng=rng)
                jobs = [jobs[i] for i in rng.permutation(len(jobs))]
                if S_total is None: S_total = len(jobs) * E; ema_step0 = int(a.ema_start * len(jobs))
                for fb in jobs:
                    lr_now = a.lr * min(1.0, (step + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(step, S_total) / S_total))
                    for g in opt.param_groups: g['lr'] = lr_now * g['mult']
                    b = va.gather(fb); X = model.dense(b); lg = model(b, X, use_pos=False); m = b['cmask']; y = b['y']
                    if kd is not None:
                        qt = torch.softmax((kd[b['ci']] / a.kd_T).masked_fill(~m, -1e4), 1) * m
                        has = (y.sum(1, keepdim=True) > 0).float()
                        if a.kd_all: y = has * ((1 - a.kd_w) * y / y.sum(1, keepdim=True).clamp_min(1) + a.kd_w * qt) + (1 - has) * qt
                        else: y = (1 - a.kd_w) * y / y.sum(1, keepdim=True).clamp_min(1) + a.kd_w * qt * has
                    loss = listwise_loss(lg, y, m, norm='batch', denom=a.max_feeds)
                    if theta0 is not None: loss = loss + 0.5 * a.l2sp * sum(((p - q) ** 2).sum() for p, q in zip(base, theta0))
                    opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
                    if a.ema > 0 and step >= ema_step0:
                        with torch.no_grad():
                            if ema is None:
                                for attr in ('aux_out', 'intent_logp'):
                                    if hasattr(model, attr): delattr(model, attr)
                                ema = copy.deepcopy(model).float(); ema_params = list(ema.parameters())
                            else: torch._foreach_lerp_(ema_params, params, 1.0 - a.ema)
                    step += 1
                    if a.bench and step >= a.bench:
                        log(f'bench: {step} steps {time.time() - ts:.1f}s = {(time.time() - ts) / step:.2f} s/step; steps/epoch {len(jobs)}'); return
                p = predict(model, va, ho_f); oof[ep] += p / a.seeds
                if s == 0: curve = (curve or [ev(oof[0], va, ho_f)]) + [ev(p, va, ho_f)]
            if ema is not None: oof[E + 1] += predict(ema, va, ho_f) / a.seeds
        done[ho_f] = True
        log(f'fold {k}: ndcg by epoch (seed0) {np.round(curve, 5).tolist()} | ema {ev(oof[E + 1], va, ho_f):.5f} | {(time.time() - t0) / 60:.1f} min')
        np.save(OUT / f'ftcpu_{a.variant}_{a.tag}_oof.npy', oof); np.save(OUT / f'ftcpu_{a.variant}_{a.tag}_done.npy', done)
    ids = np.where(done)[0]; sc = [ev(oof[e], va, ids) for e in range(E + 2)]
    log(f'OOF ndcg on {len(ids)} feeds: epochs 0..{E} {np.round(sc[:-1], 5).tolist()} | final EMA {sc[-1]:.5f} | {(time.time() - t0) / 60:.1f} min')

if __name__ == '__main__':
    main()
