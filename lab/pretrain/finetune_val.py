"""ARGUS-style step 2: fine-tune the pretrained encoder on VAL labels with 5-fold GroupKFold(user_id)
(same fold scheme as nn/finetune.py). Score(c) = scale * (<q_fe(h), v(c)> + beta * <q_new(h), v(c)>) + b(c), where
h = encoder state after the newest history event, v = output item tower, b(c) = small MLP on the
candidate's f1..f5 (optional, --fbias). Listwise softmax CE per feed. ID table frozen. The epoch count is chosen
globally on OOF NDCG; test = mean of the 5 fold models at that epoch.

Usage: CUDA_VISIBLE_DEVICES=N python pretrain/finetune_val.py --ckpt v2 --tag v2ft
Outputs: preds/pre_<tag>_{val,test}.npy (val = OOF)
"""
import argparse, copy, json, sys, time
from pathlib import Path
import numpy as np, torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold

sys.path.insert(0, "<repo>/pretrain")
sys.path.insert(0, "<repo>/src")
from seqenc import SeqEncoder
from train import counts_excluding_test, Items, FeedSplit, gather_seq, PT, PREDS, ROOT, FLAT
from metrics import ndcg_per_feed


def feed_forward(model, items, fs, S, beta):
    dev = fs.item.device
    idx, mask, Lx = gather_seq(fs.item, fs.cat, fs.et, fs.es, fs.off, fs.lens, S)
    if Lx == 0:
        idx = torch.zeros((len(S), 1), dtype=torch.long, device=dev); mask = torch.zeros_like(idx, dtype=torch.bool)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        it = torch.where(mask, fs.item[idx], 0)
        rep = items.rep(model, it, valid=mask)
        h = model.encode(rep, fs.et[idx].long() * mask, fs.es[idx].long() * mask)
    hl = h[torch.arange(len(S), device=dev), fs.lens[S]].float()
    qf, qn = model.query(hl, "fe"), model.query(hl, "new")
    Mx = int(fs.Mt[S].max().item())
    ar = torch.arange(Mx, device=dev)
    cm = ar[None, :] < fs.Mt[S][:, None]
    ci = torch.where(cm, fs.offt[S][:, None] + ar[None, :], 0)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        v = model.out_vec(items.rep(model, fs.c_item[ci]).float())
    scale = model.logit_scale.exp().clamp(max=100.0)
    s = scale * ((qf[:, None, :] * v).sum(-1) + beta * (qn[:, None, :] * v).sum(-1))
    if hasattr(model, "fb"):
        s = s + model.fb(fs.cf[ci]).squeeze(-1)
    return s, cm, ci


@torch.no_grad()
def predict(model, items, fs, beta, feeds=None, B=512):
    model.eval()
    out = np.zeros(len(fs.c_item), np.float32)
    feeds = np.arange(fs.n) if feeds is None else feeds
    order = feeds[np.argsort(fs.H_np[feeds], kind="stable")]
    for i in range(0, len(order), B):
        S = torch.from_numpy(order[i:i + B]).to(fs.item.device)
        s, cm, ci = feed_forward(model, items, fs, S, beta)
        out[ci[cm].cpu().numpy()] = s[cm].float().cpu().numpy()
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--B", type=int, default=128)
    ap.add_argument("--fbias", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = torch.device("cuda:0")
    t0 = time.time()
    ck = torch.load(PT / f"ckpt_{args.ckpt}.pt", map_location="cpu")
    ca = ck["args"]
    items = Items(dev, ca["min_cnt"], counts_excluding_test() if ca.get("exclude_test") else None)
    heads = ["fe", "new"] if ca["w_new"] > 0 else ["fe"]
    base = SeqEncoder(items.n_id, d=ca["d"], nl=ca["nl"], nh=ca["nh"], d_id=ca["d_id"], drop=ca["drop"], heads=heads, id_bias=bool(ca.get("id_bias", 0)))
    base.load_state_dict(ck["model"])
    base.id_emb.weight.requires_grad_(False)
    fss = {}
    for split in ["val", "test"]:
        fs = FeedSplit(split, dev)
        fs.Mt = torch.from_numpy(fs.M).to(dev)
        fs.offt = torch.from_numpy(fs.off_c).to(dev)
        cf = np.stack([np.load(FLAT / split / f"cand_f{i}.npy") for i in range(1, 6)], 1).astype(np.float32)
        fs.cf_raw = cf
        fss[split] = fs
    # f-feature standardization from val (log1p of abs then z) -- only used with --fbias
    mu = np.log1p(np.abs(fss["val"].cf_raw)).mean(0); sd = np.log1p(np.abs(fss["val"].cf_raw)).std(0) + 1e-6
    for fs in fss.values():
        fs.cf = torch.from_numpy(((np.log1p(np.abs(fs.cf_raw)) * np.sign(fs.cf_raw) - mu) / sd).astype(np.float32)).to(dev)
    va, te = fss["val"], fss["test"]
    y = va.y.astype(np.float32)
    yt = torch.from_numpy(y).to(dev)
    import polars as pl
    users = pl.read_parquet(FLAT / "val/feeds.parquet")["user_id"].to_numpy()
    npos = np.add.reduceat(y, va.off_c)
    E = args.epochs
    oof = np.zeros((E + 1, len(y)), np.float32)
    test = np.zeros((E + 1, len(te.c_item)), np.float32)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    fold_curves = []
    for k, (trf, hof) in enumerate(GroupKFold(n_splits=5).split(np.arange(va.n), groups=users)):
        model = copy.deepcopy(base).to(dev)
        model._id_drop = 0.0
        model.beta = nn.Parameter(torch.zeros((), device=dev))
        if args.fbias:
            model.fb = nn.Sequential(nn.Linear(5, 32), nn.GELU(), nn.Linear(32, 1)).to(dev)
            nn.init.zeros_(model.fb[2].weight); nn.init.zeros_(model.fb[2].bias)
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
        oof[0, np.isin(np.repeat(np.arange(va.n), va.M), hof)] = predict(model, items, va, model.beta, hof)[np.isin(np.repeat(np.arange(va.n), va.M), hof)]
        test[0] += predict(model, items, te, model.beta) / 5
        trp = trf[npos[trf] > 0]
        curve = []
        for ep in range(1, E + 1):
            perm = rng.permutation(trp)
            for i in range(0, len(perm), args.B):
                S = torch.from_numpy(perm[i:i + args.B]).to(dev)
                s, cm, ci = feed_forward(model, items, va, S, model.beta)
                s = s.float().masked_fill(~cm, -1e4)
                t = yt[ci] * cm
                t = t / t.sum(1, keepdim=True)
                loss = -(t * torch.log_softmax(s, 1)).sum(1).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
            rows = np.isin(np.repeat(np.arange(va.n), va.M), hof)
            oof[ep, rows] = predict(model, items, va, model.beta, hof)[rows]
            test[ep] += predict(model, items, te, model.beta) / 5
            curve.append(float(ndcg_per_feed(oof[ep][rows], y[rows], np.concatenate([[0], np.cumsum(va.M[hof])[:-1]]), va.M[hof]).mean()))
        fold_curves.append(curve)
        print(f"fold {k}: {np.round(curve, 4).tolist()} beta {model.beta.item():.3f} {time.time()-t0:.0f}s", flush=True)
        del model, opt
        torch.cuda.empty_cache()
    res = [float(ndcg_per_feed(oof[e], y, va.off_c, va.M).mean()) for e in range(E + 1)]
    best = int(np.argmax(res))
    print(f"OOF by epoch {np.round(res, 4).tolist()} -> epoch {best}", flush=True)
    np.save(PREDS / f"pre_{args.tag}_val.npy", oof[best])
    np.save(PREDS / f"pre_{args.tag}_test.npy", test[best])
    with open(ROOT / f"logs/pretrain_{args.tag}.json", "w") as fh:
        json.dump(dict(args=vars(args), oof_by_epoch=res, best=best, fold_curves=fold_curves, sec=time.time() - t0), fh)


if __name__ == "__main__":
    main()
