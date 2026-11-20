"""Pretrain the sequence encoder on the all-history corpus (see build_corpus.py), then score val/test candidates.

Loss per head (fe: next new feed-originated open; new: next new item): sampled softmax over
[positive, mixed negatives], negatives = K1 sampled fe targets + K2 sampled new targets of the batch + K3 uniform
corpus items; all logits get the logQ correction log(K1 p_fe(i) + K2 p_new(i) + K3 / N_corpus); accidental hits masked.

Usage: CUDA_VISIBLE_DEVICES=N python pretrain/train.py --tag v1 --epochs 1
Outputs: data/pretrain/ckpt_<tag>.pt, preds/pre_<tag>_<head>_{val,test}.npy, logs/pretrain_<tag>.{log,json}
"""
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np, polars as pl, numba as nb, torch
import torch.nn.functional as F

sys.path.insert(0, "<repo>/pretrain")
sys.path.insert(0, "<repo>/src")
from seqenc import SeqEncoder
from metrics import ndcg
from build_corpus import gather_rev

ROOT = Path("<repo>")
PT = ROOT / "data/pretrain"
FLAT = ROOT / "data/flat"
PREDS = ROOT / "preds"
PTC = PT  # corpus-specific arrays (seq/tok/tgt/item counts); item_tab/item_has stay global in PT


def set_corpus_dir(d):
    global PTC
    PTC = Path(d) if d else PT
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
nb.set_num_threads(8)
torch.set_num_threads(8)


@nb.njit(cache=True)
def bos_targets(off, item, et, es):
    S = len(off) - 1
    bfe = np.full(S, -1, np.int32)
    bnew = np.full(S, -1, np.int32)
    for s in range(S):
        a, b = off[s], off[s + 1]
        if b > a:
            bnew[s] = item[a]
        for j in range(a, b):
            if et[j] == 1 and es[j] == 1:
                bfe[s] = item[j]
                break
    return bfe, bnew


def log(msg, fh):
    print(msg, flush=True)
    fh.write(msg + "\n"); fh.flush()


def counts_excluding_test():
    sp = np.load(PTC / "seq_split.npy"); off = np.load(PTC / "seq_off.npy")
    keep = np.repeat(sp != 2, np.diff(off))
    G = len(np.load(PT / "item_has.npy", mmap_mode="r"))
    ti = np.load(PTC / "tok_item.npy")[keep]; tf = np.load(PTC / "tgt_fe.npy")[keep]; tn = np.load(PTC / "tgt_new.npy")[keep]
    return (np.bincount(ti, minlength=G).astype(np.int32), np.bincount(tf[tf >= 0], minlength=G).astype(np.int32),
            np.bincount(tn[tn >= 0], minlength=G).astype(np.int32))


class Items:
    """Global per-item lookup tables on GPU (indexed by item_id)."""

    def __init__(self, dev, min_cnt, counts=None, tab_name="item_tab"):
        t0 = time.time()
        tab = np.load(PT / f"{tab_name}.npy", mmap_mode="r")
        self.G = tab.shape[0]
        self.tab = torch.empty((self.G, 128), dtype=torch.int8, device=dev)
        step = 8_000_000
        for a in range(0, self.G, step):
            self.tab[a:a + step] = torch.from_numpy(np.ascontiguousarray(tab[a:a + step])).to(dev)
        self.has = torch.from_numpy(np.load(PT / f"{tab_name.replace('_tab', '_has')}.npy")).to(dev)
        cnt = np.load(PTC / "item_cnt.npy") if counts is None else counts[0]
        slot = np.zeros(self.G, np.int32)
        ids = np.where(cnt >= min_cnt)[0]
        slot[ids] = np.arange(1, len(ids) + 1, dtype=np.int32)
        self.n_id = len(ids)
        self.slot = torch.from_numpy(slot).to(dev)
        cat = np.zeros(self.G, np.int8)
        tok_item = np.load(PTC / "tok_item.npy", mmap_mode="r")
        tok_cat = np.load(PTC / "tok_cat.npy", mmap_mode="r")
        cat[np.asarray(tok_item)] = np.asarray(tok_cat)
        for s in ["train", "val", "test"]:
            cat[np.load(FLAT / s / "cand_item_id.npy")] = np.load(FLAT / s / "cand_category_id.npy")
        self.cat = torch.from_numpy(cat).to(dev)
        self.corpus_items = torch.from_numpy(np.where(cnt > 0)[0].astype(np.int32)).to(dev)
        self.cnt_fe = np.load(PTC / "item_cnt_fe.npy") if counts is None else counts[1]
        self.cnt_new = np.load(PTC / "item_cnt_new.npy") if counts is None else counts[2]
        print(f"[items] G={self.G} n_id={self.n_id} corpus={len(self.corpus_items)} {time.time()-t0:.0f}s", flush=True)

    def set_logq(self, K1, K2, K3, dev):
        q = (K1 * self.cnt_fe / self.cnt_fe.sum() + K2 * self.cnt_new / self.cnt_new.sum()
             + K3 / len(self.corpus_items)).astype(np.float32)
        self.logq = torch.from_numpy(np.log(np.maximum(q, 1e-12))).to(dev)

    def rep(self, model, items, valid=None, id_drop=0.0):
        it = items.long().clamp_min(0)
        has = self.has[it]
        if valid is not None:
            has = has & valid
        return model.item_rep(self.tab[it], has, self.cat[it].long(), self.slot[it].long(), id_drop)


class Corpus:
    def __init__(self, dev, multi=1, exclude_test=False):
        t0 = time.time()
        self.exclude_test = exclude_test
        self.off_np = np.load(PTC / "seq_off.npy")
        g = lambda n: torch.from_numpy(np.load(PTC / f"{n}.npy")).to(dev)
        self.item, self.cat, self.et, self.es = g("tok_item"), g("tok_cat"), g("tok_et"), g("tok_es")
        self.tgt = {"fe": g("tgt_fe"), "new": g("tgt_new")}
        self.multi = g("tgt_fe_multi")[:, :multi] if multi > 1 else None
        it, et, es = np.load(PTC / "tok_item.npy"), np.load(PTC / "tok_et.npy"), np.load(PTC / "tok_es.npy")
        bfe, bnew = bos_targets(self.off_np, it, et, es)
        self.bos_tgt = {"fe": torch.from_numpy(bfe).to(dev), "new": torch.from_numpy(bnew).to(dev)}
        self.off = torch.from_numpy(self.off_np).to(dev)
        self.lens_np = np.diff(self.off_np)
        self.lens = torch.from_numpy(self.lens_np).to(dev)
        self.S = len(self.lens_np)
        print(f"[corpus] {self.S} seqs {self.off_np[-1]} tokens {time.time()-t0:.0f}s", flush=True)

    def batches(self, budget, rng, vt_rep=1):
        L = self.lens_np
        sp = np.load(PTC / "seq_split.npy")
        base = np.arange(self.S)
        if self.exclude_test:
            base = base[sp != 2]
        if vt_rep > 1:
            vt = np.where((sp > 0) & ~((sp == 2) & self.exclude_test))[0]
            base = np.concatenate([base] + [vt] * (vt_rep - 1))
        perm = base[rng.permutation(len(base))]
        bk = (L[perm] + 31) // 32
        order = perm[np.argsort(bk, kind="stable")]
        bks = (L[order] + 31) // 32
        out = []
        for b in np.unique(bks):
            ids = order[bks == b]
            B = max(1, budget // (int(b) * 32 + 1))
            out += [ids[i:i + B] for i in range(0, len(ids), B)]
        return [out[k] for k in rng.permutation(len(out))]


def gather_seq(item, cat, et, es, off, lens, S):
    Lx = int(lens[S].max().item())
    ar = torch.arange(Lx, device=item.device)
    mask = ar[None, :] < lens[S][:, None]
    idx = torch.where(mask, off[S][:, None] + ar[None, :], 0)
    return idx, mask, Lx


def sampled_softmax(model, items, hq, tgt, pool, scale):
    """hq [N,d] normalized queries, tgt [N] item ids, pool [K] item ids (negatives)."""
    vp = model.out_vec(items.rep(model, tgt, id_drop=model._id_drop))
    vn = model.out_vec(items.rep(model, pool, id_drop=model._id_drop))
    lp = scale * (hq * vp).sum(-1) - items.logq[tgt.long()] + model.bias(items.slot[tgt.long()])
    ln = scale * (hq @ vn.T) - items.logq[pool.long()][None, :] + model.bias(items.slot[pool.long()])[None, :]
    ln = ln.masked_fill(pool[None, :] == tgt[:, None], -1e4)
    logits = torch.cat([lp[:, None], ln], dim=1)
    loss = torch.logsumexp(logits, dim=1) - lp
    with torch.no_grad():
        acc = (lp[:, None] > ln).float().mean(1)  # fraction of negatives beaten
    return loss.mean(), acc.mean()


# ------------------------------------------------------------------ inference on val/test feeds
class FeedSplit:
    def __init__(self, split, dev):
        f = pl.read_parquet(FLAT / split / "feeds.parquet")
        self.n = len(f)
        self.M = f["M"].to_numpy().astype(np.int64)
        self.off_c = f["cand_off"].to_numpy().astype(np.int64)
        H = f["H"].to_numpy().astype(np.int64)
        hoff = f["hist_off"].to_numpy().astype(np.int64)
        ooff = np.concatenate([[0], np.cumsum(H)[:-1]]).astype(np.int64)
        arrs = {}
        for c in ["item_id", "category_id", "event_type", "event_source"]:
            src = np.load(FLAT / split / f"hist_{c}.npy")
            dst = np.empty(int(H.sum()), src.dtype)
            gather_rev(hoff, H, src, ooff, dst)
            arrs[c] = torch.from_numpy(dst).to(dev)
        self.item, self.cat, self.et, self.es = arrs["item_id"], arrs["category_id"], arrs["event_type"], arrs["event_source"]
        self.off = torch.from_numpy(ooff).to(dev)
        self.H_np = H
        self.lens = torch.from_numpy(H).to(dev)
        self.c_item = torch.from_numpy(np.load(FLAT / split / "cand_item_id.npy")).to(dev)
        self.c_fidx = torch.from_numpy(np.load(FLAT / split / "cand_fidx.npy").astype(np.int64)).to(dev)
        self.y = np.load(FLAT / split / "cand_target.npy") if (FLAT / split / "cand_target.npy").exists() else None


@torch.no_grad()
def infer(model, items, fs, heads, budget=200_000, last_k=32):
    """Returns dict of per-candidate float32 numpy scores."""
    model.eval()
    dev = fs.item.device
    n = fs.n
    Q = {h: torch.zeros((n, model.d), device=dev) for h in heads}
    HV = torch.zeros((n, last_k, model.d), device=dev)  # out-space vectors of the last_k history items (newest first)
    HM = torch.zeros((n, last_k), dtype=torch.bool, device=dev)
    use_id = model.id_emb is not None
    if use_id:
        IV = torch.zeros((n, last_k, model.id_emb.weight.shape[1]), device=dev)
    order = np.argsort(fs.H_np, kind="stable")
    B = max(1, budget // 258)
    for i in range(0, n, B):
        ids = order[i:i + B]
        S = torch.from_numpy(ids).to(dev)
        idx, mask, Lx = gather_seq(fs.item, fs.cat, fs.et, fs.es, fs.off, fs.lens, S)
        if Lx == 0:
            idx = torch.zeros((len(ids), 1), dtype=torch.long, device=dev); mask = torch.zeros_like(idx, dtype=torch.bool)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            it = torch.where(mask, fs.item[idx], 0)
            rep = items.rep(model, it, valid=mask)
            h = model.encode(rep, fs.et[idx].long() * mask, fs.es[idx].long() * mask)
        last = fs.lens[S]  # state after the newest event = position len (0 = BOS for empty history)
        hl = h[torch.arange(len(ids), device=dev), last].float()
        for hd in heads:
            Q[hd][S] = model.query(hl, hd)
        # last_k history items in out space (newest first)
        kk = torch.arange(last_k, device=dev)
        pos = last[:, None] - 1 - kk[None, :]
        okk = pos >= 0
        ov = model.out_vec(rep.float()[torch.arange(len(ids), device=dev)[:, None], pos.clamp_min(0)])
        HV[S] = ov * okk[..., None]
        HM[S] = okk
        if use_id:
            hit = torch.where(mask, fs.item[idx], 0).long()[torch.arange(len(ids), device=dev)[:, None], pos.clamp_min(0)]
            sl = items.slot[hit].long() * okk
            IV[S] = F.normalize(model.id_emb(sl).float(), dim=-1) * (sl > 0)[..., None]
    scale = model.logit_scale.exp().clamp(max=100.0)
    out = {}
    nc = len(fs.c_item)
    v = torch.zeros((nc, model.d), device=dev)
    v_noid = torch.zeros((nc, model.d), device=dev)
    for a in range(0, nc, 500_000):
        it = fs.c_item[a:a + 500_000]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            r = items.rep(model, it)
            itl = it.long()
            r0 = model.item_rep(items.tab[itl], items.has[itl], items.cat[itl].long(), torch.zeros_like(itl))
        v[a:a + 500_000] = model.out_vec(r.float())
        v_noid[a:a + 500_000] = model.out_vec(r0.float())
    cb = model.bias(items.slot[fs.c_item.long()]) if model.id_bias is not None else torch.zeros(nc, device=dev)
    if model.id_bias is not None:
        out["ibias"] = cb.cpu().numpy().astype(np.float32)
    for hd in heads:
        qf = Q[hd][fs.c_fidx]
        out[hd] = (scale * (qf * v).sum(-1) + cb).cpu().numpy().astype(np.float32)
        out[hd + "_noid"] = (scale * (qf * v_noid).sum(-1)).cpu().numpy().astype(np.float32)
    # learned-space similarity of candidate to recent history items
    sims = []
    for a in range(0, nc, 200_000):
        fi = fs.c_fidx[a:a + 200_000]
        s = torch.einsum("nd,nkd->nk", v[a:a + 200_000], HV[fi])
        s = s.masked_fill(~HM[fi], -2.0)
        sims.append(s)
    s = torch.cat(sims)
    out["hmax32"] = s.max(1).values.cpu().numpy().astype(np.float32)
    out["hmax8"] = s[:, :8].max(1).values.cpu().numpy().astype(np.float32)
    if use_id:  # max cosine in the learned ID space (-2 when the candidate or all history items lack an ID)
        sims = []
        cs = items.slot[fs.c_item.long()].long()
        cv = F.normalize(model.id_emb(cs).float(), dim=-1)
        for a in range(0, nc, 200_000):
            fi = fs.c_fidx[a:a + 200_000]
            iv = IV[fi]
            s2 = torch.einsum("nd,nkd->nk", cv[a:a + 200_000], iv)
            s2 = s2.masked_fill((iv.abs().sum(-1) == 0) | (cs[a:a + 200_000] == 0)[:, None], -2.0)
            sims.append(s2.max(1).values)
        out["idmax32"] = torch.cat(sims).cpu().numpy().astype(np.float32)
    model.train()
    return out


def eval_val(model, items, fsv, heads):
    sc = infer(model, items, fsv, heads)
    return {k: ndcg(v, fsv.y, fsv.off_c, fsv.M) for k, v in sc.items()}, sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--nl", type=int, default=3)
    ap.add_argument("--nh", type=int, default=4)
    ap.add_argument("--d_id", type=int, default=64)
    ap.add_argument("--min_cnt", type=int, default=3)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--budget", type=int, default=131072, help="padded tokens per batch")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--id_lr", type=float, default=3e-3)
    ap.add_argument("--id_drop", type=float, default=0.3)
    ap.add_argument("--drop", type=float, default=0.05)
    ap.add_argument("--w_new", type=float, default=0.5)
    ap.add_argument("--K1", type=int, default=6144)
    ap.add_argument("--K2", type=int, default=2048)
    ap.add_argument("--K3", type=int, default=8192)
    ap.add_argument("--max_pos", type=int, default=65536, help="max supervised positions per head per step")
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--item_tab", default="item_tab", help="content table (item_tab2 = + train candidate embeddings, section 8c)")
    ap.add_argument("--corpus_dir", default="", help="corpus-specific arrays (default data/pretrain), e.g. data/pretrain/win1")
    ap.add_argument("--no_test_score", type=int, default=0, help="skip the final val/test candidate scoring (windowed models)")
    ap.add_argument("--exclude_test", type=int, default=0, help="drop all test snapshots from pretraining (val check)")
    ap.add_argument("--id_bias", type=int, default=0, help="learned scalar logit bias per ID item")
    ap.add_argument("--vt_rep", type=int, default=1, help="oversample val/test sequences x vt_rep per epoch")
    ap.add_argument("--multi", type=int, default=1, help="fe target sampled among the next R new fe items")
    args = ap.parse_args()
    dev = torch.device("cuda:0")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    set_corpus_dir(args.corpus_dir)
    fh = open(ROOT / f"logs/pretrain_{args.tag}.log", "w")
    log(json.dumps(vars(args)), fh)
    t0 = time.time()
    counts = counts_excluding_test() if args.exclude_test else None  # val-side check model: no test snapshots
    items = Items(dev, args.min_cnt, counts, tab_name=args.item_tab)
    items.set_logq(args.K1, args.K2, args.K3, dev)
    corpus = Corpus(dev, args.multi, args.exclude_test)
    fsv = FeedSplit("val", dev)
    heads = ["fe", "new"] if args.w_new > 0 else ["fe"]
    model = SeqEncoder(items.n_id, d=args.d, nl=args.nl, nh=args.nh, d_id=args.d_id, drop=args.drop, heads=heads,
                       id_bias=args.id_bias).to(dev)
    model._id_drop = args.id_drop
    dense = [p for n, p in model.named_parameters() if not n.startswith(("id_emb", "id_bias"))]
    opt = torch.optim.AdamW(dense, lr=args.lr, weight_decay=0.01, betas=(0.9, 0.98))
    sp = list(model.id_emb.parameters()) + (list(model.id_bias.parameters()) if model.id_bias is not None else [])
    opt_id = torch.optim.SparseAdam(sp, lr=args.id_lr) if model.id_emb is not None else None
    n_dense = sum(p.numel() for p in dense)
    log(f"[model] dense params {n_dense/1e6:.2f}M, id rows {items.n_id}, setup {time.time()-t0:.0f}s", fh)
    # schedule
    ep_batches = corpus.batches(args.budget, rng, args.vt_rep)
    total = int(len(ep_batches) * args.epochs)
    warm = 300
    lr_at = lambda s: min(1.0, (s + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    log(f"[sched] {len(ep_batches)} batches/epoch, total steps {total}", fh)
    curve = []
    step, ep = 0, 0
    tl = time.time()
    agg = {}
    while step < total:
        if ep > 0:
            ep_batches = corpus.batches(args.budget, rng, args.vt_rep)
        for bseq in ep_batches:
            if step >= total:
                break
            for g in opt.param_groups:
                g["lr"] = args.lr * lr_at(step)
            if opt_id is not None:
                for g in opt_id.param_groups:
                    g["lr"] = args.id_lr * lr_at(step)
            S = torch.from_numpy(bseq).to(dev)
            idx, mask, Lx = gather_seq(corpus.item, corpus.cat, corpus.et, corpus.es, corpus.off, corpus.lens, S)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                it = torch.where(mask, corpus.item[idx], 0)
                rep = items.rep(model, it, valid=mask, id_drop=args.id_drop)
                h = model.encode(rep, corpus.et[idx].long() * mask, corpus.es[idx].long() * mask)
            scale = model.logit_scale.exp().clamp(max=100.0)
            loss = 0.0
            for hd, w in [("fe", 1.0), ("new", args.w_new)]:
                if w <= 0:
                    continue
                if hd == "fe" and corpus.multi is not None:  # dense: uniform among the next R new fe items
                    tm = corpus.multi[idx]
                    nv = (tm >= 0).sum(-1)
                    r = (torch.rand(nv.shape, device=dev) * nv).long().clamp_max(tm.shape[-1] - 1)
                    tg = tm.gather(-1, r[..., None])[..., 0]
                    tg = torch.where(mask & (nv > 0), tg, -1)
                else:
                    tg = torch.where(mask, corpus.tgt[hd][idx], -1)
                tg = torch.cat([corpus.bos_tgt[hd][S][:, None], tg], dim=1)  # [B, Lx+1]
                valid = tg >= 0
                hv = h[:, :Lx + 1][valid]
                tv = tg[valid]
                if len(tv) > args.max_pos:
                    sel = torch.randperm(len(tv), device=dev)[:args.max_pos]
                    hv, tv = hv[sel], tv[sel]
                if hd == "fe":
                    fe_t = tv
                else:
                    new_t = tv
                hq = model.query(hv.float(), hd)
                if hd == "fe":
                    pool_parts = [fe_t[torch.randint(len(fe_t), (args.K1,), device=dev)]]
                    tg2 = torch.where(mask, corpus.tgt["new"][idx], -1)
                    tg2 = tg2[tg2 >= 0]
                    pool_parts.append(tg2[torch.randint(len(tg2), (args.K2,), device=dev)])
                    pool_parts.append(items.corpus_items[torch.randint(len(items.corpus_items), (args.K3,), device=dev)])
                    pool = torch.cat(pool_parts)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    l, acc = sampled_softmax(model, items, hq, tv, pool, scale)
                loss = loss + w * l
                agg.setdefault(f"loss_{hd}", []).append(l.item() if step % 50 == 0 else None)
                agg.setdefault(f"acc_{hd}", []).append(acc.item() if step % 50 == 0 else None)
            opt.zero_grad(set_to_none=True)
            if opt_id is not None:
                opt_id.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dense, 1.0)
            opt.step()
            if opt_id is not None:
                opt_id.step()
            step += 1
            if step % 200 == 0:
                m = {k: float(np.mean([x for x in v if x is not None])) for k, v in agg.items()}
                agg = {}
                tps = 200 * args.budget / (time.time() - tl)
                tl = time.time()
                log(f"step {step}/{total} ep {ep} " + " ".join(f"{k} {v:.4f}" for k, v in m.items())
                    + f" scale {scale.item():.1f} lr {args.lr*lr_at(step):.2e} {tps/1e6:.2f}M tok/s  {time.time()-t0:.0f}s", fh)
                curve.append(dict(step=step, t=time.time() - t0, **m))
            if step % args.eval_every == 0 or step == total:
                res, _ = eval_val(model, items, fsv, heads)
                log(f"[eval] step {step} " + " ".join(f"{k} {v:.4f}" for k, v in res.items()), fh)
                curve.append(dict(step=step, t=time.time() - t0, **{"val_" + k: v for k, v in res.items()}))
                json.dump(curve, open(ROOT / f"logs/pretrain_{args.tag}.json", "w"))
        ep += 1
    torch.save({"args": vars(args), "model": model.state_dict()}, PT / f"ckpt_{args.tag}.pt")
    if args.no_test_score:
        res, _ = eval_val(model, items, fsv, heads)
        log(f"[final-val-only] " + " ".join(f"{k} {v:.4f}" for k, v in res.items()) + f" total {time.time()-t0:.0f}s", fh)
        json.dump(curve, open(ROOT / f"logs/pretrain_{args.tag}.json", "w"))
        return
    # final scoring of val and test
    del corpus
    torch.cuda.empty_cache()
    res, scv = eval_val(model, items, fsv, heads)
    fst = FeedSplit("test", dev)
    sct = infer(model, items, fst, heads)
    for k in scv:
        np.save(PREDS / f"pre_{args.tag}_{k}_val.npy", scv[k])
        np.save(PREDS / f"pre_{args.tag}_{k}_test.npy", sct[k])
    log(f"[final] " + " ".join(f"{k} {v:.4f}" for k, v in res.items()) + f" total {time.time()-t0:.0f}s", fh)
    json.dump(curve, open(ROOT / f"logs/pretrain_{args.tag}.json", "w"))


if __name__ == "__main__":
    main()
