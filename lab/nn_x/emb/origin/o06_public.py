"""O6: public text-embedding models as controls for the MRL / prefix test and for geometry (anisotropy, spectrum).
Run with <repo>/.venv_llm/bin/python. Device: cuda (CUDA_VISIBLE_DEVICES=N) if available, else CPU.
Corpus: out/ru_titles.tsv (6000 templated Russian ad titles). For each model we report, on the full-D vectors and on the
128-d version (first 128 dims re-normalised = official MRL truncation for MRL models, naive truncation otherwise):
random-pair cos, ||mean||/mean-norm, participation-ratio rank, Spearman(per-dim sd, index), and top-10 neighbour overlap
with the full-128 neighbours for prefix-k vs random-k dim subsets (k = 16, 32, 64; random = mean of 20 draws).
The same statistics are computed with the same code on 6000 random dataset items (val table) for a like-for-like line.
Embeddings are saved to out/pub_<model>.npy for the category-name test (if a mapping appears)."""
import json, sys, time
import numpy as np
import torch
from scipy.stats import spearmanr
from transformers import AutoModel, AutoTokenizer

T0 = time.time()
OUT = "<repo>/nn_x/emb/origin/out"
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(12)
rows = [l.rstrip("\n").split("\t") for l in open(f"{OUT}/ru_titles.tsv")]
cats = np.array([r[0] for r in rows]); texts = [r[1] for r in rows]


def log(*a):
    print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)


def l2n(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def embed(name, texts, pool, prefix="", bs=128):
    tok = AutoTokenizer.from_pretrained(name)
    if pool == "last":
        tok.padding_side = "left"
    m = AutoModel.from_pretrained(name, torch_dtype=torch.float32).to(dev).eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            b = tok([prefix + t for t in texts[i:i + bs]], padding=True, truncation=True, max_length=64,
                    return_tensors="pt").to(dev)
            o = m(**b)
            if pool == "last":
                e = o.last_hidden_state[:, -1]
            elif pool == "mean":
                msk = b["attention_mask"].unsqueeze(-1).float()
                e = (o.last_hidden_state * msk).sum(1) / msk.sum(1)
            elif pool == "cls":
                e = o.last_hidden_state[:, 0]
            elif pool == "pooler":
                e = o.pooler_output
            outs.append(e.float().cpu().numpy())
    del m
    if dev == "cuda":
        torch.cuda.empty_cache()
    return l2n(np.concatenate(outs))


def topk_sets(x, k=10):
    s = x @ x.T
    np.fill_diagonal(s, -np.inf)
    return np.argpartition(-s, k, axis=1)[:, :k]


def overlap(a, b):
    return float(np.mean([len(set(a[i]) & set(b[i])) / a.shape[1] for i in range(len(a))]))


def geom(x, rng):
    n = len(x)
    i, j = rng.integers(0, n, 20000), rng.integers(0, n, 20000)
    keep = i != j
    rp = (x[i[keep]] * x[j[keep]]).sum(1)
    xc = x - x.mean(0)
    ev = np.linalg.eigvalsh(np.cov(xc.T))[::-1]
    ev = np.clip(ev, 0, None)
    sd = x.std(0)
    return {"dim": int(x.shape[1]), "random_pair_cos_mean": float(rp.mean()), "random_pair_cos_sd": float(rp.std()),
            "mean_vec_norm_ratio": float(np.linalg.norm(x.mean(0))),
            "pr_rank": float(ev.sum() ** 2 / (ev ** 2).sum()),
            "pca_top1": float(ev[0] / ev.sum()), "pca_top32": float(ev[:32].sum() / ev.sum()),
            "spearman_sd_vs_index": float(spearmanr(sd, np.arange(len(sd)))[0]),
            "sd_first16_over_last16": float(sd[:16].mean() / sd[-16:].mean()),
            "per_dim_mean_absmax": float(np.abs(x.mean(0)).max())}


def prefix_test(x128, rng, draws=20):
    full = topk_sets(x128)
    r = {}
    for k in (16, 32, 64):
        p = overlap(topk_sets(l2n(x128[:, :k])), full)
        rnd = [overlap(topk_sets(l2n(x128[:, np.sort(rng.choice(128, k, replace=False))])), full) for _ in range(draws)]
        r[f"k{k}"] = {"prefix": p, "random_mean": float(np.mean(rnd)), "random_sd": float(np.std(rnd)),
                      "prefix_minus_random": p - float(np.mean(rnd))}
    # also the suffix (last k dims), to see whether prefix is special or just any block
    r["k64_suffix"] = overlap(topk_sets(l2n(x128[:, 64:])), full)
    return r


MODELS = [
    ("Qwen/Qwen3-Embedding-0.6B", "last", "", "MRL (32..1024), native 1024"),
    ("intfloat/multilingual-e5-small", "mean", "passage: ", "no MRL, native 384"),
    ("deepvk/USER-base", "mean", "passage: ", "no MRL, native 768, Russian"),
    ("cointegrated/LaBSE-en-ru", "pooler", "", "no MRL, native 768, ru/en LaBSE"),
    ("deepvk/USER2-base", "mean", "search_document: ", "MRL (32..768), native 768, Russian ModernBERT"),
    ("Snowflake/snowflake-arctic-embed-l-v2.0", "cls", "", "MRL (256), native 1024, multilingual"),
    ("mixedbread-ai/mxbai-embed-large-v1", "cls", "", "MRL, native 1024, English"),
]
res = {}
rng = np.random.default_rng(0)
# dataset line: 6000 random val-table items, same code
mm = np.memmap("<repo>/data/competition/val/item_embeddings/data.memmap", dtype=np.int8, mode="r",
               shape=(2516711, 128))
idx = np.sort(rng.choice(2516711, 6000, replace=False))
xd = l2n(mm[idx].astype(np.float64))
res["DATASET_val_6000"] = {"geom": geom(xd, np.random.default_rng(1)), "prefix": prefix_test(xd, np.random.default_rng(2))}
log("dataset", json.dumps(res["DATASET_val_6000"]))
only = sys.argv[1:] or None
for name, pool, pre, note in MODELS:
    if only and name not in only:
        continue
    try:
        e = embed(name, texts, pool, pre)
    except Exception as ex:  # download / architecture failure: report and continue
        log(name, "FAILED", repr(ex)[:300]); res[name] = {"error": repr(ex)[:300]}; continue
    np.save(f"{OUT}/pub_{name.replace('/', '__')}.npy", e.astype(np.float32))
    e128 = l2n(e[:, :128])
    q = np.clip(np.round(e128 * 256), -128, 127)
    r = {"note": note, "geom_full": geom(e, np.random.default_rng(1)), "geom_128": geom(e128, np.random.default_rng(1)),
         "prefix_128": prefix_test(e128, np.random.default_rng(2)),
         "int8x256_norm_mean": float(np.linalg.norm(q, axis=1).mean())}
    # category purity on the templated corpus (kNN-1 same category), full and 128
    for tag, z in (("full", e), ("128", e128)):
        nn1 = topk_sets(z, 1)[:, 0]
        r[f"knn1_cat_acc_{tag}"] = float((cats[nn1] == cats).mean())
    res[name] = r
    log(name, json.dumps(r))
json.dump(res, open(f"{OUT}/o06{'_' + '_'.join(a.split('/')[-1] for a in only) if only else ''}.json", "w"), indent=1)
log("done")
