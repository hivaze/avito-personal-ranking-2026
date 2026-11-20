"""O7 (evidence only): dataset category-centroid geometry vs public text models' geometry of the decoded category NAMES.
Mapping: nn_x/codes/categories/out/mapping.csv (codes 2..53 -> Avito category name; parallel decoding workflow,
'code = dense rank of Avito category id'; 51 of 52 codes medium+ confidence; the embeddings were used there only to
validate vertical coherence, not to place individual names -> mild circularity, noted).
Dataset matrix: cosine between per-category mean unit vectors (val table, all items), raw and globally centred.
Behaviour matrix: history transition log-lift (train), from nn_x/codes/categories/out/hist_mats.npz (trans_all_TR).
Per public model: cos between name embeddings (raw names, and the template 'Объявления в категории «X» на Авито').
Statistic: Pearson / Spearman of the 1326 off-diagonal pairs; Mantel permutation p (10,000 relabelings); partial
correlations dataset~names | behaviour and dataset~behaviour | names (rank-based).
Run: CUDA_VISIBLE_DEVICES=N .venv_llm/bin/python o07_catnames.py"""
import json, time
import numpy as np, polars as pl, torch
from scipy.stats import spearmanr, rankdata
from transformers import AutoModel, AutoTokenizer

T0 = time.time()
R = "<repo>"; OUT = f"{R}/nn_x/emb/origin/out"; NV = 2516711
dev = "cuda" if torch.cuda.is_available() else "cpu"


def log(*a):
    print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)


mp = pl.read_csv(f"{R}/nn_x/codes/categories/out/mapping.csv")
codes = mp["code"].to_numpy(); names = mp["name_ru"].to_list(); conf = mp["confidence_vertical_name"].to_list()
# dataset centroids (val table)
x = np.asarray(np.memmap(f"{R}/data/competition/val/item_embeddings/data.memmap", dtype=np.int8, mode="r", shape=(NV, 128))).astype(np.float32)
x /= np.linalg.norm(x, axis=1, keepdims=True)
cat = np.zeros(NV, np.int64)
for kind in ("cand", "hist"):
    e = np.load(f"{R}/data/flat/val/{kind}_emb_idx.npy"); c = np.load(f"{R}/data/flat/val/{kind}_category_id.npy"); m = e >= 0
    cat[e[m]] = c[m]
C = np.stack([x[cat == c].mean(0) for c in codes]).astype(np.float64)
gm = x.mean(0).astype(np.float64)


def cosm(M):
    M = M / np.linalg.norm(M, axis=1, keepdims=True)
    return M @ M.T


mats = {"dataset_raw": cosm(C), "dataset_centred": cosm(C - gm)}
hm = np.load(f"{R}/nn_x/codes/categories/out/hist_mats.npz")
T = hm["trans_all_TR"][np.ix_(codes, codes)].astype(float).copy(); np.fill_diagonal(T, 0)
L = T * T.sum() / np.maximum(T.sum(1, keepdims=True) * T.sum(0, keepdims=True), 1e-9); L = (L + L.T) / 2
mats["behaviour_translift"] = np.log(np.maximum(L, 1e-3))
iu = np.triu_indices(len(codes), 1)


def off(M):
    return M[iu]


def mantel(A, B, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    a = rankdata(off(A)); r0 = np.corrcoef(a, rankdata(off(B)))[0, 1]
    k = len(codes); cnt = 0
    for _ in range(n):
        p = rng.permutation(k)
        if np.corrcoef(a, rankdata(B[np.ix_(p, p)][iu]))[0, 1] >= r0:
            cnt += 1
    return float(r0), (cnt + 1) / (n + 1)


def partial(a, b, z):
    a, b, z = rankdata(a), rankdata(b), rankdata(z)
    ra = a - np.polyval(np.polyfit(z, a, 1), z); rb = b - np.polyval(np.polyfit(z, b, 1), z)
    return float(np.corrcoef(ra, rb)[0, 1])


def embed(name, texts, pool, prefix=""):
    tok = AutoTokenizer.from_pretrained(name)
    if pool == "last":
        tok.padding_side = "left"
    m = AutoModel.from_pretrained(name, dtype=torch.float32).to(dev).eval()
    with torch.no_grad():
        b = tok([prefix + t for t in texts], padding=True, truncation=True, max_length=64, return_tensors="pt").to(dev)
        o = m(**b)
        if pool == "last":
            e = o.last_hidden_state[:, -1]
        elif pool == "mean":
            msk = b["attention_mask"].unsqueeze(-1).float(); e = (o.last_hidden_state * msk).sum(1) / msk.sum(1)
        elif pool == "cls":
            e = o.last_hidden_state[:, 0]
        else:
            e = o.pooler_output
    e = e.float().cpu().numpy()
    return e / np.linalg.norm(e, axis=1, keepdims=True)


MODELS = [("Qwen/Qwen3-Embedding-0.6B", "last", ""), ("intfloat/multilingual-e5-small", "mean", "query: "),
          ("deepvk/USER-base", "mean", "query: "), ("cointegrated/LaBSE-en-ru", "pooler", ""),
          ("deepvk/USER2-base", "mean", "clustering: "), ("Snowflake/snowflake-arctic-embed-l-v2.0", "cls", "")]
res = {"n_codes": int(len(codes)), "n_pairs": int(len(iu[0])), "n_low_conf": int(sum('low' in c.split('/')[1] for c in conf))}
r_b = {}
for dm in ("dataset_raw", "dataset_centred"):
    r_b[dm] = mantel(mats[dm], mats["behaviour_translift"])
res["dataset_vs_behaviour_spearman_mantelp"] = r_b
log("dataset vs behaviour", r_b)
per = {}
for name, pool, pre in MODELS:
    for tmpl in ("raw", "template"):
        tx = names if tmpl == "raw" else [f"Объявления в категории «{n}» на Авито" for n in names]
        e = embed(name, tx, pool, pre)
        N = e @ e.T
        r = {}
        for dm in ("dataset_raw", "dataset_centred"):
            rho, p = mantel(mats[dm], N)
            r[dm] = {"spearman": rho, "pearson": float(np.corrcoef(off(mats[dm]), off(N))[0, 1]), "mantel_p": p,
                     "partial_given_behaviour": partial(off(mats[dm]), off(N), off(mats["behaviour_translift"]))}
        r["names_vs_behaviour_spearman"] = float(spearmanr(off(N), off(mats["behaviour_translift"]))[0])
        r["dataset_vs_behaviour_partial_given_names"] = partial(off(mats["dataset_centred"]), off(mats["behaviour_translift"]), off(N))
        per[f"{name}|{tmpl}"] = r
        log(name, tmpl, json.dumps(r))
res["per_model"] = per
# upper bound reference: dataset raw vs centred, and val-table centroids vs codes-workflow train-pool centroids
em = np.load(f"{R}/nn_x/codes/categories/out/emb_mats.npz")
res["dataset_val_vs_trainpool_centroid_spearman"] = float(spearmanr(off(mats["dataset_raw"]), off(em["centroid_cos"][np.ix_(codes, codes)]))[0])
json.dump(res, open(f"{OUT}/o07.json", "w"), indent=1)
log("done", res["dataset_val_vs_trainpool_centroid_spearman"])
