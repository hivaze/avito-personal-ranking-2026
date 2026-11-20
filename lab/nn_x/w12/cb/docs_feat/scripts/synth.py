"""Synthetic grouped data for CatBoost 1.2.10 capability checks (CPU only, a few thousand rows).

Label model: per group g a user vector u_g, per row an item (item pool with a popularity effect and a category effect).
score = 3 * <u_g, e_item> + a_item + b_cat + 0.5 * x1 + noise ; binary label = 1 for the top-2 rows of the group by score + Gumbel noise.
soft label (teacher-like logit) = score (real valued, centred per group).
Objects returned: numeric X (4 cols), cat ids (cat_id, item_id, user_id), embeddings (item 16-d, user 16-d, product u*e 16-d),
a text column (the "history" of the user as category tokens crossed with the candidate category).
"""
import numpy as np
import pandas as pd


def make(n_groups=120, gsize=50, d=16, n_items=800, n_cat=12, seed=0):
    r = np.random.default_rng(seed)
    n = n_groups * gsize
    gid = np.repeat(np.arange(n_groups), gsize)
    U = r.normal(size=(n_groups, d)); U /= np.linalg.norm(U, axis=1, keepdims=True)
    E = r.normal(size=(n_items, d)); E /= np.linalg.norm(E, axis=1, keepdims=True)
    item_cat = r.integers(0, n_cat, n_items)
    a_item = r.normal(scale=0.5, size=n_items)
    b_cat = r.normal(scale=0.5, size=n_cat)
    item = np.concatenate([r.choice(n_items, gsize, replace=False) for _ in range(n_groups)])
    x = r.normal(size=(n, 4)).astype(np.float32)
    dot = np.einsum("ij,ij->i", U[gid], E[item])
    score = 3.0 * dot + a_item[item] + b_cat[item_cat[item]] + 0.5 * x[:, 0] + r.normal(scale=0.3, size=n)
    noisy = score + r.gumbel(scale=0.5, size=n)
    y = np.zeros(n, np.float32)
    for g in range(n_groups):
        sl = slice(g * gsize, (g + 1) * gsize)
        y[sl][np.argsort(-noisy[sl])[:2]] = 1.0
    soft = score - np.repeat(score.reshape(n_groups, gsize).mean(1), gsize)
    # user "history": 8 category tokens drawn towards the categories the user likes
    user_pref = np.stack([np.bincount(item_cat[np.argsort(-(E @ U[g]))[:40]], minlength=n_cat) for g in range(n_groups)])
    hist = [r.choice(n_cat, 8, p=(user_pref[g] + 0.2) / (user_pref[g] + 0.2).sum()) for g in range(n_groups)]
    cat = item_cat[item]
    text_hist = np.array([" ".join(f"h{h}" for h in hist[g]) for g in gid], dtype=object)
    text_cross = np.array([" ".join(f"c{cat[i]}_h{h}" for h in hist[gid[i]]) for i in range(n)], dtype=object)
    return dict(
        n=n, gid=gid, y=y, soft=soft.astype(np.float32), x=x,
        cat_id=cat.astype(np.int32), item_id=item.astype(np.int32), user_id=gid.astype(np.int32),
        emb_item=E[item].astype(np.float32), emb_user=U[gid].astype(np.float32),
        emb_prod=(U[gid] * E[item]).astype(np.float32), text_hist=text_hist, text_cross=text_cross)


def frame(D, num=True, cats=(), texts=()):
    df = pd.DataFrame(D["x"], columns=[f"x{i}" for i in range(D["x"].shape[1])]) if num else pd.DataFrame(index=np.arange(D["n"]))
    for c in cats:
        df[c] = D[c].astype(np.int32)
    for t in texts:
        df[t] = D[t]
    return df


def split(D, n_train_groups):
    """boolean masks train / eval by group id (groups stay contiguous)."""
    tr = D["gid"] < n_train_groups
    return tr, ~tr


def ndcg(score, y, gid):
    """full-list NDCG, binary gain, pessimistic ties (positives last among ties), mean over groups."""
    out = []
    for g in np.unique(gid):
        m = gid == g
        s, t = score[m], y[m]
        if t.sum() == 0:
            continue
        order = np.lexsort((t, -s))  # by score desc, ties: label asc (pessimistic)
        gains = t[order] / np.log2(np.arange(2, len(s) + 2))
        ideal = (np.sort(t)[::-1] / np.log2(np.arange(2, len(s) + 2))).sum()
        out.append(gains.sum() / ideal)
    return float(np.mean(out)), float(np.std(out) / np.sqrt(len(out))), len(out)
