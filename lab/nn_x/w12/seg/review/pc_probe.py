"""PC (DECLARATION block 2): can a 'score + w z(scalar similarity to the withheld events)' probe, nested by user, see an
effect that is KNOWN to be there (the model's own use of events 193-256 of capped users)?"""
import json
from sklearn.model_selection import GroupKFold
from rv_lib import *

TD = OUT / "trunc"
d = ROOT / "data/competition/val/item_embeddings"
meta = json.loads((d / "data.meta").read_text())
T = np.memmap(d / "data.memmap", dtype=np.int8, mode="r").reshape(meta["shape"])
ce = np.load(FLAT / "val/cand_emb_idx.npy")
he = np.load(FLAT / "val/hist_emb_idx.npy")
ccat = np.load(FLAT / "val/cand_category_id.npy").astype(np.int64)
hcat = np.load(FLAT / "val/hist_category_id.npy").astype(np.int64)


def unit(idx):
    x = np.asarray(T[np.maximum(idx, 0)], np.float32)
    x[idx < 0] = 0
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-6)


Q = {k: np.zeros(N) for k in ("max cosine to events 193-256", "mean cosine to events 193-256", "share of events 193-256 in the candidate's category", "max(0, max cos old - max cos newest 192)")}
for f in np.where(CAP)[0]:
    o, m, ho = OFF[f], M[f], HOFF[f]
    hi = he[ho:ho + 256]
    S = unit(ce[o:o + m]) @ unit(hi).T
    S[:, hi < 0] = -1.0
    S[ce[o:o + m] < 0, :] = -1.0
    old, new = S[:, 192:], S[:, :192]
    Q["max cosine to events 193-256"][o:o + m] = old.max(1)
    Q["mean cosine to events 193-256"][o:o + m] = old.mean(1)
    Q["share of events 193-256 in the candidate's category"][o:o + m] = (ccat[o:o + m, None] == hcat[ho + 192:ho + 256][None, :]).mean(1)
    Q["max(0, max cos old - max cos newest 192)"][o:o + m] = np.maximum(old.max(1) - new.max(1), 0)
WS = [0.0, 0.05, 0.1, 0.2, 0.3]
for fam, members in (("wide distilled students", ["w5_e10_kd05w192_s951", "w5_e10_kd05w192_s952"]), ("hard-label members", ["w5_e10_base_s951", "w5_e10_base_s952"])):
    S192 = np.mean([z(np.load(TD / f"{v}~K192.npy")) for v in members], 0)
    S256 = np.mean([z(np.load(TD / f"{v}~K256.npy")) for v in members], 0)
    e192, e256 = ndcg(S192), ndcg(S256)
    gt, gse, _ = cmean(e256 - e192, CAP)
    print(f"\n### {fam}: G_true = NDCG(S_256) - NDCG(S_192) at the cap = {gt:+.5f} +- {gse:.5f}")
    capi = np.where(CAP)[0]
    folds = list(GroupKFold(5).split(capi, None, CLU[capi]))
    E = {}
    for qn, q in Q.items():
        zq = z(q)
        E[qn] = np.stack([ndcg(S192 + w * zq) for w in WS], 1)       # [NF, len(WS)]
        ins = E[qn][CAP].mean(0) - e192[CAP].mean()
        nested = np.full(NF, np.nan)
        pick = []
        for tr_i, te_i in folds:
            j = int(np.argmax(E[qn][capi[tr_i]].mean(0)))
            pick.append(WS[j])
            nested[capi[te_i]] = E[qn][capi[te_i], j] - e192[capi[te_i]]
        mu, se, _ = cmean(nested, CAP)
        print(f"  probe q = {qn:52s}: in-sample gain by w {['%+.5f' % x for x in ins]} | nested {mu:+.5f} +- {se:.5f} (picked w {pick}) = {100 * mu / gt:.0f}% of G_true")
    # best q nested as well (q and w chosen together on the training folds)
    nested = np.full(NF, np.nan)
    for tr_i, te_i in folds:
        best = max(((qn, j) for qn in Q for j in range(len(WS))), key=lambda t: E[t[0]][capi[tr_i], t[1]].mean())
        nested[capi[te_i]] = E[best[0]][capi[te_i], best[1]] - e192[capi[te_i]]
    mu, se, _ = cmean(nested, CAP)
    print(f"  best (q, w) nested: {mu:+.5f} +- {se:.5f} = {100 * mu / gt:.0f}% of G_true")
