"""Step 2: inference passes of the refill replay (no label is read here; no training).
  python s2_replay.py --dev cuda            all passes, 4 held-out members, holdout + val  -> out/scores_<set>_<member>.npz
  python s2_replay.py --dev cpu --limit 60  smoke test on a few feeds (fidelity gate only, nothing saved)
Own subset loader: only the capped feeds of the 30,000 holdout feeds / of val are read (memory-mapped), with a local item
table. gather() mirrors nn_x/w5/data.py SplitData.gather line by line; the history of each feed is given per pass as a
list of events (stored or recovered from strictly earlier snapshots). nn_x/w5/model.py is imported read-only.
"""
import argparse
import json
import os
import sys
import time

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ.setdefault(_k, "12")
ap = argparse.ArgumentParser()
ap.add_argument("--dev", default="cuda")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--sets", default="holdout,val")
ARGS = ap.parse_args()
if ARGS.dev == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
else:
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "7", "GPU N only"
import numpy as np
import torch

torch.set_num_threads(12)
ROOT = "<repo>"
sys.path.insert(0, f"{ROOT}/nn_x/w5")
from model import Ranker  # noqa: E402  (read-only; common.py is NOT imported)

from align_lib import OUT, SPL, Flat  # noqa: E402

MEMBERS = ["w5_e10_kd05w192_s951", "w5_e10_kd05t25w192_s951", "w5_e10_base_s951", "w5_e10_base_s952"]
FLATD = f"{ROOT}/data/flat"
FC = f"{ROOT}/nn_x/w5/feat_cache"
DEV = torch.device("cuda:0" if ARGS.dev == "cuda" else "cpu")
SEED = 20261115
t0 = time.time()


def log(*a):
    print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)


def take(path, rows):
    """rows of a memory-mapped .npy (rows need not be sorted)."""
    x = np.load(path, mmap_mode="r")
    o = np.argsort(rows, kind="stable")
    out = np.empty(len(rows), x.dtype)
    out[o] = x[rows[o]]
    return out


def xt_column(split, name, rows):
    """exactly SplitData.attach_xt for one column, on the given candidate rows."""
    if name.startswith("has:"):
        x = (~np.isnan(take(f"{FC}/{split}/{name[4:]}.npy", rows))).astype(np.float32)
    else:
        x = take(f"{FC}/{split}/{name}.npy", rows).astype(np.float32)
    if name.startswith("has:") or name.endswith("_rk"):
        pass
    elif name.endswith("_z"):
        x = np.clip(x, -5, 5)
    elif name.startswith(("seq_cv", "it_hist")):
        x = np.log1p(np.maximum(x, 0))
    else:
        x = np.clip(x, -10, 10)
    return np.nan_to_num(x, nan=0.0)


def item_table(sc):
    d = f"{ROOT}/data/competition/{SPL[sc]}/item_embeddings"
    meta = json.loads(open(f"{d}/data.meta").read())
    return np.memmap(f"{d}/data.memmap", dtype=np.int8, mode="r", shape=tuple(meta["shape"]))


class Sub:
    """capped feeds of one set (holdout = train split, val) with stored and recovered history events."""

    def __init__(self, name, F, xt_names, prandom=None):
        self.name = name
        self.split = split = "train" if name == "holdout" else "val"
        self.sc = sc = SPL.index(split)
        z = np.load(OUT / f"align_{name}.npz")
        cap, q = z["cap"], z["fid"]
        S = z["stats"]
        self.A, self.B, self.nT, self.nU = z["A"], z["B"], S[:, 6], S[:, 7]
        if ARGS.limit:  # smoke test: a few qualifying feeds
            keep = np.where(self.A & (self.nU >= 1))[0][:ARGS.limit]
            q = q[keep]
            self.A, self.B, self.nT, self.nU = self.A[keep], self.B[keep], self.nT[keep], self.nU[keep]
            T_sc, T_row, U_sc, U_row = z["T_sc"][keep], z["T_row"][keep], z["U_sc"][keep], z["U_row"][keep]
            cap = q.copy()
        else:
            T_sc, T_row, U_sc, U_row = z["T_sc"], z["T_row"], z["U_sc"], z["U_row"]
        self.fid, self.q = cap, q
        self.qi = np.searchsorted(cap, q)
        assert (cap[self.qi] == q).all()
        self.n = n = len(cap)
        self.uid = F.uid[split][cap]
        M, coff, hoff = F.M[split][cap], F.coff[split][cap], F.hoff[split][cap]
        assert (F.H[split][cap] == 256).all()
        self.M_np = M
        self.loff = np.r_[0, np.cumsum(M)[:-1]]
        self.n_cand = int(M.sum())
        rows = np.concatenate([np.arange(o, o + m) for o, m in zip(coff, M)])
        self.rows = rows
        g = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(DEV)
        c_emb = take(f"{FLATD}/{split}/cand_emb_idx.npy", rows).astype(np.int64)
        self.c_cat = g(take(f"{FLATD}/{split}/cand_category_id.npy", rows).astype(np.int64)).to(torch.int16)
        self.c_f = torch.stack([g(take(f"{FLATD}/{split}/cand_f{i}.npy", rows)) for i in range(1, 6)], 1)
        assert self.c_f.dtype == torch.float32
        self.xt = torch.stack([g(xt_column(split, nm, rows)).to(torch.float16) for nm in xt_names], 1)
        ue = np.load(f"{FLATD}/{split}/user_emb.npy", mmap_mode="r")
        self.user_emb = g(np.stack([ue[f] for f in cap]))
        self.user0 = g(self.uid == 0)
        self.dom = None
        if prandom is not None:
            p = torch.from_numpy(prandom[cap]).to(DEV)
            self.dom = torch.stack([1 - p, p, torch.zeros_like(p)], 1)
        self.M, self.loff_t = g(M), g(self.loff)
        # ---- event universe: stored events of the capped feeds + recovered events (any split)
        st_row = (hoff[:, None] + np.arange(256)[None, :]).astype(np.int64)  # [n,256] rows of the set's own split
        code = lambda s_, r_: s_.astype(np.int64) * (1 << 40) + r_
        codes = [code(np.full(st_row.size, sc), st_row.ravel())]
        for s_, r_ in ((T_sc, T_row), (U_sc, U_row)):
            m = s_ >= 0
            codes.append(code(s_[m], r_[m]))
        uni = np.unique(np.concatenate(codes))
        esc, erow = (uni >> 40).astype(np.int64), uni & ((1 << 40) - 1)
        e_emb, e_cat = np.full(len(uni), -1, np.int64), np.zeros(len(uni), np.int64)
        e_et, e_es, e_item = np.zeros(len(uni), np.int64), np.zeros(len(uni), np.int64), np.zeros(len(uni), np.int64)
        for s_ in range(3):
            m = esc == s_
            if not m.any():
                continue
            r_ = erow[m]
            e_emb[m] = take(f"{FLATD}/{SPL[s_]}/hist_emb_idx.npy", r_)
            e_cat[m] = take(f"{FLATD}/{SPL[s_]}/hist_category_id.npy", r_)
            e_et[m] = take(f"{FLATD}/{SPL[s_]}/hist_event_type.npy", r_)
            e_es[m] = take(f"{FLATD}/{SPL[s_]}/hist_event_source.npy", r_)
            e_item[m] = take(f"{FLATD}/{SPL[s_]}/hist_item_id.npy", r_)
        # ---- local item table over (split, emb idx)
        tcodes = [code(np.full(len(c_emb), sc)[c_emb >= 0], c_emb[c_emb >= 0]), code(esc[e_emb >= 0], e_emb[e_emb >= 0])]
        tuni = np.unique(np.concatenate(tcodes))
        tsc, tidx = (tuni >> 40).astype(np.int64), tuni & ((1 << 40) - 1)
        table = np.zeros((len(tuni), 128), np.int8)
        for s_ in range(3):
            m = tsc == s_
            if m.any():
                table[m] = item_table(s_)[tidx[m]]
        self.table = g(table)
        loc = lambda s_, i_: np.where(i_ >= 0, np.searchsorted(tuni, code(s_, np.maximum(i_, 0))), -1)
        self.c_emb = g(loc(np.full(len(c_emb), sc), c_emb))
        e_loc = loc(esc, e_emb)
        self.e_emb, self.e_cat = g(e_loc), g(e_cat).to(torch.int16)
        self.e_et, self.e_es = g(e_et).to(torch.int8), g(e_es).to(torch.int8)
        ev = lambda s_, r_: np.where(s_ >= 0, np.searchsorted(uni, code(np.maximum(s_, 0).astype(np.int64), np.maximum(r_, 0))), -1)
        self.stored = ev(np.full(st_row.shape, sc), st_row)  # [n,256] local event ids
        self.EVT, self.EVU = ev(T_sc, T_row), ev(U_sc, U_row)  # [nq,128], -1 = none
        assert ((self.EVT >= 0).sum(1) == np.minimum(self.nT, 128)).all() and ((self.EVU >= 0).sum(1) == np.minimum(self.nU, 128)).all()
        # cross-table check (label-free): the same item id has the same embedding in every split table it appears in
        if (esc != sc).any():
            it_u, inv = np.unique(e_item, return_inverse=True)
            has = e_loc >= 0
            first = np.full(len(it_u), -1)
            first[inv[has][::-1]] = np.where(has)[0][::-1]
            same = (table[e_loc[has]] == table[e_loc[first[inv[has]]]]).all(1)
            cross = esc[has] != esc[first[inv[has]]]
            log(f"{name}: item-embedding identity across split tables: {int(cross.sum())} cross-table event pairs of the same item, identical vectors {100 * same[cross].mean():.3f}%")
        log(f"{name}: {n} capped feeds, {self.n_cand} candidates, {len(uni)} events, local table {table.shape[0]} rows; qualifying {len(q)}, A {int(self.A.sum())}, "
            f"R_T {int((self.A & self.B & (self.nT >= 1)).sum())}, R_U {int((self.A & (self.nU >= 1)).sum())}")

    # ------------------------------------------------------------------ passes
    def donors(self, mask, nf, rng):
        """for every read-set feed a donor of another user with >= nf recovered events (same set, same design)."""
        idx = np.where(mask)[0]
        u = self.uid[self.qi]
        out = np.full(len(mask), -1)
        for i in idx:
            c = idx[(nf[idx] >= nf[i]) & (u[idx] != u[i])]
            out[i] = rng.choice(c) if len(c) else -1
        return out

    def passes(self):
        """dict name -> (local feed idx [k], events [k,K] local ids (-1 pad), H [k], max_hist, mode)"""
        st = self.stored
        allf = np.arange(self.n)
        P = {"a": (allf, st, np.full(self.n, 256), 256, ""), "b": (allf, st[:, :128], np.full(self.n, 128), 128, ""),
             "b192": (allf, st[:, :192], np.full(self.n, 192), 192, ""),
             "g": (allf, np.concatenate([st[:, :128], st[:, 192:256]], 1), np.full(self.n, 192), 192, "")}
        rng = np.random.default_rng(SEED)
        self.meta = {}
        for tag, EV, nrec, mask in (("T", self.EVT, self.nT, self.A & self.B & (self.nT >= 1)), ("U", self.EVU, self.nU, self.A & (self.nU >= 1))):
            nf, nf64 = np.minimum(nrec, 128), np.minimum(nrec, 64)
            don = self.donors(mask, nf, rng)
            ok = mask & (don >= 0)
            r = np.where(mask)[0]
            lf = self.qi[r]
            s = st[lf]
            P["c" + tag] = (lf, np.concatenate([s[:, :128], EV[r]], 1), 128 + nf[r], 256, "")
            P["am" + tag] = (lf, s, 128 + nf[r], 256, "")
            P["c192" + tag] = (lf, np.concatenate([s[:, :192], EV[r][:, :64]], 1), 192 + nf64[r], 256, "")
            P["am192" + tag] = (lf, s, 192 + nf64[r], 256, "")
            P["d1" + tag] = (lf, np.concatenate([s, EV[r]], 1), 256 + nf[r], 384, "Xa")
            P["d2" + tag] = (lf, np.concatenate([s, EV[r]], 1), 256 + nf[r], 384, "H256")
            ro = np.where(ok)[0]
            lo = self.qi[ro]
            so = st[lo]
            P["e" + tag] = (lo, np.concatenate([so[:, :128], EV[don[ro]]], 1), 128 + nf[ro], 256, "")
            P["dP" + tag] = (lo, np.concatenate([so, EV[don[ro]]], 1), 256 + nf[ro], 384, "Xa")
            self.meta[tag] = dict(read=mask, donor=don, nf=nf)
            log(f"{self.name} design {tag}: read set {int(mask.sum())}, with a donor {int(ok.sum())}, mean fill {nf[mask].mean():.1f}, full fill {int((nf[mask] == 128).sum())}")
        return P

    def batches(self, lf, budget):
        M = self.M_np[lf]
        o = np.argsort(M, kind="stable")
        out, i = [], 0
        while i < len(o):
            mp = int(-(-M[o[i]] // 16) * 16)
            j = i
            while j < len(o) and M[o[j]] <= mp and (j - i + 1) * mp <= max(budget, mp):
                j += 1
            out.append(o[i:j])
            i = j
        return out  # positions into lf

    @torch.no_grad()
    def gather(self, lf_np, ev_np, H_np, max_hist):
        """mirror of SplitData.gather; lf_np local feed idx [B], ev_np [B,K] local event ids, H_np [B]."""
        dev = DEV
        fidx = torch.from_numpy(np.asarray(lf_np, dtype=np.int64)).to(dev)
        M = self.M[fidx]
        Mx = int(-(-self.M_np[lf_np].max() // 16) * 16)
        ar = torch.arange(Mx, device=dev)
        cmask = ar[None, :] < M[:, None]
        ci = self.loff_t[fidx][:, None] + ar[None, :]
        ci = torch.where(cmask, ci, self.loff_t[fidx][:, None])
        cemb_idx = self.c_emb[ci]
        cmiss = cemb_idx < 0
        ce_raw = self.table[cemb_idx.clamp_min(0)].float()
        ce_raw[cmiss] = 0.0
        H = torch.from_numpy(np.minimum(H_np, max_hist).astype(np.int64)).to(dev)
        Hx = int(np.minimum(H_np, max_hist).max())
        Hx = min(max(-(-Hx // 64) * 64, 64), max_hist)
        evp = np.full((len(lf_np), Hx), -1, np.int64)
        k = min(Hx, ev_np.shape[1])
        evp[:, :k] = ev_np[:, :k]
        ev = torch.from_numpy(evp).to(dev)
        ah = torch.arange(Hx, device=dev)
        hmask = ah[None, :] < H[:, None]
        assert bool((ev[hmask] >= 0).all()), "a masked-in history slot has no event"
        hi = torch.where(hmask, ev, ev[:, :1])  # padded slots point at the feed's first event, as the loader does
        hemb_idx = self.e_emb[hi]
        hmiss = (hemb_idx < 0) | ~hmask
        he_raw = self.table[hemb_idx.clamp_min(0)].float()
        he_raw[hmiss] = 0.0
        b = dict(user0=self.user0[fidx], fidx=fidx, M=M, H=H, cmask=cmask, ci=ci, hmask=hmask, ce_raw=ce_raw, cmiss=cmiss,
                 ccat=self.c_cat[ci].long(), cf=self.c_f[ci], he_raw=he_raw, hemb_ok=~hmiss, hcat=self.e_cat[hi].long(),
                 het=self.e_et[hi].long(), hes=self.e_es[hi].long(), ue_raw=self.user_emb[fidx].float(), hidx=hemb_idx)
        if self.dom is not None:
            b["dom"] = self.dom[fidx]
        b["xt"] = self.xt[ci].float()
        return b

    @torch.no_grad()
    def predict(self, model, spec, budget):
        lf, EV, H, max_hist, mode = spec
        out = torch.full((self.n_cand,), float("nan"), dtype=torch.float32, device=DEV)
        for pb in self.batches(lf, budget):
            b = self.gather(lf[pb], EV[pb], H[pb], max_hist)
            if mode == "Xa":  # tokens of the extended history, in-model dense inputs held at their stored (pass a) values
                ba = self.gather(lf[pb], self.stored[lf[pb]], np.full(len(pb), 256), 256)
                X = model.dense(ba)
                model.dense(b)  # sets b["ce"], b["he"], b["un"] for the extended history; its X is discarded
                lg = model(b, X=X, use_pos=False)
            elif mode == "H256":  # dense inputs over all events shown, H held at 256
                b["H"] = b["H"].clamp_max(256)
                lg = model(b, use_pos=False)
            else:
                lg = model(b, use_pos=False)
            m = b["cmask"]
            out[b["ci"][m]] = lg[m].float()
        return out.cpu().numpy()


def load_member(v):
    ck = torch.load(f"{ROOT}/nn_x/w5/ckpt/{v}.pt", map_location="cpu", weights_only=False)
    a = ck["args"]
    assert a["holdout"] == 30000 and a["max_hist"] == 256 and not ck["cfg"].get("id_buckets") and ck["cfg"]["arch"] == "tf"
    m = Ranker(**ck["cfg"])
    m.load_state_dict(ck["state_dict"])
    W = m.rec.weight.data
    assert W.shape[0] == 256
    # positions >= 256 (pass d only) reuse the embedding of position 255; positions < 256 are untouched
    m.rec = torch.nn.Embedding.from_pretrained(torch.cat([W, W[255:256].expand(128, -1)], 0).clone(), freeze=True)
    return m.to(DEV).eval(), ck["cfg"]


def stored_scores(sub, F, hold, v):
    """the member's stored predictions on the candidates of the capped feeds (label-free fidelity reference)."""
    if sub.name == "val":
        return np.load(f"{ROOT}/preds/nn_{v}_trainonly_val.npy")[sub.rows]
    s = np.load(f"{ROOT}/nn_x/w5/preds/holdout/nn_{v}_holdout.npy")
    Mh = F.M["train"][hold]
    offs = np.r_[0, np.cumsum(Mh)[:-1]]
    pos = np.searchsorted(hold, sub.fid)
    assert (hold[pos] == sub.fid).all() and len(s) == Mh.sum()
    return np.concatenate([s[offs[p]:offs[p] + Mh[p]] for p in pos])


def feed_corr(x, y, loff, M):
    fid = np.repeat(np.arange(len(M)), M)
    def zz(s):
        mu = np.bincount(fid, s, len(M)) / M
        sd = np.sqrt(np.maximum(np.bincount(fid, s * s, len(M)) / M - mu ** 2, 0)) + 1e-9
        return (s - mu[fid]) / sd[fid]
    return np.bincount(fid, zz(x.astype(np.float64)) * zz(y.astype(np.float64)), len(M)) / M


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    F = Flat()
    hold, pr = F.holdout()
    ck0 = torch.load(f"{ROOT}/nn_x/w5/ckpt/{MEMBERS[0]}.pt", map_location="cpu", weights_only=False)
    xt_names = ck0["cfg"]["xt_names"]
    budget = 16384 if ARGS.dev == "cuda" else 4096
    for name in ARGS.sets.split(","):
        sub = Sub(name, F, xt_names, prandom=pr if name == "holdout" else None)
        P = sub.passes()
        names = ["a"] if ARGS.limit else list(P)
        for v in MEMBERS:
            f = OUT / f"scores_{name}_{v}.npz"
            if f.exists() and not ARGS.limit:
                log(name, v, "exists")
                continue
            model, cfg = load_member(v)
            assert cfg["xt_names"] == xt_names
            res = {}
            for pn in names:
                res[pn] = sub.predict(model, P[pn], budget)
                if pn == "a":
                    ref = stored_scores(sub, F, hold, v)
                    c = feed_corr(res["a"], ref, sub.loff, sub.M_np)
                    log(f"FIDELITY {name} {v}: pass a vs stored predictions: flat corr {np.corrcoef(res['a'], ref)[0, 1]:.6f}, per-feed corr mean {c.mean():.6f} min {c.min():.6f}, max|d| {np.abs(res['a'] - ref).max():.3e}")
                    assert np.corrcoef(res["a"], ref)[0, 1] >= 0.9999, "fidelity gate failed"
            if ARGS.limit:
                # smoke: the other passes must run; d1 with the donor / own events must differ from a, am with n = 128 equals a
                for pn in ("b", "g", "cU", "amU", "eU", "d1U", "d2U", "dPU", "cT", "c192U"):
                    if len(P[pn][0]):
                        x = sub.predict(model, P[pn], budget)
                        ok = ~np.isnan(x)
                        log(f"  smoke {pn}: feeds {len(P[pn][0])}, scored rows {int(ok.sum())}, corr with a {np.corrcoef(x[ok], res['a'][ok])[0, 1]:.5f}")
                del model
                continue
            np.savez(str(f) + ".tmp.npz", **{k: x.astype(np.float32) for k, x in res.items()})
            os.replace(str(f) + ".tmp.npz", f)
            log(f"{name} {v}: {len(res)} passes saved")
            del model
            if ARGS.dev == "cuda":
                torch.cuda.empty_cache()
        if not ARGS.limit:
            np.savez(OUT / f"meta_{name}.npz", fid=sub.fid, uid=sub.uid, M=sub.M_np, loff=sub.loff, rows=sub.rows, q=sub.q, qi=sub.qi, A=sub.A, B=sub.B, nT=sub.nT, nU=sub.nU,
                     readT=sub.meta["T"]["read"], readU=sub.meta["U"]["read"], donT=sub.meta["T"]["donor"], donU=sub.meta["U"]["donor"],
                     **{f"feeds_{k}": v[0] for k, v in P.items()}, **{f"H_{k}": v[2] for k, v in P.items()})
    log("done")


if __name__ == "__main__":
    main()
