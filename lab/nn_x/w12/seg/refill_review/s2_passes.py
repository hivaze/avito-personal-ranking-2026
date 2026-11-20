"""Reviewer step 2: inference passes with my own loader and batching (NO label is read here; no training).
  python s2_passes.py --dev cuda            -> out/rv_scores_<set>_<member>.npz, out/rv_meta_<set>.npz
  python s2_passes.py --dev cpu --limit 40  smoke test (fidelity of pass a only; nothing saved)
Only nn_x/w5/model.py is imported (read-only). The study's align_*.npz are read as DATA (its recovered events); mine from out/myalign_*.npz.
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
A = ap.parse_args()
if A.dev == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
else:
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "7", "GPU N only"
import numpy as np
import torch

torch.set_num_threads(12)
from rr_lib import HERE, ROOT, SPLITS, Tables  # noqa: E402

sys.path.insert(0, f"{ROOT}/nn_x/w5")
from model import Ranker  # noqa: E402

DEV = torch.device("cuda:0" if A.dev == "cuda" else "cpu")
MEMBERS = ["w5_e10_kd05w192_s951", "w5_e10_kd05t25w192_s951", "w5_e10_base_s951", "w5_e10_base_s952"]
STUDY = f"{ROOT}/nn_x/w12/seg/refill/out"
t0 = time.time()


def log(*a):
    print(f"[{time.time() - t0:5.0f}s]", *a, flush=True)


def mm_rows(path, rows):
    x = np.load(path, mmap_mode="r")
    o = np.argsort(rows, kind="stable")
    out = np.empty((len(rows),) + x.shape[1:], x.dtype)
    out[o] = x[rows[o]]
    return out


def xt_col(split, name, rows):
    """transform of data.py attach_xt, re-implemented from its docstring; stored as fp16 there, so round-trip through fp16."""
    fc = f"{ROOT}/nn_x/w5/feat_cache/{split}"
    if name.startswith("has:"):
        x = (~np.isnan(mm_rows(f"{fc}/{name[4:]}.npy", rows))).astype(np.float32)
    else:
        x = mm_rows(f"{fc}/{name}.npy", rows).astype(np.float32)
        if name.endswith("_rk"):
            pass
        elif name.endswith("_z"):
            x = np.clip(x, -5, 5)
        elif name.startswith(("seq_cv", "it_hist")):
            x = np.log1p(np.maximum(x, 0))
        else:
            x = np.clip(x, -10, 10)
    return np.nan_to_num(x, nan=0.0).astype(np.float16).astype(np.float32)


def emb_table(sc):
    d = f"{ROOT}/data/competition/{SPLITS[sc]}/item_embeddings"
    shape = tuple(json.loads(open(f"{d}/data.meta").read())["shape"])
    return np.memmap(f"{d}/data.memmap", dtype=np.int8, mode="r", shape=shape)


class SetData:
    def __init__(self, name, T, hold, prand, xt_names):
        self.name = name
        self.split = s = "train" if name == "holdout" else "val"
        sc0 = SPLITS.index(s)
        zs, zm = np.load(f"{STUDY}/align_{name}.npz"), np.load(f"{HERE}/out/myalign_{name}.npz")
        q = zs["fid"]
        assert np.array_equal(q, zm["fid"]) and (T.uid[s][q] != 0).all() and (T.H[s][q] == 256).all()
        S = zs["stats"]
        sel = np.arange(len(q))
        if A.limit:
            sel = sel[:: max(len(q) // A.limit, 1)][:A.limit]
        self.sel = sel
        q = q[sel]
        self.q, self.n = q, len(q)
        self.uid = T.uid[s][q]
        self.read = {"Ts": (zs["A"] & zs["B"] & (S[:, 6] >= 1))[sel], "Us": (zs["A"] & (S[:, 7] >= 1))[sel],
                     "Tm": (zm["A"] & zm["B"] & (zm["nT"] >= 1))[sel], "Um": (zm["A"] & (zm["nU"] >= 1))[sel]}
        self.nrec = {"Ts": np.minimum(S[:, 6], 128)[sel], "Us": np.minimum(S[:, 7], 128)[sel], "Tm": np.minimum(zm["nT"], 128)[sel], "Um": np.minimum(zm["nU"], 128)[sel]}
        rec = {"Ts": (zs["T_sc"][sel].astype(np.int64), zs["T_row"][sel]), "Us": (zs["U_sc"][sel].astype(np.int64), zs["U_row"][sel]),
               "Tm": (zm["T_sc"][sel], zm["T_row"][sel]), "Um": (zm["U_sc"][sel], zm["U_row"][sel])}
        # ---------------- candidates
        self.M = T.M[s][q]
        self.c0 = np.r_[0, np.cumsum(self.M)[:-1]]
        rows = np.concatenate([np.arange(o, o + m) for o, m in zip(T.coff[s][q], self.M)])
        self.rows = rows
        fl = f"{ROOT}/data/flat/{s}"
        cemb = mm_rows(f"{fl}/cand_emb_idx.npy", rows).astype(np.int64)
        tab0 = emb_table(sc0)
        cvec = np.zeros((len(rows), 128), np.int8)
        ok = cemb >= 0
        cvec[ok] = mm_rows_arr(tab0, cemb[ok])
        g = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(DEV)
        self.c_vec, self.c_miss = g(cvec), g(~ok)
        self.c_cat = g(mm_rows(f"{fl}/cand_category_id.npy", rows).astype(np.int64))
        self.c_f = g(np.stack([mm_rows(f"{fl}/cand_f{i}.npy", rows).astype(np.float32) for i in range(1, 6)], 1))
        self.c_xt = g(np.stack([xt_col(s, nm, rows) for nm in xt_names], 1))
        self.u_vec = g(np.stack([np.load(f"{fl}/user_emb.npy", mmap_mode="r")[f] for f in q]).astype(np.float32))
        self.dom = None
        if name == "holdout":
            p = prand[q].astype(np.float32)
            self.dom = g(np.stack([1 - p, p, np.zeros_like(p)], 1))
        # ---------------- events: one local table over (split code, history row)
        code = lambda sc, row: sc * (1 << 40) + row
        st_rows = T.hoff[s][q][:, None] + np.arange(256)[None, :]
        allc = [code(np.full(st_rows.size, sc0, np.int64), st_rows.ravel())]
        for k, (sc, row) in rec.items():
            m = sc >= 0
            allc.append(code(sc[m], row[m]))
        uni = np.unique(np.concatenate(allc))
        esc, erow = uni >> 40, uni & ((1 << 40) - 1)
        E = len(uni)
        e_vec, e_ok = np.zeros((E, 128), np.int8), np.zeros(E, bool)
        e_cat, e_et, e_es, e_item = (np.zeros(E, np.int64) for _ in range(4))
        for c in range(3):
            m = esc == c
            if not m.any():
                continue
            fc = f"{ROOT}/data/flat/{SPLITS[c]}"
            idx = mm_rows(f"{fc}/hist_emb_idx.npy", erow[m]).astype(np.int64)
            okc = idx >= 0
            v = np.zeros((int(m.sum()), 128), np.int8)
            v[okc] = mm_rows_arr(emb_table(c), idx[okc])
            e_vec[m], e_ok[m] = v, okc
            e_cat[m] = mm_rows(f"{fc}/hist_category_id.npy", erow[m])
            e_et[m] = mm_rows(f"{fc}/hist_event_type.npy", erow[m])
            e_es[m] = mm_rows(f"{fc}/hist_event_source.npy", erow[m])
            e_item[m] = mm_rows(f"{fc}/hist_item_id.npy", erow[m])
        self.e_vec, self.e_ok, self.e_cat, self.e_et, self.e_es = g(e_vec), g(e_ok), g(e_cat), g(e_et), g(e_es)
        self.e_item = e_item
        loc = lambda sc, row: np.where(sc >= 0, np.searchsorted(uni, code(np.maximum(sc, 0), np.maximum(row, 0))), -1)
        self.stored = loc(np.full(st_rows.shape, sc0, np.int64), st_rows)
        self.rec = {k: loc(sc, row) for k, (sc, row) in rec.items()}
        for k in self.rec:
            assert (((self.rec[k] >= 0).sum(1) == self.nrec[k]) | ~self.read[k]).all(), k
        log(f"{name}: {self.n} qualifying feeds, {len(rows)} candidates, {E} events; read sets " + ", ".join(f"{k} {int(v.sum())}" for k, v in self.read.items()))

    # ------------------------------------------------------------------ passes: name -> (feed idx, events [k, K], n valid, H for the dense inputs, embedding-off slots or None)
    def build_passes(self):
        st, n = self.stored, self.n
        allf = np.arange(n)
        P = {"a": (allf, st, np.full(n, 256), np.full(n, 256), None), "b": (allf, st[:, :128], np.full(n, 128), np.full(n, 128), None)}
        rng = np.random.default_rng(777)
        self.info = {}
        for k in ("Ts", "Us", "Tm", "Um"):
            r = np.where(self.read[k])[0]
            nf = self.nrec[k][r]
            ev = np.concatenate([st[r, :128], self.rec[k][r]], 1)
            P["c" + k] = (r, ev, 128 + nf, 128 + nf, None)
            if k.endswith("m"):
                continue
            t = k[0]
            P["am" + k] = (r, st[r], 128 + nf, 128 + nf, None)
            # placebo: my donor draw
            don = np.full(len(r), -1)
            for j in range(len(r)):
                c = np.where((nf >= nf[j]) & (self.uid[r] != self.uid[r[j]]))[0]
                if len(c):
                    don[j] = rng.choice(c)
            okd = don >= 0
            P["e" + k] = (r[okd], np.concatenate([st[r[okd], :128], self.rec[k][r[don[okd]]]], 1), 128 + nf[okd], 128 + nf[okd], None)
            # length-input control
            P["bL" + t] = (r, st[r, :128], np.full(len(r), 128), 128 + nf, None)
            # metadata-only control
            off = np.zeros(ev.shape, bool)
            off[:, 128:] = True
            P["cN" + t] = (r, ev, 128 + nf, 128 + nf, off)
            # no-restore control: drop recovered events whose item occurs among the feed's stored events 129-256
            evx = np.full(ev.shape, -1, np.int64)
            evx[:, :128] = st[r, :128]
            nx = np.zeros(len(r), np.int64)
            for j in range(len(r)):
                old_items = set(self.e_item[st[r[j], 128:]].tolist())
                e = self.rec[k][r[j]][:nf[j]]
                keep = e[[x not in old_items for x in self.e_item[e].tolist()]]
                evx[j, 128:128 + len(keep)] = keep
                nx[j] = len(keep)
            P["cX" + t] = (r, evx, 128 + nx, 128 + nx, None)
            self.info[k] = dict(feeds=r, nf=nf, donor_ok=okd, nx=nx)
            log(f"{self.name} {k}: read set {len(r)}, mean fill {nf.mean():.1f}, full fill {int((nf == 128).sum())}, donors found {int(okd.sum())}; no-restore control keeps {nx.sum()} of {nf.sum()} events ({100 * nx.sum() / nf.sum():.1f}%), feeds left with 0 events {int((nx == 0).sum())}")
        return P

    @torch.no_grad()
    def run(self, model, spec, budget):
        fi, ev, nval, hin, off = spec
        out = np.full(len(self.rows), np.nan, np.float32)
        order = np.argsort(-self.M[fi], kind="stable")
        i = 0
        while i < len(order):
            mx = int(self.M[fi[order[i]]])
            B = max(1, budget // mx)
            sl = order[i:i + B]
            i += B
            f = fi[sl]
            Mb = self.M[f]
            ci = self.c0[f][:, None] + np.arange(mx)[None, :]
            cm = np.arange(mx)[None, :] < Mb[:, None]
            ci = np.where(cm, ci, self.c0[f][:, None])
            K = int(-(-nval[sl].max() // 32) * 32)
            e = np.full((len(sl), K), 0, np.int64)
            w = min(K, ev.shape[1])
            e[:, :w] = ev[sl, :w]
            hm = np.arange(K)[None, :] < nval[sl][:, None]
            assert (e[hm] >= 0).all()
            e = np.where(hm, e, 0)
            cit, cmt, et, hmt = (torch.from_numpy(x).to(DEV) for x in (ci, cm, e, hm))
            ft = torch.from_numpy(f).to(DEV)
            hok = self.e_ok[et] & hmt
            if off is not None:
                hok = hok & ~torch.from_numpy(off[sl][:, :K] if off.shape[1] >= K else np.pad(off[sl], ((0, 0), (0, K - off.shape[1])))).to(DEV)
            he = self.e_vec[et].float()
            he[~hok] = 0.0
            cmiss = self.c_miss[cit]
            ce = self.c_vec[cit].float()
            ce[cmiss] = 0.0
            b = dict(user0=torch.zeros(len(sl), dtype=torch.bool, device=DEV), M=torch.from_numpy(Mb).to(DEV), H=torch.from_numpy(hin[sl].astype(np.int64)).to(DEV),
                     cmask=cmt, hmask=hmt, ce_raw=ce, cmiss=cmiss, ccat=self.c_cat[cit], cf=self.c_f[cit], he_raw=he, hemb_ok=hok,
                     hcat=self.e_cat[et], het=self.e_et[et], hes=self.e_es[et], ue_raw=self.u_vec[ft], xt=self.c_xt[cit])
            if self.dom is not None:
                b["dom"] = self.dom[ft]
            lg = model(b, use_pos=False).float().cpu().numpy()
            out[ci[cm]] = lg[cm]
        return out


def mm_rows_arr(x, rows):
    o = np.argsort(rows, kind="stable")
    out = np.empty((len(rows),) + x.shape[1:], x.dtype)
    out[o] = x[rows[o]]
    return out


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    T = Tables()
    hold, prand = T.holdout()
    cks = {}
    for v in MEMBERS:
        ck = torch.load(f"{ROOT}/nn_x/w5/ckpt/{v}.pt", map_location="cpu", weights_only=False)
        assert ck["args"]["holdout"] == 30000 and ck["args"]["max_hist"] == 256, v
        cks[v] = ck
        log(f"member {v}: d {ck['cfg']['d']}, holdout {ck['args']['holdout']}, kd/teacher args: " + ", ".join(f"{k}={ck['args'][k]}" for k in ck["args"] if k.startswith(("kd", "teacher", "distil")))[:200])
    xt_names = cks[MEMBERS[0]]["cfg"]["xt_names"]
    assert all(cks[v]["cfg"]["xt_names"] == xt_names for v in MEMBERS)
    budget = 12000 if A.dev == "cuda" else 3000
    for name in ("holdout", "val"):
        D = SetData(name, T, hold, prand, xt_names)
        P = D.build_passes()
        for v in MEMBERS:
            ck = cks[v]
            model = Ranker(**ck["cfg"])
            model.load_state_dict(ck["state_dict"])
            model = model.to(DEV).eval()
            res = {}
            for pn in (["a"] if A.limit else list(P)):
                res[pn] = D.run(model, P[pn], budget)
                if pn == "a":
                    if name == "val":
                        ref = np.load(f"{ROOT}/preds/nn_{v}_trainonly_val.npy")[D.rows]
                    else:
                        s = np.load(f"{ROOT}/nn_x/w5/preds/holdout/nn_{v}_holdout.npy")
                        Mh = T.M["train"][hold]
                        oh = np.r_[0, np.cumsum(Mh)[:-1]]
                        pos = np.searchsorted(hold, D.q)
                        assert (hold[pos] == D.q).all() and len(s) == Mh.sum()
                        ref = np.concatenate([s[oh[p]:oh[p] + Mh[p]] for p in pos])
                    cc = float(np.corrcoef(res["a"], ref)[0, 1])
                    log(f"FIDELITY {name} {v}: pass a vs stored predictions, flat corr {cc:.6f}, max |d| {np.abs(res['a'] - ref).max():.3e}")
                    assert cc >= 0.9999, "fidelity gate failed"
            if not A.limit:
                np.savez(f"{HERE}/out/rv_scores_{name}_{v}.npz", **res)
                log(f"{name} {v}: {len(res)} passes saved")
            del model
        if not A.limit:
            np.savez(f"{HERE}/out/rv_meta_{name}.npz", q=D.q, uid=D.uid, M=D.M, c0=D.c0, rows=D.rows,
                     **{f"read_{k}": v for k, v in D.read.items()}, **{f"nrec_{k}": v for k, v in D.nrec.items()},
                     **{f"feeds_{k}": v[0] for k, v in P.items()}, **{f"nval_{k}": v[2] for k, v in P.items()})
    log("done")


if __name__ == "__main__":
    main()
