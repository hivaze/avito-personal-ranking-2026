"""trd_nb_: user-similarity-weighted population density (GPU, CUDA_VISIBLE_DEVICES=N).

For a candidate c (unit item emb v) in feed f (unit user emb u, time t), pool events e (hpos<10 events of OTHER pool feeds of
other non-zero users) in window w (same MSK day / +-3h), each event carrying its feed's user embedding u_e:
  weight      a_e = exp(G * (cos(u, u_e) - 1))                         (G = 10: neighbours in user-embedding space)
  trd_nb_k_<w>    = sum_e a_e exp(20 (cos(v,e)-1)) / sum_e a_e          neighbour-weighted kernel density around the candidate
  trd_nb_c_<w>    = sum_e a_e cos(v,e) / sum_e a_e                      neighbour-weighted mean similarity
  trd_nb_m_<w>    = max cos(v, e) over events of the TOPN most similar other users in the window
  trd_nb_kr_<w>   = log(nb_k / (unweighted mean kernel + 1e-4))        personalisation lift of the density
NaN when the user embedding is all-zero or the candidate embedding is missing.
"""
import sys, time, os
sys.path.insert(0, "<repo>/feats_v2/trd")
from trd_lib import *
import torch

KREC = 10; G = 10.0; BETA = 20.0; TOPN = 50
WINDOWS = {"s6h": (10800, 10800), "sday": "sday"}
FCH = 8    # feeds per chunk (GPU N is shared; keep memory small)


def unit(x):
    x = np.asarray(x, np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return np.where(n > 0, x / np.maximum(n, 1e-6), 0).astype(np.float32), n[:, 0] > 0


def build(split, feed_sel=None):
    dev = torch.device("cuda")
    t0 = time.time()
    f = feeds_df(split); n = len(f)
    ts = f["timestamp"].to_numpy().astype(np.int64); uid = f["user_id"].to_numpy().astype(np.int64)
    M = f["M"].to_numpy().astype(np.int64); co = f["cand_off"].to_numpy()
    ue, uok = unit(np.load(FLAT / split / "user_emb.npy"))
    pm = pool_mask(split, f)
    tbl = load_item_emb(split)
    hf = np.load(FLAT / split / "hist_fidx.npy", mmap_mode="r"); hp = np.load(FLAT / split / "hist_hpos.npy", mmap_mode="r")
    he = np.load(FLAT / split / "hist_emb_idx.npy", mmap_mode="r")
    sel = np.flatnonzero((np.asarray(hp) < KREC) & pm[np.asarray(hf)] & (np.asarray(he) >= 0))
    ev_f = np.asarray(hf[sel]).astype(np.int64)
    o = np.argsort(ts[ev_f], kind="stable"); sel = sel[o]; ev_f = ev_f[o]
    ev_t = ts[ev_f]
    ev_v, _ = unit(tbl[np.asarray(he[sel])])
    EV_V = torch.from_numpy(ev_v).to(dev, torch.float16)
    EV_F = torch.from_numpy(ev_f).to(dev); EV_T = torch.from_numpy(ev_t).to(dev); EV_DAY = (EV_T + MSK) // 86400
    EV_U = torch.from_numpy(uid[ev_f]).to(dev)
    EV_UE = torch.from_numpy(ue[ev_f]).to(dev, torch.float16)
    EV_UOK = torch.from_numpy(uok[ev_f]).to(dev)
    cemb = np.load(FLAT / split / "cand_emb_idx.npy", mmap_mode="r")
    NC = int(M.sum())
    names = [f"{k}_{w}" for w in WINDOWS for k in ["k", "c", "m", "kr"]]
    out = {k: np.full(NC, np.nan, np.float32) for k in names}
    feeds = np.arange(n) if feed_sel is None else feed_sel
    for s in range(0, len(feeds), FCH):
        fb = feeds[s:s + FCH]
        f0, f1 = fb[0], fb[-1]
        d0 = (ts[f0] + MSK) // 86400 * 86400 - MSK; d1 = (ts[f1] + MSK) // 86400 * 86400 - MSK + 86399
        a = np.searchsorted(ev_t, min(d0, ts[f0] - 10800), "left"); b = np.searchsorted(ev_t, max(d1, ts[f1] + 10800), "right")
        if b <= a:
            continue
        r0, r1 = co[f0], co[f1] + M[f1]
        rows = np.arange(r0, r1)
        cf = np.repeat(fb, M[fb])               # feed of each candidate row (fb contiguous)
        qv, qok = unit(tbl[np.maximum(np.asarray(cemb[r0:r1]), 0)])
        qok &= np.asarray(cemb[r0:r1]) >= 0
        QV = torch.from_numpy(qv).to(dev, torch.float16)
        # feed-level weights (FCH x E)
        FU = torch.from_numpy(ue[fb]).to(dev, torch.float16)
        FT = torch.from_numpy(ts[fb]).to(dev)[:, None]; FDAY = (FT + MSK) // 86400
        FUID = torch.from_numpy(uid[fb]).to(dev)[:, None]; FIDX = torch.from_numpy(fb).to(dev)[:, None]
        us = (FU @ EV_UE[a:b].T).float()
        other = (EV_F[a:b][None] != FIDX) & ((EV_U[a:b][None] != FUID) | (FUID == 0)) & EV_UOK[a:b][None]
        S = (QV @ EV_V[a:b].T).float()          # (C, E)
        K = torch.exp(BETA * (S - 1.0))
        loc = torch.from_numpy(cf - f0).to(dev)  # candidate -> local feed index? fb contiguous so cf-f0 indexes fb
        for w, wv in WINDOWS.items():
            if wv == "sday":
                tm = EV_DAY[a:b][None] == FDAY
            else:
                dt = EV_T[a:b][None] - FT
                tm = (dt >= -wv[0]) & (dt <= wv[1])
            m = other & tm                                           # (F, E)
            wgt = torch.where(m, torch.exp(G * (us - 1.0)), torch.zeros_like(us))
            wsum = wgt.sum(1, keepdim=True).clamp(min=1e-12)
            wn = (wgt / wsum)[loc]                                   # (C, E)
            mf = m.float()[loc]
            cnt = mf.sum(1).clamp(min=1)
            kw = (K * wn).sum(1); cw = (S * wn).sum(1); ku = (K * mf).sum(1) / cnt
            # top-N neighbours: threshold on user sim per feed
            usm = torch.where(m, us, torch.full_like(us, -9))
            kth = torch.topk(usm, min(TOPN, usm.shape[1]), dim=1).values[:, -1:]
            topm = (usm >= kth) & m
            mx = torch.where(topm[loc], S, torch.full_like(S, -2)).max(1).values
            has = (m.sum(1) > 0)[loc]
            res = {"k": kw, "c": cw, "m": mx, "kr": torch.log((kw + 1e-4) / (ku + 1e-4))}
            for k, v in res.items():
                v = torch.where(has, v, torch.full_like(v, float("nan"))).cpu().numpy()
                out[f"{k}_{w}"][r0:r1] = v
        bad = ~(qok & uok[cf])
        for k in names:
            out[k][r0:r1][bad] = np.nan
        if (s // FCH) % 400 == 0:
            print(split, s, len(feeds), b - a, f"{time.time()-t0:.0f}s", flush=True)
    return out


if __name__ == "__main__":
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "5"
    for sp in sys.argv[1].split(","):
        o = build(sp)
        for k, v in o.items():
            np.save(FEATS / sp / f"trd_nb_{k}.npy", v)
        print(sp, "saved", flush=True)
