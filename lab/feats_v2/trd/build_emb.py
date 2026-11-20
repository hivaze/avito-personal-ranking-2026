"""trd_ item/embedding-level population features (GPU, CUDA_VISIBLE_DEVICES=N only).

Pool events E = history events with hpos < KREC (and emb available) of POOL feeds (trd_lib.pool_mask).
For a candidate c (feed f, user u, time t, category k, unit embedding v) and a time window w around t,
using only events from OTHER feeds and other non-zero users (no same-user leakage via later histories):
  trd_ex_<w>      exact item matches: # events with item_id == candidate item      (per 1K window pool feeds)
  trd_n95_<w>     # events with cos(v, e) >= 0.95                                  (per 1K window pool feeds)
  trd_n85_<w>     # events with cos >= 0.85                                        (per 1K window pool feeds)
  trd_kde_<w>     sum exp(20 (cos - 1))                                            (per 1K window pool feeds)
  trd_max_<w>     max cos to any window event                                      (NaN if none)
  trd_msc_<w>     mean cos to window events of the SAME category                   (NaN if none)
  trd_mg_<w>      mean cos to all window events
  (+ _e1 variants restricted to event_type 1 for kde/n95)
Supply side (candidates of other pool feeds in the window, subsampled 1/SUPSUB by hash):
  trd_skde_<w>    sum exp(20 (cos - 1)) over window pool candidates                (per 1K window pool feeds, x SUPSUB)
  trd_dsr_<w>     log((kde + 0.05) / (skde + 0.5))   demand/supply ratio in embedding space
Windows: s6h = [t-3h, t+3h], sday = same MSK calendar day (24h windows crossing midnight drift: val days are isolated, test days contiguous).
"""
import sys, time, os
sys.path.insert(0, "<repo>/feats_v2/trd")
from trd_lib import *
import torch

KREC = 10
BETA = 20.0
WINDOWS = {"s6h": (10800, 10800), "sday": "sday"}   # sday = whole same MSK calendar day (identical structure in val/test)
SUPSUB = 4
CH = int(os.environ.get('TRD_CH', 4096))


def tmask(w, dt, qday, eday):
    wv = WINDOWS[w]
    if wv == "sday":
        return eday == qday
    return (dt >= -wv[0]) & (dt <= wv[1])


def unit_rows(table, idx):
    v = np.asarray(table[np.maximum(idx, 0)], dtype=np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-6
    v[idx < 0] = 0
    return v


def build(split, do_supply=True, cand_rows=None, out_suffix=""):
    dev = torch.device("cuda")
    t0 = time.time()
    f = feeds_df(split)
    n = len(f)
    ts = f["timestamp"].to_numpy().astype(np.int64)
    uid = f["user_id"].to_numpy().astype(np.int64)
    pm = pool_mask(split, f)
    tbl = load_item_emb(split)
    # ---- pool events
    hf = np.load(FLAT / split / "hist_fidx.npy", mmap_mode="r")
    hp = np.load(FLAT / split / "hist_hpos.npy", mmap_mode="r")
    he = np.load(FLAT / split / "hist_emb_idx.npy", mmap_mode="r")
    sel = np.flatnonzero((np.asarray(hp) < KREC) & pm[np.asarray(hf)] & (np.asarray(he) >= 0))
    ev_f = np.asarray(hf[sel]).astype(np.int64)
    order = np.argsort(ts[ev_f], kind="stable"); sel = sel[order]; ev_f = ev_f[order]
    ev_t = ts[ev_f]
    ev_u = uid[ev_f]
    ev_it = np.load(FLAT / split / "hist_item_id.npy", mmap_mode="r")[sel].astype(np.int64)
    ev_cat = np.load(FLAT / split / "hist_category_id.npy", mmap_mode="r")[sel].astype(np.int64)
    ev_e1 = np.load(FLAT / split / "hist_event_type.npy", mmap_mode="r")[sel] == 1
    ev_v = unit_rows(tbl, np.asarray(he[sel]))
    print(split, "pool events", len(sel), f"{time.time()-t0:.0f}s", flush=True)
    # pool feeds time index (for per-window feed counts)
    pts = np.sort(ts[pm])
    # ---- supply pool: candidates of pool feeds, subsampled
    cf_all = np.load(FLAT / split / "cand_fidx.npy", mmap_mode="r")
    cit_all = np.load(FLAT / split / "cand_item_id.npy", mmap_mode="r")
    cemb_all = np.load(FLAT / split / "cand_emb_idx.npy", mmap_mode="r")
    if do_supply:
        h = (np.asarray(cit_all).astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15) + np.asarray(cf_all).astype(np.uint64) * np.uint64(0xBF58476D1CE4E5B9)) >> np.uint64(40)
        ssel = np.flatnonzero(pm[np.asarray(cf_all)] & ((h % np.uint64(SUPSUB)) == 0) & (np.asarray(cemb_all) >= 0))
        sp_f = np.asarray(cf_all[ssel]).astype(np.int64)  # already time-sorted (rows sorted by timestamp)
        sp_t = ts[sp_f]; sp_u = uid[sp_f]
        sp_v = unit_rows(tbl, np.asarray(cemb_all[ssel]))
        print(split, "supply pool", len(ssel), f"{time.time()-t0:.0f}s", flush=True)
    # ---- query candidates
    rows = np.arange(len(cf_all)) if cand_rows is None else cand_rows
    q_f = np.asarray(cf_all[rows]).astype(np.int64)
    q_t = ts[q_f]; q_u = uid[q_f]
    q_it = np.asarray(cit_all[rows]).astype(np.int64)
    q_cat = np.load(FLAT / split / "cand_category_id.npy", mmap_mode="r")[rows].astype(np.int64)
    q_emb = np.asarray(cemb_all[rows])
    NQ = len(rows)
    names = []
    for w in WINDOWS:
        names += [f"ex_{w}", f"n95_{w}", f"n85_{w}", f"kde_{w}", f"max_{w}", f"msc_{w}", f"mg_{w}", f"kde_e1_{w}", f"n95_e1_{w}", f"nfeed_{w}"]
        if do_supply:
            names += [f"skde_{w}", f"sn95_{w}"]
    out = {k: np.full(NQ, np.nan, np.float32) for k in names}
    wmax_l = 86400; wmax_r = 86400
    EV_V = torch.from_numpy(ev_v).to(dev, torch.float16)
    EV_T = torch.from_numpy(ev_t).to(dev); EV_U = torch.from_numpy(ev_u).to(dev); EV_F = torch.from_numpy(ev_f).to(dev)
    EV_DAY = (EV_T + MSK) // 86400
    EV_IT = torch.from_numpy(ev_it).to(dev); EV_CAT = torch.from_numpy(ev_cat).to(dev); EV_E1 = torch.from_numpy(ev_e1).to(dev)
    if do_supply:
        SP_V = torch.from_numpy(sp_v).to(dev, torch.float16)
        SP_T = torch.from_numpy(sp_t).to(dev); SP_U = torch.from_numpy(sp_u).to(dev); SP_F = torch.from_numpy(sp_f).to(dev)
        SP_DAY = (SP_T + MSK) // 86400
    tbl_np = tbl
    for s in range(0, NQ, CH):
        e = min(NQ, s + CH)
        tq0, tq1 = q_t[s], q_t[e - 1]
        # superset of all windows: [min(day_start, t-3h), max(day_end, t+3h)] (tight bounds; same results as wider ones)
        lo_t = min((tq0 + MSK) // 86400 * 86400 - MSK, tq0 - 10800); hi_t = max((tq1 + MSK) // 86400 * 86400 - MSK + 86399, tq1 + 10800)
        a = np.searchsorted(ev_t, lo_t, "left"); b = np.searchsorted(ev_t, hi_t, "right")
        qv = torch.from_numpy(unit_rows(tbl_np, q_emb[s:e])).to(dev, torch.float16)
        QT = torch.from_numpy(q_t[s:e]).to(dev)[:, None]; QDAY = (QT + MSK) // 86400; QU = torch.from_numpy(q_u[s:e]).to(dev)[:, None]
        QF = torch.from_numpy(q_f[s:e]).to(dev)[:, None]
        QIT = torch.from_numpy(q_it[s:e]).to(dev)[:, None]; QCAT = torch.from_numpy(q_cat[s:e]).to(dev)[:, None]
        qmiss = torch.from_numpy(q_emb[s:e] < 0).to(dev)
        if b > a:
            S = (qv @ EV_V[a:b].T).float()                                   # (c, e)
            other = (EV_F[a:b][None] != QF) & ((EV_U[a:b][None] != QU) | (QU == 0))
            dt = EV_T[a:b][None] - QT
            K = torch.exp(BETA * (S - 1.0))
            same_it = EV_IT[a:b][None] == QIT
            same_cat = EV_CAT[a:b][None] == QCAT
            e1 = EV_E1[a:b][None]
        for w, wv in WINDOWS.items():
            lo, hi = day_or_window_bounds(pts, q_t[s:e], wv)
            nfeed = (hi - lo).astype(np.float32)
            out[f"nfeed_{w}"][s:e] = nfeed
            scale = 1000.0 / np.maximum(nfeed, 1.0)
            if b <= a:
                for k in ["ex", "n95", "n85", "kde", "kde_e1", "n95_e1"]:
                    out[f"{k}_{w}"][s:e] = 0.0
                continue
            m = other & tmask(w, dt, QDAY, EV_DAY[a:b][None])
            mf = m.float()
            S_m = torch.where(m, S, torch.tensor(-2.0, device=dev))
            res = {
                "ex": (m & same_it).float().sum(1),
                "n95": ((S >= 0.95) & m).float().sum(1),
                "n85": ((S >= 0.85) & m).float().sum(1),
                "kde": (K * mf).sum(1),
                "kde_e1": (K * (mf * e1)).sum(1),
                "n95_e1": ((S >= 0.95) & m & e1).float().sum(1),
            }
            for k, v in res.items():
                out[f"{k}_{w}"][s:e] = v.cpu().numpy() * scale
            mx = S_m.max(1).values
            cnt = mf.sum(1)
            mx = torch.where(cnt > 0, mx, torch.tensor(float("nan"), device=dev))
            out[f"max_{w}"][s:e] = mx.cpu().numpy()
            msc_n = (m & same_cat).float()
            nsc = msc_n.sum(1)
            out[f"msc_{w}"][s:e] = torch.where(nsc > 0, (S * msc_n).sum(1) / nsc.clamp(min=1), torch.tensor(float("nan"), device=dev)).cpu().numpy()
            out[f"mg_{w}"][s:e] = torch.where(cnt > 0, (S * mf).sum(1) / cnt.clamp(min=1), torch.tensor(float("nan"), device=dev)).cpu().numpy()
        if do_supply:
            a2 = np.searchsorted(sp_t, lo_t, "left"); b2 = np.searchsorted(sp_t, hi_t, "right")
            acc = {w: [torch.zeros(e - s, device=dev), torch.zeros(e - s, device=dev)] for w in WINDOWS}
            SUBC = 200000
            for a3 in range(a2, b2, SUBC):
                b3 = min(b2, a3 + SUBC)
                S2 = (qv @ SP_V[a3:b3].T).float()
                other2 = (SP_F[a3:b3][None] != QF) & ((SP_U[a3:b3][None] != QU) | (QU == 0))
                dt2 = SP_T[a3:b3][None] - QT
                K2 = torch.exp(BETA * (S2 - 1.0))
                for w in WINDOWS:
                    m2 = other2 & tmask(w, dt2, QDAY, SP_DAY[a3:b3][None])
                    acc[w][0] += (K2 * m2.float()).sum(1)
                    acc[w][1] += ((S2 >= 0.95) & m2).float().sum(1)
                del S2, other2, dt2, K2
            for w in WINDOWS:
                nfeed = out[f"nfeed_{w}"][s:e]
                scale = SUPSUB * 1000.0 / np.maximum(nfeed, 1.0)
                out[f"skde_{w}"][s:e] = acc[w][0].cpu().numpy() * scale
                out[f"sn95_{w}"][s:e] = acc[w][1].cpu().numpy() * scale
        # missing candidate embedding -> NaN for embedding-based features
        qm = q_emb[s:e] < 0
        if qm.any():
            for k in names:
                if not k.startswith(("ex_", "nfeed_")):
                    out[k][s:e][qm] = np.nan
        if (s // CH) % 50 == 0:
            print(split, s, NQ, f"{time.time()-t0:.0f}s", flush=True)
    return out


def save_all(split, out):
    d = FEATS / split
    for k, v in out.items():
        if k.startswith("nfeed_"):
            continue
        np.save(d / f"trd_{k}.npy", v.astype(np.float32))
    for w in WINDOWS:
        if f"skde_{w}" in out:
            np.save(d / f"trd_dsr_{w}.npy", np.log((out[f"kde_{w}"] + 0.05) / (out[f"skde_{w}"] + 0.5)).astype(np.float32))


if __name__ == "__main__":
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "5"
    for sp in sys.argv[1].split(","):
        o = build(sp)
        save_all(sp, o)
        print(sp, "saved", flush=True)
