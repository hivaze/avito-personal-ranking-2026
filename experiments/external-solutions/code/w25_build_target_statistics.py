# Transfer of the 2025 winning recipe, part 1 (2026-11-13): click-ratio statistics in that recipe's style.
# Counts and smoothed click rates by item, category, item cluster, user, user x category and user x cluster,
# once-per-user counts, the category of the user's last earlier click, and their crosses.
# Leak rule: a train row on day d uses train days before d only; val and test rows use all train days; a
# feed's own label and the same user's later feeds are never used.
# Run on CPU from lab/nn_x/w25 (log logs/build_te.log); the command line is not on record.
# Produced lab/nn_x/w25/data/<split>/w25_*.npy. The linear item-item scorers of the recipe are built by
# build_lae.py in the same lab folder, which follows the winners' published code and is not copied here.
# Kept as it ran: paths and names refer to the lab tree (see lab/ and docs/glossary.md).
# Original: lab/nn_x/w25/build_te.py
#
"""W25 G2/G3/G4: time-respecting target statistics in the 2025-winner style (contact ratios by node / category / user / user x
category, once-per-user counts, Bayesian ratio with C = mean impressions, last-contact category, crosses).

Leak rules: train rows on MSK day d use train days < d only (expanding window; day 0 -> NaN / prior), never the feed's own label and
never a same-user later feed; val/test rows use all train days. Sources: S1 = all train feeds, S2 = exploration-like train feeds
(p_random > 0.5). User-level keys: user_id > 0 only (user 0 -> NaN). Val-label statistics (S3) are not built here (plan.txt).
Outputs: data/<split>/w25_*.npy float32; train rows = all p_random>0.5 feeds' rows (data/train/rows.npy, same as build_lae.py).
"""
import json, sys, time
import numpy as np
sys.path.insert(0, "<repo>/src")
from common import feed_meta, FLAT, FEATS  # noqa: E402
from pathlib import Path

ROOT = Path("<repo>"); CLU = ROOT / "data/clu_work"; WD = ROOT / "nn_x/w25/data"
T0 = time.time()
NIT = 111_400_000  # item ids < 111.32M
NU = 2_483_762 + 1


def log(*a):
    print(time.strftime("%H:%M:%S"), f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def day_of(ts):
    return (ts.astype(np.int64) + 3 * 3600) // 86400


def bctr(pos, imp, p0, C):
    return ((C * p0 + pos) / (C + imp)).astype(np.float32)


class Fam:
    """dense per-key accumulators (keys are small non-negative ints)."""

    def __init__(self, n, once=False, s2=True, days=False, npd=False):
        self.n = n
        self.imp1 = np.zeros(n, np.float32); self.pos1 = np.zeros(n, np.float32); self.ex1 = np.zeros(n, np.float32)
        self.s2 = s2
        if s2:
            self.imp2 = np.zeros(n, np.float32); self.pos2 = np.zeros(n, np.float32); self.ex2 = np.zeros(n, np.float32)
        self.once = once
        if once:
            self.oimp = np.zeros(n, np.float32); self.opos = np.zeros(n, np.float32)
        self.days = days
        if days:
            self.last = np.full(n, -1, np.int16); self.first = np.full(n, -1, np.int16)
        self.npd = npd
        if npd:
            self.npos_days = np.zeros(n, np.float32)
        self.tot_imp1 = 0.0; self.tot_pos1 = 0.0; self.nseen1 = 0
        self.tot_imp2 = 0.0; self.tot_pos2 = 0.0; self.nseen2 = 0

    def priors(self):
        p1 = self.tot_pos1 / max(self.tot_imp1, 1.0); C1 = self.tot_imp1 / max(self.nseen1, 1)
        p2 = self.tot_pos2 / max(self.tot_imp2, 1.0); C2 = self.tot_imp2 / max(self.nseen2, 1)
        return p1, max(C1, 1.0), p2, max(C2, 1.0)

    def add(self, k, y, ex, p2, fo, fp, d):
        """k: keys of the day's rows; y labels; ex expected clicks; p2: S2 mask; fo/fp: first-impression / first-positive (user,key) flags."""
        n = self.n
        self.imp1 += np.bincount(k, minlength=n); self.pos1 += np.bincount(k, weights=y, minlength=n); self.ex1 += np.bincount(k, weights=ex, minlength=n)
        if self.s2:
            k2 = k[p2]
            self.imp2 += np.bincount(k2, minlength=n); self.pos2 += np.bincount(k2, weights=y[p2], minlength=n)
            self.ex2 += np.bincount(k2, weights=ex[p2], minlength=n)
        if self.once:
            self.oimp += np.bincount(k[fo], minlength=n); self.opos += np.bincount(k[fp], minlength=n)
        if self.days:
            uk = np.unique(k)
            self.last[uk] = d
            fr = self.first[uk]; self.first[uk[fr < 0]] = d
        if self.npd:
            self.npos_days[np.unique(k[y > 0])] += 1
        self.tot_imp1 = float(self.imp1.sum()); self.tot_pos1 = float(self.pos1.sum()); self.nseen1 = int((self.imp1 > 0).sum())
        if self.s2:
            self.tot_imp2 = float(self.imp2.sum()); self.tot_pos2 = float(self.pos2.sum()); self.nseen2 = int((self.imp2 > 0).sum())


def main():
    ftr, Mtr, otr = feed_meta("train")
    N = int(Mtr.sum())
    fidx = np.load(FLAT / "train/cand_fidx.npy")
    dayf = day_of(ftr["timestamp"].to_numpy()); dayf = (dayf - dayf.min()).astype(np.int16)
    assert np.all(np.diff(dayf) >= 0), "feeds must be time-sorted"
    uidf = ftr["user_id"].to_numpy().astype(np.int64)
    pr = np.load(ROOT / "eda/labels_position/out/train_prandom.npz")
    pf = np.nan_to_num(pr["p_random"].astype(np.float32), nan=0.0) > 0.5
    day = dayf[fidx]; uid = uidf[fidx]; p2 = pf[fidx]
    uid_eff = np.where(uid > 0, uid, NU + fidx.astype(np.int64))  # user-0 feeds count as distinct users
    item = np.load(FLAT / "train/cand_item_id.npy").astype(np.int64)
    cat = np.load(FLAT / "train/cand_category_id.npy").astype(np.int64)
    g4 = np.load(CLU / "cand_cl_4096_train.npy").astype(np.int64); g4 = np.where(g4 < 0, 4096, g4)  # 4096 = missing bucket
    g1 = np.load(CLU / "cand_cl_1024_train.npy").astype(np.int64); g1 = np.where(g1 < 0, 1024, g1)
    y = np.load(FLAT / "train/cand_target.npy").astype(np.float32)
    ex = np.load(CLU / "train_ex.npy").astype(np.float32)
    need = p2.copy()
    rows_need = np.where(need)[0]
    stored = np.load(WD / "train/rows.npy") if (WD / "train/rows.npy").exists() else None
    if stored is not None:
        assert (stored == rows_need).all()
    else:
        np.save(WD / "train/rows.npy", rows_need)
    log(f"train rows {N}, needed (p>0.5) {len(rows_need)}")
    first_of_feed = np.r_[True, fidx[1:] != fidx[:-1]]
    # ---- once-per-user flags: first impression / first positive of each (user, key) pair, in time order ----
    def first_flags(key, maxkey):
        pair = key * np.int64(NU + N) + uid_eff
        _, first = np.unique(pair, return_index=True)
        fo = np.zeros(N, bool); fo[first] = True
        posr = np.where(y > 0)[0]
        _, fp_ = np.unique(pair[posr], return_index=True)
        fp = np.zeros(N, bool); fp[posr[fp_]] = True
        return fo, fp
    fams = {}
    keys = {"it": item, "c": cat, "g": g4, "u": uid, "uc": uid * 64 + cat, "ucl": None}
    # ucl key via unique (uid x cl1024 is too large for a dense table)
    ucl_raw = uid * 1025 + g1
    ucl_u, ucl_inv = np.unique(ucl_raw, return_inverse=True); keys["ucl"] = ucl_inv.astype(np.int64)
    log(f"ucl keys {len(ucl_u)}")
    sizes = {"it": NIT, "c": 64, "g": 4097, "u": NU, "uc": NU * 64, "ucl": len(ucl_u)}
    flags = {}
    for fam in ("it", "c", "g"):
        flags[fam] = first_flags(keys[fam], sizes[fam]); log(f"first flags {fam}")
    fams["it"] = Fam(NIT, once=True, s2=True, days=True, npd=True)
    fams["c"] = Fam(64, once=True, s2=True)
    fams["g"] = Fam(4097, once=True, s2=True)
    fams["u"] = Fam(NU, s2=True)
    fams["uc"] = Fam(NU * 64, s2=False)
    fams["ucl"] = Fam(len(ucl_u), s2=False)
    u_feeds = np.zeros(NU, np.float32)
    u_lastcat = np.full(NU, -1, np.int16); u_lastposday = np.full(NU, -1, np.int16)
    cat_min_hpos_tr = np.load(FEATS / "train/cat_min_hpos.npy", mmap_mode="r")

    def feats_for(sel, d, split_uid, split_item, split_cat, split_g4, split_ucl, split_cmh, dcur):
        """features for rows `sel` (arrays indexed by sel already) as of state (days < dcur). d = day of the rows (scalar)."""
        out = {}
        F = fams
        # item
        p1, C1, p2_, C2 = F["it"].priors()
        it = split_item
        imp1, pos1, ex1 = F["it"].imp1[it], F["it"].pos1[it], F["it"].ex1[it]
        out["w25_it_bctr1"] = bctr(pos1, imp1, p1, C1)
        out["w25_it_once_imp1"] = F["it"].oimp[it]; out["w25_it_once_pos1"] = F["it"].opos[it]
        out["w25_it_once_bctr1"] = bctr(F["it"].opos[it], F["it"].oimp[it], p1, C1)
        ld, fd = F["it"].last[it].astype(np.float32), F["it"].first[it].astype(np.float32)
        out["w25_it_dsl"] = np.where(ld < 0, np.nan, dcur - ld).astype(np.float32)
        out["w25_it_dsf"] = np.where(fd < 0, np.nan, dcur - fd).astype(np.float32)
        out["w25_it_npd"] = F["it"].npos_days[it]
        out["w25_it_imp2"] = F["it"].imp2[it]; out["w25_it_pos2"] = F["it"].pos2[it]
        out["w25_it_bctr2"] = bctr(F["it"].pos2[it], F["it"].imp2[it], p2_, C2)
        out["w25_it_coec2"] = ((F["it"].pos2[it] + 1) / (F["it"].ex2[it] + 1)).astype(np.float32)
        # category
        c = split_cat
        p1c, C1c, p2c, C2c = F["c"].priors()
        out["w25_c_coec1"] = ((F["c"].pos1[c] + 1) / (F["c"].ex1[c] + 1)).astype(np.float32)
        out["w25_c_once_bctr1"] = bctr(F["c"].opos[c], F["c"].oimp[c], p1c, C1c)
        out["w25_c_bctr2"] = bctr(F["c"].pos2[c], F["c"].imp2[c], p2c, C2c)
        out["w25_c_coec2"] = ((F["c"].pos2[c] + 1) / (F["c"].ex2[c] + 1)).astype(np.float32)
        # group cl4096
        g = split_g4
        p1g, C1g, p2g, C2g = F["g"].priors()
        out["w25_g_imp1"] = F["g"].imp1[g]; out["w25_g_pos1"] = F["g"].pos1[g]
        out["w25_g_bctr1"] = bctr(F["g"].pos1[g], F["g"].imp1[g], p1g, C1g)
        out["w25_g_coec1"] = ((F["g"].pos1[g] + 1) / (F["g"].ex1[g] + 1)).astype(np.float32)
        out["w25_g_once_bctr1"] = bctr(F["g"].opos[g], F["g"].oimp[g], p1g, C1g)
        out["w25_g_bctr2"] = bctr(F["g"].pos2[g], F["g"].imp2[g], p2g, C2g)
        out["w25_g_coec2"] = ((F["g"].pos2[g] + 1) / (F["g"].ex2[g] + 1)).astype(np.float32)
        out["w25_grp_logratio"] = np.log1p(out["w25_g_bctr2"]).astype(np.float32)
        # user
        u = split_uid; ku = u > 0
        nanu = lambda a: np.where(ku, a, np.nan).astype(np.float32)
        p1u, C1u, p2u, C2u = F["u"].priors()
        uimp, upos = F["u"].imp1[u], F["u"].pos1[u]
        out["w25_u_pos1"] = nanu(upos)
        out["w25_u_bctr1"] = nanu(bctr(upos, uimp, p1u, C1u))
        out["w25_u_feeds1"] = nanu(u_feeds[u])
        lc = u_lastcat[u].astype(np.float32); lpd = u_lastposday[u].astype(np.float32)
        out["w25_u_lastpos_cat"] = nanu(np.where(lc < 0, np.nan, lc))
        out["w25_u_dsl_pos"] = nanu(np.where(lpd < 0, np.nan, dcur - lpd))
        out["w25_u_imp2"] = nanu(F["u"].imp2[u]); out["w25_u_bctr2"] = nanu(bctr(F["u"].pos2[u], F["u"].imp2[u], p2u, C2u))
        # user x category
        uc = u * 64 + c
        ucimp, ucpos = F["uc"].imp1[uc], F["uc"].pos1[uc]
        out["w25_uc_pos1"] = nanu(ucpos)
        out["w25_uc_share_pos"] = nanu(np.where(upos > 0, ucpos / np.maximum(upos, 1), np.nan))
        out["w25_uc_share_imp"] = nanu(np.where(uimp > 0, ucimp / np.maximum(uimp, 1), np.nan))
        out["w25_uc_coec1"] = nanu((ucpos + 1) / (F["uc"].ex1[uc] + 1))
        # user x cl1024
        p1l, C1l, _, _ = F["ucl"].priors()
        if split_ucl is not None:
            li, lp = F["ucl"].imp1[split_ucl], F["ucl"].pos1[split_ucl]
        else:
            li = np.zeros(len(u), np.float32); lp = np.zeros(len(u), np.float32)
        out["w25_ucl_imp1"] = nanu(li); out["w25_ucl_pos1"] = nanu(lp); out["w25_ucl_bctr1"] = nanu(bctr(lp, li, p1l, C1l))
        # crosses (G4)
        out["w25_x_cat_eq_lastpos"] = nanu((lc >= 0) & (lc == c)).astype(np.float32)
        out["w25_x_cat_eq_lastpos"][~ku] = np.nan
        out["w25_x_cat_eq_lastseen"] = (split_cmh == 0).astype(np.float32)
        out["w25_x_it_u"] = nanu(out["w25_it_bctr1"] * out["w25_u_bctr1"])
        out["w25_x_c_u"] = nanu(F["c"].pos1[c] / np.maximum(F["c"].imp1[c], 1) * out["w25_u_bctr1"])
        out["w25_x_g_ucl"] = nanu(out["w25_g_bctr1"] * out["w25_ucl_bctr1"])
        return out

    # ---- expanding daily loop over train ----
    ndays = int(dayf.max()) + 1
    bounds = np.searchsorted(day, np.arange(ndays + 1))
    acc = {}
    wpos = np.searchsorted(rows_need, np.arange(N))  # position of row in rows_need (valid where need)
    for d in range(ndays):
        lo, hi = bounds[d], bounds[d + 1]
        if hi <= lo:
            continue
        sl = slice(lo, hi)
        nd = np.where(need[sl])[0] + lo
        if len(nd):
            if d == 0:
                feats = None
            else:
                feats = feats_for(nd, d, uid[nd], item[nd], cat[nd], g4[nd], keys["ucl"][nd], np.asarray(cat_min_hpos_tr[nd]), d)
            for name in (feats.keys() if feats else []):
                if name not in acc:
                    acc[name] = np.full(len(rows_need), np.nan, np.float32)
                acc[name][wpos[nd]] = feats[name]
            if d == 0:
                # day 0: only the label-free cross exists
                if "w25_x_cat_eq_lastseen" not in acc:
                    acc["w25_x_cat_eq_lastseen"] = np.full(len(rows_need), np.nan, np.float32)
                acc["w25_x_cat_eq_lastseen"][wpos[nd]] = (np.asarray(cat_min_hpos_tr[nd]) == 0).astype(np.float32)
        # accumulate day d
        yd, exd, p2d = y[sl], ex[sl], p2[sl]
        for fam in ("it", "c", "g"):
            fo, fp = flags[fam]
            fams[fam].add(keys[fam][sl], yd, exd, p2d, fo[sl], fp[sl], d)
        ud = uid[sl]; ku = ud > 0
        fams["u"].add(np.where(ku, ud, 0), yd, exd, p2d, None, None, d)
        fams["uc"].add(np.where(ku, keys["uc"][sl], 0), yd, exd, p2d, None, None, d)
        fams["ucl"].add(keys["ucl"][sl], yd, exd, p2d, None, None, d)
        ff = first_of_feed[sl] & ku
        u_feeds += np.bincount(ud[ff], minlength=NU)
        pm = (yd > 0) & ku
        u_lastcat[ud[pm]] = cat[sl][pm].astype(np.int16); u_lastposday[ud[pm]] = d  # later rows overwrite (time order)
        log(f"day {d}: rows {hi - lo}, needed {len(nd)}")
    # user-0 rows: all user-keyed features NaN (handled by nanu); clean the zero-key accumulators
    for name, arr in acc.items():
        np.save(WD / f"train/{name}.npy", arr)
    log(f"train written: {len(acc)} columns")
    # ---- val / test from the full train state ----
    for split in ("val", "test"):
        f, M, off = feed_meta(split)
        fi = np.load(FLAT / f"{split}/cand_fidx.npy")
        dsplit = day_of(f["timestamp"].to_numpy()); dsplit = dsplit - day_of(ftr["timestamp"].to_numpy()).min()
        u = f["user_id"].to_numpy().astype(np.int64)[fi]
        it = np.load(FLAT / f"{split}/cand_item_id.npy").astype(np.int64); it = np.where(it < NIT, it, 0)
        c = np.load(FLAT / f"{split}/cand_category_id.npy").astype(np.int64)
        g = np.load(CLU / f"cand_cl_4096_{split}.npy").astype(np.int64); g = np.where(g < 0, 4096, g)
        gl = np.load(CLU / f"cand_cl_1024_{split}.npy").astype(np.int64); gl = np.where(gl < 0, 1024, gl)
        raw = u * 1025 + gl
        pos = np.searchsorted(ucl_u, raw); pos = np.minimum(pos, len(ucl_u) - 1)
        hit = ucl_u[pos] == raw
        ucl_idx = np.where(hit, pos, -1)
        cmh = np.load(FEATS / f"{split}/cat_min_hpos.npy")
        # days-since features need the row's day: compute per feed-day group
        out = {}
        dcur = dsplit[fi].astype(np.int64)
        for dd in np.unique(dcur):
            sel = np.where(dcur == dd)[0]
            ucl_sel = ucl_idx[sel]
            fe = feats_for(sel, dd, u[sel], it[sel], c[sel], g[sel], np.where(ucl_sel >= 0, ucl_sel, 0), cmh[sel], int(dd))
            miss = ucl_sel < 0
            for nm in ("w25_ucl_imp1", "w25_ucl_pos1"):
                fe[nm][miss] = np.where(u[sel][miss] > 0, 0.0, np.nan)
            p1l, C1l, _, _ = fams["ucl"].priors()
            fe["w25_ucl_bctr1"][miss] = np.where(u[sel][miss] > 0, bctr(np.zeros(miss.sum()), np.zeros(miss.sum()), p1l, C1l), np.nan)
            fe["w25_x_g_ucl"][miss] = fe["w25_g_bctr1"][miss] * fe["w25_ucl_bctr1"][miss]
            for nm, v in fe.items():
                if nm not in out:
                    out[nm] = np.empty(len(fi), np.float32)
                out[nm][sel] = v
        for nm, v in out.items():
            np.save(WD / f"{split}/{nm}.npy", v)
        log(f"{split} written {len(out)} columns")
    names = sorted(acc.keys())
    G3 = [n for n in names if n.startswith(("w25_it_", "w25_c_", "w25_g_", "w25_u_", "w25_uc_", "w25_ucl_"))]
    G3S2 = [n for n in G3 if n.endswith(("2",)) and "_dsl" not in n]
    G3once = [n for n in G3 if "_once_" in n]
    G4 = [n for n in names if n.startswith("w25_x_")]
    gp = WD / "groups.json"
    Gd = json.loads(gp.read_text()) if gp.exists() else {}
    Gd["G3"] = G3; Gd["G3S2"] = G3S2; Gd["G3once"] = G3once; Gd["G4"] = G4
    gp.write_text(json.dumps(Gd, indent=1))
    log("done", {k: len(v) for k, v in Gd.items()})


if __name__ == "__main__":
    main()
