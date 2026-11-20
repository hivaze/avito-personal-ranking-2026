"""Shared helpers of the fresh-exposure pre-gate (nn_x/w12/fresh/pregate). CPU only, read-only on the project.

Counting rule (DECLARATION.txt A1-A2): for a query row (item i, feed f, user u, time t) count the source feeds g that show
item i with user(g) != u, per window: a (ts < t), a24 (t - 86400 <= ts < t), b (same Moscow day), c (|ts - t| <= 86400),
d (all), e (ts > t). uid-0 feeds get a private negative user key each (they count as different users).
"""
import sys

import numba as nb
import numpy as np
import polars as pl

ROOT = "<repo>"
sys.path.insert(0, f"{ROOT}/src")
FLAT = f"{ROOT}/data/flat"
FEATS = f"{ROOT}/data/feats"
OUT = f"{ROOT}/nn_x/w12/fresh/pregate/out"
BOUND = 1790497920  # src/pmix.py: public-like = test feeds with timestamp < BOUND
WIN = ["a", "a24", "b", "c", "d", "e"]
MSK = 10800


def day_of(ts):
    return (np.asarray(ts, np.int64) + MSK) // 86400


def load_union():
    """Feed table and candidate rows of val + test (val first). Returns dict of numpy arrays."""
    fs, rows = [], []
    goff = 0
    for sp in ("val", "test"):
        f = pl.read_parquet(f"{FLAT}/{sp}/feeds.parquet")
        n = len(f)
        fs.append(dict(ts=f["timestamp"].to_numpy().astype(np.int64), uid=f["user_id"].to_numpy().astype(np.int64),
                       M=f["M"].to_numpy().astype(np.int64), off=f["cand_off"].to_numpy().astype(np.int64),
                       split=np.full(n, 0 if sp == "val" else 1, np.int8)))
        it = np.load(f"{FLAT}/{sp}/cand_item_id.npy").astype(np.int64)
        fi = np.load(f"{FLAT}/{sp}/cand_fidx.npy").astype(np.int64)
        assert (fi == np.repeat(np.arange(n), fs[-1]["M"])).all()
        rows.append(dict(item=it, gid=fi + goff))
        goff += n
    F = {k: np.concatenate([x[k] for x in fs]) for k in fs[0]}
    R = {k: np.concatenate([x[k] for x in rows]) for k in rows[0]}
    ng = len(F["ts"])
    F["eff_user"] = np.where(F["uid"] == 0, -1 - np.arange(ng), F["uid"]).astype(np.int64)
    F["day"] = day_of(F["ts"])
    F["goff"] = np.concatenate([[0], np.cumsum(F["M"])[:-1]]).astype(np.int64)  # row offset in the union row order
    R["cold"] = np.concatenate([np.load(f"{FEATS}/{sp}/tte_it_imp.npy") == 0 for sp in ("val", "test")])
    return F, R


@nb.njit(parallel=True, cache=False)
def count_kernel(q_item, q_user, q_ts, q_day, s_item_sorted, s_user, s_ts, s_day, s_ok, u_items, u_start, out):
    """out[q, 0..5] = counts for windows a, a24, b, c, d, e. Source arrays are sorted by item; u_items / u_start index the
    item groups (u_start has len(u_items) + 1 entries). s_ok = source row allowed (False = never counted)."""
    nq = len(q_item)
    for q in nb.prange(nq):
        it = q_item[q]
        k = np.searchsorted(u_items, it)
        if k >= len(u_items) or u_items[k] != it:
            continue
        t = q_ts[q]; u = q_user[q]; d = q_day[q]
        a = 0; a24 = 0; b = 0; c = 0; dd = 0; e = 0
        for j in range(u_start[k], u_start[k + 1]):
            if not s_ok[j] or s_user[j] == u:
                continue
            ts = s_ts[j]
            dd += 1
            if ts < t:
                a += 1
                if ts >= t - 86400:
                    a24 += 1
            elif ts > t:
                e += 1
            if s_day[j] == d:
                b += 1
            if abs(ts - t) <= 86400:
                c += 1
        out[q, 0] = a; out[q, 1] = a24; out[q, 2] = b; out[q, 3] = c; out[q, 4] = dd; out[q, 5] = e


def count(q_item, q_gid, F, s_item, s_gid, src_ok_feed=None):
    """Counts [nq, 6] int32 of the query rows against the source rows (item, feed) under rule A1."""
    order = np.argsort(s_item, kind="stable")
    si = s_item[order]; sg = s_gid[order]
    u_items, u_first = np.unique(si, return_index=True)
    u_start = np.concatenate([u_first, [len(si)]]).astype(np.int64)
    s_ok = np.ones(len(si), np.bool_) if src_ok_feed is None else src_ok_feed[sg]
    out = np.zeros((len(q_item), 6), np.int32)
    count_kernel(q_item, F["eff_user"][q_gid], F["ts"][q_gid], F["day"][q_gid], si, F["eff_user"][sg], F["ts"][sg],
                 F["day"][sg], s_ok, u_items, u_start, out)
    return out


@nb.njit(parallel=True, cache=False)
def feed_forms(off, M, n, r_out, z_out):
    """Within-feed percent rank of n (average ties; 0.5 if all tied) and z of log1p(n) (0 if sd == 0)."""
    for f in nb.prange(len(M)):
        o = off[f]; m = M[f]
        v = n[o:o + m]
        order = np.argsort(v)
        i = 0
        while i < m:
            j = i
            while j + 1 < m and v[order[j + 1]] == v[order[i]]:
                j += 1
            r = ((i + j) / 2.0) / (m - 1) if m > 1 else 0.5
            for k in range(i, j + 1):
                r_out[o + order[k]] = r
            i = j + 1
        s1 = 0.0; s2 = 0.0
        for k in range(m):
            l = np.log1p(v[k]); s1 += l; s2 += l * l
        mu = s1 / m; var = s2 / m - mu * mu
        if var > 1e-12:
            sd = np.sqrt(var)
            for k in range(m):
                z_out[o + k] = (np.log1p(v[k]) - mu) / sd
        else:
            for k in range(m):
                z_out[o + k] = 0.0


def forms(off, M, n):
    n = np.ascontiguousarray(n, np.float64)
    r = np.empty(len(n), np.float64); z = np.empty(len(n), np.float64)
    feed_forms(off, M, n, r, z)
    return r, z


PSI_EDGES = np.array([-0.5, 0.5, 1.5, 2.5, 4.5, 9.5, 19.5, 1e12])  # {0}, {1}, {2}, {3-4}, {5-9}, {10-19}, {20+}
PSI_LABELS = ["0", "1", "2", "3-4", "5-9", "10-19", "20+"]


def psi(a, b, edges=PSI_EDGES):
    pa = np.histogram(a, edges)[0] / max(len(a), 1); pb = np.histogram(b, edges)[0] / max(len(b), 1)
    pa, pb = np.clip(pa, 1e-6, None), np.clip(pb, 1e-6, None)
    return float(np.sum((pa - pb) * np.log(pa / pb)))


def psi_q(a, b, nbin=10):
    """PSI on quantile bins of a (unique edges)."""
    q = np.unique(np.quantile(a, np.linspace(0, 1, nbin + 1)[1:-1]))
    return psi(a, b, np.r_[-np.inf, q + 1e-9, np.inf])
