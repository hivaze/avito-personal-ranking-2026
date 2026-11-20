"""Step 1b (label-free): write the extension-block arrays from out/align_<split>.npz into
<repo>/artifacts/data/feats_new/histext_1117/<split>/ (variant k0, the main one) and .../<split>/k7/ (variant k7:
snapshot at least 7 days older, same rule for every split). New files only: refuses to overwrite anything.

Per split directory (fidx order of data/flat/<split>/feeds.parquet; holdout feeds are train rows):
 feed level  ext_off int64, ext_n int16 (0..128 events appended; 0 = not extended), ext_status int8
             (0 no admissible earlier snapshot or user 0 or H = 0 | 1 snapshot not aligned (A fails) | 2 aligned, not end-anchored
              (B fails) | 3 A and B, no older event | 4 extended), ext_snap_split int8 (-1 none; 0 train 1 val 2 test),
             ext_snap_fidx int32 (-1), ext_gap_days float32 (nan), ext_nT_full int16 (older events before the 128 cut),
             ext_capped bool (H == 256)
 event level (newest first inside a feed; event 0 = the event just older than the feed's oldest stored event)
             ext_item_id int32, ext_category_id int8, ext_event_type int8, ext_event_source int8 (same codes as data/flat),
             ext_emb_idx int32 = row of THIS split's item table (data/competition/<split>/item_embeddings) when the item is in
             it; otherwise n_table + j, a row j of ext_table.npy (the vector of the source split's table); -1 = no embedding,
             ext_src_split int8 + ext_src_row int64 (provenance: global row of data/flat/<src>/hist_* arrays)
 ext_table.npy int8 [n_new, 128] (only in the split root; shared by k0 and k7), meta.json
"""
import json
import os
import sys
import time

sys.dont_write_bytecode = True
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ[_k] = "12"
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hx_lib import ALLOWED, DATA_OUT, KEEP, OUTD, SPL, Flat  # noqa: E402

FLAT = "<repo>/data/flat"
COMP = "<repo>/data/competition"
VARIANTS = {"k0": 0.0, "k7": 7.0}
t0 = time.time()
F = Flat()


def log(*a):
    print(f"[{time.time() - t0:5.0f}s]", *a, flush=True)


def save(path, arr):
    assert not os.path.exists(path), f"refusing to overwrite {path}"
    np.save(path, arr)


def take(name, split, rows):
    x = np.load(f"{FLAT}/{split}/{name}.npy", mmap_mode="r")
    o = np.argsort(rows, kind="stable")
    out = np.empty(len(rows), x.dtype)
    out[o] = x[rows[o]]
    return out


def item_table(split):
    meta = json.loads(open(f"{COMP}/{split}/item_embeddings/data.meta").read())
    return np.memmap(f"{COMP}/{split}/item_embeddings/data.memmap", dtype=np.int8, mode="r", shape=tuple(meta["shape"]))


def item_to_row(split):
    """sorted (item_id, emb_idx) of every item that has a row in the split's table (1:1 by the readme; asserted)."""
    it = np.r_[np.load(f"{FLAT}/{split}/hist_item_id.npy"), np.load(f"{FLAT}/{split}/cand_item_id.npy")].astype(np.int64)
    em = np.r_[np.load(f"{FLAT}/{split}/hist_emb_idx.npy"), np.load(f"{FLAT}/{split}/cand_emb_idx.npy")].astype(np.int64)
    m = em >= 0
    p = np.unique(it[m] * (1 << 32) + em[m])
    items, rows = p >> 32, p & ((1 << 32) - 1)
    assert len(np.unique(items)) == len(items), f"{split}: an item has several table rows"
    no_emb = np.unique(it[~m])
    return items, rows, no_emb


log("start")
for split in SPL:
    z = np.load(f"{OUTD}/align_{split}.npz")
    KS = list(z["KS"])
    STAT = list(z["STAT"])
    ix = {k: i for i, k in enumerate(STAT)}
    S, cnt, ev_sc, ev_row = z["S"], z["cnt"], z["ev_sc"], z["ev_row"]
    n = len(F.uid[split])
    assert (z["fid"] == np.arange(n)).all()
    off_all = np.r_[0, np.cumsum(cnt.reshape(-1).astype(np.int64))[:-1]].reshape(cnt.shape)
    root = f"{DATA_OUT}/{split}"
    os.makedirs(root, exist_ok=True)
    # ---------------- events of every variant (to build one shared ext_table per split)
    per_var = {}
    for vn, k in VARIANTS.items():
        j = KS.index(k)
        ok = (S[:, j, ix["has"]] > 0) & (S[:, j, ix["A"]] > 0) & (S[:, j, ix["B"]] > 0) & (cnt[:, j] > 0)
        n_e = np.where(ok, cnt[:, j], 0).astype(np.int64)
        sel = np.where(ok)[0]
        idx = np.concatenate([np.arange(off_all[i, j], off_all[i, j] + cnt[i, j]) for i in sel]) if len(sel) else np.zeros(0, np.int64)
        per_var[vn] = (j, ok, n_e, ev_sc[idx].astype(np.int8), ev_row[idx].astype(np.int64))
    # ---------------- provenance checks on every event (label-free)
    uid, ts = F.uid[split], F.ts[split]
    for vn, (j, ok, n_e, sc, row) in per_var.items():
        fe = np.repeat(np.arange(n), n_e)
        snap_sc = S[fe, j, ix["sc"]].astype(np.int64)
        snap_f = S[fe, j, ix["fe"]].astype(np.int64)
        assert (snap_sc == sc).all()
        assert np.isin(sc, ALLOWED[split]).all(), "snapshot from a non-admissible split"
        for c in range(3):
            m = sc == c
            if not m.any():
                continue
            src = SPL[c]
            hf = take("hist_fidx", src, row[m]).astype(np.int64)
            assert (hf == snap_f[m]).all(), "event row outside the snapshot's history"
            assert (F.uid[src][snap_f[m]] == uid[fe[m]]).all() and (uid[fe[m]] != 0).all(), "other user or user 0"
            assert (F.ts[src][snap_f[m]] < ts[fe[m]]).all(), "snapshot not strictly earlier"
        assert (n_e <= KEEP).all()
        log(f"{split} {vn}: provenance OK on {len(sc)} events of {int(ok.sum())} feeds (same user != 0, strictly earlier, admissible split, row inside the snapshot)")
    # ---------------- embedding rows in this split's table
    all_sc = np.concatenate([v[3] for v in per_var.values()])
    all_row = np.concatenate([v[4] for v in per_var.values()])
    items_all = np.zeros(len(all_row), np.int64)
    srcemb_all = np.zeros(len(all_row), np.int64)
    for c in range(3):
        m = all_sc == c
        if m.any():
            items_all[m] = take("hist_item_id", SPL[c], all_row[m])
            srcemb_all[m] = take("hist_emb_idx", SPL[c], all_row[m])
    n_table = item_table(split).shape[0]
    if split == "train":
        assert (all_sc == 0).all()
        tgt_emb = srcemb_all.copy()
        ext_items, ext_src = np.zeros(0, np.int64), np.zeros((0, 2), np.int64)
        ext_table = np.zeros((0, 128), np.int8)
        n_in, n_new, n_none = int((tgt_emb >= 0).sum()), 0, int((tgt_emb < 0).sum())
    else:
        items, rows, no_emb = item_to_row(split)
        p = np.searchsorted(items, items_all)
        p = np.minimum(p, len(items) - 1)
        in_tab = items[p] == items_all
        in_noemb = np.isin(items_all, no_emb)
        tgt_emb = np.full(len(items_all), -1, np.int64)
        tgt_emb[in_tab] = rows[p[in_tab]]
        need = ~in_tab & ~in_noemb & (srcemb_all >= 0)  # item absent from this split: carry the source split's vector
        # one ext row per item id (first occurrence)
        u_it, first = np.unique(items_all[need], return_index=True)
        src_pairs = np.stack([all_sc[need][first].astype(np.int64), srcemb_all[need][first]], 1)
        ext_table = np.zeros((len(u_it), 128), np.int8)
        for c in range(3):
            m = src_pairs[:, 0] == c
            if m.any():
                tab = item_table(SPL[c])
                r = src_pairs[m, 1]
                o = np.argsort(r)
                tmp = np.empty((len(r), 128), np.int8)
                tmp[o] = tab[r[o]]
                ext_table[m] = tmp
        tgt_emb[need] = n_table + np.searchsorted(u_it, items_all[need])
        ext_items, ext_src = u_it, src_pairs
        n_in, n_new, n_none = int(in_tab.sum()), int(need.sum()), int((tgt_emb < 0).sum())
        # cross-table identity for items in both tables (label-free sanity)
        both = in_tab & (srcemb_all >= 0) & (all_sc != SPL.index(split))
        if both.any():
            kk = np.where(both)[0][:200000]
            a = np.empty((len(kk), 128), np.int8)
            b = np.empty((len(kk), 128), np.int8)
            ttab = item_table(split)
            for c in range(3):
                m = all_sc[kk] == c
                if m.any():
                    r = srcemb_all[kk][m]
                    o = np.argsort(r)
                    tmp = np.empty((len(r), 128), np.int8)
                    tmp[o] = item_table(SPL[c])[r[o]]
                    a[m] = tmp
            r = tgt_emb[kk]
            o = np.argsort(r)
            tmp = np.empty((len(r), 128), np.int8)
            tmp[o] = ttab[r[o]]
            b[:] = tmp
            same = (a == b).all(1)
            af, bf = a.astype(np.float32), b.astype(np.float32)
            cos = (af * bf).sum(1) / np.maximum(np.linalg.norm(af, axis=1) * np.linalg.norm(bf, axis=1), 1e-6)
            log(f"{split}: recovered events whose item is also in this split's table, taken from another split: {int(both.sum())}; "
                f"bit-identical vectors {same.mean():.3f}; cosine 1/5/50% {np.percentile(cos, [1, 5, 50]).round(4).tolist()}")
    log(f"{split}: events (k0 + k7) {len(all_row)}: item in this split's table {n_in} ({n_in / max(len(all_row), 1):.3f}), "
        f"carried from the source table (ext_table) {n_new} ({n_new / max(len(all_row), 1):.3f}; {len(ext_items)} distinct items), "
        f"no embedding {n_none} ({n_none / max(len(all_row), 1):.3f})")
    save(f"{root}/ext_table.npy", ext_table)
    save(f"{root}/ext_table_items.npy", ext_items.astype(np.int32))
    # ---------------- write each variant
    pos = 0
    for vn, (j, ok, n_e, sc, row) in per_var.items():
        d = root if vn == "k0" else f"{root}/{vn}"
        os.makedirs(d, exist_ok=True)
        m_ev = len(row)
        emb = tgt_emb[pos:pos + m_ev]
        it = items_all[pos:pos + m_ev]
        pos += m_ev
        cat, et, es = np.zeros(m_ev, np.int8), np.zeros(m_ev, np.int8), np.zeros(m_ev, np.int8)
        for c in range(3):
            m = sc == c
            if m.any():
                cat[m] = take("hist_category_id", SPL[c], row[m])
                et[m] = take("hist_event_type", SPL[c], row[m])
                es[m] = take("hist_event_source", SPL[c], row[m])
        has = S[:, j, ix["has"]] > 0
        A = has & (S[:, j, ix["A"]] > 0)
        B = A & (S[:, j, ix["B"]] > 0)
        status = np.zeros(n, np.int8)
        status[has] = 1
        status[A] = 2
        status[B] = 3
        status[ok] = 4
        snap_split = np.where(has, S[:, j, ix["sc"]], -1).astype(np.int8)
        snap_f = np.where(has, S[:, j, ix["fe"]], -1).astype(np.int32)
        gap = np.where(has, S[:, j, ix["gap"]], np.nan).astype(np.float32)
        nT = np.where(B, np.minimum(S[:, j, ix["nT"]], 32767), 0).astype(np.int16)
        off = np.r_[0, np.cumsum(n_e)[:-1]].astype(np.int64)
        save(f"{d}/ext_off.npy", off)
        save(f"{d}/ext_n.npy", n_e.astype(np.int16))
        save(f"{d}/ext_status.npy", status)
        save(f"{d}/ext_snap_split.npy", snap_split)
        save(f"{d}/ext_snap_fidx.npy", snap_f)
        save(f"{d}/ext_gap_days.npy", gap)
        save(f"{d}/ext_nT_full.npy", nT)
        save(f"{d}/ext_capped.npy", F.H[split] == 256)
        save(f"{d}/ext_item_id.npy", it.astype(np.int32))
        save(f"{d}/ext_category_id.npy", cat)
        save(f"{d}/ext_event_type.npy", et)
        save(f"{d}/ext_event_source.npy", es)
        save(f"{d}/ext_emb_idx.npy", emb.astype(np.int32))
        save(f"{d}/ext_src_split.npy", sc)
        save(f"{d}/ext_src_row.npy", row)
        meta = dict(split=split, variant=vn, k_days=VARIANTS[vn], n_feeds=int(n), n_extended=int(ok.sum()), n_events=int(m_ev),
                    max_events_per_feed=KEEP, n_table=int(n_table), ext_table_rows=int(len(ext_items)),
                    ext_table_path=f"{root}/ext_table.npy",
                    emb_rule="ext_emb_idx < n_table: row of this split's item table; >= n_table: row (ext_emb_idx - n_table) of ext_table.npy; -1: none",
                    admissible_snapshot_splits=[SPL[c] for c in ALLOWED[split]],
                    built_by="lab/nn_x/w12/gap1117/lifelong/s1_align_all.py + s3_build_arrays.py (2026-11-17)",
                    status_codes={"0": "no admissible earlier snapshot / user 0 / H = 0", "1": "not aligned (A)", "2": "aligned, not end-anchored (B)",
                                  "3": "A and B, no older event", "4": "extended"})
        assert not os.path.exists(f"{d}/meta.json")
        open(f"{d}/meta.json", "w").write(json.dumps(meta, indent=1))
        # consistency of what was written
        assert (np.r_[off, m_ev][1:] - off == n_e).all()
        assert (((emb >= 0) & (emb < n_table + len(ext_items))) | (emb == -1)).all()
        log(f"{split} {vn}: wrote {d}: {int(ok.sum())} extended feeds, {m_ev} events; view share {(et == 1).mean():.3f}; no-embedding {(emb < 0).mean():.3f}")
log("done")
