"""Render feeds of one split as plain-text cards for LLM review.

A card shows one feed: the user's history (newest first), and every candidate in
the order of the FD16P20 final score, with the inputs a person can read. Labels
appear only when `with_labels` is set (study cards). Nothing here reads a later
snapshot of the same user: each card uses its own feed's history only.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import polars as pl

from avito_ranker.ensemble.build import final_score, load_pool
from avito_ranker.ensemble.feeds import load_feed_index

ROOT = Path("<repo>")
FLAT = ROOT / "artifacts/data/flat"
FC = ROOT / "artifacts/feature_cache"
COMP = ROOT / "artifacts/data/competition"

ETYPE = {0: "unk", 1: "view", 2: "contact1", 3: "contact", 4: "fav", 5: "order", 6: "order", 7: "lead",
         8: "orderstep", 9: "done", 10: "guestview"}
ESRC = {0: "?", 1: "feed", 2: "search", 3: "favlist", 4: "similar", 5: "goods", 6: "rec2", 7: "realty"}


def category_names() -> dict[int, str]:
    names = {0: "Unknown"}
    for line in open(ROOT / "docs/01-competition-and-data.md"):
        m = re.match(r"^\| (\d+) \| (\d+) \| ([^|]+?) \| ([^|]+?) \| ([\d.]+)% \|", line)
        if m:
            names[int(m.group(1))] = m.group(3)
    for code, name in zip(range(34, 39), ["Dogs", "Cats", "Birds", "Aquarium", "Other animals"]):
        names[code] = name
    names[52], names[53] = "Hotels", "Shift gigs"
    return names


class Split:
    def __init__(self, split: str):
        self.split = split
        self.feeds = load_feed_index(split)
        self.table = self.feeds.table
        self.score = final_score(load_pool("FD16P20kd", split), self.feeds, 0.5).astype(np.float64)
        d = FLAT / split
        self.cat = np.load(d / "cand_category_id.npy")
        self.f = {k: np.load(d / f"cand_{k}.npy") for k in ["f1", "f2", "f3", "f4", "f5"]}
        self.cemb = np.load(d / "cand_emb_idx.npy")
        self.target = np.load(d / "cand_target.npy") if (d / "cand_target.npy").exists() and split != "test" else None
        self.hcat = np.load(d / "hist_category_id.npy")
        self.htype = np.load(d / "hist_event_type.npy")
        self.hsrc = np.load(d / "hist_event_source.npy")
        self.hemb = np.load(d / "hist_emb_idx.npy")
        self.uemb = np.load(d / "user_emb.npy").astype(np.float32)
        meta = json.load(open(COMP / split / "item_embeddings/data.meta"))
        self.items = np.memmap(COMP / split / "item_embeddings/data.memmap", dtype=np.int8, mode="r",
                               shape=tuple(meta["shape"]))
        self.hist_cnt = np.load(FC / split / "it_hist_all_luo.npy")
        self.train_exp = np.load(FC / split / "it_cand_cnt_train.npy")
        self.hour = np.load(FC / split / "hour.npy")
        self.names = category_names()

    def group_key(self) -> np.ndarray:
        """User id, with every user-0 feed its own group."""
        u = self.table["user_id"].to_numpy().astype(np.int64)
        f = self.table["fidx"].to_numpy().astype(np.int64)
        return np.where(u == 0, -(f + 1), u)

    def order(self, i: int) -> np.ndarray:
        """Rows of feed i sorted by the final score, best first (ties: stable on row order)."""
        o, m = self.feeds.offsets[i], self.feeds.sizes[i]
        return o + np.argsort(-self.score[o:o + m], kind="stable")

    def card(self, i: int, label: str, with_labels: bool) -> str:
        t = self.table.row(i, named=True)
        o, m = int(self.feeds.offsets[i]), int(self.feeds.sizes[i])
        ho, hn = int(t["hist_off"]), int(t["H"])
        rows = self.order(i)
        names = self.names
        # history, newest first (hpos 0 = newest)
        hc, ht, hs, he = (self.hcat[ho:ho + hn], self.htype[ho:ho + hn], self.hsrc[ho:ho + hn], self.hemb[ho:ho + hn])
        hvec_ok = he >= 0
        hvec = self.items[he[hvec_ok]].astype(np.float32) if hvec_ok.any() else np.zeros((0, 128), np.float32)
        hidx = np.nonzero(hvec_ok)[0]
        u = self.uemb[i]
        user_zero = not u.any()
        ts = int(t["timestamp"]) + 3 * 3600
        wd = ["Thu", "Fri", "Sat", "Sun", "Mon", "Tue", "Wed"][(ts // 86400) % 7]
        hour = (ts % 86400) // 3600
        out = []
        guest = "guest (user 0, not logged in)" if t["user_id"] == 0 else "logged-in user"
        out.append(f"=== FEED {label} | {m} candidates | history {hn} events | {guest} | "
                   f"user vector {'MISSING' if user_zero else 'present'} | {wd} {hour:02d}h Moscow ===")
        if with_labels:
            tg = self.target[rows]
            pos_ranks = (np.nonzero(tg)[0] + 1).tolist()
            from avito_ranker.metrics import ndcg_per_feed
            nd = ndcg_per_feed(self.score[o:o + m], self.target[o:o + m], np.array([0]), np.array([m]))[0]
            out.append(f"LABELS: positives at model ranks {pos_ranks}; model NDCG of this feed {nd:.3f}")
        # newest 40 events, consecutive repeats compressed
        toks = []
        for k in range(min(hn, 40)):
            tok = f"{names.get(int(hc[k]), 'cat' + str(int(hc[k])))}/{ETYPE.get(int(ht[k]), str(int(ht[k])))}"
            if int(ht[k]) in (1, 4, 7):
                tok += "@" + ESRC.get(int(hs[k]), "src" + str(int(hs[k])))
            if toks and toks[-1][0] == tok:
                toks[-1][1] += 1
            else:
                toks.append([tok, 1])
        hist_line = ", ".join(t0 if n == 1 else f"{t0} x{n}" for t0, n in toks)
        out.append(f"HISTORY newest first (first 40 of {hn}): {hist_line if hist_line else '(empty)'}")
        # category totals over the whole history
        tot = {}
        for c, e in zip(hc.tolist(), ht.tolist()):
            v = tot.setdefault(c, [0, 0, -1])
            if e in (1, 10):
                v[0] += 1
            else:
                v[1] += 1
        newest = {}
        for k, c in enumerate(hc.tolist()):
            newest.setdefault(c, k)
        top = sorted(tot.items(), key=lambda kv: -(kv[1][0] + kv[1][1]))[:10]
        out.append("HISTORY by category (views / other actions / newest event index): " +
                   ("; ".join(f"{names.get(c, c)} {v[0]}v/{v[1]}a/@{newest[c]}" for c, v in top) or "(empty)"))
        # candidates
        cv_ok = self.cemb[rows] >= 0
        cvec = np.zeros((m, 128), np.float32)
        if cv_ok.any():
            cvec[cv_ok] = self.items[self.cemb[rows][cv_ok]].astype(np.float32)
        ucos = (cvec @ u) / 65536.0
        if len(hidx):
            sim = (cvec @ hvec.T) / 65536.0                       # m x n_hist_with_vec
            best = sim.argmax(1)
            hmax = sim[np.arange(m), best]
            hmax_pos = hidx[best]
            recent = hidx < 10
            h10 = sim[:, recent].max(1) if recent.any() else np.full(m, np.nan)
        else:
            hmax = np.full(m, np.nan); hmax_pos = np.zeros(m, int); h10 = np.full(m, np.nan)
        f2 = self.f["f2"][rows]
        nz = f2[f2 > 0]
        home = np.bincount(np.unique(nz, return_inverse=True)[1]).argmax() if len(nz) else None
        home_val = np.unique(nz)[home] if home is not None else None
        out.append("CANDIDATES in model order: rank|z|category|f3|f1|f2|f4|f5|ucos|hmax@idx:type|h10|cathist|pop|exp"
                   + ("|*=POSITIVE" if with_labels else ""))

        def num(x: float) -> str:
            return f"{x:.2f}".lstrip("0") if x >= 0 else f"{x:.2f}".replace("-0.", "-.")

        for r, row in enumerate(rows):
            c = int(self.cat[row])
            ch = tot.get(c)
            cath = f"{ch[0]}v{ch[1]}a@{newest[c]}" if ch else "-"
            f2v = self.f["f2"][row]
            f2s = num(f2v) + ("H" if home_val is not None and f2v == home_val else "")
            uc = "-" if user_zero or not cv_ok[r] else num(ucos[r])
            hm = "-" if np.isnan(hmax[r]) or not cv_ok[r] else f"{num(hmax[r])}@{hmax_pos[r]}:{ETYPE.get(int(ht[hmax_pos[r]]), '?')}"
            h1 = "-" if np.isnan(h10[r]) or not cv_ok[r] else num(h10[r])
            line = (f"r{r + 1}|{self.score[row]:+.2f}|{names.get(c, 'cat' + str(c))}|{num(self.f['f3'][row])}|"
                    f"{num(self.f['f1'][row])}|{f2s}|{num(self.f['f4'][row])}|{num(self.f['f5'][row])}|"
                    f"{uc}|{hm}|{h1}|{cath}|{int(self.hist_cnt[row])}|{int(self.train_exp[row])}")
            if with_labels and self.target[row]:
                line += "|*"
            out.append(line)
        return "\n".join(out) + "\n"
