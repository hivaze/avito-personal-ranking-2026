"""Re-check: are all raw columns consumed, bit-exactly, by data/flat?  CPU only.
1. schema of every part file of every split (names, arrow types, key-value metadata) + zip listing vs extracted tree
2. val + test fully, train on 6 random parts: every raw column == its flat array (exact), incl. list order
3. nulls inside lists, NaNs, value ranges (anything the flattening could have silently mapped)
"""
import os
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "POLARS_MAX_THREADS"): os.environ[k] = "12"
import glob, json, zipfile, numpy as np, polars as pl, pyarrow.parquet as pq
R = "<repo>/data/"
FL = R + "flat/"
MAP = {"item_id": "cand_item_id", "category_id": "cand_category_id", "f1": "cand_f1", "f2": "cand_f2", "f3": "cand_f3", "f4": "cand_f4",
       "f5": "cand_f5", "pos": "cand_pos", "target": "cand_target", "item_id__embedding_idx": "cand_emb_idx",
       "history__item_id": "hist_item_id", "history__category_id": "hist_category_id", "history__event_type": "hist_event_type",
       "history__event_source": "hist_event_source", "history__item_id__embedding_idx": "hist_emb_idx"}
SCAL = ["feed_id", "timestamp", "user_id"]
# 1. zip listing vs tree
z = zipfile.ZipFile(R + "raw/competition.zip"); names = [n for n in z.namelist() if not n.endswith("/")]
ondisk = {os.path.relpath(p, R) for p in glob.glob(R + "competition/**", recursive=True) if os.path.isfile(p)}
print("zip files", len(names), "on disk", len(ondisk), "in zip not on disk", sorted(set(names) - ondisk)[:5], "on disk not in zip", sorted(ondisk - set(names))[:5])
print("non-part files in zip:", [n for n in names if "part-" not in n])
allcols = {}
for split in ("train", "val", "test"):
    files = sorted(glob.glob(R + f"competition/{split}/impressions/*.parquet"))
    sch0 = None; kv = set(); nrow = 0; nrg = 0; created = set()
    for fp in files:
        pf = pq.ParquetFile(fp); s = pf.schema_arrow
        if sch0 is None: sch0 = s
        assert s.equals(sch0, check_metadata=False), fp
        kv |= set((s.metadata or {}).keys()); nrow += pf.metadata.num_rows; nrg += pf.metadata.num_row_groups; created.add(pf.metadata.created_by)
        for i in range(len(s)):
            assert not (s.field(i).metadata), (fp, s.field(i).name, s.field(i).metadata)   # no per-field metadata
    allcols[split] = [(f.name, str(f.type)) for f in sch0]
    print(f"\n{split}: {len(files)} files, rows {nrow}, row groups {nrg}, created_by {created}, kv-metadata keys {sorted(k.decode() for k in kv)}")
    print("  columns", len(sch0), allcols[split])
    md = pq.ParquetFile(files[0]).schema_arrow.metadata or {}
    for k, v in md.items():
        v = v.decode(errors="replace"); print("  kv", k.decode(), "len", len(v), "::", v[:300].replace("\n", " "))
print("\ncolumn sets: train-val", set(allcols["train"]) - set(allcols["val"]), "val-test", set(allcols["val"]) - set(allcols["test"]), "test-val", set(allcols["test"]) - set(allcols["val"]))

def check(split, files, full):
    df = pl.read_parquet(files)
    f = pl.read_parquet(FL + f"{split}/feeds.parquet")
    if full:
        df = df.sort(["timestamp", "feed_id"]); sel = np.arange(f.height); assert df.height == f.height
    else:
        j = df.select("feed_id").join(f.select("feed_id", "fidx"), on="feed_id", how="left"); assert j["fidx"].null_count() == 0
        df = df.with_columns(j["fidx"]).sort("fidx"); sel = df["fidx"].to_numpy()
    fs = f[sel]
    res = {}
    for c in SCAL: res[c] = bool((df[c].to_numpy() == fs[c].to_numpy()).all())
    M = df["item_id"].list.len().to_numpy(); H = df["history__item_id"].list.len().to_numpy()
    res["M"] = bool((M == fs["M"].to_numpy()).all()); res["H"] = bool((H == fs["H"].to_numpy()).all())
    ue = np.load(FL + f"{split}/user_emb.npy", mmap_mode="r")
    res["user_embedding"] = bool((df["user_embedding"].list.to_array(128).to_numpy() == ue[sel]).all())
    co = fs["cand_off"].to_numpy(); ho = fs["hist_off"].to_numpy()
    crow = np.concatenate([np.arange(o, o + m) for o, m in zip(co, M)]); hrow = np.concatenate([np.arange(o, o + m) for o, m in zip(ho, H)]) if H.sum() else np.zeros(0, np.int64)
    info = {}
    for c, fn in MAP.items():
        if c not in df.columns: continue
        ser = df[c]
        nn_in = ser.list.eval(pl.element().is_null().sum()).list.first().sum()   # nulls INSIDE lists
        raw = ser.explode().drop_nulls().to_numpy() if c.startswith("history") else ser.explode().to_numpy()
        fl = np.load(FL + f"{split}/{fn}.npy", mmap_mode="r"); rows = hrow if c.startswith("history") else crow
        a = np.asarray(fl[rows]) if full is False else np.asarray(fl)
        eq = (raw == a) | ((raw != raw) & (a != a)) if raw.dtype.kind == "f" else (raw == a)
        res[c] = bool(len(raw) == len(a) and eq.all())
        info[c] = (str(ser.dtype), str(fl.dtype), int(nn_in), int(np.isnan(raw).sum()) if raw.dtype.kind == "f" else 0, raw.min(), raw.max())
    dup = df["feed_id"].n_unique() != df.height
    print(f"\n{split} ({'full' if full else str(len(files))+' parts'}; {df.height} feeds, {M.sum()} cand rows, {H.sum()} hist rows): all equal = {all(res.values())}; duplicate feed_id {dup}; scalar nulls {sum(df[c].null_count() for c in SCAL)}")
    print("  ", {k: v for k, v in res.items()})
    for c, v in info.items(): print("   %-34s raw %-14s flat %-8s nulls-in-list %d NaN %d range [%s, %s]" % ((c,) + v))
    return set(res)
cov = set()
for split in ("val", "test"):
    cov |= check(split, sorted(glob.glob(R + f"competition/{split}/impressions/*.parquet")), True)
tf = sorted(glob.glob(R + "competition/train/impressions/*.parquet")); rng = np.random.default_rng(7)
cov |= check("train", [tf[i] for i in sorted(rng.choice(len(tf), 6, replace=False))], False)
raw_names = {n for n, _ in allcols["train"]}
print("\nraw columns", len(raw_names), "verified against flat", len(raw_names & cov), "NOT verified:", sorted(raw_names - cov))
# sample_submission
ss = pl.read_parquet(R + "competition/sample_submission.parquet"); print("sample_submission", ss.schema, ss.height, "score nonzero", int((ss["score"] != 0).sum()),
      "identical to data/raw copy:", ss.equals(pl.read_parquet(R + "raw/sample_submission.parquet")))
for sp in ("train", "val", "test"): print(sp, "emb meta", open(R + f"competition/{sp}/item_embeddings/data.meta").read().strip(), "memmap bytes", os.path.getsize(R + f"competition/{sp}/item_embeddings/data.memmap"))
