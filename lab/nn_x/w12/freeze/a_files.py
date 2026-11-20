"""Audit check (a): the candidate submission files. CPU, read-only. Output: out/a_files.json + printed table.
For every candidate row:
  a1 file exists (size, mtime UTC)                  a2 sha256 == sha256 on record
  a3 column names and dtypes == those of a SCORED file (cands.SCORED_REF) and of data/raw/sample_submission.parquet
  a4 row count == scored file == sample submission  a5 no null, no NaN, no infinite value in any column
  a6 key set (feed_id, item_id): no duplicate key, equal to the key set of the sample submission and of the scored file
  a7 every test feed present exactly once per candidate row: 28,595 feeds, rows per feed == M of data/flat/test/feeds.parquet,
     item ids per feed == those of data/flat/test (same multiset, compared after sorting)
  a8 scores inside a feed are the tie-free ranks 0 .. M-1 (the form src/make_sub.py writes)
  a9 row order identical to the scored file (information; the platform's need for a row order is not on record)
"""
import sys, json, hashlib, datetime
sys.path.insert(0, '<repo>/nn_x/w12/freeze')
import numpy as np, polars as pl
from cands import ROOT, OUT, CANDS, SCORED_REF, SAMPLE

OUT.mkdir(parents=True, exist_ok=True)


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 22), b''):
            h.update(blk)
    return h.hexdigest()


def keyed(d):
    """(feed_id, item_id) as one int64 key."""
    return d['feed_id'].to_numpy().astype(np.int64) * (1 << 32) + d['item_id'].to_numpy().astype(np.int64)


ref = pl.read_parquet(ROOT / SCORED_REF); ss = pl.read_parquet(ROOT / SAMPLE)
feeds = pl.read_parquet(ROOT / 'data/flat/test/feeds.parquet')
M = feeds['M'].to_numpy().astype(np.int64); off = feeds['cand_off'].to_numpy(); fid_feed = feeds['feed_id'].to_numpy()
fidx = np.load(ROOT / 'data/flat/test/cand_fidx.npy'); item_flat = np.load(ROOT / 'data/flat/test/cand_item_id.npy')
flat_key = fid_feed[fidx].astype(np.int64) * (1 << 32) + item_flat.astype(np.int64)
ss_key = np.sort(keyed(ss)); ref_key_raw = keyed(ref); ref_key = np.sort(ref_key_raw); flat_key_sorted = np.sort(flat_key)
assert len(np.unique(fid_feed)) == len(fid_feed) == 28595
print(f'scored reference {SCORED_REF}: rows {len(ref)}, schema {dict(ref.schema)}; sample submission rows {len(ss)}; test feeds {len(M)}, candidate rows {int(M.sum())}')
print(f'key set of the sample submission == data/flat/test: {bool((ss_key == flat_key_sorted).all())}; == scored reference: {bool((ss_key == ref_key).all())}')
res = {'stamp_utc': datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'), 'scored_ref': SCORED_REF, 'rows_ref': len(ref), 'rows_sample': len(ss),
       'test_feeds': int(len(M)), 'cand_rows': int(M.sum()), 'files': {}}
for c in CANDS:
    p = ROOT / c['file']; r = {'file': c['file']}
    r['a1_exists'] = p.exists()
    if not r['a1_exists']:
        res['files'][c['id']] = r; print(c['id'], 'FILE MISSING'); continue
    st = p.stat(); r['size'] = st.st_size; r['mtime_utc'] = datetime.datetime.fromtimestamp(st.st_mtime, datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    r['sha256'] = sha256(p); r['sha_rec'] = c['sha_rec']; r['a2_sha_equals_record'] = r['sha256'] == c['sha_rec']
    d = pl.read_parquet(p)
    r['columns'] = {k: str(v) for k, v in d.schema.items()}
    r['a3_schema_equals_scored'] = dict(d.schema) == dict(ref.schema) and d.columns == ref.columns
    r['a3_schema_equals_sample'] = dict(d.schema) == dict(ss.schema) and d.columns == ss.columns
    r['rows'] = len(d); r['a4_rows_equal'] = len(d) == len(ref) == len(ss) == int(M.sum())
    sc = d['score'].to_numpy()
    r['nulls'] = int(sum(d[col].null_count() for col in d.columns)); r['nan'] = int(np.isnan(sc).sum()); r['inf'] = int(np.isinf(sc).sum())
    r['a5_no_null_nan_inf'] = r['nulls'] == 0 and r['nan'] == 0 and r['inf'] == 0
    k_raw = keyed(d); k = np.sort(k_raw)
    r['dup_keys'] = int(len(k) - len(np.unique(k)))
    r['a6_keys_equal_sample'] = len(k) == len(ss_key) and bool((k == ss_key).all())
    r['a6_keys_equal_scored'] = len(k) == len(ref_key) and bool((k == ref_key).all())
    r['a6_no_dup_key'] = r['dup_keys'] == 0
    # a7: per-feed presence
    g = d.group_by('feed_id').agg(pl.len().alias('n'))
    r['feeds_in_file'] = len(g)
    mm = feeds.select('feed_id', 'M').join(g, on='feed_id', how='left')
    r['feeds_missing'] = int(mm['n'].null_count())
    r['feeds_with_wrong_row_count'] = int((mm['n'].fill_null(-1).to_numpy() != mm['M'].to_numpy()).sum())
    r['a7_every_feed_once_per_candidate'] = (r['feeds_in_file'] == len(M) and r['feeds_missing'] == 0 and r['feeds_with_wrong_row_count'] == 0
                                             and r['dup_keys'] == 0 and len(k) == len(flat_key_sorted) and bool((k == flat_key_sorted).all()))
    # a8: ranks 0..M-1 inside each feed (align to the flat order through the key)
    o_file = np.argsort(k_raw, kind='stable'); o_flat = np.argsort(flat_key, kind='stable')
    s_flat = np.empty(len(sc), np.float64); s_flat[o_flat] = sc[o_file].astype(np.float64)        # score in data/flat/test candidate order
    ordr =np.lexsort((s_flat, fidx)); within = np.arange(len(sc)) - np.repeat(off, M)
    r['a8_scores_are_ranks_0_to_M_minus_1'] = bool((s_flat[ordr] == within).all())
    r['within_feed_ties'] = int((np.diff(s_flat[ordr]) == 0)[np.diff(fidx[ordr]) == 0].sum())
    r['a9_row_order_equals_scored'] = bool((k_raw == ref_key_raw).all())
    np.save(OUT / f'_flatscore_{c["tag"]}.npy', s_flat.astype(np.float32))                         # the file's score in flat candidate order, for check (b)
    res['files'][c['id']] = r
    print(f"{c['id']:18s} rows {r['rows']} feeds {r['feeds_in_file']} sha256 {r['sha256']} "
          f"| a2 {r['a2_sha_equals_record']} a3 {r['a3_schema_equals_scored'] and r['a3_schema_equals_sample']} a4 {r['a4_rows_equal']} a5 {r['a5_no_null_nan_inf']} "
          f"a6 {r['a6_keys_equal_sample'] and r['a6_keys_equal_scored'] and r['a6_no_dup_key']} a7 {r['a7_every_feed_once_per_candidate']} a8 {r['a8_scores_are_ranks_0_to_M_minus_1']} a9 {r['a9_row_order_equals_scored']}")
json.dump(res, open(OUT / 'a_files.json', 'w'), indent=1)
print('written', OUT / 'a_files.json')
