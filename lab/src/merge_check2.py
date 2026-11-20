"""merge_check2 (written Sun Nov 15 19:30:26 UTC 2026): check of a merge partner's score files against one of our candidates, and the fixed blend of the two.
Successor of src/merge_check.py (v1; prints the pooled Spearman of two rank files, not the correlation of the law) and of
nn_x/w12/next/merge/merge_check_v2.py (v2; right correlation, reviewed in reports/NEXT_1115/merge_verify.md). Both are left untouched.
Protocol: reports/MERGE_PROTOCOL.md. Tests: nn_x/w12/next/merge/tests_v3/. CPU only, self-contained (imports src/common.py, src/metrics.py; src/make_sub.py only for --blend).
The tool reads files and writes local files only. Sending files to a partner, receiving them and submitting are done by the owner, never by an agent.
Reviewed Sun Nov 15 20:02:27 UTC 2026: three texts corrected (the exit-code lines, the source of the refit, the source of the printed reference points); no change of behaviour.

USAGE (the partner files are UNTRUSTED data: keep them in their own directory, run with -I, nothing in them is executed)
  cd <repo> && export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 NUMBA_NUM_THREADS=12 POLARS_MAX_THREADS=12
  .venv/bin/python -I src/merge_check2.py --ours subs/cand_<TAG>_r<k>.parquet --ours_public <ours> \
      --partner <dir>/partner_test.parquet --partner_public <theirs> --partner_val <dir>/partner_val.parquet \
      [--ours_scores auto|<flat test .npy>] [--ours_val auto|<flat val .npy>] [--flip_partner] [--as_ranks] [--allow_low_agreement] \
      [--blend --out <blend.parquet> [--w eq|law] [--out_flat <prefix>] [--force]] [--json <result.json>]
  IN-HOUSE mode (our own models only; flat .npy aligned to data/flat/<split>; no public score, the VAL gap is used as in the fit of the law):
  .venv/bin/python -I src/merge_check2.py --ours subs/cand_<TAG>_r<k>.parquet --partner <arm>_test.npy --partner_val <arm>_val.npy
  OUR side of the exchange (our model scores in the same file format, verified by the same reader; the owner sends them, no agent does):
  .venv/bin/python -I src/merge_check2.py --ours subs/cand_<TAG>_r<k>.parquet --export_ours <dir>/<name>      -> <dir>/<name>_test.parquet, <dir>/<name>_val.parquet
  Exit code: 0 clean read (see DECISION) | 2 reader FAIL (no decision, no blend); the argument parser also exits with 2 on a command-line error | 3 whole file inverted / unrelated / unusable
             | 4 rank-only file, decision withheld | 5 --blend requested and refused | 1 error (a refused output path included).

FILE FORMAT expected from the partner: one parquet per split with the columns feed_id (integer), item_id (integer), score (float; larger = better; the
  model's raw output / logit, NOT a rank, NOT a probability squashed per feed), every (feed_id, item_id) of the split exactly once, finite values.

CHECKS ON RECEIPT (the same reader for the test file and for the val file; a FAIL means "no decision, no blend, ask for a corrected file")
  R1 readable parquet with the three columns; keys convertible to integers without loss                          FAIL if not
  R2 coverage: every row of the split present, no (feed_id, item_id) twice (exact duplicates included)             FAIL if not; unknown extra keys: WARN
  R3 values: no NaN, no inf, no |score| > 1e30                                                                    FAIL if any
  R4 constant feeds (one value for all candidates)                                                                FAIL above 0.1% of feeds, else WARN
  R5 tied rows inside a feed                                                                                      FAIL from 0.1% of rows, else WARN
  R6 rank-only shape (the sorted scores of a feed are a function of the rank alone: evenly spaced, or the same shape in every feed of that size)
       on >= 99% of feeds: RANK-ONLY FILE, flagged, decision withheld unless --as_ranks (owner's decision)       FLAG
       on more than 1% but fewer than 99% of feeds: mixed file                                                    FAIL
  D1 direction against our order: mean per-feed Spearman <= -0.2 with >= 80% of feeds negative = INVERTED; mean < 0.3 = UNRELATED        REFUSED (exit 3)
  D2 per-feed distribution, feeds with M >= 20: share of feeds with per-feed Spearman < 0.3                       FAIL above 0.75%, WARN above 0.2%
  D3 per-feed distribution, feeds with M < 20 (16% of test feeds): share of feeds with per-feed Spearman < 0      FAIL above 3%, WARN above 1.5%
     (strictly negative on mid-ranks; genuine models on disk against W8: neural pools 0.000%, CatBoost 0.39%, LLM 0.49%, XGBoost 1.03%)
  D4 score scale: |law-rho of the scores as sent - law-rho of their ranks restored on the shape of our scores|    FAIL above 0.01 (probabilities, exp, softmax)
  D5 test law-rho against val law-rho                                                                             WARN when they differ by more than 0.015
  V1 val optimism: (partner val - partner public) against (our val - our public)                                  WARN when the partner's is more than 0.003 larger
  D2-D4 exist because the law rewards LOW correlation: an inverted, shuffled or re-scaled part of a file reads as diversity. Lines D2-D5, R4-R6 are tool
  settings calibrated on the 10 on-disk partners of W8 (tests_v3/out/t13_calibration.txt; reports/NEXT_1115/merge.md 1.5, merge_verify.md 2.1-2.2), not rules on record.

LAW AND GATES (rules on record: reports/RECHECK_1114/SYNTHESIS.md bet 1, long_bets.md bet 1; thresholds fixed, none is fitted here)
  law-rho = FEED-MEAN Pearson of per-feed z of the MODEL SCORES of the two files on test (the definition the law on record was fitted with on val).
  LAW ON RECORD   gain of the equal z-mix over the better file = 0.0186 (1 - rho) - 0.378 |gap|;  REFIT 0.0103 (1 - rho_pooled) - 0.304 |gap| (information).
  GATE 1  partner public within 0.002 of ours.            GATE 2  law on record - 0.0005 >= 0, read with the LARGER of the two rho (as sent, restored): scale-free.
  GATE 3  the mix that would be written (--w, default eq) gains >= 2 paired SE on val over the better file.  No weight is scanned or fitted on val.

BLEND (--blend --out <path>): order of (1 - w) z(ours) + w z(partner) per feed, w = 0.5 (--w eq, the protocol's fixed form) or the gap-adjusted weight of the
  law on record (--w law; computed from rho and the public gap, not from val). z(ours) = per-feed z of our model scores (preds/ens_<TAG>_{ft,to}).
  TIE-BREAK, the same rule in the val lines and in --blend:
    1. the mix is computed in float64 and rounded to 1e-8 z units; rows of a feed with the same rounded value are tied;
    2. tied rows are ordered as in the file with the better public score (IN-HOUSE mode: better val score; equal scores: ours);
    3. rows still tied are ordered by item_id (larger item_id ranks higher). item_id is a label-free key (as a ranking key it reads random on val,
       reports/CURRENT_STATUS.md log 2026-11-14 19:03Z) and makes the order independent of row order and of any random generator.
  WRITES: nothing is written without an explicit path. --out (blend parquet, through src/make_sub.py), --out_flat <prefix> (<prefix>_test.npy and
  <prefix>_val.npy = the blend order as flat float32 ranks, the inputs src/presub_gate.py needs), --json, --export_ours <prefix>. An existing file is never
  overwritten; --json and --export_ours refuse any path under subs/ or preds/.
  A blend is refused on any reader FAIL, on a direction problem, on a withheld decision (never overridable), and unless gates 1-3 passed and D5 did not WARN, or --force is given."""
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''                       # CPU only (hard rule)
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS', 'POLARS_MAX_THREADS'):
    try: os.environ[_k] = str(min(12, int(os.environ.get(_k, '12'))))
    except ValueError: os.environ[_k] = '12'
sys.path.insert(0, '<repo>/src')
import argparse, hashlib, json, re
from pathlib import Path
import numpy as np, polars as pl
from common import feed_meta, FLAT
from metrics import ndcg_per_feed

VERSION = 'merge_check2 2026-11-15'
ROOT = Path('<repo>'); P = ROOT / 'preds'
BOUND = 1790497920                      # src/pmix.py, src/calib.py: public = test feeds with timestamp < BOUND (8,562 feeds)
LAW_REC = (0.0186, 0.378)               # law on record (solvers/out/ensemble/exp7.json "val_fit_gap<0.01": 0.018577, 0.37766), feed-mean Pearson of z
LAW_REFIT = (0.0103, 0.304)             # refit at the W8 level on 10 partners (RECHECK_1114/SYNTHESIS.md bet 1; nn_x/w12/recheck/long_bets/out/p4_partner.txt, the study of long_bets.md F13), pooled Pearson of z
MARGIN, GATE1, GATE3_Z = 0.0005, 0.002, 2.0   # rules on record (SYNTHESIS bet 1 gates 1-3); not moved here
HUGE = 1e30                             # R3: a finite |score| beyond this is treated as non-finite
CONST_FAIL = 0.001                      # R4
TIED_FAIL = 0.001                       # R5
RANK_FILE, RANK_MIXED = 0.99, 0.01      # R6
LOW_M, LOW_R, LOW_FAIL, LOW_WARN = 20, 0.3, 0.0075, 0.002     # D2 (v2 r2 settings, merge.md 1.5)
NEG_FAIL, NEG_WARN = 0.03, 0.015        # D3 (merge_verify.md 2.1: FAIL line 3%)
SCALE_FAIL = 0.01                       # D4 (merge_verify.md 2.2)
VALTEST_WARN = 0.015                    # D5 (merge_verify.md 2.1)
VAL_OPT = 0.003                         # V1: band of src/calib.py (v1 / v2 setting)
MIX_DIGITS = 8                          # tie-break step 1
DAYNAME = {20718: 'val Tue 09-22', 20720: 'val Thu 09-24', 20722: 'Sat 09-26', 20723: 'Sun 09-27', 20724: 'Mon 09-28', 20725: 'Tue 09-29'}
se = lambda d: float(np.std(d) / np.sqrt(max(len(d), 1)))
pm = lambda d, k=5: f'{np.mean(d):+.{k}f} ± {se(d):.{k}f}'
# paired SE of the val gap of on-disk partners of W8 (nn_x/w12/next/merge/out/c1_calib.txt T2): (1 - rho feed-mean, SE over 14,423 val feeds)
_GAPSE = np.array([(0.0011, 0.00034), (0.0065, 0.00055), (0.0089, 0.00063), (0.0121, 0.00067), (0.0172, 0.00073), (0.160, 0.00144), (0.227, 0.00167)])
_UNCERTAINTY = ('how far the law can be trusted (reports/NEXT_1115/merge.md 1.3, confirmed in merge_verify.md rows 13-17; val, feed bootstrap 5-95%, line + leave-one-out residual): '
                'from the 10 partners of W8 the diversity coefficient is NOT identified (a in [-0.013, +0.034]; on record 0.0186, refit 0.0089-0.0103). Equal-mix gain at rho 0.90: '
                '[-0.0014, +0.0036] at gap 0, [-0.0017, +0.0027] at gap 0.002. No on-disk pair has rho < 0.95 AND |gap| < 0.005: for a strong and different partner the law is an '
                'interpolation between two clusters, never observed. Every fit is on val gaps; the transfer to public and private was never measured.')
_RANK_COST = ('cost of a rank file on record (reports/NEXT_1115/merge.md 1.4, merge_verify.md 2.3; equal mix with a partner RANK file minus the z-mix of the model scores, val, '
              'ties to the better file): mean of 10 on-disk partners -0.00006 (7 neural pools +0.00026, 3 non-neural models -0.00080; range -0.00148 .. +0.00060); '
              'when the rank file carries the larger weight (w 0.6) -0.00131 on average, down to -0.00442. The first pass of that study read -0.0011 (ties to the weaker file); '
              'the log of 2026-11-14 23:18Z quotes 0.0005-0.003.')


# ---------------------------------------------------------------------------------------------------------------------------- split and per-feed algebra
class Split:
    """Feed metadata of 'test' or 'val' and the (feed_id, item_id) -> flat row key (data/flat/<split>, candidates of a feed contiguous)."""
    def __init__(self, name):
        f, M, off = feed_meta(name); self.name = name
        self.M = np.asarray(M, np.int64); self.off = np.asarray(off, np.int64); self.n = len(self.M); self.N = int(self.M.sum())
        self.fid = np.repeat(np.arange(self.n), self.M)
        fidx = np.load(FLAT / name / 'cand_fidx.npy'); assert len(fidx) == self.N and (fidx == self.fid).all(), 'flat order is not feed-contiguous'
        self.feed_id = f['feed_id'].to_numpy().astype(np.int64); self.item_id = np.load(FLAT / name / 'cand_item_id.npy').astype(np.int64)
        self.key = pl.DataFrame({'feed_id': self.feed_id[self.fid], 'item_id': self.item_id, 'row': np.arange(self.N, dtype=np.int64)})
        assert self.key.select('feed_id', 'item_id').is_unique().all(), 'keys of the split are not unique'
        ts = f['timestamp'].to_numpy().astype(np.int64); self.day = (ts + 3 * 3600) // 86400      # Moscow-time day, as src/presub_gate.py
        self.public = ts < BOUND if name == 'test' else np.zeros(self.n, bool)

    def groups(self):
        g = {'ALL': np.ones(self.n, bool)}
        for d in np.unique(self.day):
            m = self.day == d
            if m.sum() < 100: continue
            nm = DAYNAME.get(int(d), str(d))
            if self.name == 'test' and m[self.public].any() and m[~self.public].any():
                g[f'{nm} am (public)'] = m & self.public; g[f'{nm} pm (private)'] = m & ~self.public
            else: g[nm + ((' (public)' if m[self.public].any() else ' (private)') if self.name == 'test' else '')] = m
        if self.name == 'test': g['PUBLIC'] = self.public.copy(); g['PRIVATE'] = ~self.public
        g['M < 20'] = self.M < 20; g['M 20-49'] = (self.M >= 20) & (self.M < 50); g['M >= 50'] = self.M >= 50
        return g


_SP = {}
def SPLITS(name):
    if name not in _SP: _SP[name] = Split(name)
    return _SP[name]


def midrank(S, s):
    """Within-feed ascending rank 0..M-1, ties get the mean rank. s must be finite."""
    o = np.lexsort((s, S.fid)); ss = s[o]; ff = S.fid[o]; pos = np.arange(S.N, dtype=np.float64) - S.off[ff]
    new = np.ones(S.N, bool); new[1:] = (ff[1:] != ff[:-1]) | (ss[1:] != ss[:-1])
    run = np.cumsum(new) - 1; r = np.empty(S.N, np.float64); r[o] = (np.bincount(run, pos) / np.bincount(run))[run]
    return r


def pf_corr(S, a, b):
    """Per-feed Pearson correlation of two flat arrays (NaN where a feed has zero variance in either)."""
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    da = a - (np.bincount(S.fid, a, S.n) / S.M)[S.fid]; db = b - (np.bincount(S.fid, b, S.n) / S.M)[S.fid]
    num = np.bincount(S.fid, da * db, S.n); den = np.sqrt(np.bincount(S.fid, da * da, S.n) * np.bincount(S.fid, db * db, S.n)); ok = den > 1e-12
    return np.where(ok, num / np.where(ok, den, 1.0), np.nan)


def agg(S, r, mask=None):
    """(feed-mean, SE over feeds, pooled = M-weighted mean, feeds used) of per-feed correlations."""
    m = np.isfinite(r) if mask is None else (np.isfinite(r) & mask)
    if not m.any(): return float('nan'), float('nan'), float('nan'), 0
    x = r[m]; w = S.M[m]; return float(x.mean()), se(x), float((x * w).sum() / w.sum()), int(m.sum())


def zfeed(S, x):
    """Per-feed z of OUR scores, the convention of our builders (population sd + 1e-9; stack_val.feed_rank_z)."""
    x = np.asarray(x, np.float64); d = x - (np.bincount(S.fid, x, S.n) / S.M)[S.fid]
    return d / (np.sqrt(np.bincount(S.fid, d * d, S.n) / S.M) + 1e-9)[S.fid]


def zrel(S, x, feed_ok):
    """Per-feed z of a PARTNER file: the epsilon is relative to the file's own typical per-feed sd (median over usable feeds), so the result does not
    depend on the unit of the file (x 1e6 or x 1e-9 give the same z) and a near-constant feed is damped, not amplified. A constant feed gives 0."""
    x = np.asarray(x, np.float64); d = x - (np.bincount(S.fid, x, S.n) / S.M)[S.fid]; sd = np.sqrt(np.bincount(S.fid, d * d, S.n) / S.M)
    pos = feed_ok & (sd > 0); s0 = float(np.median(sd[pos])) if pos.any() else 1.0
    return d / (sd + 1e-9 * s0)[S.fid]


def sorted_vals(S, z): z = np.asarray(z, np.float64); return z[np.lexsort((z, S.fid))]


def shape_restore(S, rank, zs):
    """Map within-feed (mid-)ranks onto the sorted reference z of the same feed (linear interpolation for tied mid-ranks)."""
    lo = np.floor(rank).astype(np.int64); hi = np.minimum(lo + 1, S.M[S.fid] - 1); fr = rank - lo; base = S.off[S.fid]
    return zs[base + lo] * (1 - fr) + zs[base + hi] * fr


def top1(S, s): return np.lexsort((-np.asarray(s, np.float64), S.fid))[S.off]


def mix_rank(S, za, zb, w, prefer_b):
    """(1 - w) za + w zb as a TIE-FREE within-feed rank (0 = worst) and the share of rows that were tied. The tie-break of the docstring:
    rounded mix, then the preferred file (zb if prefer_b else za), then item_id. Deterministic: no random key, no dependence on row order."""
    mix = np.round((1 - w) * np.asarray(za, np.float64) + w * np.asarray(zb, np.float64), MIX_DIGITS) + 0.0
    pref = np.asarray(zb if prefer_b else za, np.float64)
    o = np.lexsort((S.item_id, pref, mix, S.fid)); r = np.empty(S.N, np.float64); r[o] = np.arange(S.N) - S.off[S.fid[o]]
    ms = mix[o]; ff = S.fid[o]; eq = np.zeros(S.N, bool); eq[1:] = (ff[1:] == ff[:-1]) & (ms[1:] == ms[:-1]); t = eq.copy(); t[:-1] |= eq[1:]
    return r, float(t.mean())


# ---------------------------------------------------------------------------------------------------------------------------- reader (R1-R6)
FAIL_NAME = {'R1': 'R1 unreadable file or wrong format', 'R2': 'R2 coverage', 'R3': 'R3 non-finite values', 'R4': 'R4 constant feeds', 'R5': 'R5 tied rows', 'R6': 'R6 mixed rank-shaped file'}


class Sub:
    def __init__(self): self.fails = []; self.warns = []; self.stats = {}; self.s = None; self.feed_cov = None; self.feed_ok = None; self.usable = False; self.is_rank = False


def read_scores(path, S, who, allow_npy=False):
    """Reader of an UNTRUSTED (feed_id, item_id, score) parquet for the split S. Returns Sub: s = flat float64 scores (NaN where missing or non-finite),
    feed_cov = feeds that are complete and finite, feed_ok = feed_cov and not constant, fails / warns, stats, is_rank (R6)."""
    R = Sub()
    def fail(msg): R.fails.append(FAIL_NAME[msg[:2]]); print(f'[FAIL] {who}: {msg}', flush=True)      # every FAIL message starts with its check (R1..R6)
    def warn(msg): R.warns.append(msg); print(f'[WARN] {who}: {msg}', flush=True)
    path = str(path)
    if path.endswith('.npy'):
        if not allow_npy: fail('R1 a flat .npy carries no keys, so coverage cannot be checked: accepted in IN-HOUSE mode only (our own models)'); return R
        try: arr = np.load(path, allow_pickle=False)
        except Exception as e: fail(f'R1 cannot read {path} as .npy ({type(e).__name__}: {str(e)[:120]})'); return R
        if arr.ndim != 1 or len(arr) != S.N: fail(f'R1 {path}: shape {arr.shape}, expected ({S.N},) = the flat {S.name} rows'); return R
        df = pl.DataFrame({'feed_id': S.key['feed_id'], 'item_id': S.key['item_id'], 'score': np.asarray(arr, np.float64)}); dt0 = str(arr.dtype)
    else:
        try:
            schema = pl.read_parquet_schema(path); low = {c.lower().strip(): c for c in schema}
            miss = [c for c in ('feed_id', 'item_id', 'score') if c not in low]
            if miss: fail(f'R1 missing column(s) {miss}; the file has {list(schema)[:12]}'); return R
            nrow = int(pl.scan_parquet(path).select(pl.len()).collect().item())
            if nrow > 4 * S.N: fail(f'R1 the file has {nrow} rows, the {S.name} set has {S.N}: not read'); return R
            extra = [c for c in schema if c.lower().strip() not in ('feed_id', 'item_id', 'score')]
            if extra: warn(f'extra columns ignored: {extra[:6]}')
            df = pl.read_parquet(path, columns=[low[c] for c in ('feed_id', 'item_id', 'score')]).select([pl.col(low[c]).alias(c) for c in ('feed_id', 'item_id', 'score')])
        except Exception as e:
            fail(f'R1 cannot read {path} as parquet ({type(e).__name__}: {str(e)[:120]})'); return R
        dt0 = str(df.schema['score'])
    n0 = len(df); R.stats['rows_in_file'] = n0
    if n0 == 0: fail('R1 the file has 0 rows'); return R
    for c in ('feed_id', 'item_id'):                          # keys: integers; anything else goes through Float64 and only exact integers are kept
        dt = df.schema[c]
        if not dt.is_integer():
            x = df[c].cast(pl.Utf8, strict=False).str.strip_chars().cast(pl.Float64, strict=False) if dt in (pl.Utf8, pl.Categorical) else df[c].cast(pl.Float64, strict=False)
            bad = int((x.is_null() | x.is_nan() | x.is_infinite() | (x != x.round())).sum())
            warn(f'{c} has dtype {dt}, expected an integer type; converted' + (f'; {bad} rows with a non-integer or unparsable key are dropped' if bad else ' without loss'))
            df = df.with_columns(pl.when(x.is_finite() & (x == x.round())).then(x).otherwise(None).cast(pl.Int64, strict=False).alias(c))
        else: df = df.with_columns(pl.col(c).cast(pl.Int64, strict=False))
    dt = df.schema['score']
    if not dt.is_numeric():
        x = df['score'].cast(pl.Utf8, strict=False).str.strip_chars().cast(pl.Float64, strict=False) if dt in (pl.Utf8, pl.Categorical) else df['score'].cast(pl.Float64, strict=False)
        warn(f'score has dtype {dt}, expected a float; converted, {int(x.is_null().sum())} values could not be parsed (treated as NaN)'); df = df.with_columns(x.alias('score'))
    else:
        if dt.is_integer(): warn(f'score has integer dtype {dt} (converted to float; integer scores tie easily)')
        df = df.with_columns(pl.col('score').cast(pl.Float64, strict=False))
    nk = int((df['feed_id'].is_null() | df['item_id'].is_null()).sum())
    if nk: fail(f'R2 {nk} rows with a null or invalid key (dropped for the diagnostics)'); df = df.filter(df['feed_id'].is_not_null() & df['item_id'].is_not_null())
    dup = df.select(pl.struct('feed_id', 'item_id').is_duplicated().alias('d'))['d']; nd = int(dup.sum()); R.stats['duplicated_rows'] = nd
    if nd:                                                    # R2: exactly once
        g = df.filter(dup).group_by(['feed_id', 'item_id']).agg(pl.col('score').n_unique().alias('nu')); conf = int((g['nu'] > 1).sum())
        fail(f'R2 {nd} rows share a (feed_id, item_id) key: {len(g)} keys are present more than once, {conf} of them with CONFLICTING scores (first occurrence kept for the diagnostics)')
        df = df.unique(subset=['feed_id', 'item_id'], keep='first', maintain_order=True)
    j = S.key.join(df.with_columns(pl.lit(True).alias('present')), on=['feed_id', 'item_id'], how='left').sort('row')
    present = j['present'].fill_null(False).to_numpy(); s = j['score'].to_numpy().astype(np.float64); s[~present] = np.nan
    n_extra = len(df) - int(present.sum()); n_missing = int((~present).sum()); R.stats.update(rows_matched=int(present.sum()), rows_missing=n_missing, rows_unknown_key=n_extra)
    if present.sum() == 0: fail(f'R2 no (feed_id, item_id) of the file belongs to the {S.name} set ({S.N} rows expected): wrong split or wrong id columns'); return R
    if n_extra: warn(f'{n_extra} rows have a key that is not in the {S.name} set (ignored)')
    feed_missing = np.bincount(S.fid, ~present, S.n) > 0
    if n_missing:
        whole = int((np.bincount(S.fid, ~present, S.n) == S.M).sum())
        fail(f'R2 {n_missing} of {S.N} {S.name} rows are missing: {int(feed_missing.sum())} feeds are incomplete ({whole} of them absent entirely); diagnostics use the complete feeds only')
    nan = present & np.isnan(s); pinf = present & np.isposinf(s); ninf = present & np.isneginf(s)
    with np.errstate(invalid='ignore'): huge = present & np.isfinite(s) & (np.abs(s) > HUGE)
    bad = nan | pinf | ninf | huge; feed_bad = np.bincount(S.fid, bad, S.n) > 0
    R.stats.update(nan=int(nan.sum()), pos_inf=int(pinf.sum()), neg_inf=int(ninf.sum()), huge=int(huge.sum()))
    if bad.any():
        fail(f'R3 non-finite scores: {int(nan.sum())} NaN, {int(pinf.sum())} +inf, {int(ninf.sum())} -inf, {int(huge.sum())} with |score| > {HUGE:g}, in {int(feed_bad.sum())} feeds (left out of the diagnostics)')
        s[bad] = np.nan
    cov = ~(feed_missing | feed_bad); s0 = np.where(np.isfinite(s), s, 0.0)
    o = np.lexsort((s0, S.fid)); ss = s0[o]; ff = S.fid[o]
    eq = np.zeros(S.N, bool); eq[1:] = (ff[1:] == ff[:-1]) & (ss[1:] == ss[:-1]); tied = eq.copy(); tied[:-1] |= eq[1:]
    const = cov & (np.minimum.reduceat(s0, S.off) == np.maximum.reduceat(s0, S.off)) & (S.M > 1); cshare = float(const.sum() / S.n)
    tied_rows = np.zeros(S.N, bool); tied_rows[o] = tied; tied_rows &= np.repeat(cov & ~const, S.M)          # the rows of a constant feed are counted by R4, not by R5
    feed_tie = (np.bincount(S.fid, tied_rows, S.n) > 0) & cov; tshare = float(tied_rows.sum() / S.N)
    R.stats.update(tied_rows=int(tied_rows.sum()), feeds_with_ties=int(feed_tie.sum()), constant_feeds=int(const.sum()))
    if const.any():
        (fail if cshare > CONST_FAIL else warn)(f'R4 {int(const.sum())} feeds ({cshare:.3%}) have ONE constant score for all their candidates (limit {CONST_FAIL:.1%}): the metric scores such a feed '
                                                 f'as the worst order; they are left out of the correlation and a blend would keep our order there')
    if tied_rows.any():
        (fail if tshare >= TIED_FAIL else warn)(f'R5 {int(tied_rows.sum())} rows ({tshare:.3%}) are tied inside their feed, in {int(feed_tie.sum())} feeds (limit {TIED_FAIL:.1%}): '
                                                f'ties are scored pessimistically by the metric; mid-ranks are used for the correlation')
    R.feed_cov = cov; R.feed_ok = cov & ~const; R.s = s; R.usable = bool(R.feed_ok.any())
    if not R.usable: fail('R1 no usable feed (complete, finite, not constant)'); return R
    # R6 rank-only shape, per feed: (a) score is an exact affine function of its own within-feed rank; (b) the sorted z of the feed equals the mean sorted z of all feeds of its size
    lin = pf_corr(S, s0, midrank(S, s0)); okr = np.repeat(R.feed_ok, S.M); z = zrel(S, s0, R.feed_ok); zs = sorted_vals(S, z)
    key = S.M[S.fid] * (int(S.M.max()) + 1) + (np.arange(S.N) - S.off[S.fid]); cnt = np.bincount(key[okr]); mean = np.bincount(key[okr], zs[okr]) / np.maximum(cnt, 1)
    dev = np.maximum.reduceat(np.abs(zs - mean[np.minimum(key, len(mean) - 1)]), S.off); nM = np.bincount(S.M[R.feed_ok], minlength=int(S.M.max()) + 1)[S.M]
    rk = R.feed_ok & ((lin > 0.999999) | ((nM >= 3) & (dev < 1e-4))); rshare = float(rk.sum() / R.feed_ok.sum())
    R.is_rank = rshare >= RANK_FILE; R.stats.update(rank_shaped_feeds=int(rk.sum()), rank_shaped_share=rshare, feeds_usable=int(R.feed_ok.sum()), feeds_total=S.n, score_dtype=dt0,
                                                    holds='within-feed ranks or a function of them (no model scores)' if R.is_rank else 'model scores')
    if RANK_MIXED < rshare < RANK_FILE: fail(f'R6 {rshare:.2%} of the feeds are rank-shaped (the score is a function of the within-feed rank alone) and the rest is not: a mixed file')
    elif 0 < rshare <= RANK_MIXED: warn(f'R6 {int(rk.sum())} feeds ({rshare:.3%}) are rank-shaped')
    print(f'[{"OK" if not R.fails else "INFO"}] {who}: {path}: {n0} rows in file, {int(present.sum())} matched; feeds complete and finite {int(cov.sum())} / {S.n}, usable {int(R.feed_ok.sum())}; '
          f'tied rows {int(tied_rows.sum())}; constant feeds {int(const.sum())}; rank-shaped feeds {rshare:.2%}; the score column holds {R.stats["holds"]} ({dt0})', flush=True)
    return R


# ---------------------------------------------------------------------------------------------------------------------------- per-feed agreement (D1-D3)
def agreement(S, rf, ok, mask=None):
    """Per-feed Spearman rf of the partner order against ours on the usable feeds: mean, share negative, D2 share (M >= 20, rf < 0.3), D3 share (M < 20, rf < 0)."""
    m = ok & np.isfinite(rf)
    if mask is not None: m = m & mask
    big = m & (S.M >= LOW_M); small = m & (S.M < LOW_M); x = rf[m]
    return dict(n=int(m.sum()), mean=float(x.mean()) if len(x) else float('nan'), neg=float((x < 0).mean()) if len(x) else float('nan'),
                n_big=int(big.sum()), low_big=float((rf[big] < LOW_R).mean()) if big.any() else float('nan'),
                n_small=int(small.sum()), neg_small=float((rf[small] < 0).mean()) if small.any() else float('nan'), nonpos_small=float((rf[small] <= 0).mean()) if small.any() else float('nan'))


def direction(A):
    if A['mean'] <= -0.2 and A['neg'] >= 0.8: return 'inverted'
    return 'unrelated' if A['mean'] < 0.3 else 'ok'


def distribution_check(S, rf, ok, who, allow):
    """D2 / D3. Returns (fail names, warn names, the agreement dict)."""
    A = agreement(S, rf, ok); fails, warns = [], []
    byg = lambda k: ', '.join(f'{g} {agreement(S, rf, ok, m)[k]:.2%}' for g, m in S.groups().items() if g in ('PUBLIC', 'PRIVATE') or g.startswith('val '))
    for name, share, n, fl, wl, txt, ref in (
            ('D2', A['low_big'], A['n_big'], LOW_FAIL, LOW_WARN, f'feeds with M >= {LOW_M} have a per-feed Spearman below {LOW_R} against our order', 'neural pools 0.000%, LLM 0.033%, CatBoost 0.084%, XGBoost 0.447%'),
            ('D3', A['neg_small'], A['n_small'], NEG_FAIL, NEG_WARN, f'feeds with M < {LOW_M} have a NEGATIVE per-feed Spearman against our order', 'neural pools 0.000%, CatBoost 0.39%, LLM 0.49%, XGBoost 1.03%')):
        if not n or not np.isfinite(share): continue
        key = 'low_big' if name == 'D2' else 'neg_small'
        if share > fl:
            lvl = 'WARN (--allow_low_agreement)' if allow else 'FAIL'; (warns if allow else fails).append(f'{name} per-feed agreement distribution')
            print(f'[{lvl}] {who}: {name} {share:.2%} of the {n} {txt} (limit {fl:.2%}; genuine models on disk against W8: {ref}). By group: {byg(key)}. '
                  f'Part of the file is probably inverted, shuffled or mis-keyed; the law would read this damage as diversity. Ask for a corrected file', flush=True)
        elif share > wl:
            warns.append(f'{name} per-feed agreement distribution'); print(f'[WARN] {who}: {name} {share:.2%} of the {n} {txt} (WARN above {wl:.2%}, FAIL above {fl:.2%}; genuine models on disk: {ref})', flush=True)
    return fails, warns, A


# ---------------------------------------------------------------------------------------------------------------------------- law, gates, our scores
def law(ab, rho, gap): return ab[0] * (1 - rho) - ab[1] * abs(gap)


def w_star(ab, rho, gap):
    """Gap-adjusted weight of the WEAKER file (reports/NEXT_1115/merge.md 1.7): argmax of G(w) = 4 w (1 - w) a (1 - rho) - 2 b |gap| w, clipped to [0, 0.5]. Not fitted on val."""
    return float(np.clip(0.5 - ab[1] * abs(gap) / (4 * ab[0] * max(1 - rho, 1e-9)), 0.0, 0.5))


def gap_se_public(rho):
    """[I] paired SE of a PUBLIC gap (8,562 feeds) at within-feed correlation rho, interpolated in log(1 - rho) from the val SEs of _GAPSE."""
    return float(np.interp(np.log(np.clip(1 - rho, 1e-4, 1.0)), np.log(_GAPSE[:, 0]), _GAPSE[:, 1]) * np.sqrt(14423 / 8562))


def our_scores(path, split, S, ref_rank=None):
    """Model scores of OUR candidate: 0.5 z(FT pool) + 0.5 z(TO pool) of preds/ens_<TAG>_{ft,to}_<split>.npy (as nn_x/w12/kd/build_r3.py, build_r4.py).
    For test the within-feed order must reproduce >= 99.99% of the ranks of the file. Returns (scores or None, source text)."""
    m = re.match(r'cand_(.+?)(_r\d+)?\.parquet$', os.path.basename(str(path)))
    if not m: return None, 'the file name is not cand_<TAG>[_r<k>].parquet'
    tag = {'Q2a8_ftbag3_memeq': 'Q2a8s3'}.get(m.group(1), m.group(1))
    if not (P / f'ens_{tag}_ft_{split}.npy').exists(): return None, f'preds/ens_{tag}_ft_{split}.npy not found'
    to = zfeed(S, np.load(P / f'ens_{tag}_to_{split}.npy')); s = to if tag == 'W8TOkd' else 0.5 * zfeed(S, np.load(P / f'ens_{tag}_ft_{split}.npy')) + 0.5 * to
    if ref_rank is not None:
        same = float((midrank(S, s) == ref_rank).mean())
        if same < 0.9999: return None, f'preds/ens_{tag}_* reproduces only {same:.4%} of the ranks of the file'
    return s, f'preds/ens_{tag}_{{ft,to}}_{split}.npy'


def guard_new(path, what, forbid=()):
    """An output path must be named by the caller, must not exist, and its folder must exist. Called before any work and again before the write."""
    p = Path(path).resolve()
    if p.exists(): sys.exit(f'REFUSED: {what} {p} exists; this tool never overwrites a file')
    if not p.parent.is_dir(): sys.exit(f'REFUSED: the folder of {what} {p} does not exist')
    for d in forbid:
        if (ROOT / d).resolve() in p.parents: sys.exit(f'REFUSED: {what} is never written under {d}/')
    return p


def table(S, ok, rf_s, rf_l, rf_h, t1same, gaps):
    print(f'  {"group":24s} {"feeds":>6s} | {"Spearman fm ± SE":>18s} {"pooled":>7s} | {"law-rho fm ± SE":>18s} {"pooled":>7s} {"restored":>8s} | {"same top-1":>10s} | {"D2 share":>8s} {"D3 share":>8s} | law on record at |gap| 0 and {gaps[1]:.6f}')
    out = {}
    for g, m in S.groups().items():
        mm = m & ok; a = agg(S, rf_s, mm); b = agg(S, rf_l, mm); h = agg(S, rf_h, mm)
        if a[3] == 0: print(f'  {g:24s} {0:6d} | no usable feed'); continue
        A = agreement(S, rf_s, ok, m)
        out[g] = dict(n=a[3], spear_fm=a[0], spear_se=a[1], spear_pl=a[2], rho_fm=b[0], rho_se=b[1], rho_pl=b[2], rho_rest_fm=h[0], rho_rest_pl=h[2], top1=float(t1same[mm].mean()), low_big=A['low_big'], neg_small=A['neg_small'])
        f_ = lambda v: '       -' if not np.isfinite(v) else f'{v:8.3%}'
        print(f'  {g:24s} {a[3]:6d} | {a[0]:8.4f} ± {a[1]:.4f} {a[2]:8.4f} | {b[0]:8.4f} ± {b[1]:.4f} {b[2]:8.4f} {h[0]:8.4f} | {t1same[mm].mean():10.1%} | {f_(A["low_big"])} {f_(A["neg_small"])} | {" ".join(f"{law(LAW_REC, b[0], g_):+.5f}" for g_ in gaps)}')
    return out


def partner_z(S, R, zs_ours, as_ranks):
    """(z used for the law and the blend, z of the ranks restored on our shape, rank kind used). A rank file or --as_ranks uses the restored z."""
    sp = np.where(np.isfinite(R.s), R.s, 0.0); rp = midrank(S, sp); zh = shape_restore(S, rp, zs_ours)
    if R.is_rank or as_ranks: return zh, zh, rp, True
    return np.where(np.repeat(R.feed_ok, S.M), zrel(S, sp, R.feed_ok), 0.0), zh, rp, False


def export_ours(a):
    """--export_ours <prefix>: OUR model scores in the exchange format, <prefix>_test.parquet and <prefix>_val.parquet (feed_id, item_id Int32; score Float64 =
    0.5 z(FT pool) + 0.5 z(TO pool) of the candidate, the quantity the blend uses). Each file is read back through the reader and must be clean, hold model
    scores and carry exactly the order of our stored scores. Nothing is sent anywhere: the exchange itself is done by the owner."""
    outs = {sp: guard_new(f'{a.export_ours}_{sp}.parquet', '--export_ours', forbid=('subs', 'preds')) for sp in ('test', 'val')}
    S, V = SPLITS('test'), SPLITS('val'); print(f'== {VERSION}  export of our model scores for {a.ours}', flush=True); O = read_scores(a.ours, S, 'ours')
    if not O.usable or O.fails or not O.feed_ok.all(): print('VERDICT: our own file fails the reader: fix it first'); return {'status': 'ours_unusable'}
    ro = midrank(S, O.s); zt, src = our_scores(a.ours, 'test', S, ro) if a.ours_scores == 'auto' else (np.load(a.ours_scores, allow_pickle=False).astype(np.float64), a.ours_scores)
    zv, vsrc = our_scores(a.ours, 'val', V) if a.ours_val == 'auto' else (np.load(a.ours_val, allow_pickle=False).astype(np.float64), a.ours_val)
    if zt is None or zv is None or float((midrank(S, zt) == ro).mean()) < 0.9999:
        print(f'VERDICT: our model scores are not available or do not reproduce the file (test: {src}; val: {vsrc})'); return {'status': 'ours_unusable'}
    RES = {'status': 'ok', 'version': VERSION, 'export': {}}
    for sp, S_, z in (('test', S, zt), ('val', V, zv)):
        assert np.abs(S_.feed_id).max() < 2 ** 31 and np.abs(S_.item_id).max() < 2 ** 31 and np.isfinite(z).all()
        pl.DataFrame({'feed_id': S_.feed_id[S_.fid].astype(np.int32), 'item_id': S_.item_id.astype(np.int32), 'score': np.asarray(z, np.float64)}).write_parquet(outs[sp])
        R = read_scores(outs[sp], S_, f'export {sp}'); good = bool(R.usable and not R.fails and R.feed_ok.all() and not R.is_rank and (midrank(S_, R.s) == midrank(S_, z)).all())
        h = hashlib.sha256(outs[sp].read_bytes()).hexdigest(); RES['export'][sp] = dict(path=str(outs[sp]), rows=S_.N, feeds=S_.n, sha256=h, verified=good)
        print(f'{"written" if good else "[FAIL] NOT VERIFIED"} {outs[sp]}: {S_.N} rows, {S_.n} feeds, score = our model scores ({src if sp == "test" else vsrc}), read back through the reader: {"clean, same order" if good else "see above"}; sha256 {h}')
        if not good: RES['status'] = 'export_failed'
    print('NEXT: the owner sends the two files (and their sha256) to the partner; no agent sends anything.')
    return RES


# ---------------------------------------------------------------------------------------------------------------------------- main
def main(argv=None):
    p = argparse.ArgumentParser(description=VERSION, usage=__doc__.split('USAGE')[1].split('FILE FORMAT')[0])
    p.add_argument('--ours', required=True, help='our candidate (subs/cand_<TAG>_r<k>.parquet); read only')
    p.add_argument('--ours_public', type=float, default=None); p.add_argument('--partner', default=''); p.add_argument('--partner_public', type=float, default=None)
    p.add_argument('--partner_val', default=''); p.add_argument('--ours_scores', default='auto'); p.add_argument('--ours_val', default='auto')
    p.add_argument('--flip_partner', action='store_true', help='negate the partner scores (only after the partner confirmed an ascending convention)')
    p.add_argument('--as_ranks', action='store_true', help="reduce the partner files to within-feed ranks restored on the shape of our scores; needed to get a decision from a rank-only file (owner's decision)")
    p.add_argument('--allow_low_agreement', action='store_true', help='turn the D2 / D3 FAIL into a WARN (only after the low-agreement feeds were inspected by hand)')
    p.add_argument('--blend', action='store_true', help='write the blend; needs --out'); p.add_argument('--out', default='', help='path of the blend parquet, named by the caller; never overwritten')
    p.add_argument('--out_flat', default='', help='prefix of the flat float32 blend ranks <prefix>_test.npy / <prefix>_val.npy (inputs of src/presub_gate.py)')
    p.add_argument('--w', default='eq', choices=('eq', 'law'), help='eq = 0.5 / 0.5 (protocol); law = gap-adjusted weight of the law on record')
    p.add_argument('--force', action='store_true', help="write the blend although gates 1-3 did not all pass (explicit owner decision; reader FAILs, direction problems and a withheld decision are never overridable)")
    p.add_argument('--json', default='', help='write the result dictionary to this path')
    p.add_argument('--export_ours', default='', help='prefix: write OUR model scores in the exchange format to <prefix>_test.parquet and <prefix>_val.parquet, verify them, and stop (no partner file needed)')
    a = p.parse_args(argv); RES = {'status': 'ok', 'version': VERSION}; inhouse = a.partner_public is None
    if a.export_ours:
        if a.partner or a.blend or a.out or a.out_flat: p.error('--export_ours is a separate action: give only --ours (and --ours_scores / --ours_val if not auto)')
        return export_ours(a)
    if not a.partner: p.error('--partner is required (or --export_ours <prefix>)')
    if inhouse and not a.partner_val: p.error('give --partner_public (partner mode), or --partner_val for IN-HOUSE mode (the val gap is used)')
    if not inhouse and a.ours_public is None: p.error('partner mode needs --ours_public (the public score of --ours, as reported by the owner)')
    if a.blend and not a.out: p.error('--blend needs --out <path>: the output is named by the caller, there is no default name')
    if (a.out or a.out_flat) and not a.blend: p.error('--out / --out_flat are used with --blend only')
    if a.blend and inhouse: p.error('--blend needs --partner_public (a blend with a partner is priced with the public gap)')
    if a.out: guard_new(a.out, '--out')
    if a.out_flat: [guard_new(f'{a.out_flat}_{sp}.npy', '--out_flat') for sp in ('test', 'val')]
    if a.json: guard_new(a.json, '--json', forbid=('subs', 'preds'))

    def finish(status=None):
        if status: RES['status'] = status
        if a.json: guard_new(a.json, '--json', forbid=('subs', 'preds')).write_text(json.dumps({k: v for k, v in RES.items() if not k.startswith('_')}, indent=1, default=float))
        return RES

    S = SPLITS('test')
    print(f'== {VERSION}  ours {a.ours} (public {"n/a" if a.ours_public is None else f"{a.ours_public:.6f}"})  partner {a.partner} (public {"none: IN-HOUSE mode, val gap" if inhouse else f"{a.partner_public:.6f}"})', flush=True)
    O = read_scores(a.ours, S, 'ours'); Pn = read_scores(a.partner, S, 'partner', allow_npy=inhouse)
    RES.update(ours_fails=O.fails, partner_fails=Pn.fails, partner_warns=Pn.warns, partner_stats=Pn.stats)
    if not O.usable or O.fails or not O.feed_ok.all(): print('VERDICT: our own file fails the reader: fix it first'); return finish('ours_unusable')
    if not Pn.usable: print('VERDICT: partner file NOT USABLE (see the FAIL lines): no correlation, no law, no blend'); return finish('partner_unusable')
    if a.flip_partner: Pn.s = -Pn.s; print('[INFO] partner scores negated on request (--flip_partner)')
    ok = Pn.feed_ok; so = O.s; ro = midrank(S, so)
    # ---- our model scores (needed for the law's correlation, for the shape of a rank file and for the blend)
    if a.ours_scores == 'auto': zo_raw, src = our_scores(a.ours, 'test', S, ro)
    else:
        zo_raw = np.load(a.ours_scores, allow_pickle=False).astype(np.float64); src = a.ours_scores; same = float((midrank(S, zo_raw) == ro).mean())
        if same < 0.9999: print(f'[FAIL] ours: --ours_scores reproduces only {same:.4%} of the ranks of {a.ours}'); zo_raw = None
    if zo_raw is None: print(f'VERDICT: our model scores are not available ({src}): give --ours_scores <flat test .npy> that reproduces the order of --ours'); return finish('ours_unusable')
    zo = zfeed(S, zo_raw); zs = sorted_vals(S, zo); print(f'[OK] ours: model scores from {src} reproduce the within-feed order of the file')
    zp, zp_h, rp, restored = partner_z(S, Pn, zs, a.as_ranks); rf_s = pf_corr(S, ro, rp)
    # ---- D1 direction
    A0 = agreement(S, rf_s, ok); d = direction(A0); RES.update(direction=d, spear_fm=A0['mean'])
    if d == 'inverted':
        print(f'[FAIL] partner: D1 the file is INVERTED against ours: mean per-feed Spearman {A0["mean"]:+.4f}, negative in {A0["neg"]:.1%} of the feeds. Its score is probably ascending '
              f'(smaller = better, for example a list position). After flipping it would read {-A0["mean"]:+.4f}. Ask the partner which convention the file uses, then re-run with --flip_partner')
        print('VERDICT: REFUSED (inverted score convention)'); return finish('inverted')
    if d == 'unrelated':
        print(f'[FAIL] partner: D1 no usable agreement with our file: mean per-feed Spearman {A0["mean"]:+.4f} (negative in {A0["neg"]:.1%} of the feeds; the weakest model on disk, XGBoost at val 0.368, '
              f'reads +0.735). Probably shuffled rows, a wrong id mapping or another task')
        print('VERDICT: REFUSED (unrelated order)'); return finish('unrelated')
    if A0['mean'] < 0.6: print(f'[WARN] partner: mean per-feed Spearman {A0["mean"]:+.4f} is below every model on disk (0.735-0.996): check the file before trusting the law'); Pn.warns.append('low mean agreement')
    # ---- D2 / D3 per-feed distribution
    fD, wD, _ = distribution_check(S, rf_s, ok, 'partner', a.allow_low_agreement); Pn.fails += fD; Pn.warns += wD; RES.update(low_big=A0['low_big'], neg_small=A0['neg_small'], nonpos_small=A0['nonpos_small'])
    # ---- R6 rank-only file, D4 score scale
    rf_l = pf_corr(S, zo, zp); rf_h = pf_corr(S, zo, zp_h); r_l = agg(S, rf_l, ok); r_h = agg(S, rf_h, ok); flagged = bool(Pn.is_rank and not a.as_ranks)
    sk = lambda z: float(np.mean((np.bincount(S.fid, z ** 3, S.n) / S.M)[ok]))
    if Pn.is_rank:
        print(f'[FLAG] partner: R6 RANK-ONLY FILE: {Pn.stats["rank_shaped_share"]:.2%} of the feeds hold a function of the within-feed rank, not model scores. This is not the agreed exchange format. '
              f'The correlation of the law is estimated by restoring the ranks on the shape of our scores (error on 10 on-disk partners <= 0.0020 in rho). ' + _RANK_COST
              + (' Accepted on request (--as_ranks).' if a.as_ranks else ' DECISION WITHHELD: ask for model scores, or re-run with --as_ranks on the owner\'s decision.'))
    elif a.as_ranks: print('[INFO] partner scores are reduced to within-feed ranks and restored on the shape of our scores on request (--as_ranks). ' + _RANK_COST)
    else:
        dsc = abs(r_l[0] - r_h[0]); RES['scale_diff'] = dsc
        print(f'[{"FAIL" if dsc > SCALE_FAIL else "OK"}] partner: D4 score scale: law-rho of the scores as sent {r_l[0]:.4f}, of their ranks restored on the shape of our scores {r_h[0]:.4f}, difference {dsc:.4f} '
              f'(limit {SCALE_FAIL}; 10 genuine on-disk partners: at most 0.0020); within-feed skewness {sk(zp):+.2f} (ours {sk(zo):+.2f})'
              + ('. Either the file does not hold logit-like scores (probabilities, softmax or another squashing) or part of it is damaged (see D2 / D3): the law is not calibrated '
                 'for such a file and reads the difference as diversity. Ask for raw model outputs, or re-run with --as_ranks on the owner\'s decision' if dsc > SCALE_FAIL else ''))
        if dsc > SCALE_FAIL: Pn.fails.append('D4 score scale')
    t1 = top1(S, zo) == top1(S, np.where(np.isfinite(Pn.s), Pn.s, -np.inf))
    if r_h[0] > 0.9999 and float(t1[ok].mean()) > 0.999: print('[WARN] partner: the order of the file is (nearly) identical to ours: it is probably our own file or a copy of it'); Pn.warns.append('identical order')

    # ---- val files (labels are used only to score)
    VAL = None; val_fails = []
    if a.partner_val:
        V = SPLITS('val'); yv = np.load(FLAT / 'val/cand_target.npy').astype(np.float64); print(f'\n-- val files (labels are used only to score; {V.n} val feeds)', flush=True)
        Pv = read_scores(a.partner_val, V, 'partner val', allow_npy=inhouse); val_fails = list(Pv.fails)
        if a.ours_val == 'auto': ov_raw, vsrc = our_scores(a.ours, 'val', V)
        else: ov_raw, vsrc = np.load(a.ours_val, allow_pickle=False).astype(np.float64), a.ours_val
        if not Pv.usable or ov_raw is None: print(f'  val check NOT possible (partner val usable: {Pv.usable}; our val scores: {vsrc})'); val_fails.append('val not usable')
        else:
            if a.flip_partner: Pv.s = -Pv.s
            okv = Pv.feed_ok; zov = zfeed(V, ov_raw); zpv, zpv_h, rpv, restored_v = partner_z(V, Pv, sorted_vals(V, zov), a.as_ranks); rfv = pf_corr(V, midrank(V, zov), rpv)
            Av = agreement(V, rfv, okv); dv = direction(Av)
            if dv != 'ok': print(f'  [FAIL] partner val: D1 the file is {dv} against ours (mean per-feed Spearman {Av["mean"]:+.4f})'); val_fails.append(f'D1 {dv}')
            else:
                fV, wV, _ = distribution_check(V, rfv, okv, 'partner val', a.allow_low_agreement); val_fails += fV
                if Pv.is_rank != Pn.is_rank and not a.as_ranks:
                    print(f'  [FAIL] partner val: the val file holds {Pv.stats["holds"]}, the test file {Pn.stats["holds"]}: the val check would not test the blend that is written'); val_fails.append('kind differs from the test file')
                rvl = agg(V, pf_corr(V, zov, zpv), okv); rvh = agg(V, pf_corr(V, zov, zpv_h), okv)
                if not restored_v and abs(rvl[0] - rvh[0]) > SCALE_FAIL:
                    print(f'  [FAIL] partner val: D4 score scale: law-rho as sent {rvl[0]:.4f}, restored on our shape {rvh[0]:.4f} (limit {SCALE_FAIL})'); val_fails.append('D4 score scale')
                ev = lambda s_: ndcg_per_feed(np.asarray(s_, np.float64), yv, V.off, V.M)[okv]
                VAL = dict(V=V, okv=okv, zov=zov, zpv=zpv, eo=ev(zov), ep=ev(zpv), ev=ev, rv=rvl, rvh=rvh, vsrc=vsrc, holds=Pv.stats['holds'], complete=bool(Pv.feed_cov.all()),
                           top1=float((top1(V, zov) == top1(V, np.where(np.isfinite(Pv.s), Pv.s, -np.inf)))[okv].mean()))
        RES['val_fails'] = val_fails
    if inhouse and VAL is None: print('VERDICT: IN-HOUSE mode needs a usable val file (the gap is the val gap)'); return finish('partner_unusable')

    # ---- gap and the better file
    if inhouse:
        gv_ = VAL['eo'] - VAL['ep']; gap = abs(float(gv_.mean())); partner_better = bool(gv_.mean() < 0); gap_note = f'the VAL gap {gap:.6f} ± {se(gv_):.5f} (IN-HOUSE mode)'
    else:
        gap = abs(a.ours_public - a.partner_public); partner_better = bool(a.partner_public > a.ours_public); gap_note = f'the public gap {gap:.6f}'
    weaker, better = ('ours', 'partner') if partner_better else ('partner', 'ours')
    kind = ('ranks of the partner file restored on the shape of our scores (approximation)' if restored else 'Pearson of per-feed z of the model scores of both files (the definition of the laws)')
    print(f'\n-- within-feed correlation with our file on test. law-rho = {kind}; restored = the same with the partner reduced to ranks on our shape. D2 = M >= {LOW_M}, Spearman < {LOW_R}; D3 = M < {LOW_M}, Spearman < 0')
    T = table(S, ok, rf_s, rf_l, rf_h, t1, (0.0, gap)); RES['groups'] = T; A = T['ALL']; rho_fm, rho_pl, rho_h, rho_hpl = A['rho_fm'], A['rho_pl'], A['rho_rest_fm'], A['rho_rest_pl']
    if 'PUBLIC' in T and 'PRIVATE' in T:
        dpp = T['PRIVATE']['rho_fm'] - T['PUBLIC']['rho_fm']; sdp = float(np.hypot(T['PRIVATE']['rho_se'], T['PUBLIC']['rho_se']))
        print(f'  private - public law-rho (feed-mean): {dpp:+.4f} ± {sdp:.4f}' + ('   <-- the two files agree LESS on the private days: the public reading of a blend does not cover that part' if dpp < -3 * sdp and dpp < -0.005 else ''))
    print('  reference points on record (test, feed-mean Pearson of z, against W8; nn_x/w12/next/merge/out/c1_calib.txt T1, test columns): S16 0.9988 | C3 0.9935 | G3 0.9931 | Pall 0.9911 | base x4 0.9880 | GRU x6 0.9828 | LLM 0.8383 | CatBoost 0.8375 | XGBoost 0.7679')
    if VAL is not None:
        dvt = rho_fm - VAL['rv'][0]; RES['val_test_rho_diff'] = dvt
        print(f'  [{"WARN" if abs(dvt) > VALTEST_WARN else "OK"}] D5 test law-rho {rho_fm:.4f} against val law-rho {VAL["rv"][0]:.4f}: difference {dvt:+.4f} (WARN beyond {VALTEST_WARN}; 10 genuine on-disk partners: at most 0.0051); same top-1 item: test {A["top1"]:.1%}, val {VAL["top1"]:.1%} (information, no line)'
              + (': the test file and the val file do not look like outputs of one model; damage in the test file alone is not seen by the val check' if abs(dvt) > VALTEST_WARN else ''))
        if abs(dvt) > VALTEST_WARN: Pn.warns.append('D5 val / test law-rho')
    # ---- law and gates
    rho_dec = max(rho_fm, rho_h); l_rec, l_ref, l_dec = law(LAW_REC, rho_fm, gap), law(LAW_REFIT, rho_pl, gap), law(LAW_REC, rho_dec, gap); ws = w_star(LAW_REC, rho_dec, gap)
    print(f'\n-- law (gain of the EQUAL z-mix over the better file; fitted on val, applied with the test correlation and {gap_note})')
    if inhouse: print(f'  val: ours {VAL["eo"].mean():.5f}, partner {VAL["ep"].mean():.5f}, |gap| {gap:.6f}')
    else:
        gse = gap_se_public(rho_fm); best = max(a.ours_public, a.partner_public)
        print(f'  public: ours {a.ours_public:.6f}, partner {a.partner_public:.6f}, |gap| {gap:.6f}; paired SE of a public gap at this rho about {gse:.5f} [I, interpolated from val SEs] -> the gap term alone is uncertain by ± {LAW_REC[1] * gse:.5f}')
    print(f'  LAW ON RECORD: 0.0186 * (1 - {rho_fm:.4f}) - 0.378 * {gap:.6f} = {l_rec:+.5f}' + ('' if inhouse else f'  -> predicted public of the equal mix {best + l_rec:.5f}'))
    print(f'  the same with the scale-free rho (ranks restored on our shape, {rho_h:.4f}): {law(LAW_REC, rho_h, gap):+.5f}')
    print(f'  REFIT (W8 level, 10 partners; information): 0.0103 * (1 - {rho_pl:.4f}) - 0.304 * {gap:.6f} = {l_ref:+.5f}')
    print(f'  with the v1 quantity (pooled Spearman {A["spear_pl"]:.4f}) the law on record would read {law(LAW_REC, A["spear_pl"], gap):+.5f} (not the definition of the law; biased towards blending)')
    print(f'  break-even at this rho: the law on record is 0 at |gap| {LAW_REC[0] * (1 - rho_dec) / LAW_REC[1]:.5f} and clears the {MARGIN} margin at |gap| <= {max((LAW_REC[0] * (1 - rho_dec) - MARGIN) / LAW_REC[1], 0):.5f}')
    print(f'  gap-adjusted weight of the weaker file ({weaker}): w* = clip(0.5 - 0.378 |gap| / (4 * 0.0186 (1 - rho)), 0, 0.5) = {ws:.3f} (used only with --w law)')
    complete = bool(Pn.feed_cov.all()); clean = (not Pn.fails) and complete
    g1 = None if inhouse else bool(gap <= GATE1 + 1e-12); g2 = bool(l_dec - MARGIN >= 0); go = bool(clean and not flagged and g2 and (g1 is not False))
    if g1 is not None:
        print(f'  GATE 1 (fixed): partner public within {GATE1} of ours -> |gap| {gap:.6f}: {"met" if g1 else "NOT met"}. Its second half ("not a history-attention neural ranker") cannot be read from a file; '
              f'law-rho >= 0.98 is the signature of our own neural family (0.983-0.999 on record), this file reads {rho_fm:.4f}')
    print(f'  GATE 2 (fixed): law on record - {MARGIN} >= 0, read with the larger of the two rho ({rho_dec:.4f}) -> {l_dec - MARGIN:+.5f}: {"met" if g2 else "NOT met"} | with the rho as sent {l_rec - MARGIN:+.5f} | '
          f'the refit gives {l_ref - MARGIN:+.5f} ({"same" if (l_ref - MARGIN >= 0) == g2 else "OPPOSITE"} reading; the rule on record decides on 0.0186 / 0.378)')
    if not clean: dec = f'NONE. Reader FAIL ({"; ".join(dict.fromkeys(Pn.fails)) or "incomplete coverage"}): the numbers above use {int(ok.sum())} of {S.n} feeds and are diagnostics only. Ask for a corrected file'
    elif flagged: dec = 'WITHHELD. Rank-only file: ask for model scores, or re-run with --as_ranks on the owner\'s decision'
    elif go: dec = 'PROCEED to the val check (gate 3) and to one public slot'
    else: dec = 'do NOT blend: keep the better solo file'
    print(f'  DECISION: {dec}' + (' [IN-HOUSE mode: gate 1 does not apply]' if inhouse else ''))
    print(f'  UNCERTAINTY: {_UNCERTAINTY}')
    RES.update(rho_fm=rho_fm, rho_pl=rho_pl, rho_rest_fm=rho_h, rho_rest_pl=rho_hpl, spear_pl=A['spear_pl'], gap=gap, law_rec=l_rec, law_rest=law(LAW_REC, rho_h, gap), law_dec=l_dec, law_refit=l_ref, w_star=ws,
               clean=clean, flagged_rank=flagged, is_rank=bool(Pn.is_rank), restored=restored, gate1=g1, gate2=g2, go=go, decision=dec.split('.')[0].split(':')[0], feeds=int(ok.sum()), better=better, top1=A['top1'])

    # ---- val check (gate 3)
    g3 = None; wv = 0.5 if a.w == 'eq' else ((1 - ws) if partner_better else ws)        # weight of the PARTNER file in the mix that would be written
    if VAL is not None:
        V, okv, zov, zpv, eo, ep, ev, rv = (VAL[k] for k in ('V', 'okv', 'zov', 'zpv', 'eo', 'ep', 'ev', 'rv')); gv = eo - ep; bt, bn = (eo, 'ours') if gv.mean() >= 0 else (ep, 'partner')
        wl = (1 - ws) if partner_better else ws; mixes = {'equal': 0.5, 'law': wl}; big = (V.M >= 50)[okv]; tue = (V.day == 20718)[okv]
        print(f'\n-- val check ({int(okv.sum())} of {V.n} feeds' + ('' if VAL['complete'] else '; the partner val file is INCOMPLETE') + f'; our val scores: {VAL["vsrc"]}; the partner val file holds {VAL["holds"]})')
        print(f'  ours {eo.mean():.5f} | partner {ep.mean():.5f} | ours - partner {pm(gv)} | val law-rho feed-mean {rv[0]:.4f} pooled {rv[2]:.4f} (test {rho_fm:.4f} / {rho_pl:.4f})')
        print(f'  law on record with the VAL correlation and the VAL gap: {law(LAW_REC, rv[0], gv.mean()):+.5f} | refit {law(LAW_REFIT, rv[2], gv.mean()):+.5f}')
        if not inhouse:
            lk_o, lk_p = eo.mean() - a.ours_public, ep.mean() - a.partner_public
            print(f'  V1 val - public: ours {lk_o:+.5f}, partner {lk_p:+.5f}' + (f'   [WARN] the partner is {lk_p - lk_o:+.5f} more optimistic on val than our file (band of src/calib.py: 0.003): '
                  f'its val predictions may have seen val labels; then this check is void' if lk_p - lk_o > VAL_OPT else ''))
            if lk_p - lk_o > VAL_OPT: Pn.warns.append('V1 val optimism of the partner')
        RES['val'] = dict(ours=float(eo.mean()), partner=float(ep.mean()), gap=float(gv.mean()), gap_se=se(gv), rho_fm=rv[0], rho_pl=rv[2], rho_rest_fm=VAL['rvh'][0], n=int(okv.sum()), better=bn, top1=VAL['top1'],
                          law_rec_val=law(LAW_REC, rv[0], gv.mean()), law_refit_val=law(LAW_REFIT, rv[2], gv.mean()))
        for nm, w in mixes.items():
            rk, tied = mix_rank(V, zov, zpv, w, prefer_b=partner_better); em = ev(rk); dd = em - bt; z_ = float(dd.mean() / max(se(dd), 1e-12))
            lab = f'{"equal mix" if nm == "equal" else "gap-adjusted mix"} (w partner {w:.3f})'
            print(f'  {lab:32s}: {em.mean():.5f} | vs the better file on val ({bn}) {pm(dd)} (z {z_:+.2f}) | vs ours {pm(em - eo)} | M>=50 {pm(dd[big])} M<50 {pm(dd[~big])} | Tue {pm(dd[tue])} Thu {pm(dd[~tue])}'
                  f' | tied rows {tied:.3%}' + (f', resolved in the order of {better}, then by item_id' if tied > 0 else ''))
            RES['val'][nm] = dict(w=float(w), ndcg=float(em.mean()), d=float(dd.mean()), se=se(dd), z=z_, tied=tied); RES[f'_pf_{nm}'] = em; RES[f'_rk_val_{nm}'] = rk
        RES['_pf_ours'] = eo; RES['_pf_partner'] = ep
        dz = RES['val'][a.w if a.w == 'law' else 'equal']['z']; val_ok = not val_fails and VAL['complete']
        g3 = bool(val_ok and dz >= GATE3_Z and 0.0 < wv < 1.0)
        print(f'  GATE 3 (fixed): the mix that would be written (--w {a.w}, weight of the partner {wv:.3f}) must gain >= {GATE3_Z:g} paired SE on val over the better file -> z {dz:+.2f}: {"PASS" if g3 else "no pass"}'
              + ('' if val_ok else f' (the val file has reader FAILs: {"; ".join(val_fails) or "incomplete"})') + f' | the other weight is printed as information. One weight is tested, the one that is written; nothing is scanned.')
    RES['gate3'] = g3
    # ---- blend
    if a.blend:
        why = []
        if not clean: why.append('reader FAIL on the partner test file (never overridable)')
        if flagged: why.append('rank-only file without --as_ranks (never overridable by --force)')
        if a.w == 'law' and ws == 0: why.append('the gap-adjusted weight of the weaker file is 0 (the blend would be the better solo file)')
        if a.partner_val and (val_fails or VAL is None or not VAL['complete']): why.append('reader FAIL on the partner val file (never overridable)')
        gates_ok = bool(go and g3)
        d5 = 'D5 val / test law-rho' in Pn.warns   # lead's setting 2026-11-15 (review of this tool): a D5 WARN blocks the blend unless --force, because a test file with noise in 25% of its feeds reads as diversity and passes every FAIL line
        if not why and (not gates_ok or d5) and not a.force:
            why += [x for x, bad in ((f'gate 1 (public within {GATE1}) not met', g1 is False), (f'gate 2 (law - {MARGIN} >= 0) not met', not g2),
                                     ('gate 3 (val check) not done: give --partner_val' if g3 is None else f'gate 3 (>= {GATE3_Z:g} paired SE on val) not passed', not g3),
                                     (f'D5: test and val law-rho differ by more than {VALTEST_WARN} (a partly damaged test file reads as diversity)', d5)) if bad] + ['add --force only on an explicit decision of the owner']
        if why: print('\nBLEND REFUSED: ' + '; '.join(why)); RES['blend'] = 'refused'; return finish('blend_refused')
        if not gates_ok: print('\n[WARN] gates 1-3 did not all pass; the blend is written because --force was given')
        rk, tied = mix_rank(S, zo, zp, wv, prefer_b=partner_better); assert np.isfinite(rk).all()
        out = guard_new(a.out, '--out'); from make_sub import make_sub
        make_sub(rk, str(out)); back = S.key.join(pl.read_parquet(out).with_columns(pl.col('feed_id').cast(pl.Int64), pl.col('item_id').cast(pl.Int64)), on=['feed_id', 'item_id'], how='left').sort('row')['score'].to_numpy().astype(np.float64)
        assert np.isfinite(back).all() and (midrank(S, back) == rk).all(), 'the written file does not hold the intended order'
        sha = hashlib.sha256(out.read_bytes()).hexdigest(); RES.update(blend=str(out), blend_tied=tied, blend_sha256=sha, blend_w_partner=wv)
        print(f'\nwritten {out} = order of {1 - wv:g} z(ours) + {wv:g} z(partner) per feed; {tied:.2%} of the rows were tied and resolved in the order of {better}, then by item_id; read back and verified; sha256 {sha}')
        if a.out_flat:
            ft = guard_new(f'{a.out_flat}_test.npy', '--out_flat'); np.save(ft, rk.astype(np.float32)); RES['flat_test'] = str(ft); msg = f'written {ft}'
            if VAL is not None: fv = guard_new(f'{a.out_flat}_val.npy', '--out_flat'); np.save(fv, RES[f'_rk_val_{"law" if a.w == "law" else "equal"}'].astype(np.float32)); RES['flat_val'] = str(fv); msg += f' and {fv}'
            print(msg + ' (flat float32 within-feed ranks of the blend, 0 = worst)')
        print('NEXT: the blend passes src/presub_gate.py like any other candidate before the owner submits it (reports/MERGE_PROTOCOL.md section 6). Nothing is submitted by this tool.')
    return finish()


def exit_code(r):
    """0 clean read | 2 reader FAIL | 3 inverted / unrelated / unusable partner file | 4 rank-only file, decision withheld | 5 blend refused | 1 error.
    Not set here: the argument parser exits with 2 on a command-line error; a refused output path exits with 1 through sys.exit."""
    st = r.get('status')
    if st in ('partner_unusable', 'inverted', 'unrelated'): return 3
    if st == 'blend_refused': return 5
    if st != 'ok': return 1
    return 2 if not r.get('clean', True) else 4 if r.get('flagged_rank') else 0


if __name__ == '__main__':
    sys.exit(exit_code(main()))
