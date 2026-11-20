"""WAVE-10 FPCD-lite (reports/DATA_PIPELINE_1113.md 5.1 'T-1', proposal P4 amended by both skeptics; written 2026-11-13 ~16:50 UTC),
used by nn_x/w10/fpcd/train.py.  OFF at the train.py default (--cd_exclude_days ''); train.py builds the object only behind
`if a.cd_exclude_days:` and uses it only through `if fpcd is not None ...:` guards, so a flag-off run executes exactly the
nn_x/w5 code path (tests/test_fpcd.py: AST proof + end-to-end identity).  Imports only numpy (no GPU guard), so the unit tests
call the very functions train.py uses.

  --cd_exclude_days "0,1,2"   Moscow calendar days (UTC+3), counted from the FIRST train day (day 0 = 2026-09-06 = absolute day
        20702 = floor((timestamp + 10800) / 86400)), whose training feeds are MASKED OUT of every cooldown batch (steps >=
        t_cool; step 0 in the CD harness) -- both the pre-computed regular batches (train.py's all_batches) and the m02
        exploration-pool batches (train.py's expl stream, incl. its reshuffles).  Mechanism under test: the Sep 6-8 feeds
        (27.45% of the training feeds, 25.8% of the pool) carry an all-NaN pre2 block (9 of the 104 inputs blank, has:pre2_fe
        = 0), a format that never occurs at inference; the arm keeps the late, memorising phase on in-format rows only.
        The skeptics' spec: NO redraw and NO refill of the batches -- the same 19,670 steps, lr and K9 schedule, each remaining
        batch simply shrinks (effective batch ~0.725x, inside wave 8's flat 0.5x-2x range), so passes per feed stay at 2
        (regular) and ~0.93 (pool), i.e. nothing moves toward the cf4 / e20 / m04 harm regions.
        Empty masked batches (a batch whose feeds are ALL excluded; only the 1-2-feed tails of an M bucket can do this, expected
        0-3 per 19,670 steps): the step is kept and filled by SPLITTING the largest masked batch of the same pass (same epoch for
        the regular batches, same reshuffle for the pool stream) into two halves, one in the donor's slot and one in the empty
        slot.  Every allowed feed is therefore still trained exactly once per pass (passes per feed unchanged, no excluded feed
        appears, the step count is unchanged); the number of such split events is logged.
        Logged at every eval: realised feeds per step (regular / pool) vs the unmasked batch, the pool share of the cooldown
        steps, pool passes, split events; leak guard: no holdout feed in the allowed set (cd_arms4.py:241 pattern).
"""
import datetime

import numpy as np

TZ_HOURS = 3  # Moscow = UTC+3 (as cd_arms4.recent_feeds)


def parse_days(s):
    days = sorted({int(t) for t in s.split(",") if t.strip()})
    assert days and all(d >= 0 for d in days), f"bad --cd_exclude_days spec {s!r} (comma list of day indices >= 0 from the first train day)"
    return days


def day_index(ts, tz_hours=TZ_HOURS):
    """Moscow calendar day of every feed, absolute (days since 1970-01-01) and relative to the first train day."""
    ts = np.asarray(ts, dtype=np.int64)
    day = (ts + tz_hours * 3600) // 86400
    return day, day - day.min()


def excluded_mask(ts, days):
    """bool [n_feeds]: True for the feeds of the excluded Moscow days (relative indices). Returns (mask, abs_days, dates)."""
    day_abs, day_rel = day_index(ts)
    excl = np.isin(day_rel, days)
    d0 = int(day_abs.min())
    abs_days = [d0 + d for d in days]
    dates = [str(datetime.date(1970, 1, 1) + datetime.timedelta(days=d)) for d in abs_days]
    return excl, abs_days, dates


def mask_batches(batches, excl, groups):
    """Masked copies of a list of batches (numpy int arrays of feed ids): every excluded feed removed. An EMPTY masked batch
    is filled by splitting the largest masked batch of the same group (groups = int per batch, e.g. the epoch) in two: the
    first half stays in the donor's slot, the second half fills the empty slot (every allowed feed still appears exactly once
    per group; no excluded feed appears; the number of batches is unchanged). Returns (masked list, number of split events)."""
    masked = [np.asarray(b)[~excl[np.asarray(b)]] for b in batches]
    groups = np.asarray(groups)
    assert len(groups) == len(masked)
    n_split = 0
    for i in [k for k, m in enumerate(masked) if len(m) == 0]:
        same = np.where(groups == groups[i])[0]
        j = max(same, key=lambda k: len(masked[k]))
        assert len(masked[j]) >= 2, f"batch {i}: no donor batch with >= 2 feeds in its group (group {groups[i]})"
        half = len(masked[j]) // 2
        masked[i], masked[j] = masked[j][half:], masked[j][:half]
        n_split += 1
    assert all(len(m) > 0 for m in masked)
    return masked, n_split


class FPCDMask:
    """Source of the masked cooldown batches (regular: pre-computed from train.py's all_batches at setup; pool: the current
    exploration stream, masked lazily when train.py builds / reshuffles it). train.py calls batch() on every cooldown step,
    after the unchanged m02 pool decision and before the gather."""

    def __init__(self, excl, all_batches, steps_per_epoch, t_cool, total):
        self.excl = excl
        self.all_batches = all_batches
        self.t_cool, self.total = t_cool, total
        groups = [i // steps_per_epoch for i in range(len(all_batches))]
        self.reg, self.n_split_reg = mask_batches(all_batches, excl, groups)
        self.pool_src, self.pool_masked, self.n_split_pool, self.pool_passes = None, None, 0, 0
        self.n_steps = [0, 0]  # cooldown steps [regular, pool]
        self.n_feeds = [0, 0]  # realised feeds [regular, pool]
        self.n_orig = [0, 0]  # unmasked feeds of the same batches
        self.pool_feeds_seen = 0

    def batch(self, fb, is_pool, step, expl, i):
        assert step >= self.t_cool
        if is_pool:
            if expl is not self.pool_src:  # new / reshuffled exploration stream: mask it once (one pass over the pool)
                self.pool_src = expl
                self.pool_masked, ns = mask_batches(expl, self.excl, np.zeros(len(expl), np.int64))
                self.n_split_pool += ns
                self.pool_passes += 1
            assert 0 <= i < len(expl) and expl[i] is fb, "FPCD: pool batch / stream index mismatch"
            out = self.pool_masked[i]
            self.pool_feeds_seen += len(out)
        else:
            assert self.all_batches[step] is fb, "FPCD: regular batch / step mismatch"
            out = self.reg[step]
        k = int(is_pool)
        self.n_steps[k] += 1
        self.n_feeds[k] += len(out)
        self.n_orig[k] += len(fb)
        assert len(out) > 0 and not self.excl[out].any(), "FPCD: excluded feed in a cooldown batch"
        return out

    def summary(self, step, n_pool_allowed):
        ns, nf, no = self.n_steps, self.n_feeds, self.n_orig
        tot = max(ns[0] + ns[1], 1)
        return (f"wave-10 FPCD step {step}: cooldown steps {ns[0] + ns[1]} (regular {ns[0]}, pool {ns[1]}, pool share "
                f"{ns[1] / tot:.4f}); realised feeds/step regular {nf[0] / max(ns[0], 1):.1f} (unmasked {no[0] / max(ns[0], 1):.1f}, "
                f"ratio {nf[0] / max(no[0], 1):.4f}) pool {nf[1] / max(ns[1], 1):.1f} (unmasked {no[1] / max(ns[1], 1):.1f}, ratio "
                f"{nf[1] / max(no[1], 1):.4f}); pool passes started {self.pool_passes}, pool passes per allowed pool feed "
                f"{self.pool_feeds_seen / max(n_pool_allowed, 1):.3f}; split events regular {self.n_split_reg} pool {self.n_split_pool}")


def setup_fpcd(a, tr, feeds, hold, pool, all_batches, steps_per_epoch, t_cool, total, log):
    """Builds the FPCDMask for train.py. Called only when --cd_exclude_days is set."""
    assert a.cooldown_mix > 0 and hold is not None, "FPCD-lite is m02 + one change: needs --cooldown_mix > 0 and --holdout"
    assert a.rec_q == 0 and not a.rho_file and a.val_mix == 0 and not a.hcap, \
        "FPCD-lite: no other cooldown-DATA arm (--rec_q / --rho_file / --val_mix / --hcap) in the same run"
    assert t_cool < total, "--cd_exclude_days: the cooldown never starts (t_cool >= total steps)"
    days = parse_days(a.cd_exclude_days)
    ts = tr.feeds["timestamp"].to_numpy()
    excl, abs_days, dates = excluded_mask(ts, days)
    allowed = feeds[~excl[feeds]]
    assert not np.isin(allowed, hold).any(), "LEAK GUARD: holdout feeds among the FPCD-allowed training feeds"
    assert not np.isin(pool, hold).any(), "LEAK GUARD: holdout feeds in the cooldown pool"
    n_pool_allowed = int((~excl[pool]).sum())
    fp = FPCDMask(excl, all_batches, steps_per_epoch, t_cool, total)
    fp.n_pool_allowed = n_pool_allowed
    sz = np.array([len(b) for b in all_batches[t_cool:]], np.float64)
    msz = np.array([len(b) for b in fp.reg[t_cool:]], np.float64)
    log(f"wave-10 FPCD-lite: --cd_exclude_days {days} = Moscow days {abs_days} {dates} masked out of every cooldown batch from "
        f"step {t_cool} (regular + pool; no redraw, no refill): excluded train feeds {int(excl.sum())} ({excl.mean():.4f} of all), "
        f"training feeds {len(feeds)} -> allowed {len(allowed)} ({len(allowed) / len(feeds):.4f}); pool {len(pool)} -> "
        f"{n_pool_allowed} ({n_pool_allowed / max(len(pool), 1):.4f}); regular cooldown batches {len(sz)}: mean size {sz.mean():.1f} "
        f"-> {msz.mean():.1f} (ratio {msz.sum() / sz.sum():.4f}; min {int(msz.min())}); split events (empty masked batches) "
        f"{fp.n_split_reg}; steps unchanged {total}; passes per allowed feed = epochs; holdout feeds in the allowed set 0")
    return fp
