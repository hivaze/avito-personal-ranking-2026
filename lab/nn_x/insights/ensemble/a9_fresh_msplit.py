"""A9: pre-declared confirmation of the M-split rule (findings.txt E1) on members NOT used when the rule was formed.
Usage: a9_fresh_msplit.py member1 member2 ...  (each must have trainonly + n2cpv3 val preds)
Reports per member and for the pool of the given members: delta of [TO-only for M<50, 0.5/0.5 for M>=50] vs standard."""
import sys
from lib import *  # noqa

ms = sys.argv[1:]
vm = meta("val"); small = vm["M"][vm["fid"]] < 50
for m in ms:
    zt = member_z(m, "val", "to").astype(np.float64); zf = member_z(m, "val", "ft").astype(np.float64)
    base = pf(0.5 * zt + 0.5 * zf); w = np.where(small, 0.0, 0.5)
    print(f"{m:24s} M-split vs standard: {fmt(pf(w * zf + (1 - w) * zt) - base)} | FT pool-level (final - TO) {fmt(base - pf(zt))}")
if len(ms) > 1:
    zt = zfeed(branch(ms, "val", "to"), "val"); zf = zfeed(branch(ms, "val", "ft"), "val")
    base = pf(0.5 * zt + 0.5 * zf); w = np.where(small, 0.0, 0.5)
    print(f"POOL of {len(ms)} fresh members: M-split vs standard {fmt(pf(w * zf + (1 - w) * zt) - base)}")
