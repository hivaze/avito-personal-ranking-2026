"""C14 cross-feed prediction consensus (PREDECLARED_ccb.txt §C14, B9).
val: cons_i = mean over OTHER users' val feeds of strictly earlier days containing item i of zf(ens_G3_to_val)[i] (test never a
source; user 0 = pooled bucket, not skipped); holdout: other holdout feeds of days < d, b_h. n = # such feeds.
cons_shr = cons * n/(n+5) (0 if n = 0), lcnt = log1p(n). Coverage gate: STOP if rows with n >= 1 < 10% of val rows.
"""
import sys
sys.path.insert(0, "/tmp/scratchpad/ccb/tester2")
from harness import *  # noqa: F401,F403

LOG = open(HERE / "logs/c14.log", "a")
log = log_to(LOG)
log("==== C14 cross-feed consensus", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
sanity(log)


def consensus(item, feed, day, user, z):
    """per row: (sum, n) over feeds g with day_g < day_f, (user_g != user_f or user_f == 0), containing item."""
    df = pl.DataFrame(dict(item=item, feed=feed, day=day, user=user, z=z, row=np.arange(len(item))))
    # one value per (feed, item)
    fi = df.group_by(["feed", "item"]).agg(pl.col("z").mean(), pl.col("day").first(), pl.col("user").first())
    S = np.zeros(len(item)); N = np.zeros(len(item))
    for d in np.unique(day):
        q = df.filter(pl.col("day") == d)
        src = fi.filter(pl.col("day") < d)
        if len(src) == 0:
            continue
        a = src.group_by("item").agg(s=pl.col("z").sum(), n=pl.len())
        au = src.filter(pl.col("user") != 0).group_by(["item", "user"]).agg(su=pl.col("z").sum(), nu=pl.len())
        j = q.join(a, on="item", how="left").join(au, on=["item", "user"], how="left").with_columns(
            pl.col("s").fill_null(0.0), pl.col("n").fill_null(0), pl.col("su").fill_null(0.0), pl.col("nu").fill_null(0))
        j = j.with_columns(su=pl.when(pl.col("user") == 0).then(0.0).otherwise(pl.col("su")),
                           nu=pl.when(pl.col("user") == 0).then(0).otherwise(pl.col("nu")))
        r = j["row"].to_numpy()
        S[r] = j["s"].to_numpy() - j["su"].to_numpy(); N[r] = j["n"].to_numpy() - j["nu"].to_numpy()
    cons = np.where(N > 0, S / np.maximum(N, 1), 0.0)
    return cons * N / (N + 5.0), np.log1p(N), N


item_v = np.load(FLAT / "val/cand_item_id.npy")
zto = zv(np.load(ROOT / "preds/ens_G3_to_val.npy"))
cs_v, lc_v, N_v = consensus(item_v, fid, day_v[fid], user_v[fid], zto)
cov = float((N_v >= 1).mean())
log(f"C14 COVERAGE (gate, printed first): val rows with n >= 1: {cov:.4%} ({int((N_v >= 1).sum())} / {len(N_v)}); "
    f"by val day: " + ", ".join(f"{d}: {(N_v[day_v[fid] == d] >= 1).mean():.4%}" for d in np.unique(day_v)))
STOP = cov < 0.10
log(f"C14 gate: {'STOP (coverage < 10%) -> p = 1, CLOSED; the LIN numbers below are INFORMATION only' if STOP else 'PASS -> test'}")
item_h = np.asarray(np.load(FLAT / "train/cand_item_id.npy", mmap_mode="r")[HOLD_ROWS])
cs_h, lc_h, N_h = consensus(item_h, fidh, HOLD_DAY_ABS[fidh], HOLD_USER[fidh], bh)
log(f"C14 holdout rows with n >= 1: {(N_h >= 1).mean():.4%}; late {(N_h[H_LATE[fidh]] >= 1).mean():.4%}")
np.save(FEATD / "c14_val.npy", np.stack([cs_v, lc_v], 1).astype(np.float32))
np.save(FEATD / "c14_hold.npy", np.stack([cs_h, lc_h], 1).astype(np.float32))
res = evaluate("C14", ["cons_shr", "lcnt"], [cs_v, lc_v], [cs_h, lc_h], log,
               extra=dict(coverage=cov, stop=bool(STOP), status="STOP p=1 CLOSED (information numbers)" if STOP else "tested"))
with open(HERE / "logs/results.jsonl", "a") as fh:
    fh.write(json.dumps(res) + "\n")
log("DONE")
