"""s0: build the per-pool score caches, verify the builders against the stored arrays and the recorded val / holdout numbers."""
from lib import *  # noqa

f, M, off = meta("val"); y = val_labels()
ft, Mt, ot = meta("test")
# 1. member-equal builder reproduces ens_G3_final (val, test)
s, _, _ = member_pool(MEMBERS["G3"], "val")
ref = np.load(P / "ens_G3_final_val.npy")
print("G3 rebuilt vs ens_G3_final_val: corr", np.corrcoef(s, ref)[0, 1], " ndcg rebuilt", ndcg_per_feed(s.astype(np.float32), y, off, M).mean(),
      "ndcg stored", ndcg_per_feed(ref, y, off, M).mean(), "ndcg pool_scores", perfeed_val("G3").mean())
st, _, _ = member_pool(MEMBERS["G3"], "test")
print("G3 rebuilt test corr with ens_G3_final_test", np.corrcoef(st, np.load(P / "ens_G3_final_test.npy"))[0, 1])
# 2. all pools: val NDCG vs recorded
print("\npool    n_mem  recool_share  m02_share  val_ndcg  recorded")
for n in SCORED + UNSCORED + ["v10", "v5"]:
    try:
        nd = perfeed_val(n)
    except FileNotFoundError as e:
        print(n, "MISSING", e); continue
    ms = MEMBERS.get(n, [])
    rc = np.mean([m in RECOOL for m in ms]) if ms else float("nan")
    m02 = np.mean([m in M02 for m in ms]) if ms else float("nan")
    print(f"{n:7s} {len(ms):5d}  {rc:11.2f}  {m02:9.2f}  {nd.mean():.5f}  {VAL_RECORDED.get(n, float('nan')):.5f}")
# S1 alternative: TO-only hfit
s1to = zf(off, M, np.load(P / "ens_hfitP11W1to_to_val.npy"))
print("S1 to-only val", ndcg_per_feed(s1to.astype(np.float32), y, off, M).mean(), " S1 ft-only",
      ndcg_per_feed(zf(off, M, np.load(P / 'ens_hfitP11W1to_ft_val.npy')).astype(np.float32), y, off, M).mean())
# 3. test caches incl. old subs
for n in SCORED + UNSCORED + OLD:
    try:
        s = pool_scores(n, "test"); print(f"test cache {n:7s} ok  len {len(s)}  nan {np.isnan(s).sum()}")
    except Exception as e:
        print("test", n, "FAILED", repr(e)[:200])
# 4. holdout
print("\nholdout per pool (train-only members):")
for n in SCORED + UNSCORED:
    nd = perfeed_holdout(n)
    print(f"  {n:7s} {'n/a' if nd is None else f'{nd.mean():.5f}'}  recorded {HOLD_RECORDED.get(n, float('nan')):.5f}")
