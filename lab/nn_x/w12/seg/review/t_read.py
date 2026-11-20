"""T read (DECLARATION block 0): cost of truncating the history to the newest K events, by H group."""
from rv_lib import *

TD = OUT / "trunc"
GROUPS = [("cap (H = 256)", CAP), ("NEAR with H 193-255", (H >= 193) & (H <= 255)), ("H 129-192", (H >= 129) & (H <= 192)), ("H 65-128", (H >= 65) & (H <= 128))]
for fam, members in (("wide distilled students (FINAL's recipe, held-out)", ["w5_e10_kd05w192_s951", "w5_e10_kd05w192_s952"]), ("hard-label members", ["w5_e10_base_s951", "w5_e10_base_s952"])):
    if not all((TD / f"{v}~K{K}.npy").exists() for v in members for K in (256, 192, 128, 64)):
        print(fam, ": files not complete yet")
        continue
    ref = {v: ndcg(np.load(TD / f"{v}~K256.npy")) for v in members}
    st = {v: ndcg(np.load(P / f"nn_{v}_trainonly_val.npy")) for v in members}
    print(f"\n### {fam}: ref (CPU fp32) NDCG " + ", ".join(f"{v} {ref[v].mean():.5f} (stored GPU file {st[v].mean():.5f})" for v in members))
    rows = []
    cost = {}
    for K in (192, 128, 64):
        cost[K] = np.mean([ndcg(np.load(TD / f"{v}~K{K}.npy")) - ref[v] for v in members], 0)
        r = [f"K = {K}"]
        for gn, g in GROUPS:
            mu, se, n = cmean(cost[K], g)
            r.append(f"{mu:+.5f}+-{se:.5f} (z {mu / se if se > 0 else 0:+.1f})")
        mu, se, _ = cmean(cost[K])
        r.append(f"{mu:+.5f}+-{se:.5f}")
        rows.append(r)
    print(tab(rows, ["newest K kept"] + [g for g, _ in GROUPS] + ["ALL feeds (pool-level cost)"]))
    # per member at the cap
    for v in members:
        print("   per member at the cap:", v, " ".join(f"K{K} {(ndcg(np.load(TD / f'{v}~K{K}.npy')) - ref[v])[CAP].mean():+.5f}" for K in (192, 128, 64)))
    # marginal value per 64 events at the cap: 256->192, 192->128, 128->64
    seg = [("events 193-256 (oldest quarter)", cost[192]), ("events 129-192", cost[128] - cost[192]), ("events 65-128", cost[64] - cost[128])]
    print("   marginal worth of each 64-event stretch at the cap (NDCG lost when it is removed, given everything newer is kept):")
    for nm, c in seg:
        mu, se, _ = cmean(c, CAP)
        print(f"      {nm}: {mu:+.5f} +- {se:.5f}")
    # split of the cap by label-free user kind (seen in train) and by the share of positives whose best match is old: descriptive
    seen = np.load(OUT / "seen_train.npy")
    for nm, g in (("cap, user seen in train", CAP & seen), ("cap, user not seen in train", CAP & ~seen)):
        print(f"   {nm}: " + " ".join("K%d %+.5f+-%.5f" % ((K,) + cmean(cost[K], g)[:2]) for K in (192, 128, 64)))
    # 2-member pool form (z-mean of the two members) at the cap
    pr = ndcg(np.mean([z(np.load(TD / f"{v}~K256.npy")) for v in members], 0))
    for K in (192, 128, 64):
        pk = ndcg(np.mean([z(np.load(TD / f"{v}~K{K}.npy")) for v in members], 0))
        mu, se, _ = cmean(pk - pr, CAP)
        print(f"   2-member pool, K = {K}, cap: {mu:+.5f} +- {se:.5f}")
