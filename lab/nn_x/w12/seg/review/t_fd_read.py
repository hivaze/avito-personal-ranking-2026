"""T-FD read (DECLARATION block 6) + pooled read over all six members."""
from rv_lib import *
TD = OUT / "trunc"
FDm = ["w5_fd10_kd05w192_s951", "w5_fd10_kd05w192_s952"]
ALL6 = FDm + ["w5_e10_kd05w192_s951", "w5_e10_kd05w192_s952", "w5_e10_base_s951", "w5_e10_base_s952"]
def cost(members, tag):
    return np.mean([ndcg(np.load(TD / f"{v}~{tag}.npy")) - ndcg(np.load(TD / f"{v}~K256.npy")) for v in members], 0)
for nm, mem in (("FINAL's own members (w5_fd10_kd05w192 s951, s952)", FDm), ("all six members (2 full-data students, 2 held-out students, 2 hard-label)", ALL6)):
    print(f"\n### {nm}")
    rows = []
    for tag in ("K192", "K128", "K64", "N192", "N128"):
        c = cost(mem, tag)
        r = [("truncate to newest " if tag[0] == "K" else "embeddings removed at index >= ") + tag[1:]]
        for g in (CAP, (H >= 193) & (H <= 255), np.ones(NF, bool)):
            mu, se, _ = cmean(c, g)
            r.append(f"{mu:+.5f}+-{se:.5f} (z {mu / se:+.1f})")
        rows.append(r)
    print(tab(rows, ["intervention", "cap", "H 193-255", "ALL feeds"]))
    print("   per member at the cap, K192:", " ".join(f"{v.split('_', 1)[1]} {cost([v], 'K192')[CAP].mean():+.5f}" for v in mem))
    c192, c128, c64 = cost(mem, "K192"), cost(mem, "K128"), cost(mem, "K64")
    for lab, c in (("events 193-256", c192), ("events 129-192", c128 - c192), ("events 65-128", c64 - c128)):
        mu, se, _ = cmean(c, CAP)
        print(f"   marginal worth at the cap, {lab}: {mu:+.5f} +- {se:.5f}")
    d, se = cdiff(c192, CAP, (H >= 193) & (H <= 255))
    print(f"   K192 cost, cap minus H 193-255: {d:+.5f} +- {se:.5f}")
