"""TN read (DECLARATION block 2): cost of removing only the EMBEDDINGS of the events at index >= K (length signals kept)."""
from rv_lib import *
TD = OUT / "trunc"
seen = np.load(OUT / "seen_train.npy")
for fam, members in (("wide distilled students (FINAL's recipe, held-out)", ["w5_e10_kd05w192_s951", "w5_e10_kd05w192_s952"]), ("hard-label members", ["w5_e10_base_s951", "w5_e10_base_s952"])):
    ref = {v: ndcg(np.load(TD / f"{v}~K256.npy")) for v in members}
    print(f"\n### {fam}")
    rows = []
    for K in (192, 128):
        cN = np.mean([ndcg(np.load(TD / f"{v}~N{K}.npy")) - ref[v] for v in members], 0)
        cT = np.mean([ndcg(np.load(TD / f"{v}~K{K}.npy")) - ref[v] for v in members], 0)
        r = [f"K = {K}"]
        for nm, g in (("cap", CAP), ("NEAR, H 193-255", (H >= 193) & (H <= 255)), ("cap, seen in train", CAP & seen), ("cap, not seen", CAP & ~seen)):
            a, sa, _ = cmean(cN, g); b, sb, _ = cmean(cT, g)
            r += [f"{a:+.5f}+-{sa:.5f} (z {a / sa:+.1f})", f"{b:+.5f}+-{sb:.5f}"]
        d, sd, _ = cmean(cT - cN, CAP)
        r.append(f"{d:+.5f}+-{sd:.5f}")
        rows.append(r)
        print("   per member at the cap, N%d: " % K + " ".join(f"{v} {(ndcg(np.load(TD / f'{v}~N{K}.npy')) - ref[v])[CAP].mean():+.5f}" for v in members))
    print(tab(rows, ["events at index >= K", "cap: embeddings removed (TN)", "cap: truncated (T)", "NEAR 193-255: TN", "NEAR 193-255: T", "cap seen: TN", "cap seen: T", "cap not seen: TN", "cap not seen: T", "cap: T minus TN (length-signal part)"]))
