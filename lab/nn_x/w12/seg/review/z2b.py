"""Z2b (DECLARATION block 4): fixed w_ft contrasts of the final composition on the cap bucket, two member sets."""
from rv_lib import *
def branches(base, pop, w=0.2):
    out = []
    for suf in ("n2cpv3s3_val_oof", "trainonly_val"):
        out.append(z((1 - w) * z(pool(base, suf)) + w * z(pool(pop, suf))))
    return out
rows = []
for n, base, pop in (("FD16P20", FD16, FDP), ("W16P20w (held-out twin)", W16, P1W)):
    ft, to = branches(base, pop)
    e5 = ndcg(0.5 * ft + 0.5 * to)
    for w in (0.3, 0.0, 0.7):
        d = ndcg(w * ft + (1 - w) * to) - e5
        a, sa, _ = cmean(d, CAP); b, sb, _ = cmean(d, REST); m, sm, _ = cmean(d); i, si = cdiff(d, CAP, REST)
        rows.append([n, f"w_ft {w} minus 0.5", f"{m:+.5f}+-{sm:.5f}", f"{a:+.5f}+-{sa:.5f} (z {a / sa:+.2f})", f"{b:+.5f}+-{sb:.5f} (z {b / sb:+.2f})", f"{i:+.5f} (z {i / si:+.2f})"])
    # the gated form implied by the cap contrast: 0.3 at the cap, 0.5 elsewhere
    wrow = np.where(CAP[FID], 0.3, 0.5)
    d = ndcg(wrow * ft + (1 - wrow) * to) - e5
    m, sm, _ = cmean(d)
    rows.append([n, "gated: 0.3 at the cap, 0.5 elsewhere (in sample, not a witness)", f"{m:+.5f}+-{sm:.5f}", "", "", ""])
print(tab(rows, ["composition", "contrast", "ALL", "cap", "H < 256", "cap minus rest"]))
