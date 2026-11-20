"""Z2 replication across member sets: FT OOF branch minus train-only branch, cap vs rest."""
from rv_lib import *
sets = {"FD16 (full-data students)": FD16, "W16 (held-out twins)": W16, "FD pop-light 4": FDP, "held-out pop-light 4": P1W,
        "FD16 kd05w192 half": FD16[:8], "FD16 kd05t25w192 half": FD16[8:], "W16 kd05w192 half": W16[:8], "W16 kd05t25w192 half": W16[8:]}
rows = []
for n, mem in sets.items():
    g = ndcg(pool(mem, "n2cpv3s3_val_oof")) - ndcg(pool(mem, "trainonly_val"))
    a, sa, _ = cmean(g, CAP); b, sb, _ = cmean(g, REST); d, se = cdiff(g, REST, CAP); m, sm, _ = cmean(g)
    rows.append([n, f"{m:+.5f}+-{sm:.5f}", f"{a:+.5f}+-{sa:.5f}", f"{b:+.5f}+-{sb:.5f}", f"{d:+.5f}+-{se:.5f} (z {d / se:+.2f})"])
g = ndcg(np.load(P / "ens_G3_ft_val.npy")) - ndcg(np.load(P / "ens_G3_to_val.npy"))
a, sa, _ = cmean(g, CAP); b, sb, _ = cmean(g, REST); d, se = cdiff(g, REST, CAP); m, sm, _ = cmean(g)
rows.append(["G3 file (hard labels)", f"{m:+.5f}+-{sm:.5f}", f"{a:+.5f}+-{sa:.5f}", f"{b:+.5f}+-{sb:.5f}", f"{d:+.5f}+-{se:.5f} (z {d / se:+.2f})"])
print(tab(rows, ["member set", "FT - TO all", "cap", "H < 256", "contrast (H < 256) - cap"]))
