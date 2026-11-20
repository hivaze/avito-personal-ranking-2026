"""Apply the pre-declared PSI exclusion rule (plan.txt amendment) to data/groups.json using out/psi_w25.json:
PSI(train,val) > 0.5 or PSI(val,test) > 0.5 or |NaN share train - val| > 0.5 -> moved to G_excl (not used by A2 / ablations)."""
import json
from pathlib import Path
WD = Path("<repo>/nn_x/w25/data"); OUT = Path("<repo>/nn_x/w25/out")
G = json.loads((WD / "groups.json").read_text()); P = json.loads((OUT / "psi_w25.json").read_text())
excl = sorted(n for n, r in P.items() if (r["psi_tr_val"] > 0.5) or (r["psi_val_test"] > 0.5) or abs(r["nan_tr"] - r["nan_val"]) > 0.5)
for g in list(G.keys()):
    if g == "G_excl":
        continue
    G[g] = [n for n in G[g] if n not in excl]
G["G_excl"] = excl
(WD / "groups.json").write_text(json.dumps(G, indent=1))
print("excluded:", excl)
print({k: len(v) for k, v in G.items()})
