import json, numpy as np
R = json.load(open("out/r7_lgb.json"))
for kind in ("WF", "STRAT"):
    P = [p for p in R["placebo"] if p["kind"] == kind]
    if not P: continue
    print(f"{kind}: {len(P)} draws")
    for r in ("5", "10", "20", "50", "100", "200"):
        g = np.array([p["A_minus_ctl"][r][0] for p in P]); real = R["real"][r]["A_minus_ctl"][0]
        gp = np.array([p["A_minus_pool"][r][0] for p in P])
        print(f"  rounds {r:>3s}: real A-ctl {real:+.5f} | placebo mean {g.mean():+.5f} sd {g.std(ddof=1):.5f} min {g.min():+.5f} max {g.max():+.5f} | real above {(real > g).sum()} of {len(g)} | (real-mean)/sd {(real - g.mean()) / g.std(ddof=1):+.2f} | placebo A-pool mean {gp.mean():+.5f}")
