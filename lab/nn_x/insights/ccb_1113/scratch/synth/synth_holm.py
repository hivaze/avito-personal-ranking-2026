"""CCB-1113 synthesiser: apply the pre-declared bar (PREDECLARED_ccb.txt section D) with Holm over the 16 items.

Reads each item's numbers from the testers' own result files (not from the workflow JSON), recomputes
z = val_delta / val_se and the one-sided p = 1 - Phi(z), applies the STOP rules (C14 coverage gate, C15 no test -> p = 1),
runs the Holm step-down at alpha 0.02275, and checks criteria (1) val >= +0.0005, (3) holdout >= 0, (4) late >= 0,
(5) pmix >= -1 SE. Prints one row per item plus batch-level information.
"""
import json, math, os

R = "/tmp/scratchpad/ccb"
ALPHA = 0.02275
NAMES = {
    "C01": "GBDT-OOF leaf-SVD64", "C02": "Streak/EWMA + per-type same-cat recency", "C03": "Latent geometry",
    "C04": "Co-vis variants + NB log-lift", "C05": "IDF/IUF/Swing/BM25 co-vis", "C06": "Bradley-Terry item strength",
    "C07": "Item trend / lifecycle", "C08": "EMDE sketch", "C09": "RP3beta walk", "C10": "User aggregates x cues (F26+F27)",
    "C11": "Typed implicit rating cvx", "C12": "Key-B kappa (residual)", "C13": "Pending favourite",
    "C14": "Cross-feed consensus", "C15": "PLE/PCGrad grad-cos pre-gate", "C16": "REGIME4 Step 0 regcorr",
}


def phi_sf(z):
    return 0.5 * math.erfc(z / math.sqrt(2.0))


items = {}
# tester1: out/results.jsonl, PRIMARY rows
t1 = {"C01_leafsvd64": "C01", "C04_covis_variants_nb_loglift": "C04", "C07_item_trend_lifecycle": "C07",
      "C10_user_agg_interactions": "C10", "C13_pending_fav": "C13", "C16_regcorr": "C16"}
for line in open(f"{R}/tester1/out/results.jsonl"):
    r = json.loads(line)
    if r["primary"] and r["name"] in t1:
        items[t1[r["name"]]] = dict(src="tester1/out/results.jsonl:" + r["name"], ncols=r["ncols"],
                                    d=r["val_delta"], se=r["val_se"], h=r["hold_delta"], hse=r["hold_se"],
                                    late=r["late_delta"], early=r["early_delta"], pm=r["pmix"], pmse=r["pmix_se"])
# tester2: logs/results.jsonl
for line in open(f"{R}/tester2/logs/results.jsonl"):
    r = json.loads(line)
    items[r["id"]] = dict(src="tester2/logs/results.jsonl:" + r["id"], ncols=len(r["theta_full"]),
                          d=r["val_delta"], se=r["val_se"], h=r["hold_delta"], hse=r["hold_se"],
                          late=r["late_delta"], early=r["early_delta"], pm=r["pmix_delta"], pmse=r["pmix_se"],
                          stop=r.get("stop", False), coverage=r.get("coverage"))
# tester3: out/C*.json
for k in ["C03", "C06", "C09", "C12"]:
    r = json.load(open(f"{R}/tester3/out/{k}.json"))
    items[k] = dict(src=f"tester3/out/{k}.json", ncols=r.get("n_cols_lin"), d=r["val_delta"], se=r["val_se"],
                    h=r["hold_delta"], hse=r["hold_se"], late=r["late_delta"], early=r["early_delta"],
                    pm=r["pmix"], pmse=r["pmix_se"])
# C15: no residual; PROCEED flag only
c15 = json.load(open(f"{R}/tester3/c15/c15_gradcos.json"))
items["C15"] = dict(src="tester3/c15/c15_gradcos.json", ncols=None, d=None, se=None, h=None, hse=None, late=None,
                    early=None, pm=None, pmse=None, notest=True, proceed=c15["proceed"])

assert sorted(items) == [f"C{i:02d}" for i in range(1, 17)], sorted(items)

for k, it in items.items():
    if it.get("notest"):
        it["z"] = None; it["p_raw"] = None; it["p"] = 1.0; it["why_p1"] = "no val delta (not a residual)"
        continue
    it["z"] = it["d"] / it["se"]
    it["p_raw"] = phi_sf(it["z"])
    if it.get("stop"):
        it["p"] = 1.0; it["why_p1"] = f"STOP: coverage {it['coverage']:.4f} < 0.10 gate"
    else:
        it["p"] = it["p_raw"]

# Holm step-down over the 16 items
order = sorted(items, key=lambda k: (items[k]["p"], k))
m = len(order)
running, still = 0.0, True
for j, k in enumerate(order, start=1):
    thr = ALPHA / (m - j + 1)
    adj = min(1.0, (m - j + 1) * items[k]["p"])
    running = max(running, adj)
    items[k]["holm_rank"] = j; items[k]["holm_thr"] = thr; items[k]["holm_p"] = running
    still = still and items[k]["p"] <= thr
    items[k]["holm_reject"] = still

for k, it in items.items():
    if it.get("notest"):
        it.update(c1=False, c3=None, c4=None, c5=None)
    else:
        it["c1"] = it["d"] >= 0.0005
        it["c3"] = it["h"] >= 0
        it["c4"] = it["late"] >= 0
        it["c5"] = it["pm"] >= -it["pmse"]
    it["c2"] = it["holm_reject"]
    it["verdict"] = "PASS" if (it["c1"] and it["c2"] and it["c3"] and it["c4"] and it["c5"]) else "CLOSED"
    fails = [n for n, c in (("1", it["c1"]), ("2", it["c2"]), ("3", it["c3"]), ("4", it["c4"]), ("5", it["c5"]))
             if c is False]
    it["fails"] = ",".join(fails)

f = lambda x, n=5: "n/a" if x is None else f"{x:+.{n}f}"
print(f"alpha {ALPHA}, m {m}; Holm order: " + " ".join(f"{k}:{items[k]['p']:.4f}" for k in order))
print("id  | name | ncols | val (SE) | z | p1 raw | p used | Holm p | hold (SE) | late | early | pmix (SE) | fails | verdict")
for k in sorted(items):
    it = items[k]
    val = "n/a" if it["d"] is None else f"{it['d']:+.5f} ({it['se']:.5f})"
    hold = "n/a" if it["h"] is None else f"{it['h']:+.5f} ({it['hse']:.5f})"
    pmx = "n/a" if it["pm"] is None else f"{it['pm']:+.5f} ({it['pmse']:.5f})"
    z = "n/a" if it["z"] is None else f"{it['z']:+.2f}"
    praw = "n/a" if it["p_raw"] is None else f"{it['p_raw']:.4f}"
    print(f"{k} | {NAMES[k]} | {it['ncols']} | {val} | {z} | {praw} | {it['p']:.4f} | {it['holm_p']:.4f} | {hold} | "
          f"{f(it['late'])} | {f(it['early'])} | {pmx} | fails ({it['fails']}) | {it['verdict']}"
          + (f" | {it['why_p1']}" if it.get("why_p1") else ""))

# batch-level information
tested = [k for k in items if not items[k].get("notest") and not items[k].get("stop")]
zs = [items[k]["z"] for k in tested]
print(f"INFO tested (non-STOP) items {len(tested)}: mean z {sum(zs)/len(zs):+.3f}, sum z/sqrt(n) {sum(zs)/math.sqrt(len(zs)):+.3f}, "
      f"z>0: {sum(z > 0 for z in zs)}, max z {max(zs):+.2f} ({max(tested, key=lambda k: items[k]['z'])})")
hs = [k for k in items if items[k]["h"] is not None]
print(f"INFO holdout point estimate < 0 on {sum(items[k]['h'] < 0 for k in hs)} / {len(hs)} items (incl. C14 information);"
      f" late < 0 on {sum(items[k]['late'] < 0 for k in hs)} / {len(hs)}")
print(f"INFO criterion (1) met by {sum(items[k]['c1'] for k in items)} items; pmix veto fires on "
      f"{sum(items[k]['c5'] is False for k in items)}: " + " ".join(k for k in sorted(items) if items[k]['c5'] is False))
print(f"INFO all of (1),(3),(4),(5) ignoring Holm: " + (" ".join(k for k in sorted(items) if items[k]['c1'] and items[k]['c3'] and items[k]['c4'] and items[k]['c5']) or "none"))
nc = [(items[k]["ncols"], items[k]["d"]) for k in tested if items[k]["ncols"]]
mx = sum(a for a, _ in nc) / len(nc); my = sum(b for _, b in nc) / len(nc)
cov = sum((a - mx) * (b - my) for a, b in nc); va = sum((a - mx) ** 2 for a, _ in nc); vb = sum((b - my) ** 2 for _, b in nc)
print(f"INFO corr(LIN ncols, val delta) over {len(nc)} tested items: {cov/math.sqrt(va*vb):+.3f}; "
      + " ".join(f"{k}:{items[k]['ncols']}" for k in sorted(tested)))
ub = {k: items[k]["d"] + 2 * items[k]["se"] for k in tested}
print("INFO val one-sided upper bound d + 2 SE: " + " ".join(f"{k}:{ub[k]:+.5f}" for k in sorted(ub))
      + f" | max {max(ub.values()):+.5f} ({max(ub, key=ub.get)}), median {sorted(ub.values())[len(ub)//2]:+.5f}")
print(f"INFO C15 PROCEED flag {items['C15']['proceed']}; C14 information z {items['C14']['z']:+.2f} (p1 raw {items['C14']['p_raw']:.4f}), coverage {items['C14']['coverage']:.4f}")
print("BATCH VERDICT: PASS " + str(sum(items[k]['verdict'] == 'PASS' for k in items)) + " / CLOSED "
      + str(sum(items[k]['verdict'] == 'CLOSED' for k in items)))
json.dump(items, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "synth_holm.json"), "w"), indent=1, default=str)
