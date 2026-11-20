import json, sys
rows = []
for f in sys.argv[1:]:
    d = json.load(open(f)); rows += d['rows'] if isinstance(d, dict) else d
print(f"{'set':10s} {'variant':26s} {'dV':>9s} {'zV':>5s} {'dVto':>9s} {'dH':>9s} {'zH':>5s} {'minSl':>6s} acc")
for r in rows:
    sl = r['slices']; ms = min((v[1] for v in sl.values()), default=0)
    mn = min(sl, key=lambda k: sl[k][1])
    dh = r.get('dh'); zh = r.get('z_h')
    print(f"{r['set']:10s} {r['var']:26s} {r['dv']:+.5f} {r['z_v']:+5.1f} {r.get('dvto', float('nan')):+.5f} "
          + (f"{dh:+.5f} {zh:+5.1f}" if dh is not None else f"{'n/a':>9s} {'':5s}") + f" {ms:+5.1f}({mn}) {'ACCEPT' if r['accept'] else '-'}")
