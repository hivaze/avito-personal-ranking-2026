import json, sys
for t in sys.argv[1:]:
    d = json.load(open(t))["res"]
    print(t)
    for b, r in d.items():
        f = lambda k: (f"{r[k][0]:+.5f} ±{r[k][1]:.5f} (z {r[k][2]:+.1f})" if k in r else "   n/a   ")
        print(f"  {b:12s} GBM val {f('gbm_val')} hold {f('gbm_hold')} ch {r.get('gbm_choice')} | ridge val {f('ridge_val')} hold {f('ridge_hold')} ch {r.get('ridge_choice')}")
