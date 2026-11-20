"""Re-prints the one-line nats.py output from the saved reads/nats_<tag>.json (same format string as nats.screen) and the val M>=50 line of fs_m50.py.
usage: python fs_lines.py <tag> [<tag> ...]"""
import sys, json
from pathlib import Path
H = Path("<repo>/nn_x/w12/cb/famS/reads")
for t in sys.argv[1:]:
    out = json.load(open(H / f"nats_{t}.json")); v, h = out["val"], out["hold"]
    print(f"{t}: level {v['level']:.5f} / {h['level']:.5f} (W16 {v['w16']:.5f} / {h['w16']:.5f}) | rho {v['rho']:.3f} / {h['rho']:.3f} | NATS {v['nats']:+.5f}+-{v['nats_se']:.5f} / {h['nats']:+.5f}+-{h['nats_se']:.5f}"
          f" (beta_X z {v['beta_x_z']:+.1f} / {h['beta_x_z']:+.1f}) -> {'PROMISING' if out['promising'] else 'not promising'} (bar 0.0016 on both)"
          f" | member rule 0.9/0.1: val {v['mix01']:+.5f}+-{v['mix01_se']:.5f}, holdout {h['mix01']:+.5f}+-{h['mix01_se']:.5f}, pooled z {out['pooled_z']:+.2f} -> {'PASS' if out['member_pass'] else 'no pass'}")
