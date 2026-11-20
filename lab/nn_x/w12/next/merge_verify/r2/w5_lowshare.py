"""w5: own computation of the reader's low-agreement share (test, per-feed Spearman of a genuine partner against the W8 order) and of the same share in the feeds the tool does not look at (M < 20)."""
import sys
sys.path.insert(0, '<repo>/nn_x/w12/next/merge_verify/r2')
from w_common import *
V, T = Split('val'), Split('test'); C = partner_table(V, T); w8 = T.z(T.rank(T.z(fin(T, 'W8kd')))); big = T.M >= 20
print(f'{"partner":9s} | mean Spearman | M >= 20: share < 0.3 | M < 20: share < 0.3, share < 0, share < -0.5 | PUBLIC M >= 20 share < 0.3')
for nm, (_, pt) in C.items():
    r = T.rf(w8, T.z(T.rank(pt)))
    print(f'{nm:9s} | {r.mean():.4f}        | {(r[big] < 0.3).mean():.3%}               | {(r[~big] < 0.3).mean():.3%}, {(r[~big] < 0).mean():.3%}, {(r[~big] < -0.5).mean():.3%}            | {(r[big & T.public] < 0.3).mean():.3%}')
