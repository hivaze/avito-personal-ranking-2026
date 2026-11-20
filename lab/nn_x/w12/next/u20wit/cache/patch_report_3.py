p = '<repo>/reports/NEXT_1115/u20wit.md'
s = open(p).read()
a = "I have no mechanism, and each gap is within 2 SE (B - A: 2.0 SE)."
b = "I have no mechanism; the gaps are 2.0 SE (B - A: -0.00172 ± 0.00084), 1.0 SE (C - A) and 1.1 SE (base - xf0)."
assert s.count(a) == 1; s = s.replace(a, b); open(p, 'w').write(s); print('ok')
