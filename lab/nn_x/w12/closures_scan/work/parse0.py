import re, collections, json
p='<repo>/docs/07-what-failed.md'
L=open(p).read().split('\n')
rows=[];theme=None;declared={}
for i,l in enumerate(L,1):
    if i<318: continue
    m=re.match(r'^### (.*) \((\d+)\)$',l)
    if m: theme=m.group(1); declared[theme]=int(m.group(2)); continue
    if l.startswith('| ') and not l.startswith('| Idea |') and not l.startswith('|---'):
        cells=[c.strip() for c in l.strip().strip('|').split(' | ')]
        if len(cells)!=4: print('BAD',i,len(cells),l[:100])
        rows.append(dict(line=i,theme=theme,idea=cells[0],outcome=cells[1],effect=cells[2],where=cells[3]))
print(len(rows))
c=collections.Counter(r['outcome'] for r in rows); print(c)
ct=collections.Counter(r['theme'] for r in rows)
for t,n in declared.items(): print(t,n,ct[t], 'MISMATCH' if n!=ct[t] else '')
json.dump(rows,open('<repo>/lab/nn_x/w12/closures_scan/work/rows.json','w'),indent=0)
