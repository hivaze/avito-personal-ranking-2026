"""R1a / R1b: my own textual + AST comparison of cd_arms6c vs cd_arms6b. Read-only."""
import ast, difflib, hashlib, json, sys
R = '<repo>'
A, B = f'{R}/nn_x/w5/cd_arms6b.py', f'{R}/nn_x/w12/stale/hz2n/cd_arms6c.py'
B2 = f'{R}/nn_x/w5/cd_arms6c.py'
sa, sb = open(A).read(), open(B).read()
md5 = lambda p: hashlib.md5(open(p, 'rb').read()).hexdigest()
res = {'md5_6b': md5(A), 'md5_6c_builder': md5(B), 'md5_6c_w5': md5(B2), 'md5_equal_copy': md5(B) == md5(B2)}
ta, tb = ast.parse(sa), ast.parse(sb)
la, lb = sa.splitlines(), sb.splitlines()
# --- R1a: region of each changed line (in the 6c numbering for + lines, 6b numbering for - lines)
def regions(tree, lines):
    reg = {}
    for i, n in enumerate(tree.body):
        if i == 0 and isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant):
            name = 'docstring'
        elif isinstance(n, ast.FunctionDef) or isinstance(n, ast.ClassDef):
            name = f'def:{n.name}'
        elif isinstance(n, ast.Assign):
            name = 'assign:' + ','.join(t.id for t in n.targets if isinstance(t, ast.Name))
        else:
            name = type(n).__name__
        for ln in range(n.lineno, n.end_lineno + 1):
            reg[ln] = name
    return reg
ra, rb = regions(ta, la), regions(tb, lb)
chg = {'minus': {}, 'plus': {}}
sm = difflib.SequenceMatcher(None, la, lb, autojunk=False)
for tag, i1, i2, j1, j2 in sm.get_opcodes():
    if tag == 'equal':
        continue
    for i in range(i1, i2):
        chg['minus'].setdefault(ra.get(i + 1, 'between(comment/blank)'), []).append(i + 1)
    for j in range(j1, j2):
        chg['plus'].setdefault(rb.get(j + 1, 'between(comment/blank)'), []).append(j + 1)
res['changed_lines_by_region'] = chg
# comment lines between statements in 6c: which top-level statement follows them
allowed = {'docstring', 'assign:ENC_TO_HZ', 'assign:ENC_GROUPS', 'def:luo_cols', 'def:setup_hzcd', 'between(comment/blank)'}
res['R1a_regions_touched'] = sorted(set(chg['minus']) | set(chg['plus']))
res['R1a_pass'] = set(res['R1a_regions_touched']) <= allowed
res['between_lines_text'] = [lb[j - 1] for j in chg['plus'].get('between(comment/blank)', [])]
# --- R1b: AST
def top(tree):
    out = {}
    for i, n in enumerate(tree.body):
        if i == 0 and isinstance(n, ast.Expr):
            key = 'docstring'
        elif isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            key = f'def:{n.name}'
        elif isinstance(n, ast.Assign):
            key = 'assign:' + ','.join(t.id for t in n.targets if isinstance(t, ast.Name))
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            key = 'import:' + ','.join(a.name for a in n.names)
        else:
            key = f'{type(n).__name__}@{i}'
        assert key not in out, key
        out[key] = n
    return out
da, db = top(ta), top(tb)
res['top_only_in_6c'] = sorted(set(db) - set(da))
res['top_only_in_6b'] = sorted(set(da) - set(db))
res['top_order_6b'] = list(da)
res['top_order_6c_minus_new'] = [k for k in db if k in da]
res['top_order_same'] = res['top_order_6b'] == res['top_order_6c_minus_new']
diff = [k for k in da if k in db and ast.dump(da[k]) != ast.dump(db[k])]
res['top_differing'] = diff
# strip the '# 6c' statements of luo_cols / setup_hzcd and compare
def strip6c(fn, lines):
    """remove every statement (at any nesting depth) all of whose first source line carries '# 6c'; returns a new FunctionDef."""
    import copy
    fn = copy.deepcopy(fn)
    removed = []
    def walk(body):
        keep = []
        for st in body:
            if '# 6c' in lines[st.lineno - 1]:
                removed.append((st.lineno, st.end_lineno, lines[st.lineno - 1].strip()[:110]))
                continue
            for attr in ('body', 'orelse', 'finalbody'):
                if hasattr(st, attr) and isinstance(getattr(st, attr), list):
                    setattr(st, attr, walk(getattr(st, attr)))
            keep.append(st)
        return keep
    fn.body = walk(fn.body)
    return fn, removed
out = {}
for name in ('luo_cols', 'setup_hzcd'):
    f6c, removed = strip6c(db[f'def:{name}'], lb)
    d6b, d6c = ast.dump(da[f'def:{name}']), ast.dump(f6c)
    eq = d6b == d6c
    resid = None
    if not eq:
        # locate the residual difference at statement level
        resid = []
        for x, y in zip(ast.walk(da[f'def:{name}']), ast.walk(f6c)):
            if type(x) is not type(y):
                resid.append((type(x).__name__, type(y).__name__)); break
            if isinstance(x, ast.Constant) and x.value != y.value:
                resid.append((repr(x.value)[:160], repr(y.value)[:160]))
            if isinstance(x, ast.Name) and x.id != y.id:
                resid.append((x.id, y.id))
        n6b, n6c = len(list(ast.walk(da[f'def:{name}']))), len(list(ast.walk(f6c)))
        resid.append(('n_nodes', n6b, n6c))
    out[name] = {'removed_6c_statements': removed, 'identical_after_strip': eq, 'residual': resid}
res['strip'] = out
res['R1b_pass'] = (res['top_only_in_6c'] == ['assign:ENC_GROUPS', 'assign:ENC_TO_HZ'] and not res['top_only_in_6b'] and res['top_order_same']
                   and set(diff) <= {'docstring', 'def:luo_cols', 'def:setup_hzcd'})
json.dump(res, open(f'{R}/nn_x/w12/stale/hz2n_review/r1_ast.json', 'w'), indent=1)
print(json.dumps(res, indent=1))
