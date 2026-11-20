"""Reviewer's own flag-off identity check (no import of the study's test code).
Constant-folds fd := None and a.holdout_in_train := False in the copy's AST and compares with the original's AST."""
import ast, sys, hashlib
def norm(tree):
    b = tree.body
    if b and isinstance(b[0], ast.Expr) and isinstance(getattr(b[0], "value", None), ast.Constant) and isinstance(b[0].value.value, str):
        tree.body = b[1:]
    return tree
def is_fd(n):
    return isinstance(n, ast.Name) and n.id == "fd"
def fd_is_none(n):   # `fd is None`
    return isinstance(n, ast.Compare) and is_fd(n.left) and len(n.ops) == 1 and isinstance(n.ops[0], ast.Is) and isinstance(n.comparators[0], ast.Constant) and n.comparators[0].value is None
def fd_not_none(n):
    return isinstance(n, ast.Compare) and is_fd(n.left) and len(n.ops) == 1 and isinstance(n.ops[0], ast.IsNot) and isinstance(n.comparators[0], ast.Constant) and n.comparators[0].value is None
def flag(n):
    return isinstance(n, ast.Attribute) and n.attr == "holdout_in_train" and isinstance(n.value, ast.Name) and n.value.id == "a"
CNT = dict(imp=0, addarg=0, fdnone=0, if_flag=0, if_fd=0, ifexp=0, andfd=0)
class Fold(ast.NodeTransformer):
    def visit_If(self, n):
        if fd_not_none(n.test) or flag(n.test):
            assert not n.orelse, "guard with else"
            CNT["if_fd" if fd_not_none(n.test) else "if_flag"] += 1
            return None                      # dead when fd is None / flag False
        self.generic_visit(n); return n
    def visit_IfExp(self, n):
        self.generic_visit(n)
        if fd_is_none(n.test):
            CNT["ifexp"] += 1; return n.body
        return n
    def visit_BoolOp(self, n):
        self.generic_visit(n)
        if isinstance(n.op, ast.And) and any(fd_is_none(v) for v in n.values):
            CNT["andfd"] += 1
            vals = [v for v in n.values if not fd_is_none(v)]
            return vals[0] if len(vals) == 1 else ast.BoolOp(op=n.op, values=vals)
        return n
    def visit_ImportFrom(self, n):
        if n.module == "cd_full":
            CNT["imp"] += 1; return None
        return n
    def visit_Assign(self, n):
        if len(n.targets) == 1 and is_fd(n.targets[0]) and isinstance(n.value, ast.Constant) and n.value.value is None:
            CNT["fdnone"] += 1; return None
        self.generic_visit(n); return n
    def visit_Expr(self, n):
        v = n.value
        if isinstance(v, ast.Call) and isinstance(v.func, ast.Attribute) and v.func.attr == "add_argument" and v.args and isinstance(v.args[0], ast.Constant) and v.args[0].value == "--holdout_in_train":
            CNT["addarg"] += 1
            assert any(k.arg == "action" and k.value.value == "store_true" for k in v.keywords)
            return None
        self.generic_visit(n); return n
ok = True
for orig, copy in ((sys.argv[1], sys.argv[2]), (sys.argv[3], sys.argv[4])):
    for k in CNT: CNT[k] = 0
    to = norm(ast.parse(open(orig).read())); tc = norm(ast.parse(open(copy).read()))
    tc = ast.fix_missing_locations(Fold().visit(tc))
    do, dc = ast.dump(to), ast.dump(tc)
    left = [n for n in ast.walk(tc) if is_fd(n) or flag(n)]
    same = do == dc
    full = (CNT['imp'], CNT['addarg'], CNT['fdnone'], CNT['if_flag']) == (1, 1, 1, 1) and CNT['if_fd'] >= 2
    ok &= same and not left and full
    print(f"{copy.split('/')[-1]} vs {orig}: AST equal after folding = {same}; residual fd/flag references = {len(left)}; folded {CNT}; orig md5 {hashlib.md5(open(orig,'rb').read()).hexdigest()} copy md5 {hashlib.md5(open(copy,'rb').read()).hexdigest()}")
print("ALL PASS" if ok else "FAIL")
