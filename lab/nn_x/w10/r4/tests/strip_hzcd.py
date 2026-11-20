"""Shared AST stripper of the hzcd statements of nn_x/w5/train.py (--hzcd, cd_arms6.py; added 2026-11-12 ~22:10 UTC).
Used by tests/test_hzcd.py (H0: train.py minus these statements == tests/ref_pre_hzcd/train.py.txt) and applied FIRST by
tests/test_w5.py and tests/test_trunc_lifts.py, so that their own flag-off proofs against their frozen copies keep holding
(chain of proofs: new == pre_hzcd + hzcd; pre_hzcd == pre_tl + trunc_lifts; pre_tl == nn_x/r6 + wave-5).
Removes exactly: `from cd_arms6 import ...`, the --hzcd add_argument, the `tr_hz = None` init, every `if a.hzcd:` block
(setup, the holdout-hz file, the release before the test split loads) and replaces the gather conditional expression
`tr_hz if tr_hz is not None and step >= t_cool else X` by X. Counts what it removed (EXPECT) so the tests assert the exact set.
"""
import ast
import collections

HZ_IF = "a.hzcd"
HZ_IFEXP = "tr_hz is not None and step >= t_cool"
EXPECT = collections.Counter({"import cd_arms6": 1, "add_argument --hzcd": 1, "tr_hz = None": 1, "if a.hzcd": 3,
                              "ifexp tr_hz": 1})


class StripHzcd(ast.NodeTransformer):
    def __init__(self):
        self.removed = collections.Counter()

    def visit_ImportFrom(self, n):
        if n.module == "cd_arms6":
            self.removed["import cd_arms6"] += 1
            return None
        return n

    def visit_Expr(self, n):
        v = n.value
        if (isinstance(v, ast.Call) and isinstance(v.func, ast.Attribute) and v.func.attr == "add_argument" and v.args
                and isinstance(v.args[0], ast.Constant) and v.args[0].value == "--hzcd"):
            self.removed["add_argument --hzcd"] += 1
            return None
        return self.generic_visit(n)

    def visit_Assign(self, n):
        if ast.unparse(n) == "tr_hz = None":
            self.removed["tr_hz = None"] += 1
            return None
        return self.generic_visit(n)

    def visit_If(self, n):
        if ast.unparse(n.test) == HZ_IF:
            self.removed["if a.hzcd"] += 1
            return None
        return self.generic_visit(n)

    def visit_IfExp(self, n):
        if ast.unparse(n.test) == HZ_IFEXP:
            self.removed["ifexp tr_hz"] += 1
            return self.visit(n.orelse)
        return self.generic_visit(n)


def strip_hzcd(tree):
    """(tree without the hzcd statements, Counter of what was removed)."""
    st = StripHzcd()
    return st.visit(tree), st.removed
