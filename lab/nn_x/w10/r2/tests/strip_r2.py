"""AST stripper of the wave-10 R2 XOFF statements of nn_x/w10/r2/train.py (--xoff / --xoff_table, cd_arms10.py; added 2026-11-13
~16:55 UTC). Used by tests/test_r2.py (C0b: train.py minus these statements == tests/ref_w5_train.py.txt, the frozen nn_x/w5/train.py).
Removes exactly: `from cd_arms10 import ...`, the two add_argument statements, the `xoff = None` init, the setup block
`if a.xoff > 0:` and the loop block `if xoff is not None:`; the variant-prefix assert (w5_ -> (w5_, w10_)) is handled by the test
itself. Counts what it removed (EXPECT) so the test asserts the exact set.
"""
import ast
import collections

R2_IFS = ("a.xoff > 0", "xoff is not None")
EXPECT = collections.Counter({"import cd_arms10": 1, "add_argument --xoff": 1, "add_argument --xoff_table": 1,
                              "xoff = None": 1, "if a.xoff > 0": 1, "if xoff is not None": 1})


class StripR2(ast.NodeTransformer):
    def __init__(self):
        self.removed = collections.Counter()

    def visit_ImportFrom(self, n):
        if n.module == "cd_arms10":
            self.removed["import cd_arms10"] += 1
            return None
        return n

    def visit_Expr(self, n):
        v = n.value
        if (isinstance(v, ast.Call) and isinstance(v.func, ast.Attribute) and v.func.attr == "add_argument" and v.args
                and isinstance(v.args[0], ast.Constant) and v.args[0].value in ("--xoff", "--xoff_table")):
            self.removed[f"add_argument {v.args[0].value}"] += 1
            return None
        return self.generic_visit(n)

    def visit_Assign(self, n):
        t, v = n.targets[0] if len(n.targets) == 1 else None, n.value
        if isinstance(t, ast.Name) and t.id == "xoff" and isinstance(v, ast.Constant) and v.value is None:
            self.removed["xoff = None"] += 1
            return None
        return self.generic_visit(n)

    def visit_If(self, n):
        t = ast.unparse(n.test)
        if t in R2_IFS:
            self.removed[f"if {t}"] += 1
            return None
        return self.generic_visit(n)


def strip_r2(tree):
    """(tree without the R2 statements, Counter of what was removed)."""
    st = StripR2()
    return st.visit(tree), st.removed
