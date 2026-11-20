"""AST stripper of the wave-10 R3 statements of nn_x/w10/r3/train.py (--bce_w, cd_arms10.py; added 2026-11-13 ~17:00 UTC).
Used by tests/test_r3.py (C0b: train.py minus these statements == tests/ref_w5_train.py.txt, the frozen nn_x/w5/train.py).
Removes exactly: `from cd_arms10 import ...`, the add_argument("--bce_w") statement, the `bce_w = 0.0` init, the setup block
`if a.bce_w > 0:` and the loop block `if bce_w > 0:`; the variant-prefix assert (w5_ -> (w5_, w10_)) is handled by the test
itself (both asserts are dropped and the new one is checked by regex). Counts what it removed (EXPECT) so the test asserts
the exact set.
"""
import ast
import collections

R3_IFS = ("a.bce_w > 0", "bce_w > 0")
EXPECT = collections.Counter({"import cd_arms10": 1, "add_argument --bce_w": 1, "bce_w = 0.0": 1, "if a.bce_w > 0": 1, "if bce_w > 0": 1})


class StripR3(ast.NodeTransformer):
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
                and isinstance(v.args[0], ast.Constant) and v.args[0].value == "--bce_w"):
            self.removed["add_argument --bce_w"] += 1
            return None
        return self.generic_visit(n)

    def visit_Assign(self, n):
        t, v = n.targets[0] if len(n.targets) == 1 else None, n.value
        if isinstance(t, ast.Name) and t.id == "bce_w" and isinstance(v, ast.Constant) and v.value == 0.0:
            self.removed["bce_w = 0.0"] += 1
            return None
        return self.generic_visit(n)

    def visit_If(self, n):
        t = ast.unparse(n.test)
        if t in R3_IFS:
            self.removed[f"if {t}"] += 1
            return None
        return self.generic_visit(n)


def strip_r3(tree):
    """(tree without the R3 statements, Counter of what was removed)."""
    st = StripR3()
    return st.visit(tree), st.removed
