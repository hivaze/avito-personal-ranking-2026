"""AST stripper of the wave-9 DW3 statements of nn_x/w9/dw3/train.py (--prand_norm / --prand_until_cool, cd_arms9.py; added
2026-11-13 ~13:30 UTC). Used by tests/test_dw3.py (C0b: train.py minus these statements == tests/ref_w5_train.py.txt, the frozen
nn_x/w5/train.py). Removes exactly: `from cd_arms9 import ...`, the two add_argument statements, the `dw3 = None` init, the setup
block `if a.prand_norm or a.prand_until_cool:` and the loop block `if dw3 is not None:`; the variant-prefix assert (w5_ -> (w5_,
w9_)) is handled by the test itself (both asserts are dropped and the new one is checked by regex). Counts what it removed
(EXPECT) so the test asserts the exact set.
"""
import ast
import collections

DW3_IFS = ("a.prand_norm or a.prand_until_cool", "dw3 is not None")
EXPECT = collections.Counter({"import cd_arms9": 1, "add_argument --prand_norm": 1, "add_argument --prand_until_cool": 1,
                              "dw3 = None": 1, "if a.prand_norm or a.prand_until_cool": 1, "if dw3 is not None": 1})


class StripDW3(ast.NodeTransformer):
    def __init__(self):
        self.removed = collections.Counter()

    def visit_ImportFrom(self, n):
        if n.module == "cd_arms9":
            self.removed["import cd_arms9"] += 1
            return None
        return n

    def visit_Expr(self, n):
        v = n.value
        if (isinstance(v, ast.Call) and isinstance(v.func, ast.Attribute) and v.func.attr == "add_argument" and v.args
                and isinstance(v.args[0], ast.Constant) and v.args[0].value in ("--prand_norm", "--prand_until_cool")):
            self.removed[f"add_argument {v.args[0].value}"] += 1
            return None
        return self.generic_visit(n)

    def visit_Assign(self, n):
        t, v = n.targets[0] if len(n.targets) == 1 else None, n.value
        if isinstance(t, ast.Name) and t.id == "dw3" and isinstance(v, ast.Constant) and v.value is None:
            self.removed["dw3 = None"] += 1
            return None
        return self.generic_visit(n)

    def visit_If(self, n):
        t = ast.unparse(n.test)
        if t in DW3_IFS:
            self.removed[f"if {t}"] += 1
            return None
        return self.generic_visit(n)


def strip_dw3(tree):
    """(tree without the DW3 statements, Counter of what was removed)."""
    st = StripDW3()
    return st.visit(tree), st.removed
