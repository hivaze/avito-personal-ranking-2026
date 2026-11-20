"""AST stripper of the WIDTH-GATE statements of nn_x/w11/width/train.py (--mup_base, cd_arms11.py; added 2026-11-14).
Used by tests/test_width.py (W0: train.py minus these statements == tests/ref_w5_train.py.txt, the frozen nn_x/w5/train.py).
Removes exactly: `from cd_arms11 import ...`, the add_argument("--mup_base") statement, the `mup = None` init, the setup block
`if a.mup_base > 0:` and the two guarded blocks `if mup is not None:` (the per-step apply and the eval-time summary); the
variant-prefix assert ("w5_" -> ("w5_", "w11_")) is handled by the test itself (both asserts are dropped and the new one is
checked by regex). Counts what it removed (EXPECT) so the test asserts the exact set.
"""
import ast
import collections

W11_IFS = ("a.mup_base > 0", "mup is not None")
EXPECT = collections.Counter({"import cd_arms11": 1, "add_argument --mup_base": 1, "mup = None": 1, "if a.mup_base > 0": 1,
                              "if mup is not None": 2})


class StripW11(ast.NodeTransformer):
    def __init__(self):
        self.removed = collections.Counter()

    def visit_ImportFrom(self, n):
        if n.module == "cd_arms11":
            self.removed["import cd_arms11"] += 1
            return None
        return n

    def visit_Expr(self, n):
        v = n.value
        if (isinstance(v, ast.Call) and isinstance(v.func, ast.Attribute) and v.func.attr == "add_argument" and v.args
                and isinstance(v.args[0], ast.Constant) and v.args[0].value == "--mup_base"):
            self.removed["add_argument --mup_base"] += 1
            return None
        return self.generic_visit(n)

    def visit_Assign(self, n):
        t, v = n.targets[0] if len(n.targets) == 1 else None, n.value
        if isinstance(t, ast.Name) and t.id == "mup" and isinstance(v, ast.Constant) and v.value is None:
            self.removed["mup = None"] += 1
            return None
        return self.generic_visit(n)

    def visit_If(self, n):
        t = ast.unparse(n.test)
        if t in W11_IFS:
            self.removed[f"if {t}"] += 1
            return None
        return self.generic_visit(n)


def strip_w11(tree):
    """(tree without the WIDTH-GATE statements, Counter of what was removed)."""
    st = StripW11()
    return st.visit(tree), st.removed
