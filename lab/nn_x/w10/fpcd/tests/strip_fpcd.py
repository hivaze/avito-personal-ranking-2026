"""AST stripper of the WAVE-10 FPCD-lite statements of nn_x/w10/fpcd/train.py (--cd_exclude_days, cd_arms10.py; added 2026-11-13
~16:50 UTC). Used by tests/test_fpcd.py: train.py minus these statements == the frozen nn_x/w5/train.py (tests/ref_w5_train.py.txt).
Removes exactly: `from cd_arms10 import setup_fpcd`, the --cd_exclude_days add_argument, the `fpcd = None` init, the setup block
`if a.cd_exclude_days:`, the loop block `if fpcd is not None and step >= t_cool:` and the eval-log block `if fpcd is not None:`.
Counts what it removed (EXPECT) so the test asserts the exact set.
"""
import ast
import collections

FPCD_IFS = ("a.cd_exclude_days", "fpcd is not None and step >= t_cool", "fpcd is not None")
EXPECT = collections.Counter({"import cd_arms10": 1, "add_argument --cd_exclude_days": 1, "fpcd = None": 1,
                              "if a.cd_exclude_days": 1, "if fpcd is not None and step >= t_cool": 1, "if fpcd is not None": 1})


class StripFPCD(ast.NodeTransformer):
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
                and isinstance(v.args[0], ast.Constant) and v.args[0].value == "--cd_exclude_days"):
            self.removed["add_argument --cd_exclude_days"] += 1
            return None
        return self.generic_visit(n)

    def visit_Assign(self, n):
        t, v = n.targets[0] if len(n.targets) == 1 else None, n.value
        if isinstance(t, ast.Name) and t.id == "fpcd" and isinstance(v, ast.Constant) and v.value is None:
            self.removed["fpcd = None"] += 1
            return None
        return self.generic_visit(n)

    def visit_If(self, n):
        t = ast.unparse(n.test)
        if t in FPCD_IFS:
            self.removed[f"if {t}"] += 1
            return None
        return self.generic_visit(n)


def strip_fpcd(tree):
    """(tree without the wave-10 FPCD statements, Counter of what was removed)."""
    st = StripFPCD()
    return st.visit(tree), st.removed
