"""Shared AST stripper of the valcd statements of nn_x/w5/train.py (--val_mix / --val_fold, cd_arms7.py; added 2026-11-12
~23:35 UTC). Used by tests/test_valcd.py (V0: train.py minus these statements == tests/ref_pre_valcd/train.py.txt) and applied
FIRST by tests/test_hzcd.py, tests/test_w5.py and tests/test_trunc_lifts.py, so that their own flag-off proofs against their
frozen copies keep holding (chain of proofs: new == pre_valcd + valcd; pre_valcd == pre_hzcd + hzcd; pre_hzcd == pre_tl +
trunc_lifts; pre_tl == nn_x/r6 + wave-5).
Removes exactly: `from cd_arms7 import ...`, the --val_mix and --val_fold add_argument lines, the `valcd = None` init, the
`if a.val_mix > 0:` setup block (which rebinds `prand` to the extended vector), the loop block `if valcd is not None and step >=
t_cool and valcd.draw(is_pool):` (val batch), the loss block `if valcd is not None and valcd.is_val:` and the two `if valcd is
not None:` log blocks (eval summary, selected-checkpoint summary). Counts what it removed (EXPECT) so the tests assert the set.
"""
import ast
import collections

VC_IFS = ("a.val_mix > 0", "valcd is not None and step >= t_cool and valcd.draw(is_pool)", "valcd is not None and valcd.is_val",
          "valcd is not None")
EXPECT = collections.Counter({"import cd_arms7": 1, "add_argument --val_mix": 1, "add_argument --val_fold": 1, "valcd = None": 1,
                              "if a.val_mix > 0": 1, "if valcd is not None and step >= t_cool and valcd.draw(is_pool)": 1,
                              "if valcd is not None and valcd.is_val": 1, "if valcd is not None": 2})


class StripValcd(ast.NodeTransformer):
    def __init__(self):
        self.removed = collections.Counter()

    def visit_ImportFrom(self, n):
        if n.module == "cd_arms7":
            self.removed["import cd_arms7"] += 1
            return None
        return n

    def visit_Expr(self, n):
        v = n.value
        if (isinstance(v, ast.Call) and isinstance(v.func, ast.Attribute) and v.func.attr == "add_argument" and v.args
                and isinstance(v.args[0], ast.Constant) and v.args[0].value in ("--val_mix", "--val_fold")):
            self.removed[f"add_argument {v.args[0].value}"] += 1
            return None
        return self.generic_visit(n)

    def visit_Assign(self, n):
        if ast.unparse(n) == "valcd = None":
            self.removed["valcd = None"] += 1
            return None
        return self.generic_visit(n)

    def visit_If(self, n):
        t = ast.unparse(n.test)
        if t in VC_IFS:
            self.removed[f"if {t}"] += 1
            return None
        return self.generic_visit(n)


def strip_valcd(tree):
    """(tree without the valcd statements, Counter of what was removed)."""
    st = StripValcd()
    return st.visit(tree), st.removed
