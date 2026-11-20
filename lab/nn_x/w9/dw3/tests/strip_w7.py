"""Shared AST stripper of the wave-7 statements of nn_x/w5/train.py (--rdrop_cd / --sam_cd, cd_arms8.py; added 2026-11-13
~05:20 UTC). Used by tests/test_w7.py (W0: train.py minus these statements == tests/ref_pre_w7/train.py.txt) and applied FIRST
by tests/test_valcd.py, tests/test_hzcd.py, tests/test_w5.py and tests/test_trunc_lifts.py, so that their own flag-off proofs
against their frozen copies keep holding (chain of proofs: new == pre_w7 + w7; pre_w7 == pre_valcd + valcd; pre_valcd ==
pre_hzcd + hzcd; pre_hzcd == pre_tl + trunc_lifts; pre_tl == nn_x/r6 + wave-5).
Removes exactly: `from cd_arms8 import ...`, the --rdrop_cd and --sam_cd add_argument statements, the `rdrop, sam = None, None`
init, the setup blocks `if a.rdrop_cd > 0:` and `if a.sam_cd > 0:`, the loop blocks `if rdrop is not None and step >= t_cool:`
(second pass + KL) and `if sam is not None and step >= t_cool:` (ascent step), and the two eval-log blocks `if rdrop is not
None:` / `if sam is not None:`. Counts what it removed (EXPECT) so the tests assert the set.
"""
import ast
import collections

W7_IFS = ("a.rdrop_cd > 0", "a.sam_cd > 0", "rdrop is not None and step >= t_cool", "sam is not None and step >= t_cool",
          "rdrop is not None", "sam is not None")
EXPECT = collections.Counter({"import cd_arms8": 1, "add_argument --rdrop_cd": 1, "add_argument --sam_cd": 1,
                              "rdrop, sam = None, None": 1, "if a.rdrop_cd > 0": 1, "if a.sam_cd > 0": 1,
                              "if rdrop is not None and step >= t_cool": 1, "if sam is not None and step >= t_cool": 1,
                              "if rdrop is not None": 1, "if sam is not None": 1})


class StripW7(ast.NodeTransformer):
    def __init__(self):
        self.removed = collections.Counter()

    def visit_ImportFrom(self, n):
        if n.module == "cd_arms8":
            self.removed["import cd_arms8"] += 1
            return None
        return n

    def visit_Expr(self, n):
        v = n.value
        if (isinstance(v, ast.Call) and isinstance(v.func, ast.Attribute) and v.func.attr == "add_argument" and v.args
                and isinstance(v.args[0], ast.Constant) and v.args[0].value in ("--rdrop_cd", "--sam_cd")):
            self.removed[f"add_argument {v.args[0].value}"] += 1
            return None
        return self.generic_visit(n)

    def visit_Assign(self, n):
        t, v = n.targets[0] if len(n.targets) == 1 else None, n.value
        if (isinstance(t, ast.Tuple) and [getattr(e, "id", None) for e in t.elts] == ["rdrop", "sam"] and isinstance(v, ast.Tuple)
                and len(v.elts) == 2 and all(isinstance(e, ast.Constant) and e.value is None for e in v.elts)):
            self.removed["rdrop, sam = None, None"] += 1  # (ast.unparse renders it as `rdrop, sam = (None, None)`)
            return None
        return self.generic_visit(n)

    def visit_If(self, n):
        t = ast.unparse(n.test)
        if t in W7_IFS:
            self.removed[f"if {t}"] += 1
            return None
        return self.generic_visit(n)


def strip_w7(tree):
    """(tree without the wave-7 statements, Counter of what was removed)."""
    st = StripW7()
    return st.visit(tree), st.removed
