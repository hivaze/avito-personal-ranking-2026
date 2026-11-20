"""AST stripper of the PSEUDO statements of lab/nn_x/w12/pseudokd/train.py (--pseudo_*, cd_pseudokd.py; 2026-11-17; = the wave-11
stripper lab/nn_x/w11/pseudo/tests/strip_pseudo.py with the helper module renamed cd_arms11 -> cd_pseudokd; frozen copy of the original:
ref/strip_pseudo_w11.py.txt). Used by tests/test_pseudokd.py (C0: train.py minus these statements == ref/train_w5.py.txt, the frozen
lab/nn_x/w5/train.py). Removes exactly: `from cd_pseudokd import ...`, the 7 add_argument("--pseudo_*") statements, the `pseudo = None`
init, the setup block `if a.pseudo_ratio > 0:`, the loop block `if pseudo is not None and step < pseudo.t_end:` and the two
`if pseudo is not None:` blocks (eval summary, final save). The variant-prefix assert is NOT changed in this copy (w5_ only). Counts what
it removed (EXPECT) so the test asserts the exact set.
"""
import ast
import collections

FLAGS = ("--pseudo_ratio", "--pseudo_gap_min", "--pseudo_gap_max", "--pseudo_w", "--pseudo_tmax", "--pseudo_head", "--pseudo_noset")
IFS = ("a.pseudo_ratio > 0", "pseudo is not None and step < pseudo.t_end", "pseudo is not None")
EXPECT = collections.Counter({"import cd_pseudokd": 1, "pseudo = None": 1, "if a.pseudo_ratio > 0": 1,
                              "if pseudo is not None and step < pseudo.t_end": 1, "if pseudo is not None": 2})
for f_ in FLAGS:
    EXPECT[f"add_argument {f_}"] = 1


class StripPseudo(ast.NodeTransformer):
    def __init__(self):
        self.removed = collections.Counter()

    def visit_ImportFrom(self, n):
        if n.module == "cd_pseudokd":
            self.removed["import cd_pseudokd"] += 1
            return None
        return n

    def visit_Expr(self, n):
        v = n.value
        if (isinstance(v, ast.Call) and isinstance(v.func, ast.Attribute) and v.func.attr == "add_argument" and v.args
                and isinstance(v.args[0], ast.Constant) and v.args[0].value in FLAGS):
            self.removed[f"add_argument {v.args[0].value}"] += 1
            return None
        return self.generic_visit(n)

    def visit_Assign(self, n):
        t, v = n.targets[0] if len(n.targets) == 1 else None, n.value
        if isinstance(t, ast.Name) and t.id == "pseudo" and isinstance(v, ast.Constant) and v.value is None:
            self.removed["pseudo = None"] += 1
            return None
        return self.generic_visit(n)

    def visit_If(self, n):
        t = ast.unparse(n.test)
        if t in IFS:
            self.removed[f"if {t}"] += 1
            return None
        return self.generic_visit(n)


def strip_pseudo(tree):
    """(tree without the PSEUDO statements, Counter of what was removed)."""
    st = StripPseudo()
    return st.visit(tree), st.removed
