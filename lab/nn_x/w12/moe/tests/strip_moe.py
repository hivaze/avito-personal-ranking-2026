"""AST strip of the w12/moe additions (tests/test_moe_cpu.py T1a): removes from lab/nn_x/w12/moe/{train.py, model_moe.py} exactly the
statements / definitions / arguments that belong to the MoE arms and returns the AST dump, to be compared with the dump of the w5
original (lab/nn_x/w5/{train.py, model.py}). Docstrings are dropped on both sides (they are not code).

Rules (applied bottom-up, so a w5 compound statement that CONTAINS an inserted statement is kept once the insertion is removed):
  * `from model_moe import X` is renamed to `from model import X` (the only changed import of a w5 name);
  * any statement whose source (after its own children were stripped) matches PAT is removed: the inserted flags, `moe = None`, the
    setup / adamw_groups / optimizer / update_bias / summary / hist blocks, the --init guard, `self.moe = None`, `self.vo = None`,
    `self.moe_on = ...`, the cfg updates, the build block, the guarded early returns and the guarded forward loop branch;
  * top-level definitions and methods named in NEW_DEFS are removed (the routed modules and the moe-only methods);
  * `**moe_kwargs(a)` keyword arguments and function parameters named in NEW_ARGS are removed.
Every removal is recorded; the caller asserts the exact expected set (nothing else may differ)."""
import ast
import re

PAT = re.compile(r"moe|\.vo\b|\bvo_|qmask|tmask|MOE_PRIME")
NEW_DEFS = {"MOE_PRIME", "quantile_shift", "RouteSite", "RoutedFFN", "SwitchVO", "_forward_moe_vo", "_forward_moe", "moe_modules"}
NEW_ARGS = {"qmask", "tmask", "moe_ffn", "moe_w", "moe_k", "moe_scale", "moe_xdrop", "moe_eta", "vo_moe", "vo_k", "vo_eta"}


class _Strip(ast.NodeTransformer):
    def __init__(self):
        self.removed = []

    def _drop_doc(self, node):
        b = getattr(node, "body", None)
        if isinstance(b, list) and b and isinstance(b[0], ast.Expr) and isinstance(b[0].value, ast.Constant) \
                and isinstance(b[0].value.value, str):
            node.body = b[1:]

    def visit(self, node):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in NEW_DEFS:
            self.removed.append(f"def {node.name}")
            return None
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)):
            self._drop_doc(node)
        if isinstance(node, ast.ImportFrom) and node.module == "model_moe":
            node.module = "model"
            self.removed.append("rename import model_moe -> model")
            return node
        node = super().visit(node)
        if node is None:
            return None
        if isinstance(node, ast.keyword) and node.arg is None and PAT.search(ast.unparse(node.value)):
            self.removed.append(f"kw **{ast.unparse(node.value)}")
            return None
        if isinstance(node, ast.arguments):
            keep = [a_ for a_ in node.args if a_.arg not in NEW_ARGS]
            nd = len(node.defaults)
            dmap = {a_.arg: d for a_, d in zip(node.args[len(node.args) - nd:], node.defaults)}
            gone = [a_.arg for a_ in node.args if a_.arg in NEW_ARGS]
            if gone:
                self.removed.append(f"args {gone}")
                node.args = keep
                node.defaults = [dmap[a_.arg] for a_ in keep if a_.arg in dmap]
            return node
        if isinstance(node, ast.stmt):
            if isinstance(node, (ast.Assign, ast.AnnAssign)) and any(isinstance(t, ast.Name) and t.id in NEW_DEFS
                                                                    for t in getattr(node, "targets", [getattr(node, "target", None)])):
                self.removed.append(f"stmt {ast.unparse(node)[:80]}")
                return None
            body = getattr(node, "body", None)
            try:
                txt = ast.unparse(node)
            except Exception:  # an emptied compound statement (only inserted statements inside)
                txt = "moe"
            if (isinstance(body, list) and not body) or PAT.search(txt):
                self.removed.append(f"stmt {txt.splitlines()[0][:110]}")
                return None
        return node


def strip(path, moe_side=True):
    """AST dump of `path` without docstrings and (moe_side) without the w12/moe additions; returns (dump, removed list)."""
    tree = ast.parse(open(path).read())
    st = _Strip()
    if moe_side:
        tree = st.visit(tree)
    else:  # the w5 side: docstrings only
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)):
                st._drop_doc(node)
    return ast.dump(tree, include_attributes=False), st.removed
