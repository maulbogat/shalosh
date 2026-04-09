from __future__ import annotations
from dataclasses import dataclass, field
from typing import Union

# Three-valued type: True, False, or "unknown"
TV = Union[bool, str]

# ---------------------------------------------------------------------------
# Module-level evaluation mode.
#
# Two-level config pattern:
#   1. Global default — set ``default_lazy`` to change the library-wide
#      behaviour without touching call sites.
#   2. Per-call override — pass ``lazy=True/False`` to ``evaluate()`` or
#      ``provenance()`` to override the global default for that one call.
#
# Eager (lazy=False, the default) evaluates every child before deciding,
# giving *complete* provenance — all load-bearing leaves are returned even
# when a short-circuit would have sufficed. This is what downstream rule-
# minimisation needs.
#
# Lazy (lazy=True) evaluates children left-to-right and stops as soon as
# the outcome is determined, giving *sufficient* provenance — only the
# leaves actually inspected before the decision was made.
# ---------------------------------------------------------------------------
default_lazy: bool = False


@dataclass
class Rule:
    """Node in a boolean decision tree.

    op:        "AND" | "OR" | "NOT" | "LEAF"
    children:  sub-rules (AND/OR take 1+, NOT takes exactly 1)
    condition: leaf label looked up in the data dict (LEAF only)
    """
    op: str
    children: list[Rule] = field(default_factory=list)
    condition: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_true(v: TV) -> bool:
    return v is True

def _is_false(v: TV) -> bool:
    return v is False

def _is_unknown(v: TV) -> bool:
    return v == "unknown"


# ---------------------------------------------------------------------------
# Core: single-pass eval + provenance
# ---------------------------------------------------------------------------

def _eval(rule: Rule, data: dict, lazy: bool) -> tuple[TV, list[str]]:
    """Evaluate *rule* against *data* and return ``(value, provenance_leaves)``.

    A single recursive traversal: no node is evaluated more than once.
    """
    if rule.op == "LEAF":
        val = data.get(rule.condition, "unknown")
        return (val, [rule.condition])

    if rule.op == "NOT":
        val, prov = _eval(rule.children[0], data, lazy)
        if _is_unknown(val):
            return ("unknown", prov)
        return (not val, prov)

    if rule.op == "AND":
        if lazy:
            # Left-to-right; return False immediately on the first False child.
            # Unknown children are noted but do not stop evaluation — a later
            # False still wins.
            true_prov: list[str] = []
            unknown_prov: list[str] = []
            for child in rule.children:
                val, prov = _eval(child, data, lazy)
                if _is_false(val):
                    return (False, prov)
                elif _is_unknown(val):
                    unknown_prov.extend(prov)
                else:
                    true_prov.extend(prov)
            # No False found
            if unknown_prov:
                return ("unknown", unknown_prov)
            return (True, true_prov)
        else:
            # Eager: evaluate all children before deciding.
            results = [_eval(c, data, lazy) for c in rule.children]
            vals = [v for v, _ in results]
            if any(_is_false(v) for v in vals):
                return (False, [leaf for v, prov in results
                                if _is_false(v) for leaf in prov])
            if any(_is_unknown(v) for v in vals):
                return ("unknown", [leaf for v, prov in results
                                    if _is_unknown(v) for leaf in prov])
            return (True, [leaf for _, prov in results for leaf in prov])

    if rule.op == "OR":
        if lazy:
            # Left-to-right; return True immediately on the first True child.
            false_prov: list[str] = []
            unknown_prov: list[str] = []
            for child in rule.children:
                val, prov = _eval(child, data, lazy)
                if _is_true(val):
                    return (True, prov)
                elif _is_unknown(val):
                    unknown_prov.extend(prov)
                else:
                    false_prov.extend(prov)
            # No True found
            if unknown_prov:
                return ("unknown", unknown_prov)
            return (False, false_prov)
        else:
            # Eager: evaluate all children before deciding.
            results = [_eval(c, data, lazy) for c in rule.children]
            vals = [v for v, _ in results]
            if any(_is_true(v) for v in vals):
                return (True, [leaf for v, prov in results
                               if _is_true(v) for leaf in prov])
            if any(_is_unknown(v) for v in vals):
                return ("unknown", [leaf for v, prov in results
                                    if _is_unknown(v) for leaf in prov])
            return (False, [leaf for _, prov in results for leaf in prov])

    raise ValueError(f"Unrecognised op: {rule.op!r}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate(rule: Rule, data: dict, *, lazy: bool | None = None) -> TV:
    """Return True, False, or "unknown" using Strong Kleene logic.

    "unknown" correctly means "outcome depends on missing features"
    only for minimal rules — rules containing no contradictions or
    tautologies in any sub-tree. For non-minimal rules, "unknown" may
    be returned in cases that are actually always True or always False.

    lazy: if True, use lazy (left-to-right short-circuit) evaluation;
          if False, use eager (evaluate all children) evaluation;
          if None (default), use the module-level ``default_lazy`` setting.

    Eager mode gives *complete* provenance — all load-bearing leaves are
    included even when a short-circuit would have sufficed. Lazy mode
    gives *sufficient* provenance — only the leaves actually inspected
    left-to-right before the decision was made.
    """
    use_lazy = default_lazy if lazy is None else lazy
    val, _ = _eval(rule, data, use_lazy)
    return val


def provenance(rule: Rule, data: dict, *, lazy: bool | None = None) -> list[str]:
    """Return the leaf condition strings that actually determined the outcome.

    For a definite result (True/False):
      AND=False  → the False children that triggered short-circuit
      AND=True   → every child (all were needed)
      OR=True    → the True children that triggered short-circuit
      OR=False   → every child (all were needed)
      NOT        → same leaves as its child

    For "unknown": the unknown leaves that prevented determination.

    lazy: if True, use lazy (left-to-right short-circuit) evaluation;
          if False, use eager (evaluate all children) evaluation;
          if None (default), use the module-level ``default_lazy`` setting.

    Eager mode gives *complete* provenance — all load-bearing leaves are
    included even when a short-circuit would have sufficed. Lazy mode
    gives *sufficient* provenance — only the leaves actually inspected
    left-to-right before the decision was made.
    """
    use_lazy = default_lazy if lazy is None else lazy
    _, prov = _eval(rule, data, use_lazy)
    return prov


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------

def leaf(condition: str) -> Rule:
    return Rule(op="LEAF", condition=condition)

def and_(*children: Rule) -> Rule:
    return Rule(op="AND", children=list(children))

def or_(*children: Rule) -> Rule:
    return Rule(op="OR", children=list(children))

def not_(child: Rule) -> Rule:
    return Rule(op="NOT", children=[child])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    def check(label: str, got, expected_eval, expected_prov):
        eval_ok = got[0] == expected_eval
        prov_ok = sorted(got[1]) == sorted(expected_prov)
        status = "PASS" if eval_ok and prov_ok else "FAIL"
        print(f"[{status}] {label}")
        if not eval_ok:
            print(f"       eval:  expected={expected_eval!r}  got={got[0]!r}")
        if not prov_ok:
            print(f"       prov:  expected={sorted(expected_prov)}  got={sorted(got[1])}")

    def run(rule, data):
        return _eval(rule, data, False)

    # 1. Simple LEAF hit
    r = leaf("sunny")
    check("leaf True",  run(r, {"sunny": True}),  True,  ["sunny"])
    check("leaf False", run(r, {"sunny": False}),  False, ["sunny"])
    check("leaf miss",  run(r, {}),                "unknown", ["sunny"])

    # 2. AND short-circuits on first False even with unknowns
    r = and_(leaf("a"), leaf("b"), leaf("c"))
    check("AND false short-circuit",
          run(r, {"a": True, "b": False}),  # c unknown
          False, ["b"])

    check("AND all true",
          run(r, {"a": True, "b": True, "c": True}),
          True, ["a", "b", "c"])

    check("AND unknown",
          run(r, {"a": True, "b": True}),  # c missing
          "unknown", ["c"])

    # 3. OR short-circuits on first True even with unknowns
    r = or_(leaf("x"), leaf("y"), leaf("z"))
    check("OR true short-circuit",
          run(r, {"y": True}),             # x, z unknown
          True, ["y"])

    check("OR all false",
          run(r, {"x": False, "y": False, "z": False}),
          False, ["x", "y", "z"])

    check("OR unknown",
          run(r, {"x": False, "y": False}),  # z missing
          "unknown", ["z"])

    # 4. NOT
    r = not_(leaf("flag"))
    check("NOT true→false", run(r, {"flag": True}),  False,     ["flag"])
    check("NOT false→true", run(r, {"flag": False}), True,      ["flag"])
    check("NOT unknown",    run(r, {}),               "unknown", ["flag"])

    # 5. Nested: (a AND b) OR (NOT c)
    r = or_(and_(leaf("a"), leaf("b")), not_(leaf("c")))
    check("nested: left branch wins",
          run(r, {"a": True, "b": True, "c": False}),
          True, ["a", "b", "c"])  # both OR branches are True; all leaves contributed

    check("nested: right branch wins",
          run(r, {"a": False, "c": False}),
          True, ["c"])

    check("nested: unknown blocks",
          run(r, {"a": False}),  # b and c unknown
          "unknown", ["c"])      # NOT(c) is the blocking unknown

    check("nested: all false",
          run(r, {"a": False, "b": False, "c": True}),
          False, ["a", "b", "c"])  # OR=False needs all children; AND=False returns all its False leaves
