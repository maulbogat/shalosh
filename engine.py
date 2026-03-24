from __future__ import annotations
from dataclasses import dataclass, field
from typing import Union

# Three-valued type: True, False, or "unknown"
TV = Union[bool, str]


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
# evaluate
# ---------------------------------------------------------------------------

def evaluate(rule: Rule, data: dict) -> TV:
    """Return True, False, or "unknown" using Strong Kleene logic."""
    if rule.op == "LEAF":
        return data.get(rule.condition, "unknown")

    if rule.op == "NOT":
        child = evaluate(rule.children[0], data)
        if _is_unknown(child):
            return "unknown"
        return not child

    vals = [evaluate(c, data) for c in rule.children]

    if rule.op == "AND":
        # A single False short-circuits regardless of unknowns
        if any(_is_false(v) for v in vals):
            return False
        if any(_is_unknown(v) for v in vals):
            return "unknown"
        return True

    if rule.op == "OR":
        # A single True short-circuits regardless of unknowns
        if any(_is_true(v) for v in vals):
            return True
        if any(_is_unknown(v) for v in vals):
            return "unknown"
        return False

    raise ValueError(f"Unrecognised op: {rule.op!r}")


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------

def provenance(rule: Rule, data: dict) -> list[str]:
    """Return the leaf condition strings that actually determined the outcome.

    For a definite result (True/False):
      AND=False  → the False children that triggered short-circuit
      AND=True   → every child (all were needed)
      OR=True    → the True children that triggered short-circuit
      OR=False   → every child (all were needed)
      NOT        → same leaves as its child

    For "unknown": the unknown leaves that prevented determination.
    """
    return _prov(rule, data)


def _prov(rule: Rule, data: dict) -> list[str]:
    if rule.op == "LEAF":
        return [rule.condition]

    if rule.op == "NOT":
        return _prov(rule.children[0], data)

    vals = [evaluate(c, data) for c in rule.children]
    outcome = evaluate(rule, data)

    if rule.op == "AND":
        if _is_false(outcome):
            # Only the False subtrees mattered
            return [leaf for c, v in zip(rule.children, vals)
                    if _is_false(v) for leaf in _prov(c, data)]
        if _is_true(outcome):
            # Every child was required
            return [leaf for c in rule.children for leaf in _prov(c, data)]
        # unknown: the unknown children blocked resolution
        return [leaf for c, v in zip(rule.children, vals)
                if _is_unknown(v) for leaf in _prov(c, data)]

    if rule.op == "OR":
        if _is_true(outcome):
            return [leaf for c, v in zip(rule.children, vals)
                    if _is_true(v) for leaf in _prov(c, data)]
        if _is_false(outcome):
            return [leaf for c in rule.children for leaf in _prov(c, data)]
        return [leaf for c, v in zip(rule.children, vals)
                if _is_unknown(v) for leaf in _prov(c, data)]

    raise ValueError(f"Unrecognised op: {rule.op!r}")


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
        return evaluate(rule, data), provenance(rule, data)

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
