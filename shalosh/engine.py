from __future__ import annotations
from dataclasses import dataclass, field
from typing import Union, Any
import ast

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


# ---------------------------------------------------------------------------
# Operator aliases
# ---------------------------------------------------------------------------

OPERATOR_ALIASES: dict[str, str] = {
    # eq
    "eq": "eq", "equal_to": "eq", "==": "eq", "=": "eq",
    # neq
    "neq": "neq", "not_equal_to": "neq", "!=": "neq", "<>": "neq",
    # gt
    "gt": "gt", "greater_than": "gt", ">": "gt",
    # gte
    "gte": "gte", "greater_than_or_equal_to": "gte", ">=": "gte",
    # lt
    "lt": "lt", "less_than": "lt", "<": "lt",
    # lte
    "lte": "lte", "less_than_or_equal_to": "lte", "<=": "lte",
    # in
    "in": "in", "is_in": "in", "contains": "in",
    # not_in
    "not_in": "not_in", "is_not_in": "not_in", "not_contains": "not_in",
    # is_true / is_false
    "is_true": "is_true",
    "is_false": "is_false",
}


# ---------------------------------------------------------------------------
# Value parser (used by Condition.from_expr)
# ---------------------------------------------------------------------------

def _parse_value(s: str) -> Any:
    """Parse a value string into an appropriate Python object."""
    s = s.strip()
    if s == "True":
        return True
    if s == "False":
        return False
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    if s.startswith("["):
        try:
            result = ast.literal_eval(s)
            if isinstance(result, list):
                return result
        except (ValueError, SyntaxError):
            pass
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    return s


# ---------------------------------------------------------------------------
# Condition
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Condition:
    """A single leaf condition: variable OP value.

    ``operator`` is normalised to its canonical short name on construction.
    Invalid operators raise ``ValueError`` immediately.
    """
    variable: str
    operator: str
    value: Any = None

    def __post_init__(self) -> None:
        if self.operator not in OPERATOR_ALIASES:
            raise ValueError(
                f"Unknown operator: {self.operator!r}. "
                f"Valid operators: {sorted(OPERATOR_ALIASES)}"
            )
        object.__setattr__(self, "operator", OPERATOR_ALIASES[self.operator])

    @classmethod
    def from_expr(cls, expr: str) -> Condition:
        """Parse a simple expression string like ``'age > 18'`` into a Condition.

        Supported formats::

            'variable operator value'

        where operator is one of: ``==``, ``!=``, ``>=``, ``<=``, ``>``, ``<``,
        ``=``, ``in``, ``not_in``.

        Values are parsed as integers, floats, booleans, lists (``[...]``), or
        strings (quotes optional, stripped if present).
        """
        # Two-character symbol operators (checked before single-char to avoid
        # e.g. '>' matching inside '>=').
        for op_sym in (">=", "<=", "!=", "=="):
            if op_sym in expr:
                idx = expr.index(op_sym)
                variable = expr[:idx].strip()
                value_str = expr[idx + len(op_sym):].strip()
                if variable:
                    return cls(variable, op_sym, _parse_value(value_str))

        # Single-character symbol operators.
        for op_sym in (">", "<", "="):
            if op_sym in expr:
                idx = expr.index(op_sym)
                variable = expr[:idx].strip()
                value_str = expr[idx + 1:].strip()
                if variable:
                    return cls(variable, op_sym, _parse_value(value_str))

        # Word operators: split into at most 3 tokens.
        tokens = expr.split(None, 2)
        if len(tokens) >= 2:
            for word_op in ("not_in", "in"):
                if tokens[1] == word_op:
                    variable = tokens[0]
                    value_str = tokens[2] if len(tokens) > 2 else ""
                    return cls(variable, word_op, _parse_value(value_str))

        raise ValueError(f"Cannot parse expression: {expr!r}")

    @classmethod
    def from_business_rules(cls, condition_dict: dict) -> Condition:
        """Parse a venmo/business-rules style condition dict.

        Example input::

            {"name": "age", "operator": "greater_than", "value": 18}
        """
        return cls(
            variable=condition_dict["name"],
            operator=condition_dict["operator"],
            value=condition_dict.get("value"),
        )


# ---------------------------------------------------------------------------
# Rule
# ---------------------------------------------------------------------------

@dataclass
class Rule:
    """Node in a boolean decision tree.

    op:        "AND" | "OR" | "NOT" | "LEAF"
    children:  sub-rules (AND/OR take 1+, NOT takes exactly 1)
    condition: leaf label looked up in the data dict (LEAF only);
               may be a plain ``str`` (legacy) or a ``Condition`` object.
    """
    op: str
    children: list[Rule] = field(default_factory=list)
    condition: str | Condition = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_true(v: TV) -> bool:
    return v is True

def _is_false(v: TV) -> bool:
    return v is False

def _is_unknown(v: TV) -> bool:
    return v == "unknown"


def _apply_condition(cond: Condition, data_val: Any) -> bool:
    """Apply ``cond``'s operator to ``data_val``.

    Raises ``TypeError`` on type mismatches, with a message that names the
    variable, operator, and the types involved.
    """
    op = cond.operator
    cmp = cond.value
    try:
        if op == "eq":       return data_val == cmp
        if op == "neq":      return data_val != cmp
        if op == "gt":       return data_val > cmp
        if op == "gte":      return data_val >= cmp
        if op == "lt":       return data_val < cmp
        if op == "lte":      return data_val <= cmp
        if op == "in":       return data_val in cmp
        if op == "not_in":   return data_val not in cmp
        if op == "is_true":  return data_val is True
        if op == "is_false": return data_val is False
    except TypeError as exc:
        raise TypeError(
            f"Type error evaluating {cond.variable!r} {op!r} {cmp!r}: "
            f"got {type(data_val).__name__!r} vs {type(cmp).__name__!r}"
        ) from exc
    raise ValueError(f"Unrecognised operator: {op!r}")  # unreachable


# ---------------------------------------------------------------------------
# Core: single-pass eval + provenance
# ---------------------------------------------------------------------------

def _eval(rule: Rule, data: dict, lazy: bool) -> tuple[TV, list]:
    """Evaluate *rule* against *data* and return ``(value, provenance_leaves)``.

    A single recursive traversal: no node is evaluated more than once.
    """
    if rule.op == "LEAF":
        cond = rule.condition
        if isinstance(cond, Condition):
            if cond.variable not in data:
                return ("unknown", [cond])
            result = _apply_condition(cond, data[cond.variable])
            return (result, [cond])
        else:
            # Legacy string key: look up directly; missing → "unknown".
            val = data.get(cond, "unknown")
            return (val, [cond])

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
            true_prov: list = []
            unknown_prov: list = []
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
                return (False, [item for v, prov in results
                                if _is_false(v) for item in prov])
            if any(_is_unknown(v) for v in vals):
                return ("unknown", [item for v, prov in results
                                    if _is_unknown(v) for item in prov])
            return (True, [item for _, prov in results for item in prov])

    if rule.op == "OR":
        if lazy:
            # Left-to-right; return True immediately on the first True child.
            false_prov: list = []
            unknown_prov: list = []
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
                return (True, [item for v, prov in results
                               if _is_true(v) for item in prov])
            if any(_is_unknown(v) for v in vals):
                return ("unknown", [item for v, prov in results
                                    if _is_unknown(v) for item in prov])
            return (False, [item for _, prov in results for item in prov])

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


def provenance(rule: Rule, data: dict, *, lazy: bool | None = None) -> list:
    """Return the leaf conditions that actually determined the outcome.

    Each element is either a plain ``str`` (legacy string key) or a
    ``Condition`` object (expression-based leaf).

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

def leaf(expr_or_key, operator=None, value=None) -> Rule:
    """Create a LEAF Rule.

    Usage::

        leaf("good_credit")              # legacy string key
        leaf("age", "gt", 18)            # explicit Condition
        leaf("age > 18")                 # parsed expression
        leaf(Condition("age", "gt", 18)) # pre-built Condition
    """
    if isinstance(expr_or_key, Condition):
        return Rule(op="LEAF", condition=expr_or_key)
    if operator is not None:
        return Rule(op="LEAF", condition=Condition(expr_or_key, operator, value))
    if isinstance(expr_or_key, str):
        try:
            cond = Condition.from_expr(expr_or_key)
            return Rule(op="LEAF", condition=cond)
        except ValueError:
            pass  # Fall through: treat as legacy string key
    return Rule(op="LEAF", condition=expr_or_key)

def and_(*children: Rule) -> Rule:
    return Rule(op="AND", children=list(children))

def or_(*children: Rule) -> Rule:
    return Rule(op="OR", children=list(children))

def not_(child: Rule) -> Rule:
    return Rule(op="NOT", children=[child])


# ---------------------------------------------------------------------------
# venmo/business-rules adapter
# ---------------------------------------------------------------------------

def from_business_rules_tree(conditions: dict) -> Rule:
    """Convert a venmo/business-rules conditions dict to a shalosh Rule tree.

    ``"all"`` maps to AND, ``"any"`` maps to OR.  Each leaf dict (with
    ``"name"``, ``"operator"``, ``"value"`` keys) becomes a Condition-based
    LEAF.

    Example input::

        {"all": [
            {"name": "age", "operator": "greater_than", "value": 18},
            {"any": [
                {"name": "gender", "operator": "equal_to", "value": "male"},
                {"name": "income", "operator": "greater_than", "value": 50000},
            ]}
        ]}
    """
    if "all" in conditions:
        children = [from_business_rules_tree(c) for c in conditions["all"]]
        return Rule(op="AND", children=children)
    if "any" in conditions:
        children = [from_business_rules_tree(c) for c in conditions["any"]]
        return Rule(op="OR", children=children)
    return Rule(op="LEAF", condition=Condition.from_business_rules(conditions))


# ---------------------------------------------------------------------------
# Inline smoke tests
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
          True, ["a", "b", "c"])

    check("nested: right branch wins",
          run(r, {"a": False, "c": False}),
          True, ["c"])

    check("nested: unknown blocks",
          run(r, {"a": False}),
          "unknown", ["c"])

    check("nested: all false",
          run(r, {"a": False, "b": False, "c": True}),
          False, ["a", "b", "c"])
