import pytest
import shalosh.engine as engine
from shalosh.engine import (
    evaluate, provenance, leaf, and_, or_, not_,
    Condition, from_business_rules_tree,
)


def ev(rule, data):
    return evaluate(rule, data)

def prov(rule, data):
    return sorted(provenance(rule, data))


# ---------------------------------------------------------------------------
# 1. LEAF
# ---------------------------------------------------------------------------

def test_leaf_true():
    r = leaf("sunny")
    assert ev(r, {"sunny": True}) is True
    assert prov(r, {"sunny": True}) == ["sunny"]

def test_leaf_false():
    r = leaf("sunny")
    assert ev(r, {"sunny": False}) is False
    assert prov(r, {"sunny": False}) == ["sunny"]

def test_leaf_missing():
    r = leaf("sunny")
    assert ev(r, {}) == "unknown"
    assert prov(r, {}) == ["sunny"]


# ---------------------------------------------------------------------------
# 2. AND
# ---------------------------------------------------------------------------

def test_and_false_short_circuit():
    r = and_(leaf("a"), leaf("b"), leaf("c"))
    assert ev(r, {"a": True, "b": False}) is False       # c unknown
    assert prov(r, {"a": True, "b": False}) == ["b"]

def test_and_all_true():
    r = and_(leaf("a"), leaf("b"), leaf("c"))
    assert ev(r, {"a": True, "b": True, "c": True}) is True
    assert prov(r, {"a": True, "b": True, "c": True}) == ["a", "b", "c"]

def test_and_unknown():
    r = and_(leaf("a"), leaf("b"), leaf("c"))
    assert ev(r, {"a": True, "b": True}) == "unknown"    # c missing
    assert prov(r, {"a": True, "b": True}) == ["c"]


# ---------------------------------------------------------------------------
# 3. OR
# ---------------------------------------------------------------------------

def test_or_true_short_circuit():
    r = or_(leaf("x"), leaf("y"), leaf("z"))
    assert ev(r, {"y": True}) is True                    # x, z unknown
    assert prov(r, {"y": True}) == ["y"]

def test_or_all_false():
    r = or_(leaf("x"), leaf("y"), leaf("z"))
    assert ev(r, {"x": False, "y": False, "z": False}) is False
    assert prov(r, {"x": False, "y": False, "z": False}) == ["x", "y", "z"]

def test_or_unknown():
    r = or_(leaf("x"), leaf("y"), leaf("z"))
    assert ev(r, {"x": False, "y": False}) == "unknown"  # z missing
    assert prov(r, {"x": False, "y": False}) == ["z"]


# ---------------------------------------------------------------------------
# 4. NOT
# ---------------------------------------------------------------------------

def test_not_true_to_false():
    r = not_(leaf("flag"))
    assert ev(r, {"flag": True}) is False
    assert prov(r, {"flag": True}) == ["flag"]

def test_not_false_to_true():
    r = not_(leaf("flag"))
    assert ev(r, {"flag": False}) is True
    assert prov(r, {"flag": False}) == ["flag"]

def test_not_unknown():
    r = not_(leaf("flag"))
    assert ev(r, {}) == "unknown"
    assert prov(r, {}) == ["flag"]


# ---------------------------------------------------------------------------
# 5. Nested: (a AND b) OR (NOT c)
# ---------------------------------------------------------------------------

@pytest.fixture
def nested():
    return or_(and_(leaf("a"), leaf("b")), not_(leaf("c")))

def test_nested_both_branches_true(nested):
    # Both OR children are True; all leaves contributed
    assert ev(nested, {"a": True, "b": True, "c": False}) is True
    assert prov(nested, {"a": True, "b": True, "c": False}) == ["a", "b", "c"]

def test_nested_right_branch_wins(nested):
    assert ev(nested, {"a": False, "c": False}) is True
    assert prov(nested, {"a": False, "c": False}) == ["c"]

def test_nested_unknown_blocks(nested):
    # b and c both unknown; NOT(c) is the blocking unknown
    assert ev(nested, {"a": False}) == "unknown"
    assert prov(nested, {"a": False}) == ["c"]

def test_nested_all_false(nested):
    # OR=False needs all children; AND=False returns all its False leaves
    assert ev(nested, {"a": False, "b": False, "c": True}) is False
    assert prov(nested, {"a": False, "b": False, "c": True}) == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# 6. Lazy mode
# ---------------------------------------------------------------------------

def lev(rule, data):
    return evaluate(rule, data, lazy=True)

def lprov(rule, data):
    return sorted(provenance(rule, data, lazy=True))


class TestLazy:
    def test_and_false_before_unknown_short_circuits(self):
        # Lazy AND: two False children — should stop at first, return only its prov.
        # Eager AND would return provenance from BOTH False children.
        r = and_(leaf("a"), leaf("b"))
        assert lev(r, {"a": False, "b": False}) is False
        assert lprov(r, {"a": False, "b": False}) == ["a"]   # only first False
        # Contrast with eager which returns both
        assert sorted(provenance(r, {"a": False, "b": False}, lazy=False)) == ["a", "b"]

    def test_and_unknown_then_false_short_circuits_on_false(self):
        # Lazy AND: unknown child first, then False — False still wins.
        r = and_(leaf("a"), leaf("b"), leaf("c"))
        # a unknown, b False, c unknown
        assert lev(r, {"b": False}) is False
        assert lprov(r, {"b": False}) == ["b"]   # only the False child's prov

    def test_or_true_before_unknown_short_circuits(self):
        # Lazy OR: two True children — should stop at first, return only its prov.
        # Eager OR would return provenance from BOTH True children.
        r = or_(leaf("x"), leaf("y"))
        assert lev(r, {"x": True, "y": True}) is True
        assert lprov(r, {"x": True, "y": True}) == ["x"]   # only first True
        # Contrast with eager which returns both
        assert sorted(provenance(r, {"x": True, "y": True}, lazy=False)) == ["x", "y"]

    def test_or_false_then_true_short_circuits_on_true(self):
        # Lazy OR: False child first, then True — True wins, stop there.
        r = or_(leaf("x"), leaf("y"), leaf("z"))
        # x False, y True, z not evaluated
        assert lev(r, {"x": False, "y": True}) is True
        assert lprov(r, {"x": False, "y": True}) == ["y"]

    def test_and_all_true_matches_eager(self):
        r = and_(leaf("a"), leaf("b"), leaf("c"))
        data = {"a": True, "b": True, "c": True}
        assert lev(r, data) is True
        assert lprov(r, data) == sorted(provenance(r, data, lazy=False))

    def test_or_all_false_matches_eager(self):
        r = or_(leaf("x"), leaf("y"), leaf("z"))
        data = {"x": False, "y": False, "z": False}
        assert lev(r, data) is False
        assert lprov(r, data) == sorted(provenance(r, data, lazy=False))

    def test_nested_lazy_left_branch_short_circuits_or(self):
        # (a AND b) OR (NOT c): lazy OR stops after left branch is True.
        # Eager returns prov from both True branches; lazy returns only left branch.
        r = or_(and_(leaf("a"), leaf("b")), not_(leaf("c")))
        data = {"a": True, "b": True, "c": False}
        assert lev(r, data) is True
        assert lprov(r, data) == ["a", "b"]               # NOT(c) never evaluated
        assert sorted(provenance(r, data, lazy=False)) == ["a", "b", "c"]

    def test_nested_lazy_right_branch_wins(self):
        # Left branch False; lazy OR continues to right branch.
        r = or_(and_(leaf("a"), leaf("b")), not_(leaf("c")))
        data = {"a": False, "c": False}
        assert lev(r, data) is True
        assert lprov(r, data) == ["c"]

    def test_default_lazy_setting(self):
        # Changing default_lazy affects calls with no lazy= kwarg.
        r = or_(leaf("x"), leaf("y"))
        data = {"x": True, "y": True}

        # Baseline: eager default → both True children in prov
        assert sorted(provenance(r, data)) == ["x", "y"]

        engine.default_lazy = True
        try:
            # Now default is lazy → only first True child
            assert sorted(provenance(r, data)) == ["x"]
        finally:
            engine.default_lazy = False   # always restore


# ---------------------------------------------------------------------------
# 7. Condition system
# ---------------------------------------------------------------------------

class TestConditions:

    # --- Condition basics ---------------------------------------------------

    def test_canonical_operator_stored(self):
        c = Condition("age", "gt", 18)
        assert c.operator == "gt"

    def test_alias_normalized_to_canonical(self):
        assert Condition("age", "greater_than", 18).operator == "gt"
        assert Condition("age", ">", 18).operator == "gt"
        assert Condition("x", "equal_to", 1).operator == "eq"
        assert Condition("x", "==", 1).operator == "eq"
        assert Condition("x", "=", 1).operator == "eq"
        assert Condition("x", "less_than", 0).operator == "lt"
        assert Condition("x", "<", 0).operator == "lt"
        assert Condition("x", "less_than_or_equal_to", 0).operator == "lte"
        assert Condition("x", "<=", 0).operator == "lte"
        assert Condition("x", "greater_than_or_equal_to", 0).operator == "gte"
        assert Condition("x", ">=", 0).operator == "gte"
        assert Condition("x", "not_equal_to", 0).operator == "neq"
        assert Condition("x", "!=", 0).operator == "neq"
        assert Condition("x", "<>", 0).operator == "neq"
        assert Condition("x", "is_in", []).operator == "in"
        assert Condition("x", "contains", []).operator == "in"
        assert Condition("x", "is_not_in", []).operator == "not_in"
        assert Condition("x", "not_contains", []).operator == "not_in"

    def test_invalid_operator_raises(self):
        with pytest.raises(ValueError, match="Unknown operator"):
            Condition("age", "bigger_than", 18)

    # --- Condition evaluation via LEAF --------------------------------------

    def test_numeric_gt_true(self):
        r = leaf(Condition("age", "gt", 18))
        assert evaluate(r, {"age": 21}) is True

    def test_numeric_gt_false(self):
        r = leaf(Condition("age", "gt", 18))
        assert evaluate(r, {"age": 16}) is False

    def test_equality_true(self):
        r = leaf(Condition("gender", "eq", "male"))
        assert evaluate(r, {"gender": "male"}) is True

    def test_equality_false(self):
        r = leaf(Condition("gender", "eq", "male"))
        assert evaluate(r, {"gender": "female"}) is False

    def test_membership_in_true(self):
        r = leaf(Condition("nationality", "in", ["American", "British"]))
        assert evaluate(r, {"nationality": "American"}) is True

    def test_membership_in_false(self):
        r = leaf(Condition("nationality", "in", ["American", "British"]))
        assert evaluate(r, {"nationality": "Indian"}) is False

    def test_not_in_true(self):
        r = leaf(Condition("nationality", "not_in", ["American", "British"]))
        assert evaluate(r, {"nationality": "Indian"}) is True

    def test_not_in_false(self):
        r = leaf(Condition("nationality", "not_in", ["American", "British"]))
        assert evaluate(r, {"nationality": "American"}) is False

    def test_is_true_operator(self):
        r = leaf(Condition("flag", "is_true"))
        assert evaluate(r, {"flag": True}) is True
        assert evaluate(r, {"flag": False}) is False

    def test_is_false_operator(self):
        r = leaf(Condition("flag", "is_false"))
        assert evaluate(r, {"flag": False}) is True
        assert evaluate(r, {"flag": True}) is False

    def test_missing_variable_unknown(self):
        r = leaf(Condition("age", "gt", 18))
        assert evaluate(r, {}) == "unknown"
        assert evaluate(r, {"other": 99}) == "unknown"

    def test_type_error_raises_with_info(self):
        r = leaf(Condition("age", "gt", 18))
        with pytest.raises(TypeError, match="age"):
            evaluate(r, {"age": "twenty"})

    # --- Provenance for Condition leaves ------------------------------------

    def test_condition_in_provenance(self):
        cond = Condition("age", "gt", 18)
        r = leaf(cond)
        prov = provenance(r, {"age": 21})
        assert cond in prov

    def test_condition_in_provenance_unknown(self):
        cond = Condition("age", "gt", 18)
        r = leaf(cond)
        prov = provenance(r, {})
        assert cond in prov

    # --- from_expr ----------------------------------------------------------

    def test_from_expr_gt_int(self):
        c = Condition.from_expr("age > 18")
        assert c.variable == "age"
        assert c.operator == "gt"
        assert c.value == 18
        assert isinstance(c.value, int)

    def test_from_expr_eq_string(self):
        c = Condition.from_expr('name == "John"')
        assert c.variable == "name"
        assert c.operator == "eq"
        assert c.value == "John"

    def test_from_expr_gte_float(self):
        c = Condition.from_expr("score >= 95.5")
        assert c.variable == "score"
        assert c.operator == "gte"
        assert c.value == 95.5

    def test_from_expr_neq_unquoted_string(self):
        c = Condition.from_expr("status != active")
        assert c.variable == "status"
        assert c.operator == "neq"
        assert c.value == "active"

    def test_from_expr_in_list(self):
        c = Condition.from_expr('role in ["admin", "editor"]')
        assert c.variable == "role"
        assert c.operator == "in"
        assert c.value == ["admin", "editor"]

    def test_from_expr_invalid_raises(self):
        with pytest.raises(ValueError, match="Cannot parse"):
            Condition.from_expr("good_credit")

    # --- from_business_rules ------------------------------------------------

    def test_from_business_rules_basic(self):
        c = Condition.from_business_rules(
            {"name": "age", "operator": "greater_than", "value": 18}
        )
        assert c.variable == "age"
        assert c.operator == "gt"
        assert c.value == 18

    def test_from_business_rules_alias_equal_to(self):
        c = Condition.from_business_rules(
            {"name": "month", "operator": "equal_to", "value": "December"}
        )
        assert c.operator == "eq"

    def test_from_business_rules_alias_less_than(self):
        c = Condition.from_business_rules(
            {"name": "days", "operator": "less_than", "value": 5}
        )
        assert c.operator == "lt"

    # --- leaf() convenience function ----------------------------------------

    def test_leaf_legacy_string(self):
        r = leaf("good_credit")
        assert r.op == "LEAF"
        assert r.condition == "good_credit"

    def test_leaf_explicit_condition(self):
        r = leaf("age", "gt", 18)
        assert r.op == "LEAF"
        assert isinstance(r.condition, Condition)
        assert r.condition.variable == "age"
        assert r.condition.operator == "gt"
        assert r.condition.value == 18

    def test_leaf_parsed_expression(self):
        r = leaf("age > 18")
        assert r.op == "LEAF"
        assert isinstance(r.condition, Condition)
        assert r.condition.variable == "age"
        assert r.condition.operator == "gt"
        assert r.condition.value == 18

    def test_leaf_prebuilt_condition(self):
        cond = Condition("age", "gt", 18)
        r = leaf(cond)
        assert r.op == "LEAF"
        assert r.condition is cond

    # --- from_business_rules_tree -------------------------------------------

    def test_tree_flat_all(self):
        tree = {
            "all": [
                {"name": "age", "operator": "greater_than", "value": 18},
                {"name": "gender", "operator": "equal_to", "value": "male"},
            ]
        }
        rule = from_business_rules_tree(tree)
        assert rule.op == "AND"
        assert evaluate(rule, {"age": 21, "gender": "male"}) is True
        assert evaluate(rule, {"age": 21, "gender": "female"}) is False

    def test_tree_flat_any(self):
        tree = {
            "any": [
                {"name": "age", "operator": "greater_than", "value": 65},
                {"name": "disability", "operator": "is_true"},
            ]
        }
        rule = from_business_rules_tree(tree)
        assert rule.op == "OR"
        assert evaluate(rule, {"age": 70, "disability": False}) is True
        assert evaluate(rule, {"age": 30, "disability": True}) is True
        assert evaluate(rule, {"age": 30, "disability": False}) is False

    def test_tree_nested_all_any(self):
        tree = {
            "all": [
                {"name": "age", "operator": "greater_than", "value": 18},
                {"any": [
                    {"name": "gender", "operator": "equal_to", "value": "male"},
                    {"name": "income", "operator": "greater_than", "value": 50000},
                ]},
            ]
        }
        rule = from_business_rules_tree(tree)
        # age ok, gender matches
        assert evaluate(rule, {"age": 21, "gender": "male", "income": 30000}) is True
        # age ok, income matches
        assert evaluate(rule, {"age": 21, "gender": "female", "income": 60000}) is True
        # age ok, neither inner condition matches
        assert evaluate(rule, {"age": 21, "gender": "female", "income": 30000}) is False
        # age fails
        assert evaluate(rule, {"age": 16, "gender": "male", "income": 60000}) is False

    def test_tree_unknown_propagation(self):
        tree = {
            "all": [
                {"name": "age", "operator": "greater_than", "value": 18},
                {"name": "income", "operator": "greater_than", "value": 50000},
            ]
        }
        rule = from_business_rules_tree(tree)
        # age present and true; income missing → unknown
        assert evaluate(rule, {"age": 21}) == "unknown"
        # age present and false → False (short-circuit in eager: all False leaves)
        assert evaluate(rule, {"age": 16}) is False

    def test_tree_provenance_contains_conditions(self):
        tree = {
            "all": [
                {"name": "age", "operator": "greater_than", "value": 18},
                {"name": "income", "operator": "greater_than", "value": 50000},
            ]
        }
        rule = from_business_rules_tree(tree)
        prov = provenance(rule, {"age": 21, "income": 60000})
        assert all(isinstance(c, Condition) for c in prov)
        variables = {c.variable for c in prov}
        assert variables == {"age", "income"}

    def test_tree_provenance_unknown_has_condition(self):
        tree = {
            "all": [
                {"name": "age", "operator": "greater_than", "value": 18},
                {"name": "income", "operator": "greater_than", "value": 50000},
            ]
        }
        rule = from_business_rules_tree(tree)
        prov = provenance(rule, {"age": 21})  # income missing
        assert len(prov) == 1
        assert isinstance(prov[0], Condition)
        assert prov[0].variable == "income"

    # --- Backward compatibility ---------------------------------------------

    def test_legacy_string_leaf_true(self):
        r = leaf("approved")
        assert evaluate(r, {"approved": True}) is True
        assert provenance(r, {"approved": True}) == ["approved"]

    def test_legacy_string_leaf_false(self):
        r = leaf("approved")
        assert evaluate(r, {"approved": False}) is False

    def test_legacy_string_leaf_missing(self):
        r = leaf("approved")
        assert evaluate(r, {}) == "unknown"

    def test_legacy_and_or_not_unchanged(self):
        r = and_(leaf("a"), or_(leaf("b"), not_(leaf("c"))))
        assert evaluate(r, {"a": True, "b": False, "c": True}) is False
        assert evaluate(r, {"a": True, "b": True, "c": True}) is True
        assert evaluate(r, {"a": True, "b": False, "c": False}) is True
