import pytest
from shalosh.engine import evaluate, provenance, leaf, and_, or_, not_


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
