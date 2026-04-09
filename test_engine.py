import pytest
import shalosh.engine as engine
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
