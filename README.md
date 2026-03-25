# shalosh

A Python library for evaluating business rules with incomplete data,
and explaining exactly why a rule fired.

## The problem

Rule engines are binary: given complete data they return true or false.
But in practice, data is often incomplete — some features are expensive
to compute, slow to fetch, or simply not yet available. Most engines
either crash, return a default, or silently ignore missing fields.

A second problem: when a rule fires, you often don't know *why*.
In a rule like `(A and B) or C`, if A=True and C=True, which one
actually caused the match? This matters for compliance, debugging,
and understanding your data.

## What this library does

**Three-valued evaluation** — given a rule and partial data, returns:
- `True` — the rule is definitely true given the available data
- `False` — the rule is definitely false given the available data
- `"unknown"` — the outcome depends on features not yet available

**Provenance tracking** — given a rule and data, returns the exact
leaf conditions that determined the outcome. Not a statistical
approximation — a structural, exact answer.

## Installation
```bash
pip install shalosh
```

## Quick start
```python
from shalosh.engine import leaf, and_, or_, not_, evaluate, provenance

# Define a credit approval rule
rule = and_(
    leaf("good_credit_history"),
    or_(
        leaf("stable_employment"),
        leaf("high_savings")
    )
)

# Evaluate with complete data
data = {"good_credit_history": True, "stable_employment": True}
print(evaluate(rule, data))      # True
print(provenance(rule, data))    # ["good_credit_history", "stable_employment"]

# Evaluate with missing data
partial = {"good_credit_history": False}
print(evaluate(rule, partial))   # False  — already determined, no need for more data
print(provenance(rule, partial)) # ["good_credit_history"]
```

## Why this matters

**Early exit on cheap features** — evaluate rules on inexpensive
features first. If the outcome is already determined, skip the
expensive API calls entirely.

**Explainability** — tell users or regulators exactly which conditions
caused a decision, with no ambiguity.

**Partial rollouts** — adding a new feature to your rules? Evaluate
on existing data to see how many decisions would be affected before
the feature is fully available.

**Data quality monitoring** — rules that return `unknown` for large
portions of your traffic signal missing data pipelines.

**Two-phase evaluation** — evaluate cheap "config" features first
to determine rule relevance, then fetch expensive "logic" features
only for relevant rules. Shalosh returns `False` early if config
features already rule it out, and `"unknown"` if the decision
depends on features not yet fetched.

## Demo

See `demo.ipynb` for a worked example on the German Credit dataset (1,000 applicants, 20 features). Key result: dropping a single feature (`age`) leaves `premium_approval` and `auto_reject` completely unaffected — they don't reference age — while `safe_profile` loses 441 resolved decisions to `unknown`. Of those 441, 396 were previously approved and 45 were previously rejected. Provenance tracking identifies `age_ge_25` as the blocking condition in every affected case.

## How it works

Rules are AND/OR/NOT trees over boolean leaf conditions. Evaluation
follows **Strong Kleene three-valued logic**: `unknown` propagates
through a tree unless the known values already determine the outcome
(a single `False` in an AND, a single `True` in an OR).

The naive approach to checking whether a rule could match given partial
data requires checking all possible assignments for missing features —
exponential in the worst case. Shalosh instead answers a more tractable
question: do the known features already determine the outcome? This
reduces to a single O(n) tree traversal. For minimal rules, `"unknown"`
precisely means "both outcomes are still possible." For non-minimal rules
containing contradictions or tautologies, this guarantee does not hold.

Provenance tracking recurses through the same tree and collects only
the leaves that were load-bearing for the result.

Both algorithms are O(n) in the number of conditions.

## Roadmap

- [ ] Rule minimization — simplify bloated rules to minimal equivalents
- [ ] Pivotal feature identification — which missing feature, if known,
      would resolve the most unknowns
- [ ] Adapter for Drools / easy-rules
- [ ] Evaluation reordering optimizer
- [ ] Provenance deduplication — merge conditions on the same feature across branches (e.g. `a>1, a<3` → `1<a<3`)
- [ ] Lazy evaluation mode — short-circuit child evaluation to match behaviour of lazy rule engines

## License

MIT
