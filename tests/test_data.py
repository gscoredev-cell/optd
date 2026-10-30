import re
from fractions import Fraction

import pytest

import data


def _fake_gsm():
    """Three configs sharing original_ids, like GSM-Symbolic (templates per GSM8K test question)."""
    rows = lambda cfg, n: [{"id": oid, "instance": k, "original_id": oid, "question": f"{cfg} q{oid}.{k}",
                            "answer": f"... #### {oid * 10 + k}"} for oid in range(n) for k in range(3)]
    return {"main": rows("main", 20), "p1": rows("p1", 16), "p2": rows("p2", 12)}


def test_split_by_original_id_across_configs():
    gsm = _fake_gsm()
    tr, te = data.make_split(gsm)
    assert tr.isdisjoint(te) and tr | te == set(range(20))
    train = data.problems(gsm, ["main", "p1"], tr)
    test = data.problems(gsm, ["main", "p1", "p2"], te)
    assert {p["oid"] for p in train}.isdisjoint({p["oid"] for p in test})   # no template leaks across the split
    assert all(p["set"] in ("main", "p1") for p in train)
    assert {p["set"] for p in test} == {"main", "p1", "p2"}
    assert data.make_split(gsm) == (tr, te)


def test_problem_fields_and_gold():
    p = data.gsm_problem("p2", {"id": 4, "instance": 2, "original_id": 77, "question": "q", "answer": "x #### 1,050"})
    assert p == {"pid": "p2:4:2", "set": "p2", "oid": 77, "question": "q", "gold": Fraction(1050)}


def test_sample_deterministic_and_sorted():
    items = [{"pid": f"main:{i}:0"} for i in range(50)]
    a = data.sample(items, 10)
    assert a == data.sample(list(reversed(items)), 10)
    assert [x["pid"] for x in a] == sorted(x["pid"] for x in a)
    assert len(data.sample(items, 99)) == 50


@pytest.mark.parametrize("expr, value", [
    ("2+3*4", 14), ("(2+3)*4", 20), ("10-4-3", 3), ("8/4/2", 1), ("2^3", 8), ("3^2*2", 18),
    ("\\frac{1}{2}+1", Fraction(3, 2)), ("(\\frac{3}{4})/(5^2)", Fraction(3, 100)), ("7-\\frac{14}{4}", Fraction(7, 2)),
])
def test_eval_expr(expr, value):
    assert data.eval_expr(expr) == value


def test_eval_expr_follows_string_not_tree():
    # the generator never parenthesises +/- operands: a - (b + c) is written a-b+c and scored as written
    assert data.eval_expr("10-2+3") == 11


def test_random_calculation():
    a = data.random_calculation(100, seed=0)
    assert a == data.random_calculation(100, seed=0)
    assert a != data.random_calculation(100, seed=1)
    assert len(a) == 100 and len({p["pid"] for p in a}) == 100
    for p in a:
        assert 4 <= p["ops"] <= 8
        assert p["ops"] == len(re.findall(r"[-+*/]", re.sub(r"\\frac\{\d+\}\{\d+\}", "F", p["expr"])))   # leaves: no ops
        assert data.eval_expr(p["expr"]) == p["gold"] and abs(p["gold"]) <= 10**12
        assert p["question"].endswith(p["expr"]) and p["set"] == "rc"
