import hashlib
import json
import math
import os
from fractions import Fraction

import pytest

import core


# strict / lenient scoring
@pytest.mark.parametrize("text, value", [
    ("step\nThe final answer is 18", Fraction(18)),
    ("The final answer is $1,234.50.", Fraction("1234.50")),
    ("The final answer is: **42**", Fraction(42)),
    ("The final answer is 35%.", Fraction(35)),
    ("The final answer is \\frac{7}{3}", Fraction(7, 3)),
    ("The final answer is $-\\dfrac{1}{2}$", Fraction(-1, 2)),
    ("The final answer is 7/3", Fraction(7, 3)),
    ("The final answer is -5", Fraction(-5)),
])
def test_strict_values(text, value):
    assert core.extract_strict(text)[0] == value


def test_strict_uses_last_phrase_and_missing_is_none():
    assert core.extract_strict("The final answer is 3\nwait\nThe final answer is 4")[0] == 4
    assert core.extract_strict("so the answer is 12") is None
    assert core.extract_strict("The final answer is unknown") is None


def test_lenient_falls_back_to_last_number():
    assert core.extract_lenient("a = 3, b = 5, so 8 total")[0] == 8
    assert core.extract_lenient("12 then The final answer is 7 and 99")[0] == 7   # strict wins
    assert core.extract_lenient("no numbers here") is None


def test_matches_precision_and_exact_forms():
    g = Fraction(7, 3)
    assert core.matches(core.extract_strict("The final answer is 2.33"), g)
    assert core.matches(core.extract_strict("The final answer is 2.3"), g)
    assert not core.matches(core.extract_strict("The final answer is 2.4"), g)
    assert core.matches(core.extract_strict("The final answer is \\frac{14}{6}"), g)
    assert not core.matches(core.extract_strict("The final answer is 2"), g)
    assert core.matches(core.extract_strict("The final answer is 18.00"), Fraction(18))
    assert not core.matches(core.extract_strict("The final answer is 17"), Fraction(18))
    assert not core.matches(None, Fraction(18))


def test_gold_gsm():
    assert core.gold_gsm("Tom has 3 + 4 = 7 apples.\n#### 1,207") == 1207


# boundaries and rollback
def test_boundaries_min_gap_and_max():
    texts = ["a"] * 40
    for i in (2, 5, 10, 11, 20, 30):
        texts[i] = "x\n"
    assert core.boundaries(texts) == [2, 10, 20, 30]          # 5 and 11 are < 8 after the previous boundary
    assert core.boundaries(texts, max_n=2) == [2, 10]
    assert core.boundaries(["\n\n"] + ["a"] * 7 + [".\n"]) == [0, 8]   # newline inside a longer token counts


def test_boundaries_on_decoded_text_not_token_ids():
    assert core.boundaries(["Step", " 1", ":", " 3", "+", "4", "=", "7", "Ċ"]) == []   # raw BPE byte, not decoded
    assert core.boundaries(["Step", " 1", ":", " 3", "+", "4", "=", "7", "\n"]) == [8]


def test_rollback_cut():
    b = [5, 14, 30]
    assert core.rollback_cut(b, 1) == 0     # restart from the prompt
    assert core.rollback_cut(b, 2) == 5     # tokens before boundary 1
    assert core.rollback_cut(b, 3) == 14


# seeds
def test_seed_for_matches_spec_formula():
    expect = int(hashlib.sha256(b"RUN-1:A2:p2:0:1").hexdigest(), 16) % 2**31
    assert core.seed_for("RUN-1", "A2", "p2:0:1") == expect
    assert core.seed_for("RUN-1", "A2", "p2:0:1") != core.seed_for("RUN-1", "A3", "p2:0:1")
    assert 0 <= core.seed_for("x", "y", 3) < 2**31


# split
def test_split_by_oid_deterministic_disjoint_halves():
    oids = [3, 1, 2, 2, 9, 7, 7, 4, 5, 8]
    tr, te = core.split_by_oid(oids)
    assert (tr, te) == core.split_by_oid(list(reversed(oids)))   # order of input does not matter
    assert tr.isdisjoint(te) and tr | te == set(oids)
    assert len(tr) == len(set(oids)) // 2
    assert core.split_by_oid(oids, seed=1) != (tr, te)


def test_holdout_by_oid():
    h = core.holdout_by_oid(range(100))
    assert len(h) == 20 and h == core.holdout_by_oid(range(100))


# thresholds
def test_tau_at_fpr_is_smallest_threshold_with_fpr_at_most_10pct():
    neg = [i / 10 for i in range(1, 11)]                       # 0.1 .. 1.0
    t = core.tau_at_fpr(neg)
    assert sum(x >= t for x in neg) / len(neg) <= 0.10
    assert sum(x >= math.nextafter(t, -math.inf) for x in neg) / len(neg) > 0.10
    ties = [0.5] * 9 + [0.9]
    t = core.tau_at_fpr(ties)
    assert sum(x >= t for x in ties) == 1


def test_auroc():
    assert core.auroc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert core.auroc([0.1], [0.9]) == 0.0
    assert core.auroc([0.5, 0.5], [0.5, 0.5]) == 0.5
    assert core.auroc([0.8, 0.4], [0.6, 0.2]) == 0.75


# atomic status
def test_write_status_atomic(tmp_path, monkeypatch):
    run = tmp_path / "RUN-1"
    run.mkdir()
    core.write_status(run, "S1")
    s = json.loads((run / "status.json").read_text())
    assert s["run_id"] == "RUN-1" and s["state"] == "RUNNING" and s["step"] == "S1" and isinstance(s["ts"], int)
    assert (run / ".step").read_text() == "S1"
    core.write_status(run, "S2", "DONE")
    assert json.loads((run / "status.json").read_text())["state"] == "DONE"
    assert sorted(os.listdir(run)) == [".step", "status.json"]   # no temp files left

    def boom(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(core.json, "dump", boom)
    with pytest.raises(OSError):
        core.write_status(run, "S3")
    assert json.loads((run / "status.json").read_text())["step"] == "S2"   # old status intact


def test_kill_plan():
    assert core.kill_plan(40) == (10, 15)
    assert core.kill_plan(16) == (10, 15)
    c, k = core.kill_plan(9)
    assert 2 <= c < k < 9 and 2 * c > k        # one checkpoint before the kill, resume replays (c, k]
    assert core.kill_plan(4) is None


def test_arm_metrics():
    rows = [dict(a0_ok=False, ok=True, flagged=True, a0_tokens=100, tokens=150, truncated=False),
            dict(a0_ok=True, ok=False, flagged=True, a0_tokens=100, tokens=120, truncated=True),
            dict(a0_ok=True, ok=True, flagged=False, a0_tokens=100, tokens=100, truncated=False),
            dict(a0_ok=False, ok=False, flagged=False, a0_tokens=100, tokens=100, truncated=False)]
    m = core.arm_metrics(rows)
    assert (m["recovered"], m["broken"], m["net"]) == (1, 1, 0)
    assert m["interv_rate"] == 0.5 and m["fpr"] == 0.5 and m["extra_tokens"] == 17.5 and m["trunc_rate"] == 0.25
