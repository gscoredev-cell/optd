"""Oracle cut, bisection (monotone and non-monotone MC vectors, level-synchronous runner with a fake generator and
an on-disk MC store), seed pairing across arms, sloppy-cut mixing and the interaction statistic."""
import random

import pytest

import oracle as O
from option_e import token_share
from train_repair import sloppy_mix


@pytest.mark.parametrize("mc, b", [
    ([0.0, 0.0], 0),                        # MC_0 < 0.5: restart
    ([0.33, 1.0, 1.0], 0),
    ([1.0, 1.0, 1.0], 2),                   # recoverable to the end: b* = B
    ([1.0, 0.67, 0.33, 0.0], 1),
    ([0.67, 0.67, 0.67, 0.33], 2),
    ([1.0, 0.0], 0),                        # the first step already loses it: b* = 0 is "the prompt", MC_0 >= 0.5
    ([0.67, 0.0, 1.0, 0.0], 2),             # non-monotone: the largest recoverable point before a drop
])
def test_oracle_cut(mc, b):
    assert O.oracle_cut(mc) == b


def test_oracle_cut_empty():
    assert O.oracle_cut([]) is None


def _vec(B, b_star, rng=None):
    """Monotone MC vector with the last recoverable point at b_star (b_star = B: never drops)."""
    return [1.0 if j <= b_star else 0.0 for j in range(B + 1)]


@pytest.mark.parametrize("B", [1, 2, 3, 5, 8, 12])
def test_bisection_matches_full_scan_on_monotone(B):
    for b_star in range(B + 1):
        mc = _vec(B, b_star)
        probes = []
        b, probed = O.bisect(lambda j: (probes.append(j), mc[j])[1], B)
        assert b == O.oracle_cut(mc) == b_star
        assert len(probes) == len(set(probes)) == len(probed)       # every point probed once
        assert len(probes) <= 2 + max(1, B.bit_length())            # ~ 2 + log2(B)


def test_bisection_restart_costs_one_probe():
    b, probed = O.bisect(lambda j: 0.0, 7)
    assert b == 0 and list(probed) == [0]


def test_bisection_to_end_costs_two_probes():
    b, probed = O.bisect(lambda j: 1.0, 7)
    assert b == 7 and sorted(probed) == [0, 7]


def test_bisection_non_monotone_is_within_one_or_flagged():
    """On a non-monotone vector bisection returns *a* recoverable point followed by an unrecoverable one; the full
    scan picks the last one. agree() is what E-S3 measures."""
    mc = [1.0, 0.0, 1.0, 1.0, 0.0, 0.0]
    b, _ = O.bisect(lambda j: mc[j], 5)
    assert mc[b] >= 0.5 and mc[b + 1] < 0.5
    assert O.oracle_cut(mc) == 3
    assert O.agree(b, 3) in (True, False)
    assert O.agree(3, 4) and not O.agree(1, 3)


def _trace(i, B, bounds_gap=9):
    return {"pid": f"p1:{i}:0", "set": "p1", "k": 0, "prompt": [100 + i, 7], "gen": list(range(1, 12 * B + 2)),
            "bounds": [bounds_gap * j + 3 for j in range(B)], "ok": False}


def _fake_gen(truth, calls):
    """Generator stand-in: a completion is 'correct' when truth[(sid, point)] >= 0.5; returns token lists."""
    def gen_fn(prefixes, seeds, cuts):
        calls.append(len(prefixes))
        return [[1, 99] for _ in prefixes], [False for _ in prefixes]

    def score(t, cut, toks, trunc):
        b = next(j for j in range(len(t["bounds"]) + 1) if O.prefix_len(t["bounds"], j) == cut)
        return truth[(O.sid(t), b)] >= 0.5
    return gen_fn, score


def test_run_bisection_level_synchronous_and_store(tmp_path):
    traces = [_trace(i, B) for i, B in enumerate((3, 5, 8, 12))]
    stars = {O.sid(t): random.Random(i).randint(0, len(t["bounds"])) for i, t in enumerate(traces)}
    truth = {(O.sid(t), j): (1.0 if j <= stars[O.sid(t)] else 0.0) for t in traces for j in range(len(t["bounds"]) + 1)}
    store = O.MCStore("m", root=tmp_path)
    calls = []
    done, rounds = O.run_bisection(_fake_gen(truth, calls), store, traces)
    assert {s: v["b"] for s, v in done.items()} == stars
    assert rounds <= 2 + 4 and all(n % O.MC_K == 0 for n in calls)       # 3 completions per probe, batched per round
    # a second run is free: everything comes from the MC store
    calls2 = []
    done2, _ = O.run_bisection(_fake_gen(truth, calls2), O.MCStore("m", root=tmp_path), traces)
    assert done2 == done and calls2 == []
    # the full scan only pays for the points bisection did not probe
    calls3 = []
    full = O.run_full_scan(_fake_gen(truth, calls3), O.MCStore("m", root=tmp_path), traces)
    assert all(O.oracle_cut(full[s]) == stars[s] for s in full) and len(full) == 4
    assert sum(calls3) == O.MC_K * (sum(len(t["bounds"]) + 1 for t in traces) - sum(len(v["probed"]) for v in done.values()))


def test_store_import_vectors(tmp_path):
    t = _trace(0, 3)
    store = O.MCStore("m", root=tmp_path)
    assert store.import_vectors([(t, [1.0, 0.67, 0.33, 0.0])]) == 4
    assert store.import_vectors([(t, [1.0, 0.67, 0.33, 0.0])]) == 0
    assert O.MCStore("m", root=tmp_path).get(t, 1) == 0.67


def test_arm_items_pair_seeds_and_clamp():
    units = [{"trace": _trace(0, 4), "b": 0}, {"trace": _trace(1, 4), "b": 2}, {"trace": _trace(2, 4), "b": 4}]
    items = {a: O.arm_items(units, a) for a in ("E0", "E1", "E2", "E3", "E4a", "E4b", "E6")}
    assert all([it["sid"] for it in items[a]] == [it["sid"] for it in items["E1"]] for a in items)   # same order
    assert [it["cut"] for it in items["E1"]] == [it["cut"] for it in items["E3"]] == [0, units[1]["trace"]["bounds"][1] + 1, units[2]["trace"]["bounds"][3] + 1]
    assert [it["cut"] for it in items["E0"]] == [0, 0, 0] == [it["cut"] for it in items["E6"]]
    assert [it["adapter"] for it in items["E3"]] == [True] * 3 and [it["adapter"] for it in items["E2"]] == [False] * 3
    assert [it["inject"] for it in items["E2"]] == [True] * 3 and not any(it["inject"] for it in items["E3"])
    assert [it["clamped"] for it in items["E4a"]] == [True, False, False]      # b* = 0 cannot go earlier
    assert [it["clamped"] for it in items["E4b"]] == [False, False, True]      # b* = B cannot go later
    assert O.chunk_seeds("r", "arms", 130, 64) == O.chunk_seeds("r", "arms", 130, 64) and len(O.chunk_seeds("r", "arms", 130, 64)) == 3
    assert O.chunk_seeds("r", "arms", 10, 64) != O.chunk_seeds("r", "e5", 10, 64)


def test_false_cut_is_seeded_and_in_range():
    t = _trace(3, 6)
    cuts = {O.false_cut(t, "run") for _ in range(5)}
    assert len(cuts) == 1 and 1 <= cuts.pop() <= 6
    assert O.false_cut({**t, "bounds": []}, "run") == 0


def test_sloppy_mix_and_token_share():
    units = [{"trace": _trace(i, 4), "b": b} for i, b in enumerate([0, 1, 2, 3, 4, 2, 3, 1])]
    mixed = sloppy_mix(units, 0.25, seed=1)
    extra = mixed[len(units):]
    assert len(extra) == round(0.25 * 7) == 2 and all(e["sloppy"] and e["b"] >= 0 for e in extra)
    assert sloppy_mix(units, 0.25, seed=1) == mixed                          # seeded
    assert token_share([1, 2, 3, 4], [1, 2, 9, 4], 4) == 0.75 and token_share([1], [1], 0) == 0.0


def test_interaction_and_cluster_bootstrap():
    rng = random.Random(0)
    rows, groups = [], []
    for i in range(200):
        g = i // 2
        e0, e6 = rng.random() < 0.2, rng.random() < 0.3        # adapter alone adds 10 points
        e1, e3 = rng.random() < 0.3, rng.random() < 0.6        # at the pointer it adds 30: I ~ +20
        rows.append({"E0": e0, "E1": e1, "E2": e1, "E3": e3, "E6": e6, "E4a": e3, "E4b": e3})
        groups.append(g)
    assert O.interaction(rows) == pytest.approx(O.delta(rows, "E3", "E1") - O.delta(rows, "E6", "E0"))
    point, ci = O.cluster_bootstrap(rows, O.interaction, groups, n=400, seed=0)
    assert ci[0] <= point <= ci[1] and ci[0] > 0
    point, ci = O.cluster_bootstrap(rows, lambda rs: O.delta(rs, "E2", "E1"), groups, n=200)
    assert point == 0 and ci == [0.0, 0.0]
