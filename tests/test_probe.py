"""probe.fit / probe.fit_step end to end on synthetic boundary features (needs numpy + scikit-learn, no torch)."""
import numpy as np
import pytest

pytest.importorskip("sklearn")
import probe


def _traces(n=240, d=16, signal=2.0, seed=0):
    """Two 'layers'; only layer index 1 carries the label. Two traces per original_id, like greedy + sample."""
    rng = np.random.default_rng(seed)
    feats, wrong, oids = [], [], []
    for i in range(n):
        w = int(rng.random() < 0.4)
        nb = int(rng.integers(3, 9))
        f = rng.normal(size=(nb, 2, d)).astype(np.float32)
        f[:, 1, 0] += signal * w
        feats.append(f), wrong.append(w), oids.append(i // 2)
    return feats, wrong, oids


def test_fit_finds_signal_layer_and_meets_gate():
    feats, wrong, oids = _traces()
    pr = probe.fit(feats, wrong, oids, layers=[21, 28])
    m = pr["metrics"]
    assert set(k.split("|")[0] for k in m["probe_table"]) == {"21", "28", "21+28"}
    assert pr["name"] in ("28", "21+28")                       # layer 28 (index 1) carries the signal
    assert m["probe_auroc"] > 0.9 and m["probe_fpr"] <= 0.10 and m["probe_gate"]
    assert all("cv_auroc" in v and "_model" not in v for v in m["probe_table"].values())
    assert sum("heldout_auroc" in v for v in m["probe_table"].values()) == 3   # one per layer set
    f = probe.flag(pr, feats[0])
    assert f is None or 1 <= f <= len(feats[0])


def test_fit_per_set_and_weighted_tau():
    feats, wrong, oids = _traces(n=400)
    sets = ["rc" if o % 4 == 0 else "main" for o in oids]
    pr = probe.fit(feats, wrong, oids, layers=[21, 28], layer_sets={"28": [28]}, c_grid=(1e-2,),
                   trace_sets=sets, eval_share={"main": 0.5, "rc": 0.5})
    m = pr["metrics"]
    assert m["probe_tau_weighted"] and set(m["probe_by_set"]) == {"main", "rc"}
    assert all(v["auroc"] > 0.85 for v in m["probe_by_set"].values())


def test_fit_with_fixed_holdout():
    feats, wrong, oids = _traces()
    hold = set(range(0, 120, 5))
    m = probe.fit(feats, wrong, oids, layers=[21, 28], layer_sets={"28": [28]}, holdout=hold)["metrics"]
    assert m["probe_holdout_traces"] == sum(o in hold for o in oids)


def test_fit_past_deadline_keeps_one_complete_pair():
    feats, wrong, oids = _traces()
    m = probe.fit(feats, wrong, oids, layers=[21, 28], deadline=0)["metrics"]   # deadline already passed
    assert list(m["probe_table"]) == ["21|C=0.0001"] and m["probe_layers"] == "21"
    assert "heldout_auroc" in m["probe_table"]["21|C=0.0001"]


def test_fit_without_signal_fails_gate():
    feats, wrong, oids = _traces(signal=0.0)
    m = probe.fit(feats, wrong, oids, layers=[21, 28])["metrics"]
    assert m["probe_auroc"] < 0.75 and not m["probe_gate"]


def test_single_class_is_reported_not_raised():
    feats, _, oids = _traces(n=40)
    pr = probe.fit(feats, [0] * 40, oids, layers=[21, 28])
    assert pr["name"] is None and not pr["metrics"]["probe_gate"] and probe.flag(pr, feats[0]) is None


def test_fit_step_finds_the_bad_boundary():
    rng = np.random.default_rng(1)
    feats, step_y, wrong, oids = [], [], [], []
    for i in range(300):
        nb = int(rng.integers(4, 10))
        f = rng.normal(size=(nb, 2, 8)).astype(np.float32)
        y = np.zeros(nb, dtype=int)
        if rng.random() < 0.4:
            j = int(rng.integers(0, nb))
            y[j] = 1
            f[j:, 0, 0] += 3.0                  # the slip shows up from the bad step on, in layer index 0
        feats.append(f), step_y.append(y), wrong.append(int(y.any())), oids.append(i // 2)
    pr = probe.fit_step(feats, step_y, wrong, oids, layers=[21, 28], layer_sets={"21": [21], "28": [28]})
    m = pr["metrics"]
    assert pr["name"] == "21" and m["step_auroc"] > 0.8 and m["step_trace_auroc"] > 0.9
    assert m["step_first_bad_within1"] > 0.5 and m["step_pos_heldout"] > 0


def test_fit_step_without_positives_reports_blocker():
    feats, _, oids = _traces(n=40)
    pr = probe.fit_step(feats, [np.zeros(len(f), int) for f in feats], [0] * 40, oids, layers=[21, 28])
    assert pr["name"] is None and "step_blocker" in pr["metrics"]
