"""probe.fit end to end on synthetic boundary features (needs numpy + scikit-learn, no torch)."""
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
    assert all("cv_auroc" in v for v in m["probe_table"].values())
    f = probe.flag(pr, feats[0])
    assert f is None or 1 <= f <= len(feats[0])


def test_fit_without_signal_fails_gate():
    feats, wrong, oids = _traces(signal=0.0)
    m = probe.fit(feats, wrong, oids, layers=[21, 28])["metrics"]
    assert m["probe_auroc"] < 0.75 and not m["probe_gate"]


def test_single_class_is_reported_not_raised():
    feats, _, oids = _traces(n=40)
    pr = probe.fit(feats, [0] * 40, oids, layers=[21, 28])
    assert pr["name"] is None and not pr["metrics"]["probe_gate"] and probe.flag(pr, feats[0]) is None
