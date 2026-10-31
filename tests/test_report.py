"""report.smoke3 renders from realistic stage metrics (full run and every stage skipped)."""
import json
import time

import numpy as np
import pytest

pytest.importorskip("sklearn")
import probe
import report

CFG = {"datacenter": "US-KS-2", "image": "img:tag", "student": "Qwen/Qwen3-1.7B", "teacher": "Qwen/Qwen3-8B",
       "phase_budget_usd": {"SMOKE3": 1.5}, "smoke3_source_run": "SMOKE-x", "rc_ops_full": 3}
SETS = ("p2", "main", "p1", "rc")


def _run_dir(tmp_path, monkeypatch):
    (tmp_path / "env.json").write_text(json.dumps({
        "run_id": "SMOKE3-1", "phase": "SMOKE3", "git_sha": "abcdef1234", "pod_id": "pod", "gpu": "L40", "tier": 48,
        "driver": "1", "max_minutes": "90", "price_per_hr": 0.82, "image_pull_sec": 6, "ssh_public_port": ""}))
    (tmp_path / "downloads.jsonl").write_text('{"repo":"Qwen/Qwen3-4B","kind":"model","cache":"download","mb":8060,'
                                              '"sec":31.0,"mb_per_s":260.0}\n')
    monkeypatch.setenv("BOOT_EPOCH", str(int(time.time()) - 600))
    return tmp_path


def _base():
    return {"pipeline_start": time.time() - 500, "torch": "2.8", "cuda": "12.8", "transformers": "4.57.1",
            "peft": "0.17.1", "capability": [8, 9], "vram_gb": 44.4, "bf16_matmul_rel_err": 0.002,
            "template_identical_ids": 20, "template_empty_think": 20, "template_n": 20,
            **{f"{s}_min": 1.0 for s in ("MCa", "MCprobe", "B1", "B2", "B3", "C2", "C3")}}


def _synthetic(n=300, seed=0):
    rng = np.random.default_rng(seed)
    feats, wrong, oids, sets, step_y = [], [], [], [], []
    for i in range(n):
        w = int(rng.random() < 0.4)
        nb = int(rng.integers(3, 8))
        f = rng.normal(size=(nb, 2, 8)).astype(np.float32)
        f[:, 0, 0] += 2 * w
        y = np.zeros(nb, int)
        if w:
            y[int(rng.integers(0, nb))] = 1
        feats.append(f), wrong.append(w), oids.append(i // 2), sets.append(SETS[i % 4]), step_y.append(y)
    return feats, wrong, oids, sets, step_y


def test_smoke3_report_full(tmp_path, monkeypatch):
    run = _run_dir(tmp_path, monkeypatch)
    feats, wrong, oids, sets, step_y = _synthetic()
    pm = probe.fit(feats, wrong, oids, [21, 28], trace_sets=sets,
                   eval_share={s: 0.25 for s in SETS})["metrics"]
    eval_fpr = {s: {"n_correct": 10, "fpr": 0.1} for s in (*SETS, "pooled")}
    pm["probe_eval_fpr"] = eval_fpr
    sm = probe.fit_step(feats, step_y, wrong, oids, [21, 28], layer_sets={"21": [21]})["metrics"]
    m = {**_base(), **sm, "step_eval_fpr": eval_fpr,
         "mc_traces": 300, "mc_of": 300, "mc_completions": 8700, "mc_tok_per_s": 900.0, "mc_step_pos": 50,
         "mc_traces_with_bad_step": 45, "mc_wrong_traces": 100, "mc_wrong_with_bad_step": 40,
         "mc_prompt_success_correct": 0.8, "mc_prompt_success_wrong": 0.3,
         "b1_tok_per_s": 300.0, "b1_n": 380, "b1_gap_p2": 0.3, "b1_gap_p2_se": 0.03,
         **{f"b1_{w}_{s}_{k}": 0.5 for w in ("student", "teacher") for s in SETS for k in ("primary", "strict", "trunc")},
         **{f"{p}_{k}": 0.5 for p in ("b2", "c2") for k in ("greedy_primary", "sampled_primary", "trunc")},
         **{f"{p}_greedy_primary_{s}": 0.5 for p in ("b2", "c2") for s in SETS},
         **{f"{p}_{k}": v for p in ("b2", "c2") for k, v in
            (("n_problems", 600), ("of", 600), ("tok_per_s", 400.0), ("mean_tokens", 200.0), ("sec", 600.0))},
         **{f"b3_{k}": v for k, v in pm.items()}, **{f"c3_{k}": v for k, v in pm.items()},
         "c3_curve": [{"total": 300, "train_traces": 240, "heldout_traces": 480, "C": 0.01, "auroc": 0.7,
                       "recall": 0.3, "by_set": {"main": 0.7, "rc": 0.6}},
                      {"total": 2400, "skipped": "stage time budget"}]}
    report.smoke3(run, m, CFG)
    text = (run / "report.md").read_text()
    for needle in ("(a) PROBE-MC", "step probe: layers 21", "(b) QWEN3-4B", "held-out AUROC per set",
                   "eval (test traces) FPR per set", "(c) DATA SCALING", "2400 traces: stage time budget",
                   "GATE", "BLOCKERS: none"):
        assert needle in text, needle
    json.loads((run / "metrics.json").read_text())


CFG4 = {**CFG, "phase_budget_usd": {"SMOKE3": 1.5}, "smoke4_branch": "A", "smoke4_stage0": "within-pair 0.75"}


def test_smoke4_report_full(tmp_path, monkeypatch):
    from test_probe import _pair_traces
    run = _run_dir(tmp_path, monkeypatch)
    feats, wrong, problem, groups = _pair_traces(n_problems=60)
    pm = probe.fit_within(feats, wrong, problem, groups, [14, 21], layer_sets={"14": [14], "21": [21]},
                          c_grid=(1e-2,))["metrics"]
    dist = {"train_p1": {"problems": 200, "hist": [40, 30, 30, 30, 70], "acc": 0.55, "contested": 0.45},
            "test_p2": {"problems": 200, "hist": [80, 20, 20, 20, 60], "acc": 0.4, "contested": 0.3}}
    test = {s: {"problems": 50, "correct": 90, "wrong": 110, "fpr": 0.1, "recall": 0.45, "pairs": 50,
                "within_auroc": 0.8, "within_ci": [0.7, 0.9]} for s in ("p1", "p2", "pooled")}
    m = {**_base(), **{f"{s}_min": 1.0 for s in ("A1", "A2", "A4", "A3")}, **pm,
         "a1_problems": 800, "a1_of": 800, "a1_tok_per_s": 700.0, "a1_bs": 128, "a1_trunc": 0.01,
         "a1_mean_tokens": 280.0, "a1_dist": dist, "a4_test": test,
         "a3_traces": 100, "a3_of": 100, "a3_completions": 3000, "a3_tok_per_s": 500.0, "a3_restart_share": 0.3,
         "a3_never_share": 0.1, "a3_mc_prompt_mean": 0.5, "a3_first_drop_hist": {"0": 10, "3": 5, "never": 2}}
    report.smoke4(run, m, CFG4)
    text = (run / "report.md").read_text()
    for needle in ("BRANCH: A", "A1 MC STRATIFICATION", "train_p1: problems 200", "WITHIN-PROBLEM AUROC",
                   "by boundary position", "A4 TEST", "pooled: problems 50", "A3 STEP LOCALIZATION",
                   "restart share 30.0%", "GATE", "BLOCKERS: none"):
        assert needle in text, needle
    assert json.loads((run / "metrics.json").read_text())["smoke4_gate"] in (True, False)


def test_smoke4_report_all_skipped(tmp_path, monkeypatch):
    run = _run_dir(tmp_path, monkeypatch)
    m = {**_base(), **{f"{s}_skipped": "skipped: not enough pod time left" for s in ("A1", "A2", "A4", "A3")}}
    report.smoke4(run, m, CFG4)
    text = (run / "report.md").read_text()
    assert "GATE (within-problem AUROC >= 0.80 held-out and recall >= 0.40 at FPR 10% on test): n/a" in text
    assert "BLOCKERS: A1 skipped" in text


def test_smoke3_report_all_skipped(tmp_path, monkeypatch):
    run = _run_dir(tmp_path, monkeypatch)
    m = {**_base(), **{f"{s}_skipped": "skipped: not enough pod time left"
                       for s in ("MCa", "MCprobe", "B1", "B2", "B3", "C2", "C3")}}
    report.smoke3(run, m, CFG)
    text = (run / "report.md").read_text()
    assert "BLOCKERS: MCa skipped" in text
    assert "b3 n/a ; c3 n/a" in text                          # no gate verdict without a probe


CFG5 = {**CFG, "phase_budget_usd": {"SMOKE3": 4}}


def test_smoke5_report_full(tmp_path, monkeypatch):
    from test_probe import _pair_traces
    run = _run_dir(tmp_path, monkeypatch)
    feats, wrong, problem, groups = _pair_traces(n_problems=60)
    pm = probe.fit_within(feats, wrong, problem, groups, [14, 21], layer_sets={"14": [14], "21": [21]},
                          c_grid=(1e-2,))["metrics"]
    rng = np.random.default_rng(3)
    step_y = [(rng.random(len(f)) < 0.3).astype(int) for f in feats]
    for f, y in zip(feats, step_y):
        f[:, 0, 1] += 2 * y
    sm = probe.fit_step(feats, step_y, wrong, groups, [14, 21], layer_sets={"14": [14]}, c_grid=(1e-2,))["metrics"]
    rising = {k: {"n": 40, "auroc": 0.5 + 0.03 * k} for k in range(1, 13)}
    flat = {str(k): {"n": 40, "pos": 9, "auroc": 0.6} for k in range(1, 13)}
    dist = {f"{sp}_{s}": {"problems": 100, "hist": [20, 20, 20, 20, 20], "acc": 0.5, "contested": 0.6}
            for sp in ("train", "test") for s in ("rc3", "rc4", "p2", "all")}
    test_a = {s: {"problems": 30, "correct": 60, "wrong": 60, "fpr": 0.1, "recall": 0.3, "pairs": 30,
                  "within_auroc": 0.78, "within_ci": [0.7, 0.86]} for s in ("rc3", "rc4", "p2", "pooled")}
    test_b = {s: {"traces": 20, "boundaries": 200, "unrec": 60, "auroc": 0.7, "fpr": 0.1, "recall": 0.4}
              for s in ("rc3", "pooled")}
    m = {**_base(), **{f"{s}_min": 1.0 for s in ("D1", "D2", "D4a", "D3", "D4b")}, **pm,
         **{f"d3_{k}": v for k, v in sm.items()}, "d2_feat_sec": 90.0,
         "d1_seed_traces": 380, "d1_seed_imported": 380, "d1_seed_used": 100, "d1_seed_prompt_ids_same": 100,
         **{f"{s}_cache_{k}": 5 for s in ("d1", "d3", "d4b") for k in ("hits", "misses")},
         **{f"{s}_tok_per_s": 600.0 for s in ("d1", "d3", "d4b")},
         "d1_problems": 700, "d1_of": 700, "d1_bs": 128, "d1_trunc": 0.01, "d1_mean_tokens": 300.0, "d1_greedy0": 100,
         "d1_dist": dist, "d4a_test": test_a, "d4a_by_position": rising,
         "d3_traces": 120, "d3_of": 120, "d3_completions": 3600, "d3_mc_prompt_mean": 0.5,
         "d3_unrec_share_correct": 0.1, "d3_unrec_share_wrong": 0.6, "d3_by_position": flat, "d3_wrong_traces": 60,
         "d3_restart_share": 0.4, "d3_never_share": 0.05, "d3_first_drop_hist": {"0": 10, "never": 3},
         "d4b_traces": 60, "d4b_of": 60, "d4b_completions": 1800, "d4b_test": test_b, "d4b_by_position": flat}
    report.smoke5(run, m, CFG5)
    text = (run / "report.md").read_text()
    for needle in ("Qwen3-8B / none (not loaded)", "GENERATION CACHE", "D1 MC STRATIFICATION", "test_p2 p_hat mixes",
                   "D2 TARGET A", "D4a TEST", "pooled: problems 30", "D3 TARGET B", "probe: layer 14",
                   "D4b TEST", "restart share 40.0%", "rises with position: yes", "rises with position: no",
                   "GATE (test within-problem AUROC >= 0.75 with 95% CI lower bound > 0.65; D4a pooled): pass",
                   "BLOCKERS: none"):
        assert needle in text, needle
    assert text.count("by boundary position") == 4
    assert json.loads((run / "metrics.json").read_text())["smoke5_gate"] is True


def test_smoke5_report_all_skipped(tmp_path, monkeypatch):
    run = _run_dir(tmp_path, monkeypatch)
    m = {**_base(), **{f"{s}_skipped": "skipped: not enough pod time left" for s in ("D1", "D2", "D4a", "D3", "D4b")}}
    report.smoke5(run, m, CFG5)
    text = (run / "report.md").read_text()
    assert "D4a pooled): n/a" in text and "BLOCKERS: D1 skipped" in text


def test_trend():
    assert report._trend({}).startswith("rises with position: n/a")
    assert "yes" in report._trend({k: {"n": 20, "auroc": 0.5 + 0.02 * k} for k in range(1, 9)})
    assert "no" in report._trend({k: {"n": 20, "auroc": 0.8 - 0.02 * k} for k in range(1, 9)})
    assert "n/a" in report._trend({k: {"n": 5, "auroc": 0.5 + 0.02 * k} for k in range(1, 9)})   # too few pairs
