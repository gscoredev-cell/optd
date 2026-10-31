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


def test_smoke3_report_all_skipped(tmp_path, monkeypatch):
    run = _run_dir(tmp_path, monkeypatch)
    m = {**_base(), **{f"{s}_skipped": "skipped: not enough pod time left"
                       for s in ("MCa", "MCprobe", "B1", "B2", "B3", "C2", "C3")}}
    report.smoke3(run, m, CFG)
    text = (run / "report.md").read_text()
    assert "BLOCKERS: MCa skipped" in text
    assert "b3 n/a ; c3 n/a" in text                          # no gate verdict without a probe
