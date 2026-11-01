"""report.option_e renders from realistic SMOKE-6 / SMOKE-7 metrics (full run and every stage skipped); the per-regime
statistics helper pairs adapter columns correctly."""
import json

import report
from option_e import V7, _stats
from test_report import CFG, _base, _run_dir

CFG6 = {**CFG, "phase_budget_usd": {"SMOKE3": 4}, "smoke4_source_run": "SMOKE4-x"}
ADS = ("rkl", "rft")


def _st(n):
    st = {"n": n, "problems": max(n // 2, 1)}
    for a, r in zip(report.E_ARMS, (0.20, 0.30, 0.28, 0.45, 0.40, 0.35, 0.25)):
        st.update({f"rec_{a}": r, f"tokens_{a}": 180.0, f"trunc_{a}": 0.01})
    for k, v in (("delta_primary", 0.15), ("e2_minus_e1", -0.02), ("e3_minus_e2", 0.17), ("interaction", 0.10)):
        st[k], st[f"{k}_ci"] = v, [v - 0.1, v + 0.1]
    return st


def _oracle(tag, n):
    return {f"{tag}_{s}": {"n": n, "restart_share": 0.45, "to_end": 3, "b_hist": {"0": 20, "1": 10, "3": 5}, "b_ge1": n // 2}
            for s in ("p1", "p2", "rc3", "pooled")}


def _train(ad):
    pre = f"{ad}_"
    return {f"{pre}train_rounds_done": 2, f"{pre}train_steps_total": 40, f"{pre}train_best_round": 2, f"{pre}train_best_step": 20,
            f"{pre}train_best_dev": 0.35, f"{pre}early_stop_fired": True, f"{pre}train_sec": 600.0, f"{pre}dev_eval_sec": 120.0,
            f"{pre}resample_sec": 200.0, f"{pre}train_steps_per_s": 0.067, f"{pre}pure_train_steps_per_s": 0.14,
            f"{pre}train_examples_r1": 300, f"{pre}train_examples_r2": 300, f"{pre}rounds_stopped": "round 3 did not improve dev (30.0% <= 35.0%)",
            f"{pre}train_rounds": [{"round": 1, "steps": 20, "curve": [[0, 0.2], [10, 0.3], [20, 0.25]], "best_step": 10,
                                    "best_dev": 0.3, "stopped": False, "loss_start": 0.1, "loss_end": 0.05},
                                   {"round": 2, "steps": 20, "curve": [[0, 0.3], [10, 0.35], [20, 0.3]], "best_step": 10,
                                    "best_dev": 0.35, "stopped": True, "loss_start": 0.1, "loss_end": 0.05}]}


def test_option_e_report_full_smoke7(tmp_path, monkeypatch):
    run = _run_dir(tmp_path, monkeypatch)
    m = {**_base(), "variant": {k: v for k, v in V7.items() if k not in ("budget", "start")},
         **{f"{s}_min": 1.0 for s in ("ES0", "ES1", "ES2", "ES4", "ES5", "ES6", "ES7", "ES8", "ES3")},
         "es0_problems": 1300, "es0_of": 1300, "es0_tok_per_s": 900.0, "es0_bs": 256, "es0_trunc": 0.01, "es0_mean_tokens": 250.0,
         "es0_train_p1": {"problems": 400, "of": 400, "hist": [100, 60, 50, 50, 140], "acc": 0.55, "contested": 0.4},
         "es0_test_p2": {"problems": 250, "of": 250, "hist": [100, 40, 30, 30, 50], "acc": 0.35, "contested": 0.4},
         "es1_a1_traces": 3200, "es1_a1_imported": 0, "es1_a3_scans": 48, "es1_a3_points_new": 0,
         "es1_contested_train": {"p1": 238, "p2": 125, "main": 23, "rc3": 60, "all": 446},
         "es1_contested_test": {"p1": 162, "p2": 185, "main": 0, "rc3": 0, "all": 347},
         "es2_greedy": 446, "es2_greedy_wrong": 250, "es2_greedy_acc": 0.44, "es2_probes_per_trace": 3.2,
         "es2_bisect_rounds": 6, "es2_bisect_traces": 250, "es2_bisect_of": 250, "es2_units": 255, "es2_units_a3": 0,
         "es2_units_reuse": 5, "es2_units_b_ge1": 130, "es2_mc_tok_per_s": 800.0, "es2_mc_cache_hits": 10,
         "es2_mc_cache_misses": 900, **_oracle("es2_oracle", 255),
         "es6_greedy": 347, "es6_greedy_wrong": 190, "es6_greedy_acc": 0.45, "es6_probes_per_trace": 3.0,
         "es6_bisect_traces": 190, "es6_bisect_of": 190, "es6_mc_tok_per_s": 810.0, "es6_e5_traces": 60,
         "es6_correct_greedy": 157, **_oracle("es6_oracle", 190),
         "es3_agree": 0.9, "es3_exact": 0.7, "es3_traces": 20, "es3_of": 20, "es3_monotone_share": 0.6,
         "es3_full_probes_per_trace": 7.5,
         "es4_units": 230, "es4_train_units_b_ge1": 120, "es4_sloppy_units": 25, "es4_dev_traces": 30, "es4_dev_b_ge1": 30,
         "es4_continuations": 920, "es4_cont_primary": 0.3, "es4_cont_trunc": 0.02, "es4_teacher_also_wrong": 0.1,
         "es4_teacher_scored": 230, "es4_mean_adv": -0.3, "es4_gen_tok_per_s": 500.0, "es4_prefill_tok_per_s": 6000.0,
         "train_examples_r1": 920, "adapter_params": 17432576, "resume_ok": True, "resume_exact": True, "train_kill_at": 15,
         "train_ckpt_every": 10, "resume_max_rel_diff": 0.0, **_train("rkl"), **_train("rft"),
         "train_steps_per_s": 0.067, "pure_train_steps_per_s": 0.14,
         "identity_n": 20, "identity_equal": 20, "identity_max_token_diff": 0, "es7_adapter_trained": True,
         "es7_adapters": list(ADS), "es7_units": 190, "es7_restart_units": 95, "es7_pointed_units": 95,
         "es7_clamped": {"E4a": 95, "E4b": 3}, "es7_arms_done": ["E0", "E1", "E2", "E3_rkl"], "es7_gen_tok_per_s": 500.0,
         "es7_e5_n": 60, "es7_plain_harm": 0.08, **{f"es7_e5_harm_{ad}": 0.1 for ad in ADS},
         **{f"es7_{ad}_{reg}_{s}": _st(n) for ad in ADS for reg in ("all", "pointed", "restart")
            for s, n in (("p1", 40), ("p2", 50), ("pooled", 90))}}
    report.option_e(run, m, CFG6)
    text = (run / "report.md").read_text()
    for needle in ("METHOD: oracle pointer (MC >= 0.25, 4 completions per probe", "DATA (E-S0, 4 samples per extra problem",
                   "train_p1: problems 400/400", "POINTED train units (b* >= 1): 130", "train rc3: n 255",
                   "ORACLE VALIDATION (E-S3, full scans vs bisection, +-1): agreement 90.0%",
                   "TRAIN DATA (round 1): units 230 (pointed 120", "TRAIN: resume check y (exact)", "TRAIN [rkl]: rounds 2",
                   "TRAIN [rft]: rounds 2", "round 3 did not improve dev", "steps/s 0.0670 overall, 0.1400 pure",
                   "IDENTITY: untrained E3 == E1 20/20", "(POINTED b* >= 1: 95, restart b* = 0: 95)",
                   "ARMS [rkl]", "ARMS [rft]", "pointed pooled (n 90, problems 45): E0 20.0%",
                   "DELTAS [rkl] (points; 95% CI by cluster bootstrap on original_id, 10,000 resamples; I on pointed units is primary):",
                   "pointed pooled (n 90): delta_primary E3-E1 +15.0 [+5.0, +25.0]", "I +10.0 [+0.0, +20.0]",
                   "E5 harm [rft]", "PROJECTION: FULL-E", "pure training steps/s",
                   "pipeline (identity 20/20, resume, early stop fired) pass",
                   "oracle (agreement >= 80%, >= 100 pointed train units) pass",
                   "signal (soft: rkl delta_primary pooled >= +5 and I >= 0 on POINTED test units, n 90) pass",
                   "harm (rkl E5 <= 1.5 x plain-retry) pass", "BLOCKERS: none"):
        assert needle in text, needle
    json.loads((run / "metrics.json").read_text())


def test_option_e_report_all_skipped(tmp_path, monkeypatch):
    run = _run_dir(tmp_path, monkeypatch)
    m = {**_base(), **{f"{s}_skipped": "skipped: not enough pod time left" for s in ("ES1", "ES2", "ES4", "ES5", "ES6", "ES7", "ES8", "ES3")}}
    report.option_e(run, m, CFG6)
    text = (run / "report.md").read_text()
    assert "BLOCKERS: ES1 skipped" in text and "GATES: pipeline (identity 20/20, resume, early stop fired) n/a" in text
    assert "ORACLE VALIDATION (E-S3, full scans vs bisection, +-1): skipped" in text


def test_stats_uses_adapter_columns():
    rows = []
    for i in range(40):
        r = {"group": i // 2, "E0": i % 5 == 0, "E1": i % 4 == 0, "E2": i % 3 == 0, "tok_E0": 10, "tok_E1": 10, "tok_E2": 12,
             "trunc_E0": False, "trunc_E1": False, "trunc_E2": False}
        for a, p in (("E3", 2), ("E4a", 3), ("E4b", 4), ("E6", 5)):
            r[f"{a}_rkl"], r[f"tok_{a}_rkl"], r[f"trunc_{a}_rkl"] = i % p == 0, 20, False
            r[f"{a}_rft"], r[f"tok_{a}_rft"], r[f"trunc_{a}_rft"] = False, 30, True
        rows.append(r)
    st = _stats(rows, [r["group"] for r in rows], "_rkl")
    assert st["rec_E3"] == 0.5 and st["tokens_E3"] == 20 and st["rec_E1"] == 0.25
    assert abs(st["delta_primary"] - 0.25) < 1e-9 and st["delta_primary_ci"][0] <= 0.25 <= st["delta_primary_ci"][1]
    st2 = _stats(rows, [r["group"] for r in rows], "_rft")
    assert st2["rec_E3"] == 0.0 and st2["trunc_E6"] == 1.0 and st2["rec_E1"] == st["rec_E1"]
