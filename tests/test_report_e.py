"""report.option_e renders from realistic SMOKE-6 metrics (full run and every stage skipped)."""
import json

import report
from test_report import CFG, _base, _run_dir

CFG6 = {**CFG, "phase_budget_usd": {"SMOKE3": 4}, "smoke4_source_run": "SMOKE4-x"}


def _stats(n):
    st = {"n": n, "problems": n // 2}
    for a, r in zip(report.E_ARMS, (0.20, 0.30, 0.28, 0.45, 0.40, 0.35, 0.25)):
        st.update({f"rec_{a}": r, f"tokens_{a}": 180.0, f"trunc_{a}": 0.01})
    for k, v in (("delta_primary", 0.15), ("e2_minus_e1", -0.02), ("e3_minus_e2", 0.17), ("interaction", 0.10)):
        st[k], st[f"{k}_ci"] = v, [v - 0.1, v + 0.1]
    return st


def _oracle(tag, n):
    return {f"{tag}_{s}": {"n": n, "restart_share": 0.45, "to_end": 3, "b_hist": {"0": 20, "1": 10, "3": 5}, "b_ge1": n // 2}
            for s in ("p1", "p2", "pooled")}


def test_option_e_report_full(tmp_path, monkeypatch):
    run = _run_dir(tmp_path, monkeypatch)
    m = {**_base(), **{f"{s}_min": 1.0 for s in ("ES1", "ES2", "ES4", "ES5", "ES6", "ES7", "ES8", "ES3")},
         "es1_a1_traces": 3200, "es1_a1_imported": 3200, "es1_a3_scans": 48, "es1_a3_points_new": 300,
         "es1_contested_train": {"p1": 78, "p2": 45, "main": 23, "all": 146},
         "es1_contested_test": {"p1": 72, "p2": 85, "main": 0, "all": 157},
         "es2_greedy": 146, "es2_greedy_wrong": 80, "es2_greedy_acc": 0.45, "es2_probes_per_trace": 4.2,
         "es2_bisect_rounds": 6, "es2_bisect_traces": 80, "es2_bisect_of": 80, "es2_units": 130, "es2_units_a3": 48,
         "es2_units_reuse": 2, "es2_units_b_ge1": 70, "es2_mc_tok_per_s": 300.0, "es2_mc_cache_hits": 10,
         "es2_mc_cache_misses": 900, **_oracle("es2_oracle", 130),
         "es6_greedy": 157, "es6_greedy_wrong": 90, "es6_greedy_acc": 0.43, "es6_probes_per_trace": 4.0,
         "es6_bisect_traces": 90, "es6_bisect_of": 90, "es6_mc_tok_per_s": 310.0, "es6_e5_traces": 60,
         "es6_correct_greedy": 67, **_oracle("es6_oracle", 90),
         "es3_agree": 0.9, "es3_exact": 0.7, "es3_traces": 20, "es3_of": 20, "es3_monotone_share": 0.8,
         "es3_full_probes_per_trace": 7.5,
         "es4_units": 90, "es4_sloppy_units": 12, "es4_dev_traces": 60, "es4_dev_b_ge1": 30, "es4_continuations": 360,
         "es4_cont_primary": 0.3, "es4_cont_trunc": 0.02, "es4_teacher_also_wrong": 0.4, "es4_teacher_scored": 90,
         "es4_mean_adv": 0.1, "es4_gen_tok_per_s": 400.0, "es4_prefill_tok_per_s": 4000.0,
         "train_rounds_done": 2, "train_steps_total": 40, "train_best_round": 2, "train_best_step": 20,
         "train_best_dev": 0.35, "early_stop_fired": True, "resume_ok": True, "train_kill_at": 15, "train_ckpt_every": 10,
         "resume_max_rel_diff": 1e-6, "adapter_params": 17000000, "train_steps_per_s": 0.5,
         "train_rounds": [{"round": 1, "steps": 23, "curve": [[0, 0.2], [10, 0.3], [20, 0.25], [23, 0.25]],
                           "best_step": 10, "best_dev": 0.3, "stopped": False, "loss_start": 0.1, "loss_end": 0.05},
                          {"round": 2, "steps": 23, "curve": [[0, 0.3], [10, 0.35], [20, 0.3]], "best_step": 10,
                           "best_dev": 0.35, "stopped": True, "loss_start": 0.1, "loss_end": 0.05}],
         "identity_n": 20, "identity_equal": 20, "identity_max_token_diff": 0, "es7_adapter_trained": True,
         "es7_units": 90, "es7_restart_units": 40, "es7_clamped": {"E4a": 40, "E4b": 3}, "es7_arms_done": list(report.E_ARMS),
         "es7_gen_tok_per_s": 420.0, "es7_e5_n": 60, "es7_e5_harm": 0.1, "es7_plain_harm": 0.08,
         **{f"es7_{s}": _stats(n) for s, n in (("p1", 40), ("p2", 50), ("pooled", 90))}}
    report.option_e(run, m, CFG6)
    text = (run / "report.md").read_text()
    for needle in ("METHOD:", "CACHE: SMOKE-4 A1 traces imported 3200/3200", "DATA: contested train problems p1/p2/main 78/45/23",
                   "ORACLE (train): bisection probes per trace 4.20", "train p1: n 130 | restart share 45.0%",
                   "ORACLE VALIDATION (E-S3, full scans vs bisection, +-1): agreement 90.0%", "TRAIN DATA (round 1): units 90",
                   "early stopping fired y", "round 2: steps 23", "IDENTITY: untrained E3 == E1 20/20",
                   "pooled (n 90, problems 45): E0 20.0%", "E5 harm (correct -> wrong, adapter on, random cut) 10.0%",
                   "pooled: delta_primary E3-E1 +15.0 [+5.0, +25.0]", "I +10.0 [+0.0, +20.0]", "PROJECTION: FULL-E",
                   "pipeline (identity 20/20, resume, early stop fired) pass", "oracle (agreement >= 80%, >= 100 train units with b* >= 1) fail",
                   "signal (soft: delta_primary pooled >= +5 and I >= 0) pass", "harm (E5 <= 1.5 x plain-retry) pass",
                   "BLOCKERS: none"):
        assert needle in text, needle
    g = json.loads((run / "metrics.json").read_text())["option_e_gates"]
    assert g["signal (soft: delta_primary pooled >= +5 and I >= 0)"] is True


def test_option_e_report_all_skipped(tmp_path, monkeypatch):
    run = _run_dir(tmp_path, monkeypatch)
    m = {**_base(), **{f"{s}_skipped": "skipped: not enough pod time left" for s in ("ES1", "ES2", "ES4", "ES5", "ES6", "ES7", "ES8", "ES3")}}
    report.option_e(run, m, CFG6)
    text = (run / "report.md").read_text()
    assert "BLOCKERS: ES1 skipped" in text and "GATES: pipeline (identity 20/20, resume, early stop fired) n/a" in text
    assert "ORACLE VALIDATION (E-S3, full scans vs bisection, +-1): skipped" in text
