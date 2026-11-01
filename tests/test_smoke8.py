"""SMOKE-8 pure helpers (set selection, stratified dev split, teacher prompt, arm statistics) and report rendering."""
import json

import report
import smoke8 as S
from core import SYSTEM
from data import random_calculation, rc_reference
from test_report import CFG, _base, _run_dir

CFG8 = {**CFG, "phase_budget_usd": {"SMOKE3": 6}, "smoke8_model": "Qwen/Qwen3.5-9B"}


def test_choose_sets_and_sizes():
    acc = {"p1": 0.8, "p2": 0.5, "rc3": 0.72, "rc4": 0.35, "rc5": 0.2, "rc6": None}
    assert S.choose_sets(acc) == ["p2", "rc4"]
    assert S.split_sizes(400, ["p2", "rc4"]) == {"p2": 200, "rc4": 200} and S.split_sizes(400, []) == {}


def test_dev_split_never_takes_a_scarce_stratum():
    units = [{"pid": f"p1:{i}", "set": "p1"} for i in range(90)] + [{"pid": f"rc4:{i}", "set": "rc4"} for i in range(10)]
    dev, train = S.dev_split(units, n=30)
    by = {s: sum(u["set"] == s for u in dev) for s in ("p1", "rc4")}
    assert by["p1"] == 27 and by["rc4"] <= 3                     # 30% cap on the 10-unit stratum
    assert {u["pid"] for u in dev}.isdisjoint({u["pid"] for u in train}) and len(dev) + len(train) == 100
    assert S.dev_split(units, n=30)[0] == dev                      # seeded


def test_teacher_system_and_rc_reference():
    p = random_calculation(1, seed=803, min_ops=3, max_ops=3)[0]
    ref = rc_reference(p)
    assert ref.endswith(f"The final answer is {ref.splitlines()[-2].split('= ')[-1]}") or "The final answer is" in ref
    sysline = S.teacher_system(ref)
    assert sysline.startswith(SYSTEM) and "Reference solution:" in sysline and ref in sysline
    assert S.common_suffix([1, 2, 3, 9, 8], [7, 7, 3, 9, 8]) == 3 and S.common_suffix([1], [2]) == 0


def test_arm_stats():
    rows = [{"group": i // 2, "E0": i % 4 == 0, "E2": i % 3 == 0, "E6": i % 2 == 0, "E6h": i % 2 == 0,
             **{f"tok_{a}": 100 for a in S.ARMS}, **{f"trunc_{a}": False for a in S.ARMS}} for i in range(40)]
    st = S.arm_stats(rows, [r["group"] for r in rows])
    assert st["rec_E6"] == 0.5 and st["rec_E0"] == 0.25 and abs(st["e6_minus_e0"] - 0.25) < 1e-9
    assert st["e6h_minus_e6"] == 0 and st["e6h_minus_e6_ci"] == [0.0, 0.0]
    assert st["e6_minus_e0_ci"][0] <= 0.25 <= st["e6_minus_e0_ci"][1]


def _st(n):
    st = {"n": n, "problems": n // 2}
    for a, r in zip(S.ARMS, (0.3, 0.35, 0.5, 0.52)):
        st.update({f"rec_{a}": r, f"tokens_{a}": 300.0, f"trunc_{a}": 0.01})
    for k, v in (("e6_minus_e0", 0.2), ("e2_minus_e0", 0.05), ("e6h_minus_e0", 0.22), ("e6h_minus_e6", 0.02), ("e6_minus_e2", 0.15)):
        st[k], st[f"{k}_ci"] = v, [v - 0.1, v + 0.1]
    return st


def test_smoke8_report_full(tmp_path, monkeypatch):
    run = _run_dir(tmp_path, monkeypatch)
    m = {**_base(), **{f"{s}_min": 1.0 for s in ("Z0", "Z1", "Z2", "Z3", "Z4", "Z5", "Z6", "Z7")},
         "z0_transformers": "5.22.1", "z0_transformers_ge5": True, "z0_fla": "0.6.0", "z0_causal_conv1d": "1.7.0", "z0_triton": "3.9.0",
         "z0_peft": "0.21.2", "z0_model_class": "Qwen3_5ForConditionalGeneration", "z0_params_b": 9.4, "z0_load_sec": 40.0,
         "z0_vram_gb_after_load": 18.5, "z0_fast_path_warning": False, "z0_module_fast_path_flags": {"is_fast_path_available": True},
         "z0_load_warnings": [], "z0_suffix_covers_user_turn": 20, "z0_common_suffix_min": 60, "z0_user_tokens_mean": 55.0,
         "z0_think_block": 20, "z0_hint_roundtrip": True, "z0_prompt_tail": "'<think>\\n\\n</think>\\n\\n'", "z0_gen_smoke_tokens": 32,
         "z0_gen_smoke_sec": 2.0,
         "z1_cal": {s: {"n": 60, "acc": a, "strict": a, "trunc": 0.0, "mean_tokens": 300.0} for s, a in
                    (("p1", 0.8), ("p2", 0.6), ("rc3", 0.75), ("rc4", 0.5), ("rc5", 0.3), ("rc6", 0.2))},
         "z1_chosen": ["p2", "rc4"], "z1_tok_per_s": 500.0,
         "z2_problems": 640, "z2_of": 640, "z2_sizes_train": {"p2": 200, "rc4": 200}, "z2_sizes_test": {"p2": 120, "rc4": 120},
         "z2_tok_per_s": 600.0, "z2_trunc": 0.01, "z2_mean_tokens": 320.0, "z2_test_units": 90, "z2_harm_traces": 60,
         "z2_cache_hits": 0, "z2_cache_misses": 2800,
         "z2_train_p2": {"problems": 200, "hist": [40, 30, 30, 40, 60], "acc": 0.55, "contested": 0.5},
         "z2_test_p2": {"problems": 120, "hist": [20, 20, 20, 20, 40], "acc": 0.55, "contested": 0.5, "greedy_acc": 0.6,
                        "wrong_greedy_contested": 45},
         "z3_teacher_n": 200, "z3_teacher_acc": 0.98, "z3_teacher_acc_p2": 0.99, "z3_teacher_tok_per_s": 500.0,
         "z3_gap_correct": 0.05, "z3_gap_wrong": 0.4, "z3_gap_n_correct": 100, "z3_gap_n_wrong": 100,
         "z3_student_logp_correct": -0.3, "z3_student_logp_wrong": -0.5, "z3_prefill_tok_per_s": 5000.0,
         "z4_units": 170, "z4_dev": 30, "z4_dev_by_set": {"p2": 15, "rc4": 15}, "z4_units_by_set": {"p2": 85, "rc4": 85},
         "adapter_params": 40000000, "z4_micro": 4, "train_rounds_done": 2, "executed_steps": 90, "train_best_round": 2,
         "train_best_step": 30, "train_best_dev": 0.5, "early_stop_fired": True, "resume_ok": True, "resume_exact": True,
         "train_kill_at": 15, "train_ckpt_every": 10, "resume_max_rel_diff": 0.0, "train_sec": 900.0, "dev_eval_sec": 200.0,
         "resample_sec": 300.0, "pure_train_steps_per_s": 0.2, "train_steps_per_s": 0.1,
         "train_rounds": [{"round": 1, "steps": 43, "examples": 680, "curve": [[0, 0.3], [10, 0.4], [20, 0.45]], "best_step": 20,
                           "best_dev": 0.45, "stopped": False, "cut": False, "loss_start": 0.2, "loss_end": 0.1},
                          {"round": 2, "steps": 43, "examples": 680, "curve": [[0, 0.45], [10, 0.5], [20, 0.4]], "best_step": 10,
                           "best_dev": 0.5, "stopped": True, "cut": False, "loss_start": 0.2, "loss_end": 0.1}],
         "round3_skipped": "stage time budget",
         **{f"z4r{r}_{k}": v for r in (1, 2) for k, v in (("continuations", 680), ("cont_primary", 0.3), ("cont_trunc", 0.01),
                                                          ("teacher_also_wrong", 0.02), ("mean_adv", -0.2), ("prefill_tok_per_s", 5000.0),
                                                          ("gen_tok_per_s", 550.0))},
         "z5_arms_done": list(S.ARMS), "z5_units": 90, "z5_adapter_trained": True, "z5_arms_cut": None, "z5_gen_tok_per_s": 550.0,
         "z5_harm_n": 60, "z5_harm_adapter": 0.1, "z5_harm_plain": 0.12,
         **{f"z5_{s}": _st(n) for s, n in (("p2", 45), ("rc4", 45), ("pooled", 90))},
         "z6_n": 240, "z6_of": 240, "z6_adapter_trained": True, "z6_tok_per_s": 600.0, "z6_a6_trunc": 0.0, "z6_a6_mean_tokens": 300.0,
         **{f"z6_{s}": {"n": n, "base": 0.55, "a6": 0.6, "delta": 0.05, "delta_ci": [0.0, 0.1]} for s, n in (("p2", 120), ("rc4", 120), ("pooled", 240))}}
    report.smoke8(run, m, CFG8)
    text = (run / "report.md").read_text()
    for needle in ("Qwen3.5-9B / Qwen3.5-9B + reference", "Z0 ENV: transformers 5.22.1 (>= 5: y)", "fast-path warning none",
                   "TEMPLATES (20 prompts", "common suffix >= user turn 20/20", "Z1 CALIBRATION", "chosen ['p2', 'rc4']",
                   "Z2 DATA (4 samples per problem", "test_p2: problems 120", "wrong-greedy contested 45",
                   "Z3 SELF-TEACHER: teacher greedy accuracy with the reference 98.0%", "wrong traces 0.400",
                   "Z4 TRAIN: units 170", "resume y (exact)", "round 1: continuations 680", "round 3: stage time budget",
                   "Z5 ARMS", "pooled (n 90, problems 45): E0 30.0%", "HARM (correct greedy test traces", "adapter 10.0% vs plain 12.0%",
                   "pooled (n 90): E6-E0 +20.0 [+10.0, +30.0]", "Z6 PASS@1", "pooled base 55.0% -> A6 60.0% (+5.0 [+0.0, +10.0], n 240)",
                   "PROJECTION: FULL", "env (transformers >= 5, fast path, template suffix 20/20) pass",
                   "signal (E6-E0 pooled CI lower bound > 0) pass", "harm (adapter <= 1.5 x plain) pass", "pass@1 (A6 >= base pooled) pass",
                   "BLOCKERS: none"):
        assert needle in text, needle
    json.loads((run / "metrics.json").read_text())


def test_smoke8_report_all_skipped(tmp_path, monkeypatch):
    run = _run_dir(tmp_path, monkeypatch)
    m = {**_base(), **{f"{s}_skipped": "skipped: not enough pod time left" for s in ("Z0", "Z1", "Z2", "Z3", "Z4", "Z5", "Z6", "Z7")}}
    report.smoke8(run, m, CFG8)
    text = (run / "report.md").read_text()
    assert "BLOCKERS: Z0 skipped" in text and "env (transformers >= 5, fast path, template suffix 20/20) n/a" in text
