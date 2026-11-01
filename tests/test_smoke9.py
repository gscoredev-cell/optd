"""SMOKE-9 pure helpers (GRPO grouping with DAPO filtering, brevity) and the report rendering with SMOKE-8 numbers."""
import json

import pytest

import report
import smoke9 as S9
from test_report import CFG, _base, _run_dir

CFG9 = {**CFG, "phase_budget_usd": {"SMOKE3": 6}, "smoke8_model": "Qwen/Qwen3.5-9B", "smoke8_source_run": "SMOKE8-x"}


def test_grpo_groups_dapo_and_advantage():
    ex = []
    for pid, oks in (("a", [1, 1, 1, 1]), ("b", [0, 0, 0, 0]), ("c", [1, 0, 0, 0]), ("d", [1, 1, 0, 0]), ("e", [1])):
        ex += [{"pid": pid, "ok": bool(o), "cont": [1]} for o in oks]
    kept, st = S9.grpo_groups(ex)
    assert st == {"groups": 5, "kept": 2, "dropped_all_correct": 1, "dropped_all_wrong": 1, "incomplete": 1}
    assert len(kept) == 8 and all("adv" in e for e in kept)
    c = [e["adv"] for e in kept if e["pid"] == "c"]
    assert c[0] == pytest.approx(3 ** 0.5) and c[1] == pytest.approx(-1 / 3 ** 0.5)   # (1-.25)/.433, (0-.25)/.433
    d = [e["adv"] for e in kept if e["pid"] == "d"]
    assert d[0] == pytest.approx(1.0) and d[2] == pytest.approx(-1.0) and sum(d) == pytest.approx(0.0)


def test_brevity():
    assert S9.brevity(0.468, 0.10) == pytest.approx(0.368) and S9.brevity(None, 0.1) is None


def _pass1(n, base, a6):
    return {"n": n, "base": base, "base_ci": [base - 0.05, base + 0.05], "a6": a6, "a6_ci": [a6 - 0.05, a6 + 0.05], "delta": a6 - base,
            "delta_ci": [a6 - base - 0.08, a6 - base + 0.08], "base_tokens": 900.0, "a6_tokens": 300.0, "base_trunc": 0.05, "a6_trunc": 0.0}


def _arms(n, e0, e6):
    return {"n": n, "E0": e0, "E6": e6, "delta": e6 - e0, "delta_ci": [e6 - e0 - 0.15, e6 - e0 + 0.15], "tok_E0": 1000.0, "tok_E6": 250.0,
            "trunc_E0": 0.1, "trunc_E6": 0.0}


def test_smoke9_report_full(tmp_path, monkeypatch):
    run = _run_dir(tmp_path, monkeypatch)
    m8 = {"z6_pooled": {"n": 240, "base": 0.562, "a6": 0.742, "delta": 0.179, "delta_ci": [0.096, 0.262]},
          "z5_pooled": {"n": 47, "rec_E0": 0.319, "rec_E6": 0.787, "tokens_E0": 658.0, "tokens_E6": 219.0, "trunc_E0": 0.553, "trunc_E6": 0.0,
                        "e6_minus_e0": 0.468, "e6_minus_e0_ci": [0.277, 0.638]},
          "z5_harm_adapter": 0.217, "z5_harm_plain": 0.217, "z2_trunc": 0.386, "z2_mean_tokens": 622.0,
          "z1_cal": {"rc4": {"acc": 0.183}}}
    m = {**_base(), **{f"{s}_min": 1.0 for s in ("Y0", "Y1", "Y2", "Y3", "Y4", "Y5")},
         "y0_transformers": "5.22.1", "y0_model_class": "Qwen3_5ForConditionalGeneration", "y0_load_sec": 5.0, "y0_vram_gb_after_load": 17.5,
         "y0_fast_path_warning": False, "y0_lp_bs": 4, "y0_micro": 2, "y0_sets_from_smoke8": ["rc3"], "y0_smoke8_run": "SMOKE8-x",
         "rkl_base_cache_hits": 156, "rkl_base_cache_misses": 84, "rkl_base_tok_per_s": 300.0, "rkl_a6_tok_per_s": 500.0,
         "rkl_pass1_n": 240, "rkl_pass1_of": 240,
         "y1_rc4_cal_n": 60, "y1_rc4_cal_acc": 0.45, "y1_rc4_cal_trunc": 0.1, "y1_rc4_cal_tokens": 1500.0, "y1_rc4_added": True,
         "y2_problems": 320, "y2_of": 320, "y2_tok_per_s": 700.0, "y2_mean_tokens": 1400.0, "y2_test_units": 30, "y2_harm": 40,
         "y2_train_rc4": {"problems": 200, "hist": [40, 40, 40, 40, 40], "acc": 0.5, "contested": 0.6, "trunc": 0.1},
         "y2_test_rc4": {"problems": 120, "hist": [24, 24, 24, 24, 24], "acc": 0.5, "contested": 0.6, "trunc": 0.1, "greedy_acc": 0.5,
                         "wrong_greedy_contested": 30},
         "y3_units": 250, "y3_units_by_set": {"rc3": 162, "rc4": 88}, "y3_dev": 45, "y3_dev_by_set": {"rc3": 30, "rc4": 15}, "y3_lr": 5e-5,
         "y3_clip": 0.2, "y3_micro": 2, "train_rounds_done": 2, "executed_steps": 60, "train_best_round": 2, "train_best_step": 20,
         "train_best_dev": 0.6, "early_stop_fired": True, "resume_ok": True, "resume_exact": False, "resume_max_rel_diff": 1e-5,
         "train_sec": 1500.0, "dev_eval_sec": 600.0, "resample_sec": 500.0, "pure_train_steps_per_s": 0.15, "adapter_params": 43278336,
         "y3r1_reused_smoke8": 648, "y3r1_groups": {"groups": 162, "kept": 90, "dropped_all_correct": 30, "dropped_all_wrong": 42, "incomplete": 0},
         "y3r2_groups": {"groups": 250, "kept": 120, "dropped_all_correct": 80, "dropped_all_wrong": 50, "incomplete": 0},
         "y3r2_continuations": 1000, "y3r2_cont_primary": 0.6, "y3r2_cont_trunc": 0.02, "y3r2_mean_tokens": 500.0, "y3r2_gen_tok_per_s": 600.0,
         "train_rounds": [{"round": 1, "steps": 23, "examples": 360, "curve": [[0, 0.4], [10, 0.5], [20, 0.55]], "best_step": 20, "best_dev": 0.55,
                           "stopped": False, "cut": False, "loss_start": 0.01, "loss_end": -0.02},
                          {"round": 2, "steps": 30, "examples": 480, "curve": [[0, 0.55], [10, 0.6], [20, 0.6], [30, 0.5]], "best_step": 10,
                           "best_dev": 0.6, "stopped": True, "cut": False, "loss_start": 0.01, "loss_end": -0.01}],
         "round3_skipped": "stage time budget"}
    for tag, a6, e6 in (("rkl", 0.70, 0.5), ("grpo", 0.74, 0.6)):
        for s, n in (("rc3", 240), ("rc4", 120), ("pooled", 360)):
            m[f"{tag}_pass1_{s}"] = _pass1(n, 0.60, a6)
        for s, n in (("rc3", 47), ("rc4", 30), ("pooled", 77)):
            m[f"{tag}_arms_{s}"] = _arms(n, 0.35, e6)
        m.update({f"{tag}_harm_n": 100, f"{tag}_harm_adapter": 0.15, f"{tag}_harm_plain": 0.16, f"{tag}_harm_adapter_rc3": 0.15,
                  f"{tag}_harm_plain_rc3": 0.16, f"{tag}_harm_adapter_rc4": 0.15, f"{tag}_harm_plain_rc4": 0.16})
    report.smoke9(run, m, CFG9, m8)
    text = (run / "report.md").read_text()
    for needle in ("METHOD: cap 3200 everywhere", "SMOKE-8 AT CAP 800 (for comparison): pass@1 pooled base 56.2% -> A6 74.2% (+17.9 [+9.6, +26.2], n 240)",
                   "Y1 SMOKE-8 rkl ADAPTER AT CAP 3200:", "pass@1 rc3 (n 240): base 60.0%", "E0 vs E6 pooled (n 77): E0 35.0%",
                   "RC4 CALIBRATION at 3200", "yes, RC4 added", "Y2 RC4 DATA (cap 3200", "Y3 GRPO: units 250",
                   "round 1: reused SMOKE-8 continuations 648 (dropped None truncated at 800) | groups 162 kept 90 (dropped all-correct 30, all-wrong 42, incomplete 0) -> examples 360",
                   "round 3: stage time budget", "Y4 GRPO ADAPTER AT CAP 3200:", "SUMMARY AT CAP 3200",
                   "pooled: pass@1 base 60.0% | A6 rkl 70.0% (+10.0 [+2.0, +18.0]) | A6 grpo 74.0% (+14.0 [+6.0, +22.0]) ; E6-E0 rkl +15.0",
                   "BREVITY: of SMOKE-8's gain at cap 800, the part that disappears at 3200 with the same adapter = E6-E0 +46.8 - +15.0 = +31.8 points ; A6 pass@1 +17.9 - +10.0 = +7.9 points",
                   "PROJECTION: FULL at cap 3200", "rkl A6 > base at 3200 (CI lower bound > 0) pass", "grpo A6 > base at 3200 (CI lower bound > 0) pass",
                   "harm (each adapter <= 1.5 x plain) pass", "BLOCKERS: none"):
        assert needle in text, needle
    j = json.loads((run / "metrics.json").read_text())
    assert j["smoke9_brevity"]["e6_minus_e0"] == pytest.approx(0.318)


def test_smoke9_report_all_skipped(tmp_path, monkeypatch):
    run = _run_dir(tmp_path, monkeypatch)
    m = {**_base(), **{f"{s}_skipped": "skipped: not enough pod time left" for s in ("Y0", "Y1", "Y3", "Y4", "Y5")},
         "Y2_skipped": "not needed: RC4 not in the accuracy band at 3200"}
    report.smoke9(run, m, CFG9, {})
    text = (run / "report.md").read_text()
    assert "BLOCKERS: Y0 skipped" in text and "Y2 skipped" not in text and "env (fast path, model loaded) n/a" in text
