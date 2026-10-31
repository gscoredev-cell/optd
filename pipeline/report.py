"""report.md + metrics.json (§12). Shared header for every phase; ENVCHECK and SMOKE bodies."""
import json
import math
import os
import time

# Fallbacks for stages SMOKE-2 may skip, measured in SMOKE-20261030-1807-6462834 (L40, tier 48).
SMOKE1_TEACHER_PREFILL_TOK_PER_S = 4425.3
SMOKE1_TRAIN_SEC_PER_STEP = 1 / 0.4729


def _bad(x):
    return x is None or (isinstance(x, float) and math.isnan(x))


def _p(x):
    """Rate as percent with one decimal."""
    return "nan" if _bad(x) else f"{100 * x:.1f}%"


def _n(x, nd=3):
    return "nan" if _bad(x) else f"{x:.{nd}f}"


def header(run_dir, m, cfg):
    """RUN / IMAGE PULL / SSH / ENV / DEVICE lines; adds elapsed, cost and cache numbers to m."""
    env = json.loads((run_dir / "env.json").read_text())
    dl = [json.loads(l) for l in (run_dir / "downloads.jsonl").read_text().splitlines() if l.strip()]
    hits = sum(x["cache"] == "hit" for x in dl)
    got = [x for x in dl if x["cache"] == "download"]
    mbps = f"{sum(x['mb'] for x in got) / max(sum(x['sec'] for x in got), 1e-3):.1f}" if got else "n/a (all cache hits)"
    boot = int(os.environ["BOOT_EPOCH"])
    elapsed = (time.time() - boot) / 60
    price = float(env["price_per_hr"])
    pull = env.get("image_pull_sec", 0) / 60
    port = env.get("ssh_public_port", "")
    m.update({"elapsed_min": round(elapsed, 2), "cost_usd": round(price * elapsed / 60, 3), "cache_hits": hits,
              "downloads": len(got), "bootstrap_min": round((m["pipeline_start"] - boot) / 60, 2),
              "image_pull_min": round(pull, 2), "image_pull_cost_usd": round(price * pull / 60, 3),
              "ssh_public_port": port or None, "price_per_hr": price})
    cap = ".".join(map(str, m["capability"]))
    return env, dl, [
        f"RUN: {env['run_id']} | {env['phase']} | {env['git_sha'][:7]} | {env['pod_id']} | {env['gpu']} | "
        f"{m['vram_gb']} GB | {cfg['datacenter']} | ${price:.2f}/h | {elapsed:.1f} min | ${price * elapsed / 60:.2f}",
        f"IMAGE PULL: {pull:.1f} min (pod create -> bootstrap start) | cost of pull ${price * pull / 60:.2f} | "
        f"billed total est. {pull + elapsed:.1f} min ${price * (pull + elapsed) / 60:.2f}",
        f"SSH: public port assigned {'yes (' + port + ')' if port else 'no'}",
        f"ENV: {cfg['image']} | torch {m['torch']} (cuda {m['cuda']}) | transformers {m['transformers']} | "
        f"peft {m['peft']} | {cfg['student'].split('/')[-1]} / {cfg['teacher'].split('/')[-1]} | "
        f"cache hits {hits} / downloads {len(got)} | download MB/s {mbps}",
        f"DEVICE: capability {cap} | bf16 matmul ok (rel err {m['bf16_matmul_rel_err']}) | tier {env['tier']} | "
        f"driver {env['driver']}",
        f"TEMPLATES: identical token ids {m['template_identical_ids']}/{m['template_n']} | "
        f"empty think block {m['template_empty_think']}/{m['template_n']}",
    ]


def _write(run_dir, lines, m):
    (run_dir / "report.md").write_text("\n".join(lines) + "\n")
    (run_dir / "metrics.json").write_text(json.dumps(m, indent=1, default=str))
    print("\n".join(lines))


def envcheck(run_dir, m, cfg):
    env, dl, lines = header(run_dir, m, cfg)
    price = m["price_per_hr"]
    lines += [
        f"TIMING (min): bootstrap {m['bootstrap_min']} | env {m['env_min']} | templates {m['templates_min']} | "
        f"gen {m['gen_min']} | teacher_prefill {m['teacher_prefill_min']} | total {m['elapsed_min']:.2f}",
        f"THROUGHPUT: gen tok/s {m['gen_tok_per_s']} (batch {m['gen_batch']}, {m['gen_tokens']} tokens, "
        f"truncated {m['gen_truncated']}/{m['gen_batch']}, peak {m['gen_peak_vram_gb']} GB) | "
        f"teacher prefill tok/s {m['prefill_tok_per_s']} (batch {m['prefill_batch']}, {m['prefill_tokens']} tokens, "
        f"peak {m['prefill_peak_vram_gb']} GB, mean logp {m['teacher_mean_logp']})",
        "DOWNLOADS: " + " ; ".join(f"{x['repo']} cache hit" if x["cache"] == "hit"
                                   else f"{x['repo']} {x['mb']} MB {x['mb_per_s']} MB/s" for x in dl),
        f"BUDGET: max_minutes {env['max_minutes']} x ${price:.2f}/h = ${price * int(env['max_minutes']) / 60:.2f} "
        f"<= ${cfg['phase_budget_usd'][env['phase']]}",
        "BLOCKERS: none",
    ]
    _write(run_dir, lines, m)


def _projection(m):
    """Rough FULL projection per seed (§10 FULL sizes: 7000 train prompts, greedy + 1 sample, 3 rounds x 4
    continuations of flagged-wrong traces, 1800 test problems, arms A0-A6 with 2 interventions, A5 k = 5) from SMOKE
    token counts and measured tokens/s. Returns (hours, parts in hours)."""
    T = m["s2_mean_tokens"]
    final = m["s1_max_new_final"]
    gen_g, gen_s = m["s2_greedy_tok_per_s"], m["s2_sample_tok_per_s"]
    prefill = m.get("s6_prefill_tok_per_s") or SMOKE1_TEACHER_PREFILL_TOK_PER_S
    step = 1 / m["train_steps_per_s"] if m.get("train_steps_per_s") else SMOKE1_TRAIN_SEC_PER_STEP
    flag_wrong = m["s3_flagged_wrong_greedy"] / 300
    interv = m["s8_flagged"] / max(sum(m["s1_n"].values()), 1)
    n_train, n_test = 7000, 1800
    conts = 3 * flag_wrong * n_train * 4
    parts = {
        "rollouts": n_train * T / gen_g + n_train * T / gen_s,
        "features": m["s3_feat_sec"] * 2 * n_train / m["s2_n"],
        "continuations": conts * T / gen_s,
        "teacher_scoring": conts * (2 * T + 150) / prefill,
        "training": conts / 16 * step,
        "headroom": n_test * T / m[f"s1_student_tok_per_s_{final}"] + n_test * T / m[f"s1_teacher_tok_per_s_{final}"],
        "arms": n_test * T * (4 * 2 * interv + 5) / gen_s + n_test * T / gen_g,
    }
    parts = {k: v / 3600 for k, v in parts.items()}
    return sum(parts.values()), parts


def smoke(run_dir, m, cfg):
    env, _, lines = header(run_dir, m, cfg)
    price = m["price_per_hr"]
    sets = ("p2", "main", "p1", "rc")
    n = m["s1_n"]
    trained = m.get("s8_adapter_trained")

    def acc(who):
        return " / ".join(f"{s} (n={n[s]}) {_p(m[f's1_{who}_{s}_primary'])}/{_p(m[f's1_{who}_{s}_strict'])}"
                          for s in sets)

    trunc = " ; ".join(f"{k} tokens: " + ", ".join(f"{w} {_p(v)}" for w, v in m[f"s1_trunc_{k}"].items())
                       for k in m["s1_max_new_tried"])
    rc = " ; ".join(f"{k} ops: student {_p(m[f'rc_cal_{k}']['student_acc'])} (trunc {_p(m[f'rc_cal_{k}']['student_trunc'])})"
                    f", teacher {_p(m[f'rc_cal_{k}']['teacher_acc'])} (trunc {_p(m[f'rc_cal_{k}']['teacher_trunc'])})"
                    for k in (3, 4, 5))
    table = m.get("probe_table", {})
    rows = []
    for name in dict.fromkeys(k.split("|")[0] for k in table):
        cells = [f"C={k.split('C=')[1]} cv {_n(v['cv_auroc'])}" + (f" held-out {_n(v['heldout_auroc'])}"
                 if "heldout_auroc" in v else "") for k, v in table.items() if k.startswith(name + "|")]
        rows.append(f"  {name}: " + " ; ".join(cells))
    restart = m.get("probe_restart_share")
    tol_l = sorted(m.get("tol_gen_vs_b1_bf16", {}), key=int)
    pick = [l for l in tol_l if int(l) in (7, 14, 21, 28)]
    hours, parts = _projection(m)
    blockers = [m[k] for k in ("probe_blocker", "train_blocker") if m.get(k)]
    if not m.get("identity_equal"):
        blockers.append(f"identity failed (n {m.get('identity_n')}, max token diff {m.get('identity_max_token_diff')})")
    if m.get("resume_ok") is False:
        blockers.append(f"resume mismatch (max rel diff {_n(m.get('resume_max_rel_diff'), 5)})")
    if m.get("rc_ops_choice") is None:
        blockers.append("RandomCalculation: no ops level with teacher >= 60% and truncation <= 5%")
    if not m.get("probe_gate"):
        blockers.append("probe gate failed: FULL not cleared (decide probe-mc labels vs Qwen3-4B student)")
    gates = {
        "headroom (gap >= 8 pts)": m["s1_gap_p2"] >= 0.08,
        "base rate in [20%, 85%]": 0.20 <= m["s2_greedy_primary"] <= 0.85,
        "probe AUROC >= 0.80 and recall >= 0.40 at FPR 10%": bool(m.get("probe_gate")),
        "identity": bool(m.get("identity_equal")),
        "resume": m.get("resume_ok") is True if trained else None,
        "RC calibrated": m.get("rc_ops_choice") is not None,
        "timing (<= 900 min/pod)": hours * 60 <= 900,
    }
    eval_lines = []
    for arm in ("A0", "A2", "A3", "A6"):
        e = m[f"eval_{arm}"]
        per_set = ", ".join(f"{s} {_p(m[f'eval_{arm}_{s}_primary'])}/{_p(m[f'eval_{arm}_{s}_strict'])}" for s in sets)
        eval_lines.append(f"  {arm}: primary/strict {per_set} ; recovered {e['recovered']} ; broken {e['broken']} ; "
                          f"net {e['net']} ; interv. rate {_p(e['interv_rate'])} ; FPR {_p(e['fpr'])} ; "
                          f"extra tokens {_n(e['extra_tokens'], 1)} ; truncation {_p(e['trunc_rate'])}")
    final = m["s1_max_new_final"]
    skipped = "skipped (probe gate failed)"
    lines += [
        "SCORING: primary = lenient (strict answer, else last number), truncated = wrong ; strict = secondary",
        "TIMING (min): " + " | ".join(f"{s} {m[f'{s}_min']}" for s in
                                      ("S1", "S1rc", "S2", "S3", "S4", "S5", "S6", "S7", "S8"))
        + f" | total {m['elapsed_min']:.2f}",
        f"HEADROOM primary/strict: student {acc('student')} ; teacher {acc('teacher')} ; "
        f"gap p2 (primary) {100 * m['s1_gap_p2']:.1f} ± {100 * m['s1_gap_p2_se']:.1f} pts (paired SE) ; "
        f"max_new_tokens tried {m['s1_max_new_tried']} -> {final} ; truncation {trunc}",
        f"RC CALIBRATION (60 per level, 800 tokens, primary): {rc} ; choice for FULL eval and train prompts: "
        f"{m.get('rc_ops_choice') if m.get('rc_ops_choice') is not None else 'none qualifies'} ops "
        f"(rule: largest level with teacher >= 60% and truncation <= 5%)",
        f"ROLLOUTS: n {m['s2_n']} | primary acc greedy {_p(m['s2_greedy_primary'])} / sampled "
        f"{_p(m['s2_sampled_primary'])} (strict greedy {_p(m['s2_greedy_strict'])}) | mean tokens "
        f"{_n(m['s2_mean_tokens'], 1)} | mean boundaries {_n(m['s2_mean_boundaries'], 2)} | truncation {_p(m['s2_trunc'])}",
        f"PROBE: layers {m.get('probe_layers')} | C {m.get('probe_C')} | AUROC held-out {_n(m.get('probe_auroc'))} "
        f"(cv {_n(m.get('probe_cv_auroc'))}) | tau {m.get('probe_tau')} | FPR {_p(m.get('probe_fpr'))} | recall "
        f"{_p(m.get('probe_recall'))} | restart share {_p(restart)}"
        f"{' (> 60%: the detector is acting as a difficulty estimator)' if not _bad(restart) and restart > 0.6 else ''}"
        f" | first-flag histogram {m.get('probe_first_flag_hist')} | held-out traces {m.get('probe_holdout_traces')} "
        f"({m.get('probe_holdout_wrong')} wrong) | features fp32 prefill, batch 8",
        "PROBE AUROC TABLE (5-fold CV grouped by original_id on train ids; held-out at each set's best C):",
        *rows,
        f"IDENTITY: A3-untrained == A2 {'y' if m.get('identity_equal') else 'n'} (n {m.get('identity_n')}, max token "
        f"diff {m.get('identity_max_token_diff')})",
        f"STATE TOLERANCE (informational; max rel L2 vs generation states at {m.get('tol_boundaries')} boundaries): "
        f"batch-1 unpadded bf16 prefill max {_n(m.get('tol_max_b1'), 5)} (layers "
        + ", ".join(f"{l}: {_n(m['tol_gen_vs_b1_bf16'][l], 5)}" for l in pick)
        + f") ; fp32 batch-8 prefill max {_n(m.get('tol_max_fp32'), 5)} (layers "
        + ", ".join(f"{l}: {_n(m['tol_gen_vs_fp32_batched'][l], 5)}" for l in pick) + ")",
        ("REPAIR DATA: " + skipped) if m.get("skipped") and not trained else
        f"REPAIR DATA: flagged-wrong {m.get('s5_flagged_wrong')} (used {m.get('s5_problems')}) | continuations "
        f"{m.get('s5_continuations')} (primary {_p(m.get('s5_primary'))}, truncation {_p(m.get('s5_trunc'))}) | "
        f"teacher-also-wrong {_p(m.get('s6_teacher_also_wrong'))}",
        ("TRAIN: " + skipped) if not trained else
        f"TRAIN: steps {m.get('train_steps')} | loss start/end {_n(m.get('loss_start'), 4)}/{_n(m.get('loss_end'), 4)} "
        f"| resume check {'y' if m.get('resume_ok') else 'n' if m.get('resume_ok') is False else 'n/a'} (kill at "
        f"{m.get('train_kill_at')}, checkpoint every {m.get('train_ckpt_every')}, max rel diff "
        f"{_n(m.get('resume_max_rel_diff'), 6)}) | adapter params {m.get('adapter_params')}",
        f"EVAL ({sum(n.values())}){'' if trained else ' — adapter untrained (S7 skipped): A3 ~ A2, A6 ~ A0'}:",
        *eval_lines,
        f"THROUGHPUT (HF generate, tok/s): S1 student {_n(m[f's1_student_tok_per_s_{final}'], 1)} / teacher "
        f"{_n(m[f's1_teacher_tok_per_s_{final}'], 1)} | S2 greedy {_n(m['s2_greedy_tok_per_s'], 1)} / sampled "
        f"{_n(m['s2_sample_tok_per_s'], 1)} | S8 A6 {_n(m.get('s8_a6_tok_per_s'), 1)} | teacher prefill "
        f"{_n(m.get('s6_prefill_tok_per_s'), 1) if m.get('s6_prefill_tok_per_s') else f'{SMOKE1_TEACHER_PREFILL_TOK_PER_S} (SMOKE-1)'}"
        f" | train steps/s {_n(m.get('train_steps_per_s'), 4) if m.get('train_steps_per_s') else f'{1 / SMOKE1_TRAIN_SEC_PER_STEP:.4f} (SMOKE-1)'}",
        f"PROJECTION: FULL {hours:.1f} h and ${hours * price:.2f} per seed on this tier (rough; "
        + ", ".join(f"{k} {v:.1f} h" for k, v in parts.items()) + "; ceiling 15 h/pod)",
        "GATES: " + " ; ".join(f"{k} {'pass' if v else 'n/a' if v is None else 'fail'}" for k, v in gates.items()),
        f"BUDGET: max_minutes {env['max_minutes']} x ${price:.2f}/h = ${price * int(env['max_minutes']) / 60:.2f} "
        f"<= ${cfg['phase_budget_usd'][env['phase']]}",
        "BLOCKERS: " + ("; ".join(blockers) if blockers else "none"),
    ]
    _write(run_dir, lines, m)
