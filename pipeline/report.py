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


def _table(prefix, m, key):
    """AUROC table lines from a probe_table / step_table dict."""
    table = m.get(prefix + key) or {}
    out = []
    for name in dict.fromkeys(k.split("|")[0] for k in table):
        cells = [f"C={k.split('C=')[1]} cv {_n(v['cv_auroc'])}" + (f" held-out {_n(v['heldout_auroc'])}"
                 if "heldout_auroc" in v else "") for k, v in table.items() if k.startswith(name + "|")]
        out.append(f"    {name}: " + " ; ".join(cells))
    return out


def _fpr(d):
    if not d:
        return "n/a"
    return " ; ".join(f"{s} {_p(v['fpr'])} (n {v['n_correct']})" for s, v in d.items())


def _by_set(d, key):
    return " ; ".join(f"{s} {_n(v[key]) if key == 'auroc' else _p(v[key])} (n {v['n']}, wrong {v['wrong']})"
                      for s, v in (d or {}).items()) or "n/a"


def _probe_lines(p, m, label):
    if m.get(p + "probe_layers") is None:
        return [f"  {label}: {m.get(p + 'probe_blocker', 'not run')}"]
    return [
        f"  {label}: layers {m[p + 'probe_layers']} | C {m[p + 'probe_C']} | held-out AUROC {_n(m[p + 'probe_auroc'])} "
        f"(cv {_n(m[p + 'probe_cv_auroc'])}) | recall {_p(m[p + 'probe_recall'])} at held-out FPR "
        f"{_p(m[p + 'probe_fpr'])} (tau {_n(m[p + 'probe_tau'], 4)}, eval-mix weighted "
        f"{'y' if m.get(p + 'probe_tau_weighted') else 'n'}) | restart share {_p(m.get(p + 'probe_restart_share'))} | "
        f"gate {'pass' if m.get(p + 'probe_gate') else 'fail'}",
        f"    held-out AUROC per set: {_by_set(m.get(p + 'probe_by_set'), 'auroc')}",
        f"    held-out FPR per set at tau: {_by_set(m.get(p + 'probe_by_set'), 'fpr')}",
        f"    eval (test traces) FPR per set at tau: {_fpr(m.get(p + 'probe_eval_fpr'))}",
        "    AUROC table (grouped 5-fold CV on train ids; held-out at each set's best C):",
        *_table(p, m, "probe_table"),
    ]


def smoke3(run_dir, m, cfg):
    env, _, lines = header(run_dir, m, cfg)
    price = m["price_per_hr"]
    sets = ("p2", "main", "p1", "rc")
    stages = ("MCa", "MCprobe", "B1", "B2", "B3", "C2", "C3")
    skipped = {s: m.get(f"{s}_skipped") for s in stages if m.get(f"{s}_skipped")}
    lines += [
        f"SOURCE: SMOKE-2 artifacts from {cfg['smoke3_source_run']} (S2 greedy traces, S1 A0 traces, S1 teacher) ; "
        f"scoring primary (lenient, truncated = wrong) ; RC for FULL: {cfg['rc_ops_full']} ops",
        "TIMING (min): " + " | ".join(f"{s} {m.get(f'{s}_min')}" for s in stages) + f" | total {m['elapsed_min']:.2f}",
        "(a) PROBE-MC (Qwen3-1.7B, SMOKE-2 greedy train traces, 3 completions per point, T 0.8, cap 800):",
    ]
    if "MCa_skipped" in m:
        lines.append(f"  {m['MCa_skipped']}")
    else:
        lines += [
            f"  labels: traces {m.get('mc_traces')}/{m.get('mc_of')} | completions {m.get('mc_completions')} | "
            f"tok/s {_n(m.get('mc_tok_per_s'), 1)} | bad steps {m.get('mc_step_pos')} | traces with a bad step "
            f"{m.get('mc_traces_with_bad_step')} | wrong traces {m.get('mc_wrong_traces')} (with a bad step "
            f"{m.get('mc_wrong_with_bad_step')}) | MC success at the prompt: correct traces "
            f"{_n(m.get('mc_prompt_success_correct'))}, wrong traces {_n(m.get('mc_prompt_success_wrong'))}",
        ]
        if m.get("step_layers") is None:
            lines.append(f"  step probe: {m.get('MCprobe_skipped') or m.get('step_blocker', 'not run')} "
                         f"(positives train {m.get('step_pos_train')}, held-out {m.get('step_pos_heldout')})")
        else:
            lines += [
                f"  step probe: layers {m['step_layers']} | C {m['step_C']} | step AUROC held-out "
                f"{_n(m['step_auroc'])} (cv {_n(m['step_cv_auroc'])}; positives train {m['step_pos_train']}/"
                f"{m['step_boundaries_train']}, held-out {m['step_pos_heldout']}/{m['step_boundaries_heldout']}) | "
                f"first bad step within +-1 {_p(m['step_first_bad_within1'])} (n {m['step_heldout_traces_with_bad']}) | "
                f"implied trace AUROC {_n(m['step_trace_auroc'])} | tau {_n(m['step_tau'], 4)}",
                f"    eval (test traces) FPR per set at step tau: {_fpr(m.get('step_eval_fpr'))}",
                "    step AUROC table:", *_table("", m, "step_table"),
            ]
    lines.append("(b) QWEN3-4B STUDENT:")
    if "B1_skipped" in m:
        lines.append(f"  S1: {m['B1_skipped']}")
    else:
        lines.append(
            "  S1 (380, 800 tokens, teacher from SMOKE-2) primary/strict, student vs teacher: " + " / ".join(
                f"{s} {_p(m[f'b1_student_{s}_primary'])}/{_p(m[f'b1_student_{s}_strict'])} vs "
                f"{_p(m[f'b1_teacher_{s}_primary'])} (trunc {_p(m[f'b1_student_{s}_trunc'])})" for s in sets)
            + f" ; gap p2 (primary) {100 * m['b1_gap_p2']:.1f} ± {100 * m['b1_gap_p2_se']:.1f} pts ; "
              f"tok/s {_n(m['b1_tok_per_s'], 1)} ; 4B/8B prompt token ids identical {m.get('b1_template_identical')}/20")
    if "B2_skipped" in m:
        lines.append(f"  S2: {m['B2_skipped']}")
    else:
        lines.append(
            f"  S2: problems {m['b2_n_problems']}/{m['b2_of']} x 2 | base rate (greedy primary) "
            f"{_p(m['b2_greedy_primary'])} (" + ", ".join(f"{s} {_p(m[f'b2_greedy_primary_{s}'])}"
                                                           for s in ("main", "p1", "p2", "rc"))
            + f") | sampled {_p(m['b2_sampled_primary'])} | truncation {_p(m['b2_trunc'])} | mean tokens "
              f"{_n(m['b2_mean_tokens'], 1)} | tok/s {_n(m['b2_tok_per_s'], 1)}")
    lines += [f"  S3: {m['B3_skipped']}"] if "B3_skipped" in m else _probe_lines("b3_", m, "S3")
    lines.append("(c) DATA SCALING (Qwen3-1.7B):")
    if "C2_skipped" in m:
        lines.append(f"  S2: {m['C2_skipped']}")
    else:
        lines.append(
            f"  S2: problems {m['c2_n_problems']}/{m['c2_of']} x 2 | base rate (greedy primary) "
            f"{_p(m['c2_greedy_primary'])} (" + ", ".join(f"{s} {_p(m[f'c2_greedy_primary_{s}'])}"
                                                           for s in ("main", "p1", "p2", "rc"))
            + f") | truncation {_p(m['c2_trunc'])} | mean tokens {_n(m['c2_mean_tokens'], 1)} | tok/s "
              f"{_n(m['c2_tok_per_s'], 1)}")
    if "C3_skipped" in m:
        lines.append(f"  S3: {m['C3_skipped']}")
    else:
        lines.append("  curve (layer 21, C by grouped CV, fixed held-out groups; AUROC pooled | per set | recall):")
        for c in m.get("c3_curve", []):
            if c.get("skipped"):
                lines.append(f"    {c['total']} traces: {c['skipped']}")
                continue
            per = ", ".join(f"{s} {_n(v)}" for s, v in (c.get("by_set") or {}).items())
            lines.append(f"    {c['total']} traces (train {c['train_traces']}, held-out {c['heldout_traces']}, C "
                         f"{c['C']}): AUROC {_n(c['auroc'])} | {per} | recall {_p(c['recall'])}")
        lines += _probe_lines("c3_", m, "full protocol at 2,400 traces")
    gates = {k: m.get(k) for k in ("b3_probe_gate", "c3_probe_gate")}
    lines += [
        "GATE (AUROC >= 0.80 and recall >= 0.40 at FPR 10% on held-out ids): "
        + " ; ".join(f"{k.split('_')[0]} {'pass' if v else 'fail' if v is not None else 'n/a'}"
                     for k, v in gates.items()),
        f"BUDGET: max_minutes {env['max_minutes']} x ${price:.2f}/h = ${price * int(env['max_minutes']) / 60:.2f} "
        f"<= ${cfg['phase_budget_usd'][env['phase']]}",
        "BLOCKERS: " + ("; ".join(f"{s} {v}" for s, v in skipped.items()) if skipped else "none"),
    ]
    _write(run_dir, lines, m)


def _ci(c):
    return f"[{_n(c[0])}, {_n(c[1])}]" if c else "[nan, nan]"


def smoke4(run_dir, m, cfg):
    env, _, lines = header(run_dir, m, cfg)
    price = m["price_per_hr"]
    stages = ("A1", "A2", "A4", "A3")
    skipped = {s: m.get(f"{s}_skipped") for s in stages if m.get(f"{s}_skipped")}
    lines += [
        f"BRANCH: {cfg['smoke4_branch']} (Qwen3-1.7B) chosen by STAGE 0 (local, SMOKE-2 features): {cfg['smoke4_stage0']}",
        "TIMING (min): " + " | ".join(f"{s} {m.get(f'{s}_min')}" for s in stages) + f" | total {m['elapsed_min']:.2f}",
        "A1 MC STRATIFICATION (4 samples, T 0.8, cap 800; contested = 1..3 of 4 correct, the widest bounds that give "
        "a correct and a wrong trace):",
    ]
    if "A1_skipped" in m:
        lines.append(f"  {m['A1_skipped']}")
    else:
        lines.append(f"  problems {m['a1_problems']}/{m['a1_of']} | tok/s {_n(m['a1_tok_per_s'], 1)} (batch "
                     f"{m['a1_bs']}) | truncation {_p(m['a1_trunc'])} | mean tokens {_n(m['a1_mean_tokens'], 1)}")
        for k, v in m["a1_dist"].items():
            lines.append(f"  {k}: problems {v['problems']} | #correct of 4 (0..4) {v['hist']} | accuracy "
                         f"{_p(v['acc'])} | contested {_p(v['contested'])}")
        p1 = m["a1_dist"].get("train_p1", {}).get("contested")
        if p1 is not None and p1 < 0.30:
            lines.append(f"  NOTE: contested share of train p1 {_p(p1)} < 30%; 1..3 of 4 is already the widest bound")
    lines.append("A2 WITHIN-PROBLEM PROBE (one wrong + one correct per contested train problem; fp32 features; "
                 "layer, C, pooling by grouped 5-fold CV on train ids):")
    if "A2_skipped" in m or m.get("within_layers") is None:
        lines.append(f"  {m.get('A2_skipped') or m.get('within_blocker', 'not run')} (pairs train "
                     f"{m.get('within_train_pairs')}, held-out {m.get('within_heldout_pairs')})")
    else:
        lines += [
            f"  pairs train {m['within_train_pairs']} | held-out {m['within_heldout_pairs']} | chosen: layer "
            f"{m['within_layers']}, C {m['within_C']}, pooling {m['within_pool']} (cv {_n(m['within_cv_auroc'])})",
            f"  held-out WITHIN-PROBLEM AUROC {_n(m['within_auroc'])} {_ci(m['within_ci'])} | pooled trace AUROC "
            f"{_n(m['within_trace_auroc'])} | tau {_n(m['within_tau'], 4)} (FPR 10% on held-out contested correct): "
            f"FPR {_p(m['within_heldout_fpr'])}, recall {_p(m['within_heldout_recall'])}",
            "    by pooling (chosen layer): " + " ; ".join(f"{k} {_n(v['auroc'])} {_ci(v['ci'])}"
                                                       for k, v in m["within_by_pool"].items()),
            "    by layer (each at its best C; max / mean / last): " + " ; ".join(
                f"{k} (C {v['C']}) {_n(v['max'])} / {_n(v['mean'])} / {_n(v['last'])}"
                for k, v in m["within_by_layers"].items()),
            "    by boundary position (chosen layer): " + " ; ".join(
                f"{k}: {_n(v['auroc'])} (n {v['n']})" for k, v in m["within_by_position"].items()),
            "    CV table (max / mean / last): " + " ; ".join(
                f"{k} {_n(v['cv']['max'])}/{_n(v['cv']['mean'])}/{_n(v['cv']['last'])}"
                for k, v in m["within_table"].items()),
        ]
    lines.append("A4 TEST (all traces of contested TEST problems; tau from A2):")
    if "A4_skipped" in m:
        lines.append(f"  {m['A4_skipped']}")
    else:
        for k, v in m["a4_test"].items():
            lines.append(f"  {k}: problems {v['problems']} (correct {v['correct']}, wrong {v['wrong']}) | FPR "
                         f"{_p(v['fpr'])} | recall {_p(v['recall'])} | within-pair AUROC {_n(v['within_auroc'])} "
                         f"{_ci(v['within_ci'])} (pairs {v['pairs']})")
    lines.append(f"A3 STEP LOCALIZATION (contested wrong train traces, 3 completions per point; first point with MC "
                 f"success < 0.25; restart = first drop at the prompt or boundary 1):")
    if "A3_skipped" in m:
        lines.append(f"  {m['A3_skipped']}")
    else:
        lines.append(f"  traces {m['a3_traces']}/{m['a3_of']} | completions {m['a3_completions']} | tok/s "
                     f"{_n(m['a3_tok_per_s'], 1)} | MC at prompt mean {_n(m['a3_mc_prompt_mean'])} | restart share "
                     f"{_p(m['a3_restart_share'])} | never below 0.25 {_p(m['a3_never_share'])} | first drop "
                     f"(0 = prompt) {m['a3_first_drop_hist']}")
    pooled = (m.get("a4_test") or {}).get("pooled", {})
    gate = None if m.get("within_auroc") is None or not pooled else bool(
        m["within_auroc"] >= 0.80 and pooled.get("recall", 0) >= 0.40)
    lines += [
        "GATE (within-problem AUROC >= 0.80 held-out and recall >= 0.40 at FPR 10% on test): "
        + ("n/a" if gate is None else "pass" if gate else "fail"),
        f"BUDGET: max_minutes {env['max_minutes']} x ${price:.2f}/h = ${price * int(env['max_minutes']) / 60:.2f} "
        f"<= ${cfg['phase_budget_usd'][env['phase']]}",
        "BLOCKERS: " + ("; ".join(f"{s} {v}" for s, v in skipped.items()) if skipped else "none"),
    ]
    m["smoke4_gate"] = gate
    _write(run_dir, lines, m)


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
