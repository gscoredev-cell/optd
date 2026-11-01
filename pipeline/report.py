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
        f"TEMPLATES: identical token ids {m.get('template_identical_ids', 'n/a')}/{m.get('template_n', 'n/a')} | "
        f"empty think block {m.get('template_empty_think', 'n/a')}/{m.get('template_n', 'n/a')}",
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


def _positions(by_pos):
    return " ; ".join(f"{k}: {_n(v['auroc'])} (n {v['n']}" + (f", unrec {v['pos']})" if "pos" in v else ")")
                      for k, v in sorted(by_pos.items(), key=lambda kv: int(kv[0])))


def _trend(by_pos, min_n=10):
    """One line: does AUROC rise with boundary position? Spearman rho over positions with n >= min_n; yes when
    rho >= 0.5 and the mean over positions 5..12 exceeds the mean over 1..4 by >= 0.05."""
    pts = sorted((int(k), v["auroc"]) for k, v in by_pos.items() if v["n"] >= min_n and not _bad(v["auroc"]))
    if len(pts) < 3:
        return f"rises with position: n/a (fewer than 3 positions with n >= {min_n})"
    from scipy.stats import spearmanr
    rho = float(spearmanr([k for k, _ in pts], [a for _, a in pts])[0])
    early, late = [a for k, a in pts if k <= 4], [a for k, a in pts if k >= 5]
    diff = sum(late) / len(late) - sum(early) / len(early) if early and late else float("nan")
    verdict = "n/a" if _bad(diff) else "yes" if rho >= 0.5 and diff >= 0.05 else "no"
    return (f"rises with position: {verdict} (Spearman rho {_n(rho, 2)} over positions "
            f"{pts[0][0]}..{pts[-1][0]} with n >= {min_n}; mean 5..12 minus mean 1..4 = {_n(diff)})")


def smoke5(run_dir, m, cfg):
    env, _, lines = header(run_dir, m, {**cfg, "student": "Qwen/Qwen3-8B", "teacher": "none (not loaded)"})
    price = m["price_per_hr"]
    stages = ("D1", "D2", "D4a", "D3", "D4b")
    skipped = {s: m.get(f"{s}_skipped") for s in stages if m.get(f"{s}_skipped")}
    hm = " ; ".join(f"{s} {m[f'{s.lower()}_cache_hits']}/{m[f'{s.lower()}_cache_misses']}"
                    for s in ("D1", "D3", "D4b") if f"{s.lower()}_cache_hits" in m)
    lines += [
        "STUDENT: Qwen3-8B, no teacher | features bf16 prefill, batch 8 | HF generate, T 0.8, top_p 0.95, cap 800",
        "TIMING (min): " + " | ".join(f"{s} {m.get(f'{s}_min')}" for s in stages) + f" | total {m['elapsed_min']:.2f}",
        f"GENERATION CACHE (/workspace/gencache, key model | prompt sha256 | T | top_p | seed, cap rule): seeded from "
        f"SMOKE-2 S1 ({cfg['smoke3_source_run']}): {m.get('d1_seed_traces')} Qwen3-8B greedy traces, "
        f"{m.get('d1_seed_imported')} new, {m.get('d1_seed_used')} are D1 problems (prompt ids identical "
        f"{m.get('d1_seed_prompt_ids_same')}/{m.get('d1_seed_used')}) | hits/misses per stage: {hm or 'n/a'}",
        "D1 MC STRATIFICATION (4 per problem; contested = 1..3 of 4 correct):",
    ]
    if "D1_skipped" in m:
        lines.append(f"  {m['D1_skipped']}")
    else:
        lines.append(f"  problems {m['d1_problems']}/{m['d1_of']} | tok/s {_n(m['d1_tok_per_s'], 1)} (batch "
                     f"{m['d1_bs']}, generated rows only) | truncation {_p(m['d1_trunc'])} | mean tokens "
                     f"{_n(m['d1_mean_tokens'], 1)}")
        for k, v in m["d1_dist"].items():
            lines.append(f"  {k}: problems {v['problems']} | #correct of 4 (0..4) {v['hist']} | accuracy "
                         f"{_p(v['acc'])} | contested {_p(v['contested'])}")
        lines.append(f"  NOTE: test_p2 p_hat mixes 1 greedy (sample 0, SMOKE-2 S1 via the cache) + 3 sampled for "
                     f"{m['d1_greedy0']} problems; not directly comparable with train_p2 (4 sampled)")
    lines.append("D2 TARGET A, WITHIN-PROBLEM PROBE (one wrong + one correct per contested train problem; layer, C, "
                 "pooling by grouped 5-fold CV on train groups; held-out = 20% of train groups):")
    if "D2_skipped" in m or m.get("within_layers") is None:
        lines.append(f"  {m.get('D2_skipped') or m.get('within_blocker', 'not run')} (pairs train "
                     f"{m.get('within_train_pairs')}, held-out {m.get('within_heldout_pairs')})")
    else:
        lines += [
            f"  pairs train {m['within_train_pairs']} | held-out {m['within_heldout_pairs']} | chosen: layer "
            f"{m['within_layers']}, C {m['within_C']}, pooling {m['within_pool']} (cv {_n(m['within_cv_auroc'])}) | "
            f"features {_n(m['d2_feat_sec'] / 60, 1)} min",
            f"  held-out WITHIN-PROBLEM AUROC {_n(m['within_auroc'])} {_ci(m['within_ci'])} | pooled trace AUROC "
            f"{_n(m['within_trace_auroc'])} | tau {_n(m['within_tau'], 4)} (FPR 10% on held-out contested correct): "
            f"FPR {_p(m['within_heldout_fpr'])}, recall {_p(m['within_heldout_recall'])}",
            "    by pooling (chosen layer): " + " ; ".join(f"{k} {_n(v['auroc'])} {_ci(v['ci'])}"
                                                       for k, v in m["within_by_pool"].items()),
            "    by layer (each at its best C; max / mean / last): " + " ; ".join(
                f"{k} (C {v['C']}) {_n(v['max'])} / {_n(v['mean'])} / {_n(v['last'])}"
                for k, v in m["within_by_layers"].items()),
            "    by boundary position (held-out pairs): " + _positions(m["within_by_position"]),
            "    " + _trend(m["within_by_position"]),
            "    CV table (max / mean / last): " + " ; ".join(
                f"{k} {_n(v['cv']['max'])}/{_n(v['cv']['mean'])}/{_n(v['cv']['last'])}"
                for k, v in m["within_table"].items()),
        ]
    lines.append("D4a TEST, TARGET A (all traces of contested TEST problems; fresh pairs; tau from D2):")
    if "D4a_skipped" in m:
        lines.append(f"  {m['D4a_skipped']}")
    else:
        for k, v in m["d4a_test"].items():
            lines.append(f"  {k}: problems {v['problems']} (correct {v['correct']}, wrong {v['wrong']}) | FPR "
                         f"{_p(v['fpr'])} | recall {_p(v['recall'])} | within-pair AUROC {_n(v['within_auroc'])} "
                         f"{_ci(v['within_ci'])} (pairs {v['pairs']})")
        lines += ["  by boundary position (test pairs): " + _positions(m["d4a_by_position"]),
                  "  " + _trend(m["d4a_by_position"])]
    lines.append(f"D3 TARGET B, RECOVERABILITY (contested train pairs, 3 completions from the prompt and each boundary; "
                 f"boundary unrecoverable = MC success < 0.5; boundary probe by grouped 5-fold CV, held-out = 20% of "
                 f"groups):")
    if "D3_skipped" in m:
        lines.append(f"  {m['D3_skipped']}")
    else:
        lines.append(f"  traces {m['d3_traces']}/{m['d3_of']} | completions {m['d3_completions']} | tok/s "
                     f"{_n(m['d3_tok_per_s'], 1)} | MC at prompt mean {_n(m['d3_mc_prompt_mean'])} | unrecoverable "
                     f"boundaries: correct traces {_p(m['d3_unrec_share_correct'])}, wrong traces "
                     f"{_p(m['d3_unrec_share_wrong'])}")
        if m.get("d3_step_layers") is None:
            lines.append(f"  probe: {m.get('d3_step_blocker', 'not fitted')}")
        else:
            lines += [
                f"  probe: layer {m['d3_step_layers']}, C {m['d3_step_C']} (cv {_n(m['d3_step_cv_auroc'])}) | held-out "
                f"boundary AUROC {_n(m['d3_step_auroc'])} (boundaries {m['d3_step_boundaries_heldout']}, unrecoverable "
                f"{m['d3_step_pos_heldout']}) | tau {_n(m['d3_step_tau'], 4)} (FPR 10% on held-out recoverable)",
                "    by layer (held-out at best C): " + " ; ".join(
                    f"{k} {_n(v['heldout_auroc'])}" for k, v in m["d3_step_table"].items() if "heldout_auroc" in v),
                "    by boundary position (held-out): " + _positions(m.get("d3_by_position", {})),
                "    " + _trend(m.get("d3_by_position", {})),
            ]
        lines.append(f"  wrong traces {m['d3_wrong_traces']}: first point with MC < 0.25 (SMOKE-4 rule) | restart "
                     f"share {_p(m['d3_restart_share'])} | never below 0.25 {_p(m['d3_never_share'])} | first drop "
                     f"(0 = prompt) {m['d3_first_drop_hist']}")
    lines.append("D4b TEST, TARGET B (contested TEST pairs with MC labels, 3 completions per point; tau from D3):")
    if "D4b_skipped" in m:
        lines.append(f"  {m['D4b_skipped']}")
    else:
        lines.append(f"  traces {m['d4b_traces']}/{m['d4b_of']} | completions {m['d4b_completions']} | tok/s "
                     f"{_n(m['d4b_tok_per_s'], 1)}")
        for k, v in m["d4b_test"].items():
            lines.append(f"  {k}: traces {v['traces']} | boundaries {v['boundaries']} (unrecoverable {v['unrec']}) | "
                         f"boundary AUROC {_n(v['auroc'])} | FPR {_p(v['fpr'])} | recall {_p(v['recall'])}")
        lines += ["  by boundary position (test): " + _positions(m["d4b_by_position"]),
                  "  " + _trend(m["d4b_by_position"])]
    pooled = (m.get("d4a_test") or {}).get("pooled")
    gate = None if not pooled or _bad(pooled["within_auroc"]) else bool(
        pooled["within_auroc"] >= 0.75 and pooled["within_ci"][0] > 0.65)
    lines += [
        "GATE (test within-problem AUROC >= 0.75 with 95% CI lower bound > 0.65; D4a pooled): "
        + ("n/a" if gate is None else "pass" if gate else "fail"),
        f"BUDGET: max_minutes {env['max_minutes']} x ${price:.2f}/h = ${price * int(env['max_minutes']) / 60:.2f} "
        f"<= ${cfg['phase_budget_usd'][env['phase']]}",
        "BLOCKERS: " + ("; ".join(f"{s} {v}" for s, v in skipped.items()) if skipped else "none"),
    ]
    m["smoke5_gate"] = gate
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


# ---- Option E (OPTION_E_SPEC.md §8; SMOKE-7 addendum: per adapter and per regime) -----------------------------------

E_ARMS = ("E0", "E1", "E2", "E3", "E4a", "E4b", "E6")
E_SETS = ("p1", "p2", "main", "rc3", "pooled")


def _oracle_line(m, tag):
    parts = []
    for s in E_SETS:
        v = m.get(f"{tag}_{s}")
        if v:
            parts.append(f"{s}: n {v['n']} | restart share {_p(v['restart_share'])} | b* >= 1 {v['b_ge1']} | "
                         f"to end {v['to_end']} | b* hist {v['b_hist']}")
    return parts or ["n/a"]


def _pt(x):
    return "nan" if _bad(x) else f"{100 * x:+.1f}"


def _delta(st, key):
    ci = st.get(key + "_ci") or [None, None]
    return f"{_pt(st.get(key))} [{_pt(ci[0])}, {_pt(ci[1])}]"


def e_projection(m, price):
    """FULL-E per seed (§6 FULL-E) from the measured rates: ~1,800 contested train problems, 4 rounds, test = all
    contested test p1/p2 (~1,950 problems at the SMOKE-4 rates) + RC3 300, 2 continuations per arm. Training uses the
    pure steps/s (dev evaluation and resampling excluded). Returns (hours, parts)."""
    T = 250.0
    mc_rate = m.get("es2_mc_tok_per_s") or m.get("es6_mc_tok_per_s") or 300.0
    gen = m.get("es7_gen_tok_per_s") or m.get("es4_gen_tok_per_s") or 400.0
    prefill = m.get("es4_prefill_tok_per_s") or SMOKE1_TEACHER_PREFILL_TOK_PER_S
    sps = m.get("pure_train_steps_per_s") or m.get("train_steps_per_s")
    step = 1 / sps if sps else SMOKE1_TRAIN_SEC_PER_STEP
    wrong_rate = ((m.get("es2_greedy_wrong") or 0) / max(m.get("es2_greedy") or 1, 1)) or 0.5
    probes = m.get("es2_probes_per_trace") or 5.0
    k = (m.get("variant") or {}).get("k", 3)
    n_train_units = 1800 * wrong_rate * 1.5      # greedy wrong + sampled wrong with own scans
    n_test = 1950 * wrong_rate + 300 * wrong_rate
    n_arms = 3 + 5 * len((m.get("variant") or {}).get("adapters", ("rkl",)))
    parts = {
        "oracle_train": n_train_units * probes * k * T / mc_rate,
        "continuations_4_rounds": 4 * n_train_units * 5 * T / gen,
        "teacher_scoring": 4 * n_train_units * 5 * (2 * T + 150) / prefill,
        "training": 4 * n_train_units * 5 / 16 * step,
        "dev_evals": 4 * 5 * (m.get("rkl_dev_eval_sec") or m.get("dev_eval_sec") or 60.0),
        "oracle_test": n_test * probes * k * T / mc_rate,
        "arms": n_test * n_arms * 2 * T / gen,
    }
    parts = {k2: v / 3600 for k2, v in parts.items()}
    return sum(parts.values()), parts


def _arms_block(m, ad, label):
    """ARMS + DELTAS lines for one adapter: regimes all / pointed (b* >= 1) / restart (b* = 0), per set and pooled."""
    lines = [f"ARMS [{label}] (test; recovery rate wrong -> correct; new tokens incl. injected text; truncation):"]
    for reg in ("all", "pointed", "restart"):
        for s in E_SETS:
            st = m.get(f"es7_{ad}_{reg}_{s}")
            if not st or not st["n"]:
                continue
            lines.append(f"  {reg} {s} (n {st['n']}, problems {st['problems']}): " + " ; ".join(
                f"{a} {_p(st.get(f'rec_{a}'))} ({_n(st.get(f'tokens_{a}'), 0)} tok, trunc {_p(st.get(f'trunc_{a}'))})"
                for a in E_ARMS))
    lines.append(f"  E5 harm [{label}] (correct -> wrong, adapter on, random cut) {_p(m.get(f'es7_e5_harm_{ad}'))} vs "
                 f"plain-retry harm {_p(m.get('es7_plain_harm'))} (n {m.get('es7_e5_n')})")
    lines.append(f"DELTAS [{label}] (points; 95% CI by cluster bootstrap on original_id, 10,000 resamples; I on pointed "
                 "units is primary):")
    for reg in ("pointed", "restart", "all"):
        for s in E_SETS:
            st = m.get(f"es7_{ad}_{reg}_{s}")
            if st and "delta_primary" in st:
                lines.append(f"  {reg} {s} (n {st['n']}): delta_primary E3-E1 {_delta(st, 'delta_primary')} | E2-E1 "
                             f"{_delta(st, 'e2_minus_e1')} | E3-E2 {_delta(st, 'e3_minus_e2')} | I {_delta(st, 'interaction')}")
    return lines


def _train_block(m, ad, label):
    pre = f"{ad}_"
    if f"{pre}train_rounds" not in m:
        return [f"TRAIN [{label}]: {m.get(f'{ad}_skipped') or m.get(f'{pre}train_blocker', 'not run')}"]
    lines = [
        f"TRAIN [{label}]: rounds {m.get(f'{pre}train_rounds_done')} | steps {m.get(f'{pre}train_steps_total')} | best: round "
        f"{m.get(f'{pre}train_best_round')} step {m.get(f'{pre}train_best_step')} dev E3 recovery {_p(m.get(f'{pre}train_best_dev'))} | "
        f"early stopping fired {'y' if m.get(f'{pre}early_stop_fired') else 'n'} | train {_n(m.get(f'{pre}train_sec'), 0)} s "
        f"(dev evals {_n(m.get(f'{pre}dev_eval_sec'), 0)} s, resampling {_n(m.get(f'{pre}resample_sec'), 0)} s) | steps/s "
        f"{_n(m.get(f'{pre}train_steps_per_s'), 4)} overall, {_n(m.get(f'{pre}pure_train_steps_per_s'), 4)} pure"
        + (f" | {m[f'{pre}rounds_stopped']}" if m.get(f"{pre}rounds_stopped") else "")
        + (f" | {m[f'{pre}train_blocker']}" if m.get(f"{pre}train_blocker") else "")]
    for r in m.get(f"{pre}train_rounds") or []:
        curve = " ; ".join(f"{s}: {_p(v)}" for s, v in r["curve"])
        n_ex = m.get(f"{pre}train_examples_r{r['round']}")
        lines.append(f"  round {r['round']}: steps {r['steps']} (examples {n_ex}) | "
                     f"loss start/end {_n(r['loss_start'], 4)}/{_n(r['loss_end'], 4)} | dev E3 recovery per eval (step: rate) "
                     f"{curve or 'n/a'} | best step {r['best_step']} | stopped early {'y' if r['stopped'] else 'n'}")
    for r in range(2, 5):
        if m.get(f"{pre}round{r}_skipped"):
            lines.append(f"  round {r}: {m[f'{pre}round{r}_skipped']}")
    return lines


def option_e(run_dir, m, cfg):
    env, _, lines = header(run_dir, m, cfg)
    price = m["price_per_hr"]
    v = m.get("variant") or {"k": 3, "thresh": 0.5, "lr": 1e-4, "clip": None, "dev_n": 60, "rounds": 2, "adapters": ["rkl"],
                             "extra_data": False}
    adapters = [a for a in v["adapters"]]
    stages = (("ES0",) if v.get("extra_data") else ()) + ("ES1", "ES2", "ES4", "ES5", "ES6", "ES7", "ES8", "ES3")
    skipped = {s: m.get(f"{s}_skipped") for s in stages if m.get(f"{s}_skipped")}
    hm = " ; ".join(f"{k[:-11]} {m[k]}/{m[k[:-4] + 'misses']}" for k in sorted(m) if k.endswith("_cache_hits"))
    c_tr, c_te = m.get("es1_contested_train") or {}, m.get("es1_contested_test") or {}
    lines += [
        f"METHOD: oracle pointer (MC >= {v['thresh']}, {v['k']} completions per probe, bisection) -> LoRA repair adapter(s) "
        f"{'/'.join(adapters)} (lr {v['lr']}, PPO clip {v['clip']}, early stopping on {v['dev_n']} dev traces, up to "
        f"{v['rounds']} rounds) ; first attempts greedy via the cache ; arms E0-E6 + E5 per adapter ; A6 not run",
        "TIMING (min): " + " | ".join(f"{s} {m.get(f'{s}_min')}" for s in stages) + f" | total {m['elapsed_min']:.2f}",
        f"CACHE: SMOKE-4 A1 traces imported {m.get('es1_a1_imported')}/{m.get('es1_a1_traces')} | A3 scans imported as MC "
        f"vectors {m.get('es1_a3_scans')} ({m.get('es1_a3_points_new')} new points; k = 3 store only) | hits/misses per "
        f"stage: {hm or 'n/a'}",
    ]
    if v.get("extra_data"):
        if "ES0_skipped" in m:
            lines.append(f"DATA (E-S0): {m['ES0_skipped']}")
        else:
            lines.append(f"DATA (E-S0, 4 samples per extra problem, T 0.8, cap 800): problems {m.get('es0_problems')}/"
                         f"{m.get('es0_of')} | tok/s {_n(m.get('es0_tok_per_s'), 1)} (batch {m.get('es0_bs')}) | truncation "
                         f"{_p(m.get('es0_trunc'))} | mean tokens {_n(m.get('es0_mean_tokens'), 1)}")
            for k in sorted(k for k in m if k.startswith("es0_t") and isinstance(m[k], dict)):
                d = m[k]
                lines.append(f"  {k[4:]}: problems {d['problems']}/{d['of']} | #correct of 4 (0..4) {d['hist']} | accuracy "
                             f"{_p(d['acc'])} | contested {_p(d['contested'])}")
    lines += [
        f"DATA: contested train problems p1/p2/main/rc3 {c_tr.get('p1')}/{c_tr.get('p2')}/{c_tr.get('main')}/{c_tr.get('rc3')} "
        f"| contested test p1/p2 {c_te.get('p1')}/{c_te.get('p2')} | wrong greedy train traces "
        f"{m.get('es2_greedy_wrong')} of {m.get('es2_greedy')} (acc {_p(m.get('es2_greedy_acc'))}) | wrong greedy test "
        f"traces {m.get('es6_greedy_wrong')} of {m.get('es6_greedy')} (acc {_p(m.get('es6_greedy_acc'))}) | correct test "
        f"traces for E5 {m.get('es6_e5_traces')} (of {m.get('es6_correct_greedy')})",
        f"ORACLE (train): bisection probes per trace {_n(m.get('es2_probes_per_trace'), 2)} (rounds "
        f"{m.get('es2_bisect_rounds')}; traces {m.get('es2_bisect_traces')}/{m.get('es2_bisect_of')}) | units "
        f"{m.get('es2_units')} = bisected + A3 own scans {m.get('es2_units_a3')} + sampled reusing the greedy cut "
        f"{m.get('es2_units_reuse')} | POINTED train units (b* >= 1): {m.get('es2_units_b_ge1')} | MC tok/s "
        f"{_n(m.get('es2_mc_tok_per_s'), 1)}",
        *[f"  train {x}" for x in _oracle_line(m, "es2_oracle")],
        f"ORACLE (test): probes per trace {_n(m.get('es6_probes_per_trace'), 2)} | traces {m.get('es6_bisect_traces')}/"
        f"{m.get('es6_bisect_of')} | MC tok/s {_n(m.get('es6_mc_tok_per_s'), 1)}",
        *[f"  test {x}" for x in _oracle_line(m, "es6_oracle")],
        "ORACLE VALIDATION (E-S3, full scans vs bisection, +-1): " + (
            m["ES3_skipped"] if "ES3_skipped" in m else
            f"agreement {_p(m.get('es3_agree'))} (exact {_p(m.get('es3_exact'))}) on {m.get('es3_traces')}/"
            f"{m.get('es3_of')} traces | MC vectors monotone {_p(m.get('es3_monotone_share'))} | full-scan points per "
            f"trace {_n(m.get('es3_full_probes_per_trace'), 1)}"),
    ]
    if "ES4_skipped" in m:
        lines.append(f"TRAIN DATA: {m['ES4_skipped']}")
    else:
        lines.append(
            f"TRAIN DATA (round 1): units {m.get('es4_units')} (pointed {m.get('es4_train_units_b_ge1')}, sloppy b*-1 copies "
            f"{m.get('es4_sloppy_units')}) | dev traces {m.get('es4_dev_traces')} (b* >= 1: {m.get('es4_dev_b_ge1')}) | "
            f"continuations {m.get('es4_continuations')} (primary {_p(m.get('es4_cont_primary'))}, truncation "
            f"{_p(m.get('es4_cont_trunc'))}) | teacher-also-wrong {_p(m.get('es4_teacher_also_wrong'))} (n "
            f"{m.get('es4_teacher_scored')}) | mean advantage {_n(m.get('es4_mean_adv'))}")
    if "ES5_skipped" in m:
        lines.append(f"TRAIN: {m['ES5_skipped']}")
    else:
        lines.append(f"TRAIN: resume check {'y' if m.get('resume_ok') else 'n' if m.get('resume_ok') is False else 'n/a'}"
                     f"{' (exact)' if m.get('resume_exact') else ''} (kill at {m.get('train_kill_at')}, ckpt every "
                     f"{m.get('train_ckpt_every')}, max rel diff {_n(m.get('resume_max_rel_diff'), 6)}) | adapter params "
                     f"{m.get('adapter_params')} | round-1 examples {m.get('train_examples_r1')}"
                     + (f" | {m['train_blocker']}" if m.get("train_blocker") else ""))
        for ad in adapters:
            lines += _train_block(m, ad, ad)
    if "ES7_skipped" in m:
        lines.append(f"ARMS: {m['ES7_skipped']}")
    else:
        lines.append(
            f"IDENTITY: untrained E3 == E1 {m.get('identity_equal')}/{m.get('identity_n')} (max token diff "
            f"{m.get('identity_max_token_diff')}) | adapters {m.get('es7_adapters')} | units {m.get('es7_units')} "
            f"(POINTED b* >= 1: {m.get('es7_pointed_units')}, restart b* = 0: {m.get('es7_restart_units')}) | E4a/E4b "
            f"clamped at 0/B {m.get('es7_clamped')} | arms done {m.get('es7_arms_done')}"
            + (f" | {m['es7_arms_cut']}" if m.get("es7_arms_cut") else ""))
        for ad in m.get("es7_adapters") or []:
            if f"es7_{ad}_all_pooled" in m:
                lines += _arms_block(m, ad, ad)
    hours, parts = e_projection(m, price)
    first = adapters[0] if adapters else "rkl"
    pooled = m.get(f"es7_{first}_pointed_pooled") or {}
    verdict = {True: "pass", False: "fail", None: "n/a"}
    gates = {
        "pipeline (identity 20/20, resume, early stop fired)": None if "ES7_skipped" in m or m.get("resume_ok") is None
        else bool(m.get("identity_n") and m.get("identity_equal") == m.get("identity_n") and m.get("resume_ok")
                  and any(m.get(f"{a}_early_stop_fired") for a in adapters)),
        "oracle (agreement >= 80%, >= 100 pointed train units)": None if m.get("es2_units_b_ge1") is None
        or _bad(m.get("es3_agree")) else bool(m["es3_agree"] >= 0.80 and m["es2_units_b_ge1"] >= 100),
        f"signal (soft: {first} delta_primary pooled >= +5 and I >= 0 on POINTED test units, n "
        f"{pooled.get('n')})": None if "delta_primary" not in pooled else bool(
            pooled["delta_primary"] >= 0.05 and pooled["interaction"] >= 0),
        f"harm ({first} E5 <= 1.5 x plain-retry)": None if _bad(m.get(f"es7_e5_harm_{first}")) or _bad(m.get("es7_plain_harm"))
        else bool(m[f"es7_e5_harm_{first}"] <= 1.5 * m["es7_plain_harm"]),
        "timing (FULL-E <= 12 h per seed)": hours <= 12,
    }
    blockers = [f"{s} {v2}" for s, v2 in skipped.items()]
    if "delta_primary" in pooled and pooled["delta_primary"] <= -0.05:
        blockers.append("delta_primary on pointed units <= -5 points: stop")
    if m.get("es7_adapter_trained") is False:
        blockers.append("no trained adapter in E-S7")
    lines += [
        f"THROUGHPUT: gen tok/s {_n(m.get('es7_gen_tok_per_s') or m.get('es4_gen_tok_per_s'), 1)} | data tok/s "
        f"{_n(m.get('es0_tok_per_s'), 1)} | MC tok/s {_n(m.get('es2_mc_tok_per_s'), 1)} | teacher prefill tok/s "
        f"{_n(m.get('es4_prefill_tok_per_s'), 1)} | train steps/s pure {_n(m.get('pure_train_steps_per_s'), 4)} "
        f"(overall {_n(m.get('train_steps_per_s'), 4)})",
        f"PROJECTION: FULL-E {hours:.1f} h and ${hours * price:.2f} per seed on this tier (rough, pure training steps/s; "
        + ", ".join(f"{k} {v2:.1f} h" for k, v2 in parts.items()) + "; cut train size, then rounds, then seeds if > 12 h)",
        "GATES: " + " ; ".join(f"{k} {verdict[v2]}" for k, v2 in gates.items()),
        f"BUDGET: max_minutes {env['max_minutes']} x ${price:.2f}/h = ${price * int(env['max_minutes']) / 60:.2f} "
        f"<= ${cfg['phase_budget_usd'][env['phase']]}",
        "BLOCKERS: " + ("; ".join(blockers) if blockers else "none"),
    ]
    m["option_e_gates"] = gates
    _write(run_dir, lines, m)


# ---- SMOKE-8 (Qwen3.5-9B self-teacher; no pointer arms) --------------------------------------------------------------

S8_ARMS = ("E0", "E2", "E6", "E6h")


def _d8(st, key):
    ci = st.get(key + "_ci") or [None, None]
    return f"{_pt(st.get(key))} [{_pt(ci[0])}, {_pt(ci[1])}]"


def s8_projection(m, price):
    """FULL per seed from the SMOKE-8 rates: 1,500 contested train problems x 4 continuations x 3 rounds (+ teacher
    scoring and training at the pure steps/s), 600 test units x 4 arms + harm, A6 on 1,500 test problems."""
    T = m.get("z2_mean_tokens") or 300.0
    gen = m.get("z5_gen_tok_per_s") or m.get("z4r1_gen_tok_per_s") or m.get("z2_tok_per_s") or 300.0
    prefill = m.get("z4r1_prefill_tok_per_s") or m.get("z3_prefill_tok_per_s") or 3000.0
    sps = m.get("pure_train_steps_per_s") or 0.3
    n_units, n_test, n_a6 = 1500, 600, 1500
    parts = {"continuations_3_rounds": 3 * n_units * 4 * T / gen,
             "teacher_scoring": 3 * n_units * 4 * 2 * (2 * T + 120) / prefill,
             "training": 3 * n_units * 4 / 16 / sps, "dev_evals": 3 * 6 * (m.get("dev_eval_sec") or 300) / max(m.get("train_rounds_done") or 1, 1),
             "arms_and_harm": (n_test * 4 + 120) * T / gen, "a6": n_a6 * T / gen,
             "sampling_4x": (n_units + n_test) * 4 * T / gen}
    parts = {k: v / 3600 for k, v in parts.items()}
    return sum(parts.values()), parts


def smoke8(run_dir, m, cfg):
    env, _, lines = header(run_dir, m, {**cfg, "student": cfg["smoke8_model"], "teacher": cfg["smoke8_model"] + " + reference"})
    price = m["price_per_hr"]
    stages = ("Z0", "Z1", "Z2", "Z3", "Z4", "Z5", "Z6", "Z7")
    skipped = {s: m.get(f"{s}_skipped") for s in stages if m.get(f"{s}_skipped")}
    hm = " ; ".join(f"{k[:-11]} {m[k]}/{m[k[:-4] + 'misses']}" for k in sorted(m) if k.endswith("_cache_hits"))
    lines += [
        "METHOD: Qwen3.5-9B student; self-teacher = same weights with a 'Reference solution' system line (dataset answer / "
        "RC evaluation steps); training units = contested train problems, restart from the prompt, 4 continuations with "
        "the adapter on over the whole sequence; rkl + PPO clip 0.2, lr 5e-5, dev 30 traces every 10 steps, <= 3 rounds; "
        "arms E0 / E2 (hint) / E6 (adapter) / E6h (adapter + hint); A6 = adapter always on, greedy",
        "TIMING (min): " + " | ".join(f"{s} {m.get(f'{s}_min')}" for s in stages) + f" | total {m['elapsed_min']:.2f}",
        "Z0 ENV: " + (m["Z0_skipped"] if "Z0_skipped" in m else
                      f"transformers {m.get('z0_transformers')} (>= 5: {'y' if m.get('z0_transformers_ge5') else 'n'}) | fla "
                      f"{m.get('z0_fla')} | causal_conv1d {m.get('z0_causal_conv1d')} | triton {m.get('z0_triton')} | peft "
                      f"{m.get('z0_peft')} | class {m.get('z0_model_class')} | params {_n(m.get('z0_params_b'), 2)} B | load "
                      f"{_n(m.get('z0_load_sec'), 0)} s | VRAM after load {_n(m.get('z0_vram_gb_after_load'), 1)} GB | fast-path "
                      f"warning {'YES' if m.get('z0_fast_path_warning') else 'none'} | module flags {m.get('z0_module_fast_path_flags')} "
                      f"| load warnings {m.get('z0_load_warnings')}"),
    ]
    if "Z0_skipped" not in m:
        lines.append(f"  TEMPLATES (20 prompts, student vs teacher variant, enable_thinking=False): common suffix >= user turn "
                     f"{m.get('z0_suffix_covers_user_turn')}/20 (min common suffix {m.get('z0_common_suffix_min')} tokens, user "
                     f"tokens mean {_n(m.get('z0_user_tokens_mean'), 1)}) | think block in render {m.get('z0_think_block')}/20 | "
                     f"hint round-trips {m.get('z0_hint_roundtrip')} | prompt tail {m.get('z0_prompt_tail')} | smoke generation "
                     f"{m.get('z0_gen_smoke_tokens')} tokens in {_n(m.get('z0_gen_smoke_sec'), 1)} s")
    lines.append("Z1 CALIBRATION (student greedy, 60 per set; keep accuracy in [35%, 70%]): " + (
        m["Z1_skipped"] if "Z1_skipped" in m else " ; ".join(
            f"{s} {_p(v['acc'])} (strict {_p(v['strict'])}, trunc {_p(v['trunc'])}, n {v['n']}, {_n(v['mean_tokens'], 0)} tok)"
            for s, v in (m.get("z1_cal") or {}).items()) + f" | chosen {m.get('z1_chosen')} | tok/s {_n(m.get('z1_tok_per_s'), 1)}"))
    if "Z2_skipped" in m:
        lines.append(f"Z2 DATA: {m['Z2_skipped']}")
    else:
        lines.append(f"Z2 DATA (4 samples per problem, T 0.8, cap 800; greedy on test): problems {m.get('z2_problems')}/{m.get('z2_of')} "
                     f"(sizes train {m.get('z2_sizes_train')}, test {m.get('z2_sizes_test')}) | tok/s {_n(m.get('z2_tok_per_s'), 1)} | "
                     f"truncation {_p(m.get('z2_trunc'))} | mean tokens {_n(m.get('z2_mean_tokens'), 1)} | test units (wrong greedy "
                     f"of contested test) {m.get('z2_test_units')} | harm traces {m.get('z2_harm_traces')} | cache hits/misses {hm or 'n/a'}")
        for k in sorted(k for k in m if (k.startswith("z2_train_") or k.startswith("z2_test_")) and isinstance(m[k], dict)):
            v = m[k]
            extra = f" | greedy acc {_p(v['greedy_acc'])} | wrong-greedy contested {v['wrong_greedy_contested']}" if "greedy_acc" in v else ""
            lines.append(f"  {k[3:]}: problems {v['problems']} | #correct of 4 {v['hist']} | accuracy {_p(v['acc'])} | contested "
                         f"{_p(v['contested'])}{extra}")
    lines.append("Z3 SELF-TEACHER: " + (m["Z3_skipped"] if "Z3_skipped" in m else
                 f"teacher greedy accuracy with the reference {_p(m.get('z3_teacher_acc'))} (n {m.get('z3_teacher_n')}; "
                 + ", ".join(f"{k[15:]} {_p(v)}" for k, v in sorted(m.items()) if k.startswith("z3_teacher_acc_"))
                 + f") | teacher tok/s {_n(m.get('z3_teacher_tok_per_s'), 1)} | mean per-token log-prob gap teacher - student: "
                 f"correct traces {_n(m.get('z3_gap_correct'), 3)} (n {m.get('z3_gap_n_correct')}), wrong traces "
                 f"{_n(m.get('z3_gap_wrong'), 3)} (n {m.get('z3_gap_n_wrong')}) | student mean log-prob correct/wrong "
                 f"{_n(m.get('z3_student_logp_correct'), 3)}/{_n(m.get('z3_student_logp_wrong'), 3)} | prefill tok/s "
                 f"{_n(m.get('z3_prefill_tok_per_s'), 1)}"))
    if "Z4_skipped" in m:
        lines.append(f"Z4 TRAIN: {m['Z4_skipped']}")
    else:
        lines.append(f"Z4 TRAIN: units {m.get('z4_units')} {m.get('z4_units_by_set')} | dev {m.get('z4_dev')} {m.get('z4_dev_by_set')} | "
                     f"rounds {m.get('train_rounds_done')} | executed steps {m.get('executed_steps')} | best round {m.get('train_best_round')} "
                     f"step {m.get('train_best_step')} dev E6 recovery {_p(m.get('train_best_dev'))} | early stopping fired "
                     f"{'y' if m.get('early_stop_fired') else 'n'} | resume {'y' if m.get('resume_ok') else 'n' if m.get('resume_ok') is False else 'n/a'}"
                     f"{' (exact)' if m.get('resume_exact') else ''} (kill at {m.get('train_kill_at')}, ckpt every {m.get('train_ckpt_every')}, "
                     f"max rel diff {_n(m.get('resume_max_rel_diff'), 6)}) | train {_n(m.get('train_sec'), 0)} s (dev evals "
                     f"{_n(m.get('dev_eval_sec'), 0)} s, resampling + scoring {_n(m.get('resample_sec'), 0)} s) | steps/s pure "
                     f"{_n(m.get('pure_train_steps_per_s'), 4)} (overall {_n(m.get('train_steps_per_s'), 4)}) | adapter params "
                     f"{m.get('adapter_params')} | micro {m.get('z4_micro')}"
                     + (f" | {m['rounds_stopped']}" if m.get("rounds_stopped") else "")
                     + (f" | {m['train_blocker']}" if m.get("train_blocker") else ""))
        for r in m.get("train_rounds") or []:
            tag = f"z4r{r['round']}"
            curve = " ; ".join(f"{s}: {_p(v)}" for s, v in r["curve"])
            lines.append(f"  round {r['round']}: continuations {m.get(f'{tag}_continuations')} (primary {_p(m.get(f'{tag}_cont_primary'))}, "
                         f"trunc {_p(m.get(f'{tag}_cont_trunc'))}, teacher-also-wrong {_p(m.get(f'{tag}_teacher_also_wrong'))}, mean adv "
                         f"{_n(m.get(f'{tag}_mean_adv'))}) | steps {r['steps']} | loss start/end {_n(r['loss_start'], 4)}/{_n(r['loss_end'], 4)} | "
                         f"dev per eval (step: rate) {curve or 'n/a'} | best step {r['best_step']} | stopped early {'y' if r['stopped'] else 'n'}"
                         f"{' | CUT by deadline' if r.get('cut') else ''}")
        for r in range(2, 4):
            if m.get(f"round{r}_skipped"):
                lines.append(f"  round {r}: {m[f'round{r}_skipped']}")
    if "Z5_skipped" in m:
        lines.append(f"Z5 ARMS: {m['Z5_skipped']}")
    else:
        lines.append(f"Z5 ARMS (test units = wrong greedy traces of contested test problems; one sampled continuation per arm, "
                     f"identical batches and seeds; adapter {'trained' if m.get('z5_adapter_trained') else 'UNTRAINED'}; arms done "
                     f"{m.get('z5_arms_done')}{' | ' + m['z5_arms_cut'] if m.get('z5_arms_cut') else ''}):")
        for s in sorted(k[3:] for k in m if k.startswith("z5_") and isinstance(m[k], dict) and "rec_E0" in m[k]):
            st = m[f"z5_{s}"]
            lines.append(f"  {s} (n {st['n']}, problems {st['problems']}): " + " ; ".join(
                f"{a} {_p(st.get(f'rec_{a}'))} ({_n(st.get(f'tokens_{a}'), 0)} tok, trunc {_p(st.get(f'trunc_{a}'))})" for a in S8_ARMS))
        lines.append(f"  HARM (correct greedy test traces resampled from the prompt, n {m.get('z5_harm_n')}): adapter "
                     f"{_p(m.get('z5_harm_adapter'))} vs plain {_p(m.get('z5_harm_plain'))}")
        lines.append("  DELTAS (points; 95% CI by cluster bootstrap on original_id, 10,000 resamples):")
        for s in sorted(k[3:] for k in m if k.startswith("z5_") and isinstance(m[k], dict) and "e6_minus_e0" in m[k]):
            st = m[f"z5_{s}"]
            lines.append(f"    {s} (n {st['n']}): E6-E0 {_d8(st, 'e6_minus_e0')} | E2-E0 {_d8(st, 'e2_minus_e0')} | E6h-E0 "
                         f"{_d8(st, 'e6h_minus_e0')} | E6h-E6 {_d8(st, 'e6h_minus_e6')} | E6-E2 {_d8(st, 'e6_minus_e2')}")
    lines.append("Z6 PASS@1 (greedy on all test problems, base vs adapter always on): " + (
        m["Z6_skipped"] if "Z6_skipped" in m else " ; ".join(
            f"{k[3:]} base {_p(v['base'])} -> A6 {_p(v['a6'])} ({_pt(v['delta'])} [{_pt(v['delta_ci'][0])}, {_pt(v['delta_ci'][1])}], n {v['n']})"
            for k, v in sorted(m.items()) if k.startswith("z6_") and isinstance(v, dict))
        + f" | A6 tok/s {_n(m.get('z6_tok_per_s'), 1)} | A6 truncation {_p(m.get('z6_a6_trunc'))}"
        + (f" | {m['z6_cut']}" if m.get("z6_cut") else "")))
    hours, parts = s8_projection(m, price)
    pooled = m.get("z5_pooled") or {}
    verdict = {True: "pass", False: "fail", None: "n/a"}
    gates = {
        "env (transformers >= 5, fast path, template suffix 20/20)": None if "Z0_skipped" in m else bool(
            m.get("z0_transformers_ge5") and not m.get("z0_fast_path_warning") and m.get("z0_suffix_covers_user_turn") == 20),
        "calibration (>= 1 set in band)": None if "Z1_skipped" in m else bool(m.get("z1_chosen")),
        "pipeline (resume, early stop fired)": None if "Z4_skipped" in m or m.get("resume_ok") is None else bool(
            m.get("resume_ok") and m.get("early_stop_fired")),
        "teacher (accuracy with reference >= 90%)": None if _bad(m.get("z3_teacher_acc")) else bool(m["z3_teacher_acc"] >= 0.9),
        "signal (E6-E0 pooled CI lower bound > 0)": None if "e6_minus_e0" not in pooled else bool(pooled["e6_minus_e0_ci"][0] > 0),
        "harm (adapter <= 1.5 x plain)": None if _bad(m.get("z5_harm_adapter")) or _bad(m.get("z5_harm_plain")) else bool(
            m["z5_harm_adapter"] <= 1.5 * m["z5_harm_plain"]),
        "pass@1 (A6 >= base pooled)": None if "z6_pooled" not in m or m.get("z6_cut") or not m.get("z6_adapter_trained")
        else bool(m["z6_pooled"]["delta"] >= 0),
        "timing (FULL <= 12 h per seed)": hours <= 12,
    }
    lines += [
        f"THROUGHPUT: sampling tok/s {_n(m.get('z2_tok_per_s'), 1)} | adapter gen tok/s {_n(m.get('z5_gen_tok_per_s'), 1)} | teacher "
        f"prefill tok/s {_n(m.get('z4r1_prefill_tok_per_s') or m.get('z3_prefill_tok_per_s'), 1)} | train steps/s pure "
        f"{_n(m.get('pure_train_steps_per_s'), 4)}",
        f"PROJECTION: FULL {hours:.1f} h and ${hours * price:.2f} per seed on this tier (rough; "
        + ", ".join(f"{k} {v:.1f} h" for k, v in parts.items()) + ")",
        "GATES: " + " ; ".join(f"{k} {verdict[v]}" for k, v in gates.items()),
        f"BUDGET: max_minutes {env['max_minutes']} x ${price:.2f}/h = ${price * int(env['max_minutes']) / 60:.2f} "
        f"<= ${cfg['phase_budget_usd'][env['phase']]}",
        "BLOCKERS: " + ("; ".join(f"{s} {v}" for s, v in skipped.items()) if skipped else "none"),
    ]
    m["smoke8_gates"] = gates
    _write(run_dir, lines, m)


# ---- SMOKE-9 (cap 3200 re-evaluation + GRPO; SMOKE-8 numbers at cap 800 side by side) ------------------------------

def _ci9(v, key):
    ci = v.get(key + "_ci") or [None, None]
    return f"{_pt(v.get(key))} [{_pt(ci[0])}, {_pt(ci[1])}]"


def s9_projection(m, price, m8):
    """FULL per seed at cap 3200 from the measured rates: 1,500 contested train problems x 4 continuations x 3 rounds
    (GRPO: no teacher), training at the pure steps/s, dev evals as measured, 600 test units x 2 arms + harm, A6 and base
    greedy on 1,500 test problems, 4x sampling of 2,100 problems."""
    T = m.get("y3r2_mean_tokens") or m.get("y2_mean_tokens") or (m8.get("z2_mean_tokens") or 600.0) * 1.5
    gen = m.get("y3r2_gen_tok_per_s") or m.get("rkl_a6_tok_per_s") or m.get("y2_tok_per_s") or 400.0
    sps = m.get("pure_train_steps_per_s") or 0.1
    n_units, n_test, n_a6 = 1500, 600, 1500
    rounds = max(m.get("train_rounds_done") or 1, 1)
    parts = {"sampling_4x": (n_units + n_test) * 4 * T / gen, "continuations_3_rounds": 3 * n_units * 4 * T / gen,
             "logprobs": 3 * n_units * 4 * (T + 120) / (m.get("y3r2_lp_tok_per_s") or 4000.0),
             "training": 3 * n_units * 4 * 0.6 / 16 / sps, "dev_evals": 3 * 6 * (m.get("dev_eval_sec") or 600) / rounds / 5,
             "arms_and_harm": (n_test * 2 + 240) * T / gen, "a6_and_base": 2 * n_a6 * T / gen}
    parts = {k: v / 3600 for k, v in parts.items()}
    return sum(parts.values()), parts


def _eval_block(m, tag, label, sets):
    lines = [f"{label} AT CAP 3200:"]
    for s in sets + ["pooled"]:
        v = m.get(f"{tag}_pass1_{s}")
        if v:
            lines.append(f"  pass@1 {s} (n {v['n']}): base {_p(v['base'])} [{_p(v['base_ci'][0])}, {_p(v['base_ci'][1])}] -> A6 "
                         f"{_p(v['a6'])} [{_p(v['a6_ci'][0])}, {_p(v['a6_ci'][1])}] | delta {_ci9(v, 'delta')} | tokens base/A6 "
                         f"{_n(v['base_tokens'], 0)}/{_n(v['a6_tokens'], 0)} | truncation base/A6 {_p(v['base_trunc'])}/{_p(v['a6_trunc'])}")
    for s in sets + ["pooled"]:
        v = m.get(f"{tag}_arms_{s}")
        if v:
            lines.append(f"  E0 vs E6 {s} (n {v['n']}): E0 {_p(v['E0'])} ({_n(v['tok_E0'], 0)} tok, trunc {_p(v['trunc_E0'])}) ; E6 {_p(v['E6'])} "
                         f"({_n(v['tok_E6'], 0)} tok, trunc {_p(v['trunc_E6'])}) | E6-E0 {_ci9(v, 'delta')}")
    if m.get(f"{tag}_arms_cut"):
        lines.append(f"  arms: {m[f'{tag}_arms_cut']}")
    hs = " ; ".join(f"{s} {_p(m.get(f'{tag}_harm_adapter_{s}'))} vs {_p(m.get(f'{tag}_harm_plain_{s}'))}" for s in sets)
    lines.append(f"  HARM (n {m.get(f'{tag}_harm_n')}): adapter {_p(m.get(f'{tag}_harm_adapter'))} vs plain {_p(m.get(f'{tag}_harm_plain'))}"
                 + (f" | per set {hs}" if hs else ""))
    return lines


def smoke9(run_dir, m, cfg, m8):
    env, _, lines = header(run_dir, m, {**cfg, "student": cfg["smoke8_model"], "teacher": "none (GRPO) / SMOKE-8 adapter re-evaluated"})
    price = m["price_per_hr"]
    stages = ("Y0", "Y1", "Y2", "Y3", "Y4", "Y5")
    skipped = {s: m.get(f"{s}_skipped") for s in stages if m.get(f"{s}_skipped") and not str(m.get(f"{s}_skipped")).startswith("not needed")}
    sets = list(m.get("y0_sets_from_smoke8") or ["rc3"]) + (["rc4"] if m.get("y1_rc4_added") else [])
    hm = " ; ".join(f"{k[:-11]} {m[k]}/{m[k[:-4] + 'misses']}" for k in sorted(m) if k.endswith("_cache_hits"))
    z6, z5 = m8.get("z6_pooled") or {}, m8.get("z5_pooled") or {}
    lines += [
        f"METHOD: cap 3200 everywhere; SMOKE-8 run {m.get('y0_smoke8_run')} re-evaluated (rkl adapter, no training) and a GRPO adapter "
        f"(reward = primary correctness, groups of 4, DAPO dynamic sampling, no KL, clip 0.2, lr 5e-5) on the same units; no hint arms",
        "TIMING (min): " + " | ".join(f"{s} {m.get(f'{s}_min')}" for s in stages) + f" | total {m['elapsed_min']:.2f}",
        "Y0 ENV: " + (m["Y0_skipped"] if "Y0_skipped" in m else
                      f"transformers {m.get('y0_transformers')} | class {m.get('y0_model_class')} | load {_n(m.get('y0_load_sec'), 0)} s | "
                      f"VRAM {_n(m.get('y0_vram_gb_after_load'), 1)} GB | fast-path warning {'YES' if m.get('y0_fast_path_warning') else 'none'} | "
                      f"log-prob batch {m.get('y0_lp_bs')} | micro {m.get('y0_micro')} | cache hits/misses {hm or 'n/a'}"),
        f"SMOKE-8 AT CAP 800 (for comparison): pass@1 pooled base {_p(z6.get('base'))} -> A6 {_p(z6.get('a6'))} ({_ci9(z6, 'delta')}, n {z6.get('n')}) | "
        f"E0 {_p(z5.get('rec_E0'))} ({_n(z5.get('tokens_E0'), 0)} tok, trunc {_p(z5.get('trunc_E0'))}) vs E6 {_p(z5.get('rec_E6'))} "
        f"({_n(z5.get('tokens_E6'), 0)} tok, trunc {_p(z5.get('trunc_E6'))}), E6-E0 {_ci9(z5, 'e6_minus_e0')} (n {z5.get('n')}) | harm adapter "
        f"{_p(m8.get('z5_harm_adapter'))} vs plain {_p(m8.get('z5_harm_plain'))} | sampled-trace truncation {_p(m8.get('z2_trunc'))}, mean tokens "
        f"{_n(m8.get('z2_mean_tokens'), 0)}",
    ]
    if "Y1_skipped" in m:
        lines.append(f"Y1: {m['Y1_skipped']}")
    else:
        lines += _eval_block(m, "rkl", "Y1 SMOKE-8 rkl ADAPTER", sets)
        lines.append(f"  RC4 CALIBRATION at 3200 (student greedy, n {m.get('y1_rc4_cal_n')}): accuracy {_p(m.get('y1_rc4_cal_acc'))} | truncation "
                     f"{_p(m.get('y1_rc4_cal_trunc'))} | tokens {_n(m.get('y1_rc4_cal_tokens'), 0)} | in [35%, 70%]: "
                     f"{'yes, RC4 added' if m.get('y1_rc4_added') else 'no'} (at 800: {_p(((m8.get('z1_cal') or {}).get('rc4') or {}).get('acc'))})")
    if "Y2_skipped" in m:
        lines.append(f"Y2 RC4 DATA: {m['Y2_skipped']}")
    else:
        lines.append(f"Y2 RC4 DATA (cap 3200, 4 samples; greedy on test): problems {m.get('y2_problems')}/{m.get('y2_of')} | tok/s "
                     f"{_n(m.get('y2_tok_per_s'), 1)} | mean tokens {_n(m.get('y2_mean_tokens'), 0)} | test units {m.get('y2_test_units')} | harm "
                     f"{m.get('y2_harm')}" + (f" | {m['y2_cut']}" if m.get("y2_cut") else ""))
        for sp in ("train", "test"):
            v = m.get(f"y2_{sp}_rc4")
            if v:
                extra = f" | greedy acc {_p(v['greedy_acc'])} | wrong-greedy contested {v['wrong_greedy_contested']}" if "greedy_acc" in v else ""
                lines.append(f"  {sp}: problems {v['problems']} | #correct of 4 {v['hist']} | accuracy {_p(v['acc'])} | contested {_p(v['contested'])} | "
                             f"truncation {_p(v['trunc'])}{extra}")
    if "Y3_skipped" in m:
        lines.append(f"Y3 GRPO: {m['Y3_skipped']}")
    else:
        lines.append(f"Y3 GRPO: units {m.get('y3_units')} {m.get('y3_units_by_set')} | dev {m.get('y3_dev')} {m.get('y3_dev_by_set')} | rounds "
                     f"{m.get('train_rounds_done')} | executed steps {m.get('executed_steps')} | best round {m.get('train_best_round')} step "
                     f"{m.get('train_best_step')} dev E6 recovery {_p(m.get('train_best_dev'))} | early stopping fired {'y' if m.get('early_stop_fired') else 'n'} | "
                     f"resume {'y' if m.get('resume_ok') else 'n' if m.get('resume_ok') is False else 'n/a'}{' (exact)' if m.get('resume_exact') else ''} "
                     f"(max rel diff {_n(m.get('resume_max_rel_diff'), 6)}) | train {_n(m.get('train_sec'), 0)} s (dev evals {_n(m.get('dev_eval_sec'), 0)} s, "
                     f"sampling + log-probs {_n(m.get('resample_sec'), 0)} s) | steps/s pure {_n(m.get('pure_train_steps_per_s'), 4)} | adapter params "
                     f"{m.get('adapter_params')}" + (f" | {m['rounds_stopped']}" if m.get("rounds_stopped") else "")
                     + (f" | {m['train_blocker']}" if m.get("train_blocker") else ""))
        for r in m.get("train_rounds") or []:
            tag = f"y3r{r['round']}"
            g = m.get(f"{tag}_groups") or {}
            src = (f"reused SMOKE-8 continuations {m.get('y3r1_reused_smoke8')} (dropped {m.get('y3r1_dropped_truncated')} truncated at 800)" if r["round"] == 1 else
                   f"sampled {m.get(f'{tag}_continuations')} (primary {_p(m.get(f'{tag}_cont_primary'))}, trunc {_p(m.get(f'{tag}_cont_trunc'))}, "
                   f"mean tokens {_n(m.get(f'{tag}_mean_tokens'), 0)})")
            curve = " ; ".join(f"{s}: {_p(v)}" for s, v in r["curve"])
            lines.append(f"  round {r['round']}: {src} | groups {g.get('groups')} kept {g.get('kept')} (dropped all-correct {g.get('dropped_all_correct')}, "
                         f"all-wrong {g.get('dropped_all_wrong')}, incomplete {g.get('incomplete')}) -> examples {r['examples']} | steps {r['steps']} | loss start/end "
                         f"{_n(r['loss_start'], 4)}/{_n(r['loss_end'], 4)} | dev per eval (step: rate) {curve or 'n/a'} | best step {r['best_step']} | "
                         f"stopped early {'y' if r['stopped'] else 'n'}{' | CUT by deadline' if r.get('cut') else ''}")
        for r in range(2, 4):
            if m.get(f"round{r}_skipped"):
                lines.append(f"  round {r}: {m[f'round{r}_skipped']}")
    if "Y4_skipped" in m:
        lines.append(f"Y4 GRPO ADAPTER: {m['Y4_skipped']}")
    else:
        lines += _eval_block(m, "grpo", "Y4 GRPO ADAPTER", sets)
    # summary per set at 3200 and the brevity line
    lines.append("SUMMARY AT CAP 3200 (pass@1 base / A6 rkl / A6 grpo ; E6-E0 rkl / grpo ; harm rkl / grpo):")
    for s in sets + ["pooled"]:
        r, g = m.get(f"rkl_pass1_{s}") or {}, m.get(f"grpo_pass1_{s}") or {}
        ra, ga = m.get(f"rkl_arms_{s}") or {}, m.get(f"grpo_arms_{s}") or {}
        lines.append(f"  {s}: pass@1 base {_p(r.get('base'))} | A6 rkl {_p(r.get('a6'))} ({_ci9(r, 'delta')}) | A6 grpo {_p(g.get('a6'))} "
                     f"({_ci9(g, 'delta')}) ; E6-E0 rkl {_ci9(ra, 'delta')} | grpo {_ci9(ga, 'delta')} ; harm rkl "
                     f"{_p(m.get(f'rkl_harm_adapter_{s}') if s != 'pooled' else m.get('rkl_harm_adapter'))} / grpo "
                     f"{_p(m.get(f'grpo_harm_adapter_{s}') if s != 'pooled' else m.get('grpo_harm_adapter'))} (plain "
                     f"{_p(m.get(f'rkl_harm_plain_{s}') if s != 'pooled' else m.get('rkl_harm_plain'))})")
    b_e6 = brev(z5.get("e6_minus_e0"), (m.get("rkl_arms_pooled") or {}).get("delta"))
    b_a6 = brev(z6.get("delta"), (m.get("rkl_pass1_pooled") or {}).get("delta"))
    lines.append(f"BREVITY: of SMOKE-8's gain at cap 800, the part that disappears at 3200 with the same adapter = E6-E0 {_pt(z5.get('e6_minus_e0'))} - "
                 f"{_pt((m.get('rkl_arms_pooled') or {}).get('delta'))} = {_pt(b_e6)} points ; A6 pass@1 {_pt(z6.get('delta'))} - "
                 f"{_pt((m.get('rkl_pass1_pooled') or {}).get('delta'))} = {_pt(b_a6)} points")
    hours, parts = s9_projection(m, price, m8)
    rk, gp = m.get("rkl_pass1_pooled") or {}, m.get("grpo_pass1_pooled") or {}
    verdict = {True: "pass", False: "fail", None: "n/a"}
    gates = {
        "env (fast path, model loaded)": None if "Y0_skipped" in m else bool(not m.get("y0_fast_path_warning")),
        "re-evaluation complete (rkl pass@1, arms, harm)": None if "Y1_skipped" in m else bool(
            m.get("rkl_pass1_n") == m.get("rkl_pass1_of") and "rkl_arms_pooled" in m and not _bad(m.get("rkl_harm_plain"))),
        "pipeline (resume, early stop fired)": None if "Y3_skipped" in m or m.get("resume_ok") is None else bool(m.get("resume_ok") and m.get("early_stop_fired")),
        "rkl A6 > base at 3200 (CI lower bound > 0)": None if "delta" not in rk else bool(rk["delta_ci"][0] > 0),
        "grpo A6 > base at 3200 (CI lower bound > 0)": None if "delta" not in gp else bool(gp["delta_ci"][0] > 0),
        "harm (each adapter <= 1.5 x plain)": None if _bad(m.get("rkl_harm_adapter")) or _bad(m.get("rkl_harm_plain")) else bool(
            m["rkl_harm_adapter"] <= 1.5 * m["rkl_harm_plain"] and (_bad(m.get("grpo_harm_adapter")) or m["grpo_harm_adapter"] <= 1.5 * m["rkl_harm_plain"])),
        "timing (FULL <= 12 h per seed at 3200)": hours <= 12,
    }
    lines += [
        f"THROUGHPUT: base greedy tok/s {_n(m.get('rkl_base_tok_per_s'), 1)} | A6 tok/s {_n(m.get('rkl_a6_tok_per_s'), 1)} | GRPO sampling tok/s "
        f"{_n(m.get('y3r2_gen_tok_per_s'), 1)} | train steps/s pure {_n(m.get('pure_train_steps_per_s'), 4)}",
        f"PROJECTION: FULL at cap 3200 {hours:.1f} h and ${hours * price:.2f} per seed on this tier (rough; "
        + ", ".join(f"{k} {v:.1f} h" for k, v in parts.items()) + ")",
        "GATES: " + " ; ".join(f"{k} {verdict[v]}" for k, v in gates.items()),
        f"BUDGET: max_minutes {env['max_minutes']} x ${price:.2f}/h = ${price * int(env['max_minutes']) / 60:.2f} <= ${cfg['phase_budget_usd'][env['phase']]}",
        "BLOCKERS: " + ("; ".join(f"{s} {v}" for s, v in skipped.items()) if skipped else "none"),
    ]
    m["smoke9_gates"] = gates
    m["smoke9_brevity"] = {"e6_minus_e0": b_e6, "a6_pass1": b_a6}
    _write(run_dir, lines, m)


def brev(d800, d3200):
    return None if d800 is None or d3200 is None else d800 - d3200
