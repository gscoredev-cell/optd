"""report.md + metrics.json (§12). Shared header for every phase; ENVCHECK and SMOKE bodies."""
import json
import math
import os
import time


def _p(x):
    """Rate as percent with one decimal."""
    return "nan" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.1f}%"


def _n(x, nd=3):
    return "nan" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


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
    """Rough linear FULL projection per seed from SMOKE unit times (§10 FULL sizes; 3 rounds; 2 interventions)."""
    per_gen = m["s2_sec"] / max(m["s2_n"], 1)
    per_probe = m["S3_min"] * 60 / max(m["s2_n"], 1)
    final = m["s1_max_new_final"]
    per_s1 = (m[f"s1_student_sec_{final}"] + m[f"s1_teacher_sec_{final}"]) / max(sum(m["s1_n"].values()), 1)
    per_cont = (m["S5_min"] + m["S6_min"]) * 60 / max(m["s5_continuations"], 1)
    per_step = 1 / m["train_steps_per_s"] if m.get("train_steps_per_s") else 0.0
    per_eval = m["s8_sec"] / max(m["s8_traces"], 1)
    flag_wrong = m["s5_flagged_wrong"] / 300
    interv = m["s8_flagged"] / max(sum(m["s1_n"].values()), 1)
    train_prompts, test = 7000, 1800
    conts = flag_wrong * train_prompts * 4
    sec = (per_gen * 2 * train_prompts + per_probe * 2 * train_prompts + 3 * (per_cont * conts + per_step * conts / 16)
           + per_s1 * test + per_eval * test * (5 + 1 + 4 * 2 * interv))
    return sec / 3600


def smoke(run_dir, m, cfg):
    env, _, lines = header(run_dir, m, cfg)
    price = m["price_per_hr"]
    sets = ("p2", "main", "p1", "rc")
    n = m["s1_n"]

    def acc(who):
        return " / ".join(f"{s} (n={n[s]}) {_p(m[f's1_{who}_{s}_strict'])}/{_p(m[f's1_{who}_{s}_lenient'])}"
                          for s in sets)

    trunc = " ; ".join(f"{k} tokens: " + ", ".join(f"{w} {_p(v)}" for w, v in m[f"s1_trunc_{k}"].items())
                       for k in m["s1_max_new_tried"])
    restart = m.get("probe_restart_share")
    hours = _projection(m)
    blockers = [m[k] for k in ("probe_blocker", "train_blocker") if m.get(k)]
    if not m.get("identity_equal"):
        blockers.append(f"identity failed (max token diff {m.get('identity_max_token_diff')})")
    if m.get("resume_ok") is False:
        blockers.append(f"resume mismatch (max rel diff {_n(m.get('resume_max_rel_diff'), 5)})")
    if m.get("s3_state_tol", 0) > 1e-2:
        blockers.append(f"prefill-vs-generation state tolerance {_n(m['s3_state_tol'], 4)} > 1e-2")
    gates = {
        "headroom (gap >= 8 pts)": m["s1_gap_p2"] >= 0.08,
        "base rate in [20%, 85%]": 0.20 <= m["s2_greedy_strict"] <= 0.85,
        "probe AUROC >= 0.80 and recall >= 0.40": (m.get("probe_auroc") or 0) >= 0.80 and (m.get("probe_recall") or 0) >= 0.40,
        "identity": bool(m.get("identity_equal")),
        "resume": m.get("resume_ok") is True,
        "timing (<= 900 min/pod)": hours * 60 <= 900,
    }
    eval_lines = []
    for arm in ("A0", "A2", "A3", "A6"):
        e = m[f"eval_{arm}"]
        per_set = ", ".join(f"{s} {_p(m[f'eval_{arm}_{s}_strict'])}" for s in sets)
        eval_lines.append(f"  {arm}: strict {per_set} ; recovered {e['recovered']} ; broken {e['broken']} ; "
                          f"net {e['net']} ; interv. rate {_p(e['interv_rate'])} ; FPR {_p(e['fpr'])} ; "
                          f"extra tokens {_n(e['extra_tokens'], 1)} ; truncation {_p(e['trunc_rate'])}")
    lines += [
        f"TIMING (min): S1 {m['S1_min']} | S2 {m['S2_min']} | S3 {m['S3_min']} | S4 {m['S4_min']} | S5 {m['S5_min']} | "
        f"S6 {m['S6_min']} | S7 {m['S7_min']} | S8 {m['S8_min']} | total {m['elapsed_min']:.2f}",
        f"HEADROOM strict/lenient: student {acc('student')} ; teacher {acc('teacher')} ; "
        f"gap p2 {100 * m['s1_gap_p2']:.1f} ± {100 * m['s1_gap_p2_se']:.1f} pts (paired SE) ; "
        f"max_new_tokens tried {m['s1_max_new_tried']} -> {m['s1_max_new_final']} ; truncation {trunc}",
        f"ROLLOUTS: n {m['s2_n']} | strict acc greedy {_p(m['s2_greedy_strict'])} / sampled {_p(m['s2_sampled_strict'])} | "
        f"mean tokens {_n(m['s2_mean_tokens'], 1)} | mean boundaries {_n(m['s2_mean_boundaries'], 2)} | "
        f"truncation {_p(m['s2_trunc'])}",
        f"PROBE: layer {m.get('probe_layer')} | AUROC {_n(m.get('probe_auroc'))} (by layer "
        f"{ {k: round(v, 3) for k, v in m.get('probe_auroc_by_layer', {}).items()} }) | tau {_n(m.get('probe_tau'), 4)} | "
        f"FPR {_p(m.get('probe_fpr'))} | recall {_p(m.get('probe_recall'))} | restart share {_p(restart)}"
        f"{' (> 60%: the detector is acting as a difficulty estimator)' if restart and restart > 0.6 else ''} | "
        f"first-flag histogram {m.get('probe_first_flag_hist')} | held-out traces {m.get('probe_holdout_traces')}",
        f"IDENTITY: A3-untrained == A2 {'y' if m.get('identity_equal') else 'n'} (n {m.get('identity_n')}, max token diff "
        f"{m.get('identity_max_token_diff')}) ; prefill-vs-generation state tolerance {_n(m.get('s3_state_tol'), 5)} "
        f"(max rel L2, limit 1e-2)",
        f"REPAIR DATA: flagged-wrong {m['s5_flagged_wrong']} (used {m['s5_problems']}) | continuations "
        f"{m['s5_continuations']} (strict {_p(m['s5_strict'])}, truncation {_p(m['s5_trunc'])}) | teacher-also-wrong "
        f"{_p(m['s6_teacher_also_wrong'])}",
        f"TRAIN: steps {m['train_steps']} | loss start/end {_n(m.get('loss_start'), 4)}/{_n(m.get('loss_end'), 4)} | "
        f"resume check {'y' if m.get('resume_ok') else 'n' if m.get('resume_ok') is False else 'n/a'} "
        f"(kill at {m.get('train_kill_at')}, checkpoint every {m.get('train_ckpt_every')}, max rel diff "
        f"{_n(m.get('resume_max_rel_diff'), 6)}) | adapter params {m['adapter_params']}",
        f"EVAL ({sum(n.values())}):",
        *eval_lines,
        f"THROUGHPUT: gen tok/s {_n(m['s2_n'] * m['s2_mean_tokens'] / m['s2_sec'], 1)} (S2, batch incl. sampling) | "
        f"teacher prefill tok/s {_n(m.get('s6_prefill_tok_per_s'), 1)} | train steps/s {_n(m.get('train_steps_per_s'), 4)}",
        f"PROJECTION: FULL {hours:.1f} h and ${hours * price:.2f} per seed on this tier (rough linear estimate; "
        f"ceiling 15 h/pod)",
        "GATES (informational in SMOKE): " + " ; ".join(f"{k} {'pass' if v else 'fail'}" for k, v in gates.items()),
        f"BUDGET: max_minutes {env['max_minutes']} x ${price:.2f}/h = ${price * int(env['max_minutes']) / 60:.2f} "
        f"<= ${cfg['phase_budget_usd'][env['phase']]}",
        "BLOCKERS: " + ("; ".join(blockers) if blockers else "none"),
    ]
    _write(run_dir, lines, m)
