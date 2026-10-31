# Option D: results (2026-10-30 to 2026-10-31)

Hypothesis (OPTION_D_SPEC.md §0): a small model can learn to recover from its own reasoning slips if an external
hidden-state detector decides *when* to repair and a trained adapter learns *how*. Before any repair training, the
detector has to flag the instance-level slip: the wrong trace among several traces of the same problem. It does not.
No repair adapter was trained and FULL was never started.

## Pipeline

- **Orchestration:** local Claude Code drives RunPod Secure Cloud.
  - `orchestrator/launch.ps1` creates the pods, `watch.ps1` polls `status.json` on the network volume's S3 API, and `pull.ps1` copies results back.
  - Four cost stops:
    - an in-pod deadline watchdog;
    - pod self-terminate at DONE/FAILED;
    - the `watch.ps1` stall rule;
    - the local scheduled task `OptD-RunPod-Watchdog`.
- **Pod:** `pod/bootstrap.sh` is fetched by commit SHA. It restores a cached venv, models and data from the volume, then runs the staged pipeline `pipeline/run_phase.py`. Each stage saves its outputs to the volume, so a restarted pod resumes.
- **Data:**
  - GSM-Symbolic (main/p1/p2), split by `original_id`;
  - RandomCalculation with 3–4 operations.
  - Answers are scored by a primary rule: last-number extraction plus a truncation check.
- **Detector:**
  - Hidden states at reasoning-step boundaries are extracted with a bf16 or fp32 prefill. The layers used are half, three quarters and all of the model's depth.
  - These go into a standardized L2 logistic regression. The layer and C are chosen by 5-fold CV grouped by `original_id`, and the threshold tau is set at 10% FPR on held-out correct traces.
  - **Across-problem probe:** boundary label = trace wrong.
  - **Within-problem probe:** contested problems only (1–3 of 4 sampled traces correct), scored by within-pair AUROC (wrong trace above its correct twin).
  - **Recoverability:** Monte Carlo (MC) success from each boundary, 3 completions at T 0.8; a boundary with MC < 0.5 is unrecoverable.

## Runs

| Run | Commit | Student | What it tested | GPU | Min | Cost |
|---|---|---|---|---|---|---|
| SMOKE | 6462834 | Qwen3-1.7B | Pipeline check; across-problem probe on greedy traces | L40 | 55.1 | $0.75 |
| SMOKE-2 | ccc68bc | Qwen3-1.7B (teacher Qwen3-8B) | Primary scoring; CV-tuned across-problem probe; RC calibration | L40 | 65.2 | $0.89 |
| SMOKE-3 | 0250c6d | Qwen3-1.7B probe-mc; Qwen3-4B student | Step labels from MC; 4B across-problem probe | L40 | 71.6 | $0.98 |
| SMOKE-4 | da91eb5 | Qwen3-1.7B | Within-problem probe; step localization (MC) | L40 | 50.4 | $0.69 |
| SMOKE-5 | 977e096 | Qwen3-8B (no teacher) | Within-problem probe (target a); recoverability probe (target b) | L40 | 88.4 | $1.21 |

The five runs total 331 min and $4.52. The ENVCHECK and WATCHDOGTEST runs come on top; the two ENVCHECK runs that wrote a report cost $0.05 and $0.01.

Every gate failed:

- **SMOKE and SMOKE-2:** across-problem AUROC 0.746 and 0.747, under the 0.80 gate.
- **SMOKE-3:** the 4B reached 0.810, but recall was 31% (gate ≥ 40%), and 27 of its 29 held-out wrong traces were RandomCalculation.
- **SMOKE-4:** 1.7B within-problem AUROC 0.599.
- **SMOKE-5:** 8B within-problem AUROC 0.378.

## Within-problem vs across-problem AUROC

Positive control, computed locally from saved features (`P.fit`: trace label on every boundary, score = max over boundaries, layer and C by grouped CV). The same probe is scored two ways on the same held-out traces:

- **across-problem:** trace-level AUROC;
- **within-pair:** the wrong trace against its correct twin.

The two SMOKE-4/5 rows train on the D2/A2 pair traces and hold out every trace of the contested test problems. The train/test split is disjoint by `original_id`.

| Model | Features (run) | Traces in scope | Held-out traces (wrong) | Across-problem AUROC | Within-pair AUROC [95% CI] (pairs) |
|---|---|---|---|---|---|
| 1.7B | SMOKE-2 S2, layers 21/28 (recomputed; reproduces the reported probe: layer 21, C 0.01) | all main + p1 problems, greedy + 1 sampled | 108 (47) | **0.747** | 0.455 [0.18, 0.73] (11) |
| 4B | SMOKE-3 S3 probe (reported, not recomputed) | all problems, greedy; 27 of 29 held-out wrong are RC | 90 (29) | 0.810 | n/a |
| 1.7B | SMOKE-4 A2, fp32 | contested only | 628 (307) | 0.539 | 0.561 [0.478, 0.637] (157) |
| 8B | SMOKE-5 D2, bf16 | contested only | 180 (98) | 0.446 | 0.556 [0.422, 0.689] (45) |

The probes trained by the within-problem objective, which is the gate statistic, scored:

| Model | Run | Within-pair test AUROC [95% CI] (pairs) |
|---|---|---|
| 1.7B | SMOKE-4 A4 | 0.599 [0.522, 0.675] (157) |
| 8B | SMOKE-5 D4a | 0.378 [0.244, 0.511] (45) |

Neither AUROC rises with boundary position:

- 1.7B: Spearman rho 0.38; positions 5–12 average 0.024 higher than 1–4.
- 8B: rho −0.29; positions 5–12 average 0.053 lower.

The problem-level Spearman correlation between the mean probe score and the share of correct traces came out as:

| Run | rho | Problems |
|---|---|---|
| 1.7B, SMOKE-2 all problems | −0.52 | 54 |
| 1.7B, SMOKE-4 contested | −0.10 | 157 |
| 8B, SMOKE-5 contested | +0.26 | 45 |

A negative rho means higher scores on harder problems, which is the expected direction for a difficulty signal.

## Restart share

Wrong traces whose MC success falls below 0.25 at the prompt or the first boundary, so the problem was effectively lost from the start:

| Model | Run | Wrong traces | Restart share | Never below 0.25 |
|---|---|---|---|---|
| 1.7B | SMOKE-4 A3 | 48 | 50.0% | 2.1% |
| 8B | SMOKE-5 D3 | 12 | 16.7% | 25.0% |

SMOKE to SMOKE-3 report a probe-based "restart share" of 0% (the probe's first flag at boundary 1). That is a different quantity.

## Conclusion

**Hidden-state error probes measure difficulty, not instance-level slips, for Qwen3-1.7B/8B on GSM-Symbolic and
RandomCalculation.**

What the evidence supports:

- **Instance-level slips, both models:** shown. Within-problem AUROC is 0.599 for the 1.7B and 0.378 for the 8B. Neither rises with boundary position, and the 8B CI excludes the 0.65 bar.
- **Difficulty, 1.7B:** shown. Across all problems the probe reaches 0.747. The same probe on the same held-out traces ranks the 11 wrong traces against their correct twins at 0.455. Its problem-level scores track how hard the problem is (rho −0.52). Within contested problems it drops to 0.54–0.60. This control covers GSM-Symbolic main/p1 only; RandomCalculation is not in it.
- **Difficulty, 8B: inferred, not measured.** No 8B features over all problems were saved; SMOKE-5 extracted features for contested problems only. On those, across-problem AUROC is 0.446, but contested problems span only 1–3 of 4 correct, so this does not test difficulty.
- **The 8B direct control:** it needs one pod. It would prefill SMOKE-5's 280 D1 problems (1120 traces, already generated and cached on the volume) and fit `P.fit` across all of them.
- **Front-loading:** the 1.7B is front-loaded (restart share 50%). The 8B is less so (16.7%), but on only 12 wrong traces.
- **Target b (recoverability, 8B):** no result. The D3 held-out boundary AUROC is 0.412 on 24 traces. D4b scored 0 test traces, because the 24-trace MC group could not finish within D4b's 17-minute budget on the L40.
