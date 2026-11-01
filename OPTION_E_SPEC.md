# Option E: trained repair under an external trigger — specification for agents

Version 1.6 · 2026-10-31 · supersedes the method sections of `OPTION_D_SPEC.md` v1.5. Infrastructure sections of v1.5 (§1–§6: RunPod, volume, launcher, watchdogs, cache, tiers, permissions) remain in force unchanged. Audience: the worker agent (local Claude Code). No human reader assumed.

## 0. Purpose and question

Option D showed that this student's hidden states do not reveal *when* it slips. Option E drops autonomous detection and tests the other half of the idea: **given an external pointer to where a solution went wrong, can a small trained adapter learn to redo it better than the model's own retry?**

In deployment the pointer is a user ("that's wrong, from step 3"), a unit test, or a verifier. In the experiment it is an **oracle** computed by sampling: the last point in the trace from which the model could still have succeeded.

Primary question: does **E3** (cut at the oracle point, continue with the repair adapter) recover more wrong first attempts than **E1** (same cut, plain retry)? Secondary: is the gain specific to the pointer (interaction test, §5.4), does it survive a sloppy pointer (E4), and does a false pointer harm correct answers (E5)?

Published baselines: untrained models get *worse* when told where their error is (arXiv 2601.00828); self-correction prompts without training fail (arXiv 2605.09502, 2605.31561); trained correction exists only for whole-answer rewrites (Welleck 2022) or for safety backtracking with a critic (arXiv 2602.08377). A trained redo from a pointed step on reasoning tasks is unclaimed.

## 1. What is reused (no changes)

- RunPod pipeline, volume `5l0v4bx88l`, S3 results, launcher/watch/kill/watchdog, generation cache, time budgets, report mechanics (v1.5 §1–§6, §13–§15).
- Student `Qwen/Qwen3-1.7B`, teacher `Qwen/Qwen3-8B`, non-thinking, identical rendered token ids.
- Data: GSM-Symbolic `p1` + `p2` split by `original_id` (seed 0) and RandomCalculation 3 ops (seeds as in `run_defaults.yaml`); prompt, decoding (greedy first attempts; retries T=0.8, top_p 0.95; cap 800), primary scoring (final-answer line, else last number; truncated = wrong).
- Contested stratum: problems with 1–3 of 4 sampled attempts correct (SMOKE-4 A1 labels and traces for 400 train + 400 test problems exist on the volume; import them into the cache rather than regenerating).
- Offline rollback (re-prefill the kept prefix), adapter-off prefix KV, LoRA r=16 (B=0), `rkl` loss with `rft` ablation, identity and resume checks, boundary definition (decoded `\n`, ≥ 8 tokens apart, max 12).
- Oracle labels already computed: SMOKE-4 A3 (48 wrong 1.7B traces, full boundary scans) and SMOKE-5 D3 (8B; not used here).

## 2. Oracle pointer

For a wrong first attempt (greedy) with boundaries 1..B, define `MC_b` = fraction of 3 sampled completions from the prefix ending at boundary b that reach the correct answer (`MC_0` = from the prompt).

- **Oracle cut** `b*` = the largest b with `MC_b ≥ 0.5` such that `MC_{b+1} < 0.5` (last recoverable point). If `MC_0 < 0.5`: `b* = 0` (restart). If `MC_B ≥ 0.5`: the trace is "recoverable to the end" (the error is in the final line) → `b* = B`.
- **Cost control: bisection.** Recoverability is assumed monotone non-increasing along a wrong trace. Probe `MC` at the prompt, then bisect over 1..B (≈ 4 probes instead of B). Validate on 20 traces with full scans: report the fraction of traces where the full scan agrees with bisection within ±1 boundary. If agreement < 80%, use full scans for the rest and record it.
- **Sloppy pointers** for robustness arms: `b*−1` (one step too early) and `b*+1` (one step too late, i.e., the prefix already contains the bad step).
- **Restart share** = fraction of wrong traces with `b* = 0`. Report per set. Traces with `b* = 0` stay in the evaluation (for them E1 ≡ E0 and E3 ≡ E6).

## 3. Arms

Unit of evaluation = one wrong greedy first attempt on a contested **test** problem, with its oracle cut. Every arm uses the same retry seed per trace; one continuation per arm in SMOKE-6, two in FULL (averaged).

| Arm | Start point | Adapter | Text injected | Purpose |
| --- | --- | --- | --- | --- |
| E0 | prompt (restart) | off | – | floor: plain resample |
| E1 | prefix(b*) | off | – | plain retry at the pointer |
| E2 | prefix(b*) | off | "\nWait — let me re-check that step.\n" | untrained text-level correction at the pointer |
| E3 | prefix(b*) | **on** | – | **trained repair at the pointer** |
| E4a | prefix(b*−1) | on | – | sloppy pointer, early |
| E4b | prefix(b*+1) | on | – | sloppy pointer, late |
| E6 | prompt (restart) | on | – | adapter without the pointer (distillation effect) |
| E5 | **correct** greedy traces, random cut uniform over 1..B | on | – | false-trigger harm |
| A6 | prompt, greedy | on (whole answer) | – | adapter always on (optional in SMOKE-6, required in FULL) |

Metrics per arm: **recovery rate** (wrong → correct) on the unit set; for E5 **harm rate** (correct → wrong); mean extra tokens; truncation rate. Report per set (p1, p2, RC3) and pooled.

## 4. Adapter training

- **Training units:** wrong attempts (greedy + sampled) on contested **train** problems, each with its oracle cut. One oracle scan per problem (the greedy wrong trace); sampled wrong traces of the same problem reuse the greedy cut only if their first B boundaries share ≥ 70% of tokens with the greedy trace (else they get their own scan in FULL; dropped in SMOKE-6).
- **Continuations:** 4 per unit from `prefix(b*)` with the adapter on (T=0.8, cap 800). Shared-prefix prefill once per unit. Also add 25% of units with a sloppy cut (`b*−1`) so the adapter sees imprecise pointers.
- **Loss `rkl`** (default): teacher prefill gives `logp_T` on continuation tokens; `A_t = −(logp_S_old − logp_T)` clipped ±5; loss = −mean A_t · logp_S_θ. **`rft`** ablation: SFT on continuations with a correct final answer only. Teacher-also-wrong units kept and counted.
- **Schedule:** AdamW lr 1e-4, cosine, 3% warmup, effective batch 16, grad checkpointing. **Early stopping** on a dev set (60 wrong train traces not used for training): every 10 steps evaluate E3 recovery on the dev set with one continuation each; keep the best checkpoint; stop after 3 evaluations without improvement. Rounds: SMOKE-6 = 2 rounds (re-sample continuations with the current adapter between rounds); FULL = up to 4.
- Checkpoint every 10 steps; `--resume` exercised once in SMOKE-6 (kill at step 15).
- The adapter is never active while building a prefix KV, never active in E0/E1/E2, and inactive for the first attempt everywhere except A6.

## 5. Analysis

### 5.1 Primary
`Δ_primary = recovery(E3) − recovery(E1)` on contested test wrong traces, paired by trace, bootstrap clustered by `original_id` (10,000 resamples), per set and pooled.

### 5.2 Baselines
`recovery(E2) − recovery(E1)` (does the untrained text hint help or hurt at the pointer); `recovery(E3) − recovery(E2)`.

### 5.3 Robustness
`recovery(E4a)`, `recovery(E4b)` vs `recovery(E3)`; `harm(E5)` vs a plain-retry false trigger (same cuts, adapter off).

### 5.4 Interaction (is the gain pointer-specific or just distillation?)
`I = [recovery(E3) − recovery(E1)] − [recovery(E6) − recovery(E0)]`. `I > 0` with CI lower bound > 0 means the adapter benefits from being told where to resume beyond making the model generally better. This is the claim that would be new.

### 5.5 Decision (FULL)
- **Pass:** `Δ_primary` CI lower bound > 0 on p1 and p2, and `I` CI lower bound > 0 pooled, Holm-corrected over the two; `harm(E5)` ≤ 1.5 × plain-retry harm.
- **Kill:** `Δ_primary` ≤ 0 pooled.
- **Distillation-only:** `Δ_primary` > 0 but `I` ≤ 0 → report as "adapter helps because it is a better model, not because of the pointer".

## 6. Phases

### SMOKE-6 (pipeline + first signal; A100 preferred, else L40; cap 120 min; budget $4)

| Step | Job | Size | Est. L40 |
| --- | --- | --- | --- |
| E-S1 | Import SMOKE-4 A1 traces/labels into the cache; select contested train/test problems; confirm counts | – | 3 min |
| E-S2 | Oracle labels, train: bisection on greedy wrong traces of contested train problems (reuse 48 from A3) | ≤ 200 traces | 25 min |
| E-S3 | Bisection validation: full scans on 20 traces; agreement rate | 20 | 6 min |
| E-S4 | Continuations ×4 (+25% sloppy) from oracle prefixes; teacher scoring | ≤ 200 units | 8 min |
| E-S5 | Train `rkl`, 2 rounds, early stopping on 60-trace dev set; kill/resume at step 15 | – | 10 min |
| E-S6 | Oracle labels, test: bisection on greedy wrong traces of contested test problems | 120 traces | 15 min |
| E-S7 | Arms E0–E6, E5 on 60 correct test traces, 1 continuation each, identical seeds | 120 wrong + 60 correct | 12 min |
| E-S8 | Report + FULL projection | – | 2 min |

Order for the deadline: E-S1, E-S2, E-S4, E-S5, E-S6, E-S7, E-S8, then E-S3 (validation is last because it is the only optional step; if it is cut, say so).

### FULL-E (after approval; ceilings)
Contested train problems from the full split (≈ 1,800 of 5,000 prompts at the measured 36% rate); oracle scans on all their wrong greedy traces plus sampled wrong traces per §4; up to 4 rounds with early stopping; test = all contested test problems (p1, p2) + RC3, 2 continuations per arm; seeds 0,1,2 (one pod each). Projection from SMOKE-6 rates; cut train size first, then rounds, then seeds; never below 150 test traces per set.

## 7. Gates

| Gate | Condition | If it fails |
| --- | --- | --- |
| Pipeline | identity (untrained E3 ≡ E1, 20/20), resume check, early stopping fired at least once | fix before FULL |
| Oracle | bisection–full-scan agreement ≥ 80% (±1); restart share reported; ≥ 100 train units with `b* ≥ 1` | full scans; or rebalance toward p1 (fewer restarts) |
| Signal (soft, SMOKE-6) | `Δ_primary` pooled ≥ +5 points and `I` ≥ 0 (point estimates) | still proceed to FULL if ≥ 0; stop if `Δ_primary` ≤ −5 |
| Harm | `harm(E5)` ≤ 1.5 × plain-retry harm | add correct-trace continuations to training (adapter must learn "nothing to fix") |
| Timing | FULL-E projection ≤ 12 h per seed | cut per §6 |

## 8. Report template (`report.md`)

```
RUN | phase | sha | pod | GPU | $/h | min | cost | cache hits/misses
DATA: contested train problems (p1/p2/RC3) | contested test | wrong greedy train traces | wrong greedy test traces | correct test traces for E5
ORACLE: bisection probes per trace | agreement with full scan (n=20) | restart share per set | b* histogram | traces recoverable to the end
TRAIN: units | continuations | teacher-also-wrong % | rounds | steps | best step (early stopping) | dev E3 recovery per eval | resume check
ARMS (test, per set and pooled): E0 | E1 | E2 | E3 | E4a | E4b | E6 recovery rates; E5 harm vs plain-retry harm; extra tokens; truncation
DELTAS: Δ_primary [CI] per set, pooled | E2−E1 | E3−E2 | I [CI] pooled
THROUGHPUT: gen tok/s | MC tok/s | teacher prefill tok/s | train steps/s
PROJECTION: FULL-E hours and $ per seed on this tier
GATES: pipeline | oracle | signal | harm | timing
BLOCKERS
```

## 9. Implementation notes

- New modules: `pipeline/oracle.py` (bisection + full scan, cached completions), `pipeline/option_e.py` (stages E-S1…E-S8), `train_repair.py` gains `--early-stop-dev` and sloppy-cut mixing; `eval.py` gains the E arms. Tests for bisection logic (synthetic monotone/non-monotone MC vectors), seed pairing, and the interaction statistic.
- Bisection must use the generation cache so repeated probes at the same prefix are free.
- E2's injected text is appended to the prefix *before* re-prefill; it counts toward extra tokens.
- E5 uses correct greedy traces of contested test problems; the random cut is seeded per trace.
- Early-stopping evaluation on the dev set must use the same seeds each time so the curve is comparable across steps.
- All stage outputs to `runs/<RUN_ID>/stage_*`; `.done` markers; status heartbeat; results via S3; pod self-terminates.
- Never copy question text out of the pod; commit only `report.md` and `metrics.json`.

## 10. Session start checklist (fresh session, same repo directory)

Read, in order: `CLAUDE.md`, this file, `OPTION_D_SPEC.md` §1–§6 and §13–§15, `docs/HANDOFF.md` (project history, infrastructure facts, what is on the volume), `docs/RESULTS.md`. Then verify before writing any code:

1. `git status` clean on `main`; `git log -1` is at or after `f90f384`.
2. `runpodctl pod list` returns `[]`; `kill.ps1 --All` reports 0 pods. Scheduled task `OptD-RunPod-Watchdog` exists and is enabled (`launch.ps1` refuses otherwise).
3. `aws --profile runpod s3api list-objects-v2 --bucket 5l0v4bx88l --prefix runs/ --max-keys 20` succeeds and shows SMOKE-2…SMOKE-5 run folders.
4. `pytest` passes (59 tests before Option E).
5. Known facts to honor without re-discovering them: runpodctl 3.0.1 has no `--terminate-after` (deadline watchdog in pod + local task instead); `aws s3 cp/sync` fail on this endpoint, use `s3api`; the local clock is slow, scripts use the S3 `Date` header; the pod's `runpodctl` is older (`remove pod` first); HF sampling is per batch, so cache seeds are per row and not bit-replayable; never copy question text out of the pod.
6. Reusable artifacts on the volume: SMOKE-4 A1 (400 train + 400 test problems × 4 samples, 1.7B) and A3 (48 oracle scans), SMOKE-2 S2 traces and features, generation cache under `/workspace/gencache`. Import, do not regenerate.

## 11. Risks named up front

- The oracle is cleaner than any user; E4a/E4b bound how much that matters.
- With ≈ 200 training units the adapter may learn little in SMOKE-6; the soft gate reflects that. FULL-E is where the test happens.
- Restart share near 50% on p2 (SMOKE-4) means half the test units have `b* = 0`; p1 (lower restart share expected) is the more informative set. Report both.
- Teacher log-probs on student continuations from an oracle prefix are off-distribution for the teacher only mildly (the prefix is the student's own correct-so-far text); still count teacher-also-wrong units.

## 12. SMOKE-7 addendum (2026-10-31, after SMOKE-6)

SMOKE-6 findings: the adapter learns after ~10 steps (E3 66%, E6 59% vs E0 33%); the E2 hint adds +15; round 2 over-optimizes; the pointer regime is starved (restart share 78–80% at MC ≥ 0.5); the projection was inflated by dev evaluations. SMOKE-7 (phase `SMOKE7`; A100 first, else L40; cap 120 min; budget $4):

1. **Oracle:** 4 completions per probe; recoverable = at least 1 of 4 correct (MC ≥ 0.25); bisection; re-validate on 20 traces with full scans; restart share per set. The MC store is keyed by k (`oracle_<model>_k4.jsonl`); A3's k = 3 scans are not reused as units.
2. **Data (E-S0):** 4 samples for 800 additional train problems (p1 400, p2 200, RC3 200 at seed 703; train templates, disjoint from SMOKE-4's) and 500 additional test problems (p1 250, p2 250; test templates), all through the cache; contested = 1..3 of 4. Target ≥ 120 pointed train units (b* ≥ 1); report the count. Caps: 300 train / 200 test traces for bisection.
3. **Training:** lr 5e-5; PPO-style ratio clipping (ε 0.2) on `rkl` against the stored sampling log-probs; a second adapter trained with `rft` on the same units (ablation); dev evaluation every 10 steps on 30 fixed-seed traces (pointed traces first); up to 3 rounds; early stopping across rounds keeps the best checkpoint and stops once a round fails to improve dev; the RNG is seeded before each optimizer step so the resume check is exact; training and dev-eval time reported separately.
4. **Arms:** §3 with both adapters (E3/E4a/E4b/E6 and E5 for `rkl` and for `rft`; E0/E1/E2 shared). One continuation per arm, identical batches and seeds.
5. **Analysis:** every delta on pointed units (b* ≥ 1) and on restart units (b* = 0) separately, with n and cluster-bootstrap CIs; `I` on pointed units is primary. Projection from pure training steps/s. Gates per §7 with the pointed-unit counts.

## 13. SMOKE-8 addendum (2026-10-31, after SMOKE-7): pivot, no pointer arms

SMOKE-7 closed the pointer: the oracle is degenerate (b* = last line at MC ≥ 0.25), the adapter's gain replicates in the restart regime (+18) and the hint replicates. SMOKE-8 (phase `SMOKE8`; Qwen/Qwen3.5-9B; A100 first; cap 150 min; budget $6; `pod/requirements_SMOKE8.txt`: transformers 5.22.1, peft 0.21.2, flash-linear-attention 0.6.0, causal-conv1d 1.7.0 installed by `bootstrap.sh`):

0. **Environment (Z0):** transformers ≥ 5; `fla` and `causal_conv1d` importable; no "fast path is not available" warning while loading; bf16 matmul; the student prompt and the teacher prompt (system line + "Reference solution") render with `enable_thinking=False` and share the user-turn suffix token for token. Language model only (text inputs to the multimodal class). Generation and log-probs use plain HF calls, no prefix cache (hybrid cache object).
1. **Self-teacher with privileged information (Z3):** teacher = the same weights with the system line `Reference solution:\n<dataset answer>` (GSM-Symbolic `answer`; the evaluation steps + result for RandomCalculation). Student = the same model without it. Report teacher greedy accuracy with the reference and the mean per-token teacher − student log-prob gap on correct vs wrong student traces.
2. **Difficulty calibration (Z1):** student greedy on 60 problems each of p1, p2 (train templates) and RC at 3/4/5/6 ops; keep the sets with accuracy in [35%, 70%]; report the contested fraction (4 samples) on them (Z2: 400 train and 240 test problems split evenly over the kept sets).
3. **Training (Z4):** units = contested train problems, restart from the prompt; 4 continuations each with the adapter on over the whole sequence; `rkl` with PPO ratio clipping (ε 0.2), lr 5e-5; dev every 10 steps on 30 traces (split by problem, stratified by set, never more than 30% of a set); up to 3 rounds; the best checkpoint is kept; kill/resume once.
4. **Arms (Z5):** on wrong greedy traces of contested test problems: E0 plain resample, E2 resample with the hint line, E6 resample with the adapter, E6h adapter + hint; identical batches and seeds. Harm: adapter vs plain resample on 60 correct greedy test traces. **Z6:** greedy pass@1 on all test problems, base vs adapter always on (A6), per set.
5. **Report (Z7):** per set and pooled recovery deltas with cluster-bootstrap CIs, harm, A6 vs base pass@1, tokens, throughput, FULL projection (pure training steps/s). Stop. No FULL.
