RUN: SMOKE6-20261031-1953-f1771e9 | SMOKE6 | f1771e9 | 8u5i4p4yx5rvjb | NVIDIA A100 80GB PCIe | 79.3 GB | US-KS-2 | $1.79/h | 40.3 min | $1.20
IMAGE PULL: 2.9 min (pod create -> bootstrap start) | cost of pull $0.09 | billed total est. 43.1 min $1.29
SSH: public port assigned yes (12123)
ENV: runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04 | torch 2.8.0.dev20250319+cu128 (cuda 12.8) | transformers 4.57.1 | peft 0.17.1 | Qwen3-1.7B / Qwen3-8B | cache hits 4 / downloads 0 | download MB/s n/a (all cache hits)
DEVICE: capability 8.0 | bf16 matmul ok (rel err 0.00165) | tier 80 | driver 595.91.07
TEMPLATES: identical token ids 20/20 | empty think block 20/20
METHOD: oracle pointer (MC >= 0.5, 3 completions, bisection) -> LoRA repair adapter (rkl, early stopping on 60 dev traces, 2 rounds) ; first attempts greedy via the cache ; arms E0-E6 + E5 ; A6 not run (optional in SMOKE-6)
TIMING (min): ES1 0.01 | ES2 4.49 | ES4 5.32 | ES5 9.66 | ES6 6.89 | ES7 9.3 | ES8 0.0 | ES3 3.61 | total 40.26
CACHE: SMOKE-4 A1 traces imported 3200/3200 | A3 scans imported as MC vectors 48 (517 new points; completions were not stored by SMOKE-4) | hits/misses per stage: es2_greedy 0/146 ; es2_mc 0/510 ; es3_mc 0/540 ; es6_greedy 0/157 ; es6_mc 0/591
DATA: contested train problems p1/p2/main 78/45/23 (RC3 none in SMOKE-6) | contested test p1/p2 72/85 | wrong greedy train traces 84 of 146 (acc 42.5%) | wrong greedy test traces 85 of 157 (acc 45.9%) | correct test traces for E5 60 (of 72)
ORACLE (train): bisection probes per trace 2.02 (rounds 6; traces 84/84) | units 136 = bisected + A3 own scans 48 + sampled reusing the greedy cut 4 | units with b* >= 1: 27 | MC tok/s 523.9
  train p1: n 73 | restart share 76.7% | b* >= 1 17 | to end 0 | b* hist {'0': 56, '1': 6, '2': 1, '3': 2, '4': 1, '5': 2, '6': 2, '7': 1, '8': 2}
  train p2: n 43 | restart share 86.0% | b* >= 1 6 | to end 1 | b* hist {'0': 37, '1': 1, '4': 1, '5': 2, '9': 1, '12': 1}
  train main: n 20 | restart share 80.0% | b* >= 1 4 | to end 0 | b* hist {'0': 16, '1': 1, '3': 1, '5': 2}
  train pooled: n 136 | restart share 80.1% | b* >= 1 27 | to end 1 | b* hist {'0': 109, '1': 8, '2': 1, '3': 3, '4': 2, '5': 6, '6': 2, '7': 1, '8': 2, '9': 1, '12': 1}
ORACLE (test): probes per trace 2.32 | traces 85/85 | MC tok/s 417.4
  test p1: n 35 | restart share 77.1% | b* >= 1 8 | to end 0 | b* hist {'0': 27, '1': 2, '2': 1, '3': 3, '5': 1, '10': 1}
  test p2: n 50 | restart share 78.0% | b* >= 1 11 | to end 2 | b* hist {'0': 39, '1': 2, '2': 1, '3': 1, '5': 1, '6': 1, '7': 1, '9': 1, '10': 1, '12': 2}
  test pooled: n 85 | restart share 77.6% | b* >= 1 19 | to end 2 | b* hist {'0': 66, '1': 4, '2': 2, '3': 4, '5': 2, '6': 1, '7': 1, '9': 1, '10': 2, '12': 2}
ORACLE VALIDATION (E-S3, full scans vs bisection, +-1): agreement 85.0% (exact 85.0%) on 20/20 traces | MC vectors monotone 30.0% | full-scan points per trace 11.4
TRAIN DATA (round 1): units 66 (sloppy b*-1 copies 4) | dev traces 60 (b* >= 1: 11) | continuations 264 (primary 32.2%, truncation 0.0%) | teacher-also-wrong 7.6% (n 66) | mean advantage -0.320
TRAIN: rkl | rounds 2 | steps 34 | best: round 1 step 10 dev E3 recovery 58.3% | early stopping fired n | resume check n (kill at 15, ckpt every 10, max rel diff 0.003968) | adapter params 17432576 | train steps/s 0.0588
  round 1: steps 17 | loss start/end -0.0997/-2.6314 | dev E3 recovery per eval (step: rate) 0: 38.3% ; 10: 58.3% ; 17: 53.3% | best step 10 | stopped early n
  round 2: steps 17 | loss start/end -0.0740/-3.6113 | dev E3 recovery per eval (step: rate) 0: 58.3% ; 10: 40.0% ; 17: 20.0% | best step 0 | stopped early n
IDENTITY: untrained E3 == E1 20/20 (max token diff 0) | adapter trained | units 85 (restart b* = 0: 66) | E4a/E4b clamped at 0/B {'E4a': 66, 'E4b': 2} | arms done ['E0', 'E1', 'E2', 'E3', 'E4a', 'E4b', 'E6']
ARMS (test; recovery rate wrong -> correct; new tokens incl. injected text; truncation):
  p1 (n 35, problems 22): E0 31.4% (237 tok, trunc 2.9%) ; E1 28.6% (235 tok, trunc 0.0%) ; E2 60.0% (270 tok, trunc 0.0%) ; E3 74.3% (406 tok, trunc 5.7%) ; E4a 71.4% (418 tok, trunc 5.7%) ; E4b 57.1% (259 tok, trunc 2.9%) ; E6 60.0% (457 tok, trunc 8.6%)
  p2 (n 50, problems 20): E0 34.0% (313 tok, trunc 2.0%) ; E1 38.0% (275 tok, trunc 0.0%) ; E2 42.0% (335 tok, trunc 2.0%) ; E3 60.0% (465 tok, trunc 0.0%) ; E4a 60.0% (463 tok, trunc 2.0%) ; E4b 52.0% (338 tok, trunc 2.0%) ; E6 58.0% (515 tok, trunc 4.0%)
  pooled (n 85, problems 32): E0 32.9% (282 tok, trunc 2.4%) ; E1 34.1% (258 tok, trunc 0.0%) ; E2 49.4% (308 tok, trunc 1.2%) ; E3 65.9% (441 tok, trunc 2.4%) ; E4a 64.7% (444 tok, trunc 3.5%) ; E4b 54.1% (305 tok, trunc 2.4%) ; E6 58.8% (491 tok, trunc 5.9%)
  E5 harm (correct -> wrong, adapter on, random cut) 13.3% vs plain-retry harm 20.0% (n 60)
DELTAS (points; 95% CI by cluster bootstrap on original_id, 10,000 resamples):
  p1: delta_primary E3-E1 +45.7 [+27.3, +61.5] | E2-E1 +31.4 [+13.5, +48.4] | E3-E2 +14.3 [-6.5, +34.4] | I +17.1 [-5.6, +39.4]
  p2: delta_primary E3-E1 +22.0 [-2.1, +43.6] | E2-E1 +4.0 [-11.8, +22.5] | E3-E2 +18.0 [-8.5, +44.2] | I -2.0 [-25.0, +20.8]
  pooled: delta_primary E3-E1 +31.8 [+14.9, +47.1] | E2-E1 +15.3 [+3.9, +27.7] | E3-E2 +16.5 [-1.2, +34.1] | I +5.9 [-11.1, +22.5]
THROUGHPUT: gen tok/s 475.3 | MC tok/s 523.9 | teacher prefill tok/s 6761.5 | train steps/s 0.0588
PROJECTION: FULL-E 20.5 h and $36.75 per seed on this tier (rough; oracle_train 1.3 h, continuations_4_rounds 4.5 h, teacher_scoring 0.8 h, training 9.8 h, oracle_test 1.0 h, arms 3.0 h; cut train size, then rounds, then seeds if > 12 h)
GATES: pipeline (identity 20/20, resume, early stop fired) fail ; oracle (agreement >= 80%, >= 100 train units with b* >= 1) fail ; signal (soft: delta_primary pooled >= +5 and I >= 0) pass ; harm (E5 <= 1.5 x plain-retry) pass ; timing (FULL-E <= 12 h per seed) fail
BUDGET: max_minutes 120 x $1.79/h = $3.58 <= $4
BLOCKERS: none
