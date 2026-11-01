RUN: SMOKE7-20261031-2103-6f32b18 | SMOKE7 | 6f32b18 | i701035pca81yr | NVIDIA A100-SXM4-80GB | 79.3 GB | US-KS-2 | $1.79/h | 92.4 min | $2.76
IMAGE PULL: 0.1 min (pod create -> bootstrap start) | cost of pull $0.00 | billed total est. 92.4 min $2.76
SSH: public port assigned yes (18545)
ENV: runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04 | torch 2.8.0.dev20250319+cu128 (cuda 12.8) | transformers 4.57.1 | peft 0.17.1 | Qwen3-1.7B / Qwen3-8B | cache hits 4 / downloads 0 | download MB/s n/a (all cache hits)
DEVICE: capability 8.0 | bf16 matmul ok (rel err 0.00166) | tier 80 | driver 595.91.07
TEMPLATES: identical token ids 20/20 | empty think block 20/20
METHOD: oracle pointer (MC >= 0.25, 4 completions per probe, bisection) -> LoRA repair adapter(s) rkl/rft (lr 5e-05, PPO clip 0.2, early stopping on 30 dev traces, up to 3 rounds) ; first attempts greedy via the cache ; arms E0-E6 + E5 per adapter ; A6 not run
TIMING (min): ES0 29.76 | ES1 0.02 | ES2 12.54 | ES4 5.14 | ES5 11.28 | ES6 13.04 | ES7 12.64 | ES8 0.0 | ES3 6.39 | total 92.36
CACHE: SMOKE-4 A1 traces imported 0/3200 | A3 scans imported as MC vectors 48 (0 new points; k = 3 store only) | hits/misses per stage: es0 0/3072 ; es2_greedy 146/177 ; es2_mc 402/1314 ; es3_mc 120/580 ; es6_greedy 157/109 ; es6_mc 462/1342
DATA (E-S0, 4 samples per extra problem, T 0.8, cap 800): problems 768/1300 | tok/s 477.0 (batch 256) | truncation 0.8% | mean tokens 277.2
  test_p1: problems 153/250 | #correct of 4 (0..4) [36, 17, 16, 20, 64] | accuracy 59.6% | contested 34.6%
  test_p2: problems 145/250 | #correct of 4 (0..4) [67, 30, 16, 10, 22] | accuracy 31.0% | contested 38.6%
  train_p1: problems 237/400 | #correct of 4 (0..4) [72, 28, 27, 32, 78] | accuracy 51.7% | contested 36.7%
  train_p2: problems 121/200 | #correct of 4 (0..4) [61, 17, 24, 9, 10] | accuracy 27.3% | contested 41.3%
  train_rc3: problems 112/200 | #correct of 4 (0..4) [58, 19, 10, 11, 14] | accuracy 28.6% | contested 35.7%
DATA: contested train problems p1/p2/main/rc3 165/95/23/40 | contested test p1/p2 125/141 | wrong greedy train traces 176 of 323 (acc 45.5%) | wrong greedy test traces 146 of 266 (acc 45.1%) | correct test traces for E5 60 (of 120)
ORACLE (train): bisection probes per trace 1.26 (rounds 3; traces 57/176) | units 61 = bisected + A3 own scans 0 + sampled reusing the greedy cut 4 | POINTED train units (b* >= 1): 19 | MC tok/s 350.9
  train p1: n 27 | restart share 66.7% | b* >= 1 9 | to end 9 | b* hist {'0': 18, '5': 1, '7': 3, '10': 2, '12': 3}
  train p2: n 21 | restart share 71.4% | b* >= 1 6 | to end 6 | b* hist {'0': 15, '9': 1, '12': 5}
  train main: n 4 | restart share 100.0% | b* >= 1 0 | to end 0 | b* hist {'0': 4}
  train rc3: n 9 | restart share 55.6% | b* >= 1 4 | to end 4 | b* hist {'0': 5, '12': 4}
  train pooled: n 61 | restart share 68.9% | b* >= 1 19 | to end 19 | b* hist {'0': 42, '5': 1, '7': 3, '9': 1, '10': 2, '12': 12}
ORACLE (test): probes per trace 1.39 | traces 51/146 | MC tok/s 339.6
  test p1: n 22 | restart share 77.3% | b* >= 1 5 | to end 3 | b* hist {'0': 17, '2': 1, '3': 1, '12': 3}
  test p2: n 29 | restart share 72.4% | b* >= 1 8 | to end 8 | b* hist {'0': 21, '12': 8}
  test pooled: n 51 | restart share 74.5% | b* >= 1 13 | to end 11 | b* hist {'0': 38, '2': 1, '3': 1, '12': 11}
ORACLE VALIDATION (E-S3, full scans vs bisection, +-1): agreement 100.0% (exact 100.0%) on 2/20 traces | MC vectors monotone 50.0% | full-scan points per trace 7.5
TRAIN DATA (round 1): units 31 (pointed 0, sloppy b*-1 copies 0) | dev traces 30 (b* >= 1: 19) | continuations 124 (primary 11.3%, truncation 0.8%) | teacher-also-wrong 25.8% (n 31) | mean advantage -0.291
TRAIN: resume check y (exact) (kill at 4, ckpt every 3, max rel diff 0.000000) | adapter params 17432576 | round-1 examples 124
TRAIN [rkl]: rounds 3 | steps 24 | best: round 2 step 8 dev E3 recovery 46.7% | early stopping fired n | train 458 s (dev evals 224 s, resampling 195 s) | steps/s 0.0546 overall, 0.6307 pure | round 3 did not improve dev (46.7% <= 46.7%)
  round 1: steps 8 (examples 124) | loss start/end 0.2802/0.2611 | dev E3 recovery per eval (step: rate) 0: 20.0% ; 8: 30.0% | best step 8 | stopped early n
  round 2: steps 8 (examples 124) | loss start/end 0.3695/0.2210 | dev E3 recovery per eval (step: rate) 0: 30.0% ; 8: 46.7% | best step 8 | stopped early n
  round 3: steps 8 (examples 124) | loss start/end 0.2197/0.1952 | dev E3 recovery per eval (step: rate) 0: 46.7% ; 8: 36.7% | best step 0 | stopped early n
TRAIN [rft]: rounds 2 | steps 2 | best: round 1 step 0 dev E3 recovery 20.0% | early stopping fired n | train 216 s (dev evals 124 s, resampling 89 s) | steps/s 0.0093 overall, 0.7115 pure | round 2 did not improve dev (20.0% <= 20.0%)
  round 1: steps 1 (examples 14) | loss start/end 0.0650/0.0650 | dev E3 recovery per eval (step: rate) 0: 20.0% ; 1: 20.0% | best step 0 | stopped early n
  round 2: steps 1 (examples 14) | loss start/end 0.0604/0.0604 | dev E3 recovery per eval (step: rate) 0: 20.0% ; 1: 20.0% | best step 0 | stopped early n
IDENTITY: untrained E3 == E1 20/20 (max token diff 0) | adapters ['rkl', 'rft'] | units 51 (POINTED b* >= 1: 13, restart b* = 0: 38) | E4a/E4b clamped at 0/B {'E4a': 38, 'E4b': 11} | arms done ['E0', 'E1', 'E2', 'E3_rkl', 'E4a_rkl', 'E4b_rkl', 'E6_rkl', 'E3_rft', 'E4a_rft', 'E4b_rft', 'E6_rft']
ARMS [rkl] (test; recovery rate wrong -> correct; new tokens incl. injected text; truncation):
  all p1 (n 22, problems 12): E0 31.8% (271 tok, trunc 4.5%) ; E1 27.3% (250 tok, trunc 9.1%) ; E2 36.4% (262 tok, trunc 0.0%) ; E3 27.3% (314 tok, trunc 4.5%) ; E4a 50.0% (315 tok, trunc 4.5%) ; E4b 27.3% (252 tok, trunc 4.5%) ; E6 50.0% (377 tok, trunc 4.5%)
  all p2 (n 29, problems 18): E0 37.9% (328 tok, trunc 0.0%) ; E1 27.6% (272 tok, trunc 0.0%) ; E2 41.4% (329 tok, trunc 0.0%) ; E3 44.8% (363 tok, trunc 0.0%) ; E4a 41.4% (360 tok, trunc 3.4%) ; E4b 34.5% (287 tok, trunc 0.0%) ; E6 51.7% (423 tok, trunc 3.4%)
  all pooled (n 51, problems 25): E0 35.3% (303 tok, trunc 2.0%) ; E1 27.5% (263 tok, trunc 3.9%) ; E2 39.2% (300 tok, trunc 0.0%) ; E3 37.3% (342 tok, trunc 2.0%) ; E4a 45.1% (340 tok, trunc 3.9%) ; E4b 31.4% (272 tok, trunc 2.0%) ; E6 51.0% (403 tok, trunc 3.9%)
  pointed p1 (n 5, problems 3): E0 40.0% (303 tok, trunc 0.0%) ; E1 20.0% (170 tok, trunc 0.0%) ; E2 80.0% (251 tok, trunc 0.0%) ; E3 0.0% (210 tok, trunc 20.0%) ; E4a 40.0% (155 tok, trunc 0.0%) ; E4b 0.0% (202 tok, trunc 20.0%) ; E6 60.0% (416 tok, trunc 0.0%)
  pointed p2 (n 8, problems 4): E0 75.0% (360 tok, trunc 0.0%) ; E1 37.5% (193 tok, trunc 0.0%) ; E2 62.5% (275 tok, trunc 0.0%) ; E3 50.0% (192 tok, trunc 0.0%) ; E4a 25.0% (204 tok, trunc 0.0%) ; E4b 50.0% (192 tok, trunc 0.0%) ; E6 75.0% (444 tok, trunc 0.0%)
  pointed pooled (n 13, problems 6): E0 61.5% (338 tok, trunc 0.0%) ; E1 30.8% (184 tok, trunc 0.0%) ; E2 69.2% (265 tok, trunc 0.0%) ; E3 30.8% (199 tok, trunc 7.7%) ; E4a 30.8% (185 tok, trunc 0.0%) ; E4b 30.8% (196 tok, trunc 7.7%) ; E6 69.2% (433 tok, trunc 0.0%)
  restart p1 (n 17, problems 11): E0 29.4% (262 tok, trunc 5.9%) ; E1 29.4% (273 tok, trunc 11.8%) ; E2 23.5% (266 tok, trunc 0.0%) ; E3 35.3% (345 tok, trunc 0.0%) ; E4a 52.9% (362 tok, trunc 5.9%) ; E4b 35.3% (267 tok, trunc 0.0%) ; E6 47.1% (365 tok, trunc 5.9%)
  restart p2 (n 21, problems 16): E0 23.8% (316 tok, trunc 0.0%) ; E1 23.8% (303 tok, trunc 0.0%) ; E2 33.3% (350 tok, trunc 0.0%) ; E3 42.9% (428 tok, trunc 0.0%) ; E4a 47.6% (420 tok, trunc 4.8%) ; E4b 28.6% (324 tok, trunc 0.0%) ; E6 42.9% (414 tok, trunc 4.8%)
  restart pooled (n 38, problems 24): E0 26.3% (292 tok, trunc 2.6%) ; E1 26.3% (289 tok, trunc 5.3%) ; E2 28.9% (312 tok, trunc 0.0%) ; E3 39.5% (390 tok, trunc 0.0%) ; E4a 50.0% (394 tok, trunc 5.3%) ; E4b 31.6% (298 tok, trunc 0.0%) ; E6 44.7% (392 tok, trunc 5.3%)
  E5 harm [rkl] (correct -> wrong, adapter on, random cut) 20.0% vs plain-retry harm 26.7% (n 60)
DELTAS [rkl] (points; 95% CI by cluster bootstrap on original_id, 10,000 resamples; I on pointed units is primary):
  pointed p1 (n 5): delta_primary E3-E1 -20.0 [-50.0, +0.0] | E2-E1 +60.0 [+50.0, +100.0] | E3-E2 -80.0 [-100.0, -50.0] | I -40.0 [-100.0, +100.0]
  pointed p2 (n 8): delta_primary E3-E1 +12.5 [+0.0, +27.3] | E2-E1 +25.0 [-33.3, +77.8] | E3-E2 -12.5 [-66.7, +33.3] | I +12.5 [-33.3, +42.9]
  pointed pooled (n 13): delta_primary E3-E1 +0.0 [-25.0, +16.7] | E2-E1 +38.5 [+0.0, +69.2] | E3-E2 -38.5 [-80.0, +0.0] | I -7.7 [-63.6, +35.3]
  restart p1 (n 17): delta_primary E3-E1 +5.9 [-23.1, +42.9] | E2-E1 -5.9 [-26.7, +14.3] | E3-E2 +11.8 [-10.0, +41.7] | I -11.8 [-50.0, +22.2]
  restart p2 (n 21): delta_primary E3-E1 +19.0 [-5.3, +45.0] | E2-E1 +9.5 [-17.4, +41.2] | E3-E2 +9.5 [-27.8, +41.7] | I +0.0 [-31.6, +28.6]
  restart pooled (n 38): delta_primary E3-E1 +13.2 [-6.2, +35.5] | E2-E1 +2.6 [-15.6, +22.2] | E3-E2 +10.5 [-11.4, +34.2] | I -5.3 [-25.0, +15.6]
  all p1 (n 22): delta_primary E3-E1 +0.0 [-21.1, +35.7] | E2-E1 +9.1 [-9.1, +27.8] | E3-E2 -9.1 [-29.6, +25.0] | I -18.2 [-48.6, +16.7]
  all p2 (n 29): delta_primary E3-E1 +17.2 [-2.5, +38.5] | E2-E1 +13.8 [-8.3, +35.1] | E3-E2 +3.4 [-22.2, +30.8] | I +3.4 [-23.8, +25.0]
  all pooled (n 51): delta_primary E3-E1 +9.8 [-5.0, +29.2] | E2-E1 +11.8 [-5.5, +26.3] | E3-E2 -2.0 [-19.3, +21.3] | I -5.9 [-24.4, +13.5]
ARMS [rft] (test; recovery rate wrong -> correct; new tokens incl. injected text; truncation):
  all p1 (n 22, problems 12): E0 31.8% (271 tok, trunc 4.5%) ; E1 27.3% (250 tok, trunc 9.1%) ; E2 36.4% (262 tok, trunc 0.0%) ; E3 27.3% (250 tok, trunc 9.1%) ; E4a 31.8% (237 tok, trunc 0.0%) ; E4b 27.3% (208 tok, trunc 0.0%) ; E6 31.8% (271 tok, trunc 4.5%)
  all p2 (n 29, problems 18): E0 37.9% (328 tok, trunc 0.0%) ; E1 27.6% (272 tok, trunc 0.0%) ; E2 41.4% (329 tok, trunc 0.0%) ; E3 27.6% (272 tok, trunc 0.0%) ; E4a 20.7% (276 tok, trunc 0.0%) ; E4b 20.7% (279 tok, trunc 0.0%) ; E6 37.9% (328 tok, trunc 0.0%)
  all pooled (n 51, problems 25): E0 35.3% (303 tok, trunc 2.0%) ; E1 27.5% (263 tok, trunc 3.9%) ; E2 39.2% (300 tok, trunc 0.0%) ; E3 27.5% (263 tok, trunc 3.9%) ; E4a 25.5% (259 tok, trunc 0.0%) ; E4b 23.5% (248 tok, trunc 0.0%) ; E6 35.3% (303 tok, trunc 2.0%)
  pointed p1 (n 5, problems 3): E0 40.0% (303 tok, trunc 0.0%) ; E1 20.0% (170 tok, trunc 0.0%) ; E2 80.0% (251 tok, trunc 0.0%) ; E3 20.0% (170 tok, trunc 0.0%) ; E4a 40.0% (172 tok, trunc 0.0%) ; E4b 20.0% (158 tok, trunc 0.0%) ; E6 40.0% (303 tok, trunc 0.0%)
  pointed p2 (n 8, problems 4): E0 75.0% (360 tok, trunc 0.0%) ; E1 37.5% (193 tok, trunc 0.0%) ; E2 62.5% (275 tok, trunc 0.0%) ; E3 37.5% (193 tok, trunc 0.0%) ; E4a 25.0% (197 tok, trunc 0.0%) ; E4b 37.5% (193 tok, trunc 0.0%) ; E6 75.0% (360 tok, trunc 0.0%)
  pointed pooled (n 13, problems 6): E0 61.5% (338 tok, trunc 0.0%) ; E1 30.8% (184 tok, trunc 0.0%) ; E2 69.2% (265 tok, trunc 0.0%) ; E3 30.8% (184 tok, trunc 0.0%) ; E4a 30.8% (187 tok, trunc 0.0%) ; E4b 30.8% (179 tok, trunc 0.0%) ; E6 61.5% (338 tok, trunc 0.0%)
  restart p1 (n 17, problems 11): E0 29.4% (262 tok, trunc 5.9%) ; E1 29.4% (273 tok, trunc 11.8%) ; E2 23.5% (266 tok, trunc 0.0%) ; E3 29.4% (273 tok, trunc 11.8%) ; E4a 29.4% (256 tok, trunc 0.0%) ; E4b 29.4% (223 tok, trunc 0.0%) ; E6 29.4% (262 tok, trunc 5.9%)
  restart p2 (n 21, problems 16): E0 23.8% (316 tok, trunc 0.0%) ; E1 23.8% (303 tok, trunc 0.0%) ; E2 33.3% (350 tok, trunc 0.0%) ; E3 23.8% (303 tok, trunc 0.0%) ; E4a 19.0% (306 tok, trunc 0.0%) ; E4b 14.3% (312 tok, trunc 0.0%) ; E6 23.8% (316 tok, trunc 0.0%)
  restart pooled (n 38, problems 24): E0 26.3% (292 tok, trunc 2.6%) ; E1 26.3% (289 tok, trunc 5.3%) ; E2 28.9% (312 tok, trunc 0.0%) ; E3 26.3% (289 tok, trunc 5.3%) ; E4a 23.7% (284 tok, trunc 0.0%) ; E4b 21.1% (272 tok, trunc 0.0%) ; E6 26.3% (292 tok, trunc 2.6%)
  E5 harm [rft] (correct -> wrong, adapter on, random cut) 26.7% vs plain-retry harm 26.7% (n 60)
DELTAS [rft] (points; 95% CI by cluster bootstrap on original_id, 10,000 resamples; I on pointed units is primary):
  pointed p1 (n 5): delta_primary E3-E1 +0.0 [+0.0, +0.0] | E2-E1 +60.0 [+50.0, +100.0] | E3-E2 -60.0 [-100.0, -50.0] | I +0.0 [+0.0, +0.0]
  pointed p2 (n 8): delta_primary E3-E1 +0.0 [+0.0, +0.0] | E2-E1 +25.0 [-33.3, +77.8] | E3-E2 -25.0 [-77.8, +33.3] | I +0.0 [+0.0, +0.0]
  pointed pooled (n 13): delta_primary E3-E1 +0.0 [+0.0, +0.0] | E2-E1 +38.5 [+0.0, +69.2] | E3-E2 -38.5 [-69.2, +0.0] | I +0.0 [+0.0, +0.0]
  restart p1 (n 17): delta_primary E3-E1 +0.0 [+0.0, +0.0] | E2-E1 -5.9 [-26.7, +14.3] | E3-E2 +5.9 [-14.3, +26.7] | I +0.0 [+0.0, +0.0]
  restart p2 (n 21): delta_primary E3-E1 +0.0 [+0.0, +0.0] | E2-E1 +9.5 [-17.4, +41.2] | E3-E2 -9.5 [-41.2, +17.4] | I +0.0 [+0.0, +0.0]
  restart pooled (n 38): delta_primary E3-E1 +0.0 [+0.0, +0.0] | E2-E1 +2.6 [-15.6, +22.2] | E3-E2 -2.6 [-22.2, +15.6] | I +0.0 [+0.0, +0.0]
  all p1 (n 22): delta_primary E3-E1 +0.0 [+0.0, +0.0] | E2-E1 +9.1 [-9.1, +27.8] | E3-E2 -9.1 [-27.8, +9.1] | I +0.0 [+0.0, +0.0]
  all p2 (n 29): delta_primary E3-E1 +0.0 [+0.0, +0.0] | E2-E1 +13.8 [-8.3, +35.1] | E3-E2 -13.8 [-35.1, +8.3] | I +0.0 [+0.0, +0.0]
  all pooled (n 51): delta_primary E3-E1 +0.0 [+0.0, +0.0] | E2-E1 +11.8 [-5.5, +26.3] | E3-E2 -11.8 [-26.3, +5.5] | I +0.0 [+0.0, +0.0]
THROUGHPUT: gen tok/s 304.9 | data tok/s 477.0 | MC tok/s 350.9 | teacher prefill tok/s 6632.8 | train steps/s pure 0.6307 (overall 0.0546)
PROJECTION: FULL-E 19.5 h and $34.93 per seed on this tier (rough, pure training steps/s; oracle_train 1.5 h, continuations_4_rounds 6.7 h, teacher_scoring 0.8 h, training 0.8 h, dev_evals 1.2 h, oracle_test 1.2 h, arms 7.3 h; cut train size, then rounds, then seeds if > 12 h)
GATES: pipeline (identity 20/20, resume, early stop fired) fail ; oracle (agreement >= 80%, >= 100 pointed train units) fail ; signal (soft: rkl delta_primary pooled >= +5 and I >= 0 on POINTED test units, n 13) fail ; harm (rkl E5 <= 1.5 x plain-retry) pass ; timing (FULL-E <= 12 h per seed) fail
BUDGET: max_minutes 120 x $1.79/h = $3.58 <= $4
BLOCKERS: none
