RUN: SMOKE5-20261031-1722-977e096 | SMOKE5 | 977e096 | 86rvpwifqk9b7m | NVIDIA L40 | 44.4 GB | US-KS-2 | $0.82/h | 88.4 min | $1.21
IMAGE PULL: 0.1 min (pod create -> bootstrap start) | cost of pull $0.00 | billed total est. 88.4 min $1.21
SSH: public port assigned yes (49206)
ENV: runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04 | torch 2.8.0.dev20250319+cu128 (cuda 12.8) | transformers 4.57.1 | peft 0.17.1 | Qwen3-8B / none (not loaded) | cache hits 4 / downloads 0 | download MB/s n/a (all cache hits)
DEVICE: capability 8.9 | bf16 matmul ok (rel err 0.00166) | tier 48 | driver 595.91.07
TEMPLATES: identical token ids 20/20 | empty think block 20/20
STUDENT: Qwen3-8B, no teacher | features bf16 prefill, batch 8 | HF generate, T 0.8, top_p 0.95, cap 800
TIMING (min): D1 30.15 | D2 3.38 | D4a 0.29 | D3 35.39 | D4b 18.46 | total 88.39
GENERATION CACHE (/workspace/gencache, key model | prompt sha256 | T | top_p | seed, cap rule): seeded from SMOKE-2 S1 (SMOKE-20261030-2118-ccc68bc): 380 Qwen3-8B greedy traces, 380 new, 100 are D1 problems (prompt ids identical 100/100) | hits/misses per stage: D1 100/1088 ; D3 0/1192 ; D4b 0/576
D1 MC STRATIFICATION (4 per problem; contested = 1..3 of 4 correct):
  problems 280/700 | tok/s 260.2 (batch 64, generated rows only) | truncation 3.6% | mean tokens 432.2
  train_rc3: problems 60 | #correct of 4 (0..4) [16, 5, 19, 9, 11] | accuracy 47.5% | contested 55.0%
  train_rc4: problems 62 | #correct of 4 (0..4) [26, 15, 9, 9, 3] | accuracy 29.0% | contested 53.2%
  train_p2: problems 50 | #correct of 4 (0..4) [9, 5, 6, 7, 23] | accuracy 65.0% | contested 36.0%
  train_all: problems 172 | #correct of 4 (0..4) [51, 25, 34, 25, 37] | accuracy 45.9% | contested 48.8%
  test_rc3: problems 36 | #correct of 4 (0..4) [10, 7, 6, 3, 10] | accuracy 47.2% | contested 44.4%
  test_rc4: problems 38 | #correct of 4 (0..4) [17, 9, 5, 5, 2] | accuracy 27.6% | contested 50.0%
  test_p2: problems 34 | #correct of 4 (0..4) [0, 3, 4, 3, 24] | accuracy 85.3% | contested 29.4%
  test_all: problems 108 | #correct of 4 (0..4) [27, 19, 15, 11, 36] | accuracy 52.3% | contested 41.7%
  NOTE: test_p2 p_hat mixes 1 greedy (sample 0, SMOKE-2 S1 via the cache) + 3 sampled for 34 problems; not directly comparable with train_p2 (4 sampled)
D2 TARGET A, WITHIN-PROBLEM PROBE (one wrong + one correct per contested train problem; layer, C, pooling by grouped 5-fold CV on train groups; held-out = 20% of train groups):
  pairs train 67 | held-out 17 | chosen: layer 18, C 0.1, pooling mean (cv 0.623) | features 2.2 min
  held-out WITHIN-PROBLEM AUROC 0.529 [0.294, 0.765] | pooled trace AUROC 0.588 | tau 0.7025 (FPR 10% on held-out contested correct): FPR 5.9%, recall 29.4%
    by pooling (chosen layer): max 0.588 [0.353, 0.824] ; mean 0.529 [0.294, 0.765] ; last 0.471 [0.235, 0.706]
    by layer (each at its best C; max / mean / last): 18 (C 0.1) 0.588 / 0.529 / 0.471 ; 27 (C 0.0001) 0.529 / 0.471 / 0.471 ; 36 (C 0.1) 0.588 / 0.588 / 0.353
    by boundary position (held-out pairs): 1: 0.412 (n 17) ; 2: 0.529 (n 17) ; 3: 0.647 (n 17) ; 4: 0.529 (n 17) ; 5: 0.412 (n 17) ; 6: 0.588 (n 17) ; 7: 0.529 (n 17) ; 8: 0.625 (n 16) ; 9: 0.562 (n 16) ; 10: 0.600 (n 15) ; 11: 0.533 (n 15) ; 12: 0.429 (n 14)
    rises with position: no (Spearman rho 0.21 over positions 1..12 with n >= 10; mean 5..12 minus mean 1..4 = 0.005)
    CV table (max / mean / last): 18|C=0.0001 0.622/0.606/0.551 ; 18|C=0.001 0.550/0.562/0.521 ; 18|C=0.01 0.580/0.579/0.484 ; 18|C=0.1 0.592/0.623/0.594 ; 18|C=1 0.534/0.595/0.562 ; 27|C=0.0001 0.552/0.606/0.519 ; 27|C=0.001 0.534/0.594/0.508 ; 27|C=0.01 0.428/0.561/0.570 ; 27|C=0.1 0.496/0.536/0.512 ; 27|C=1 0.475/0.523/0.527 ; 36|C=0.0001 0.576/0.534/0.554 ; 36|C=0.001 0.567/0.577/0.527 ; 36|C=0.01 0.450/0.574/0.545 ; 36|C=0.1 0.478/0.531/0.604 ; 36|C=1 0.450/0.496/0.588
D4a TEST, TARGET A (all traces of contested TEST problems; fresh pairs; tau from D2):
  rc3: problems 16 (correct 28, wrong 36) | FPR 25.0% | recall 25.0% | within-pair AUROC 0.438 [0.188, 0.688] (pairs 16)
  rc4: problems 19 (correct 34, wrong 42) | FPR 35.3% | recall 19.0% | within-pair AUROC 0.421 [0.211, 0.632] (pairs 19)
  p2: problems 10 (correct 20, wrong 20) | FPR 0.0% | recall 10.0% | within-pair AUROC 0.200 [0.000, 0.500] (pairs 10)
  pooled: problems 45 (correct 82, wrong 98) | FPR 23.2% | recall 19.4% | within-pair AUROC 0.378 [0.244, 0.511] (pairs 45)
  by boundary position (test pairs): 1: 0.533 (n 45) ; 2: 0.467 (n 45) ; 3: 0.444 (n 45) ; 4: 0.600 (n 45) ; 5: 0.422 (n 45) ; 6: 0.467 (n 45) ; 7: 0.477 (n 44) ; 8: 0.500 (n 44) ; 9: 0.439 (n 41) ; 10: 0.450 (n 40) ; 11: 0.421 (n 38) ; 12: 0.486 (n 37)
  rises with position: no (Spearman rho -0.29 over positions 1..12 with n >= 10; mean 5..12 minus mean 1..4 = -0.053)
D3 TARGET B, RECOVERABILITY (contested train pairs, 3 completions from the prompt and each boundary; boundary unrecoverable = MC success < 0.5; boundary probe by grouped 5-fold CV, held-out = 20% of groups):
  traces 24/120 | completions 936 | tok/s 226.7 | MC at prompt mean 0.611 | unrecoverable boundaries: correct traces 41.0%, wrong traces 58.3%
  probe: layer 18, C 0.0001 (cv 0.445) | held-out boundary AUROC 0.412 (boundaries 48, unrecoverable 25) | tau 0.5744 (FPR 10% on held-out recoverable)
    by layer (held-out at best C): 18|C=0.0001 0.412
    by boundary position (held-out): 1: 0.000 (n 4, unrec 3) ; 2: 1.000 (n 4, unrec 2) ; 3: 0.500 (n 4, unrec 2) ; 4: 0.000 (n 4, unrec 1) ; 5: 0.333 (n 4, unrec 3) ; 6: 0.000 (n 4, unrec 3) ; 7: 0.500 (n 4, unrec 2) ; 8: 0.500 (n 4, unrec 2) ; 9: 0.500 (n 4, unrec 2) ; 10: 0.333 (n 4, unrec 3) ; 11: 0.500 (n 4, unrec 2) ; 12: nan (n 4, unrec 0)
    rises with position: n/a (fewer than 3 positions with n >= 10)
  wrong traces 12: first point with MC < 0.25 (SMOKE-4 rule) | restart share 16.7% | never below 0.25 25.0% | first drop (0 = prompt) {'0': 1, '1': 1, '2': 1, '3': 2, '4': 1, '7': 2, '12': 1, 'never': 3}
D4b TEST, TARGET B (contested TEST pairs with MC labels, 3 completions per point; tau from D3):
  traces 0/60 | completions 0 | tok/s 219.5
  by boundary position (test): 
  rises with position: n/a (fewer than 3 positions with n >= 10)
GATE (test within-problem AUROC >= 0.75 with 95% CI lower bound > 0.65; D4a pooled): fail
BUDGET: max_minutes 120 x $0.82/h = $1.64 <= $4
BLOCKERS: none
