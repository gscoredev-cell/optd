RUN: SMOKE4-20261031-0142-da91eb5 | SMOKE4 | da91eb5 | qb1sw03h4zljua | NVIDIA L40 | 44.4 GB | US-KS-2 | $0.82/h | 50.4 min | $0.69
IMAGE PULL: 0.1 min (pod create -> bootstrap start) | cost of pull $0.00 | billed total est. 50.4 min $0.69
SSH: public port assigned yes (41675)
ENV: runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04 | torch 2.8.0.dev20250319+cu128 (cuda 12.8) | transformers 4.57.1 | peft 0.17.1 | Qwen3-1.7B / Qwen3-8B | cache hits 4 / downloads 0 | download MB/s n/a (all cache hits)
DEVICE: capability 8.9 | bf16 matmul ok (rel err 0.00166) | tier 48 | driver 595.91.07
TEMPLATES: identical token ids 20/20 | empty think block 20/20
BRANCH: A (Qwen3-1.7B) chosen by STAGE 0 (local, SMOKE-2 features): within-pair AUROC 0.750 [0.625, 0.875] (layer 21, last pooling; best of layers 21/28 x max/mean/last) on 48 contested greedy/sampled pairs (main 16, p1 32) >= 0.70 with n >= 40; max 0.417, mean 0.479; layer 14 n/a locally
TIMING (min): A1 31.01 | A2 2.36 | A4 0.03 | A3 16.16 | total 50.37
A1 MC STRATIFICATION (4 samples, T 0.8, cap 800; contested = 1..3 of 4 correct, the widest bounds that give a correct and a wrong trace):
  problems 800/800 | tok/s 433.2 (batch 128) | truncation 0.3% | mean tokens 249.6
  train_p1: problems 200 | #correct of 4 (0..4) [51, 28, 23, 27, 71] | accuracy 54.9% | contested 39.0%
  train_p2: problems 100 | #correct of 4 (0..4) [47, 15, 16, 14, 8] | accuracy 30.2% | contested 45.0%
  train_main: problems 100 | #correct of 4 (0..4) [16, 10, 3, 10, 61] | accuracy 72.5% | contested 23.0%
  train_all: problems 400 | #correct of 4 (0..4) [114, 53, 42, 51, 140] | accuracy 53.1% | contested 36.5%
  test_p1: problems 200 | #correct of 4 (0..4) [46, 21, 19, 32, 82] | accuracy 60.4% | contested 36.0%
  test_p2: problems 200 | #correct of 4 (0..4) [89, 31, 27, 27, 26] | accuracy 33.8% | contested 42.5%
  test_all: problems 400 | #correct of 4 (0..4) [135, 52, 46, 59, 108] | accuracy 47.1% | contested 39.2%
A2 WITHIN-PROBLEM PROBE (one wrong + one correct per contested train problem; fp32 features; layer, C, pooling by grouped 5-fold CV on train ids):
  pairs train 129 | held-out 17 | chosen: layer 21, C 0.01, pooling mean (cv 0.635)
  held-out WITHIN-PROBLEM AUROC 0.471 [0.235, 0.706] | pooled trace AUROC 0.516 | tau 0.7317 (FPR 10% on held-out contested correct): FPR 5.9%, recall 5.9%
    by pooling (chosen layer): max 0.471 [0.235, 0.706] ; mean 0.471 [0.235, 0.706] ; last 0.471 [0.235, 0.706]
    by layer (each at its best C; max / mean / last): 14 (C 1.0) 0.647 / 0.471 / 0.706 ; 21 (C 0.01) 0.471 / 0.471 / 0.471
    by boundary position (chosen layer): 1: 0.529 (n 17) ; 2: 0.235 (n 17) ; 3: 0.353 (n 17) ; 4: 0.412 (n 17) ; 5: 0.412 (n 17) ; 6: 0.643 (n 14) ; 7: 0.643 (n 14) ; 8: 0.500 (n 10) ; 9: 0.400 (n 10) ; 10: 0.600 (n 10) ; 11: 0.286 (n 7) ; 12: 0.333 (n 6)
    CV table (max / mean / last): 14|C=0.0001 0.429/0.428/0.502 ; 14|C=0.001 0.469/0.509/0.491 ; 14|C=0.01 0.465/0.507/0.470 ; 14|C=0.1 0.503/0.475/0.499 ; 14|C=1 0.523/0.487/0.489 ; 21|C=0.0001 0.441/0.489/0.527 ; 21|C=0.001 0.537/0.571/0.519 ; 21|C=0.01 0.535/0.635/0.556 ; 21|C=0.1 0.443/0.581/0.551 ; 21|C=1 0.449/0.474/0.564
A4 TEST (all traces of contested TEST problems; tau from A2):
  p1: problems 72 (correct 155, wrong 133) | FPR 1.3% | recall 3.8% | within-pair AUROC 0.597 [0.486, 0.708] (pairs 72)
  p2: problems 85 (correct 166, wrong 174) | FPR 0.6% | recall 2.9% | within-pair AUROC 0.600 [0.494, 0.706] (pairs 85)
  pooled: problems 157 (correct 321, wrong 307) | FPR 0.9% | recall 3.3% | within-pair AUROC 0.599 [0.522, 0.675] (pairs 157)
A3 STEP LOCALIZATION (contested wrong train traces, 3 completions per point; first point with MC success < 0.25; restart = first drop at the prompt or boundary 1):
  traces 48/100 | completions 1551 | tok/s 267.4 | MC at prompt mean 0.417 | restart share 50.0% | never below 0.25 2.1% | first drop (0 = prompt) {'0': 13, '1': 11, '2': 5, '3': 6, '4': 2, '5': 4, '6': 2, '7': 2, '8': 1, '10': 1, 'never': 1}
GATE (within-problem AUROC >= 0.80 held-out and recall >= 0.40 at FPR 10% on test): fail
BUDGET: max_minutes 90 x $0.82/h = $1.23 <= $1.5
BLOCKERS: none
