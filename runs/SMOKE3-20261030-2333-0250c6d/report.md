RUN: SMOKE3-20261030-2333-0250c6d | SMOKE3 | 0250c6d | bm9to7ym8ae6sl | NVIDIA L40 | 44.4 GB | US-KS-2 | $0.82/h | 71.6 min | $0.98
IMAGE PULL: 0.1 min (pod create -> bootstrap start) | cost of pull $0.00 | billed total est. 71.6 min $0.98
SSH: public port assigned yes (46177)
ENV: runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04 | torch 2.8.0.dev20250319+cu128 (cuda 12.8) | transformers 4.57.1 | peft 0.17.1 | Qwen3-1.7B / Qwen3-8B | cache hits 3 / downloads 1 | download MB/s 321.4
DEVICE: capability 8.9 | bf16 matmul ok (rel err 0.00166) | tier 48 | driver 595.91.07
TEMPLATES: identical token ids 20/20 | empty think block 20/20
SOURCE: SMOKE-2 artifacts from SMOKE-20261030-2118-ccc68bc (S2 greedy traces, S1 A0 traces, S1 teacher) ; scoring primary (lenient, truncated = wrong) ; RC for FULL: 3 ops
TIMING (min): MCa 18.28 | MCprobe 6.01 | B1 9.41 | B2 12.27 | B3 11.13 | C2 13.19 | C3 0.0 | total 71.56
(a) PROBE-MC (Qwen3-1.7B, SMOKE-2 greedy train traces, 3 completions per point, T 0.8, cap 800):
  labels: traces 144/300 | completions 4167 | tok/s 487.9 | bad steps 44 | traces with a bad step 27 | wrong traces 49 (with a bad step 11) | MC success at the prompt: correct traces 0.870, wrong traces 0.150
  step probe: layers 21 | C 0.0001 | step AUROC held-out 0.436 (cv 0.607; positives train 42/1080, held-out 2/165) | first bad step within +-1 0.0% (n 1) | implied trace AUROC 0.697 | tau 0.0463
    eval (test traces) FPR per set at step tau: n/a
    step AUROC table:
    21: C=0.0001 cv 0.607 held-out 0.436 ; C=0.001 cv 0.566 ; C=0.01 cv 0.511 ; C=0.1 cv 0.500 ; C=1 cv 0.503
(b) QWEN3-4B STUDENT:
  S1 (380, 800 tokens, teacher from SMOKE-2) primary/strict, student vs teacher: p2 75.0%/75.0% vs 82.0% (trunc 0.0%) / main 85.0%/85.0% vs 97.5% (trunc 2.5%) / p1 90.0%/90.0% vs 87.5% (trunc 0.0%) / rc 18.0%/19.0% vs 21.0% (trunc 21.0%) ; gap p2 (primary) 7.0 ± 3.4 pts ; tok/s 291.5 ; 4B/8B prompt token ids identical 20/20
  S2: problems 256/600 x 2 | base rate (greedy primary) 66.8% (main 95.2%, p1 80.0%, p2 50.0%, rc 59.5%) | sampled 69.5% | truncation 1.6% | mean tokens 330.3 | tok/s 229.8
  S3: layers 27 | C 0.01 | held-out AUROC 0.810 (cv 0.706) | recall 31.0% at held-out FPR 9.8% (tau 0.8803, eval-mix weighted y) | restart share 0.0% | gate fail
    held-out AUROC per set: main 0.385 (n 14, wrong 1) ; p1 nan (n 10, wrong 0) ; p2 0.857 (n 8, wrong 1) ; rc 0.762 (n 58, wrong 27)
    held-out FPR per set at tau: main 0.0% (n 14, wrong 1) ; p1 20.0% (n 10, wrong 0) ; p2 0.0% (n 8, wrong 1) ; rc 12.9% (n 58, wrong 27)
    eval (test traces) FPR per set at tau: n/a
    AUROC table (grouped 5-fold CV on train ids; held-out at each set's best C):
    27: C=0.001 cv 0.688 ; C=0.01 cv 0.706 held-out 0.810
(c) DATA SCALING (Qwen3-1.7B):
  S2: problems 576/1200 x 2 | base rate (greedy primary) 45.5% (main 67.4%, p1 48.3%, p2 36.4%, rc 30.6%) | truncation 0.8% | mean tokens 257.7 | tok/s 375.1
  S3: skipped: not enough pod time left
GATE (AUROC >= 0.80 and recall >= 0.40 at FPR 10% on held-out ids): b3 fail ; c3 n/a
BUDGET: max_minutes 90 x $0.82/h = $1.23 <= $1.5
BLOCKERS: C3 skipped: not enough pod time left
