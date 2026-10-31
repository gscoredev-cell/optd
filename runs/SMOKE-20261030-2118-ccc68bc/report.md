RUN: SMOKE-20261030-2118-ccc68bc | SMOKE | ccc68bc | 33tntixfi4nufh | NVIDIA L40 | 44.4 GB | US-KS-2 | $0.82/h | 65.2 min | $0.89
IMAGE PULL: 0.1 min (pod create -> bootstrap start) | cost of pull $0.00 | billed total est. 65.3 min $0.89
SSH: public port assigned yes (17005)
ENV: runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04 | torch 2.8.0.dev20250319+cu128 (cuda 12.8) | transformers 4.57.1 | peft 0.17.1 | Qwen3-1.7B / Qwen3-8B | cache hits 3 / downloads 0 | download MB/s n/a (all cache hits)
DEVICE: capability 8.9 | bf16 matmul ok (rel err 0.00166) | tier 48 | driver 595.91.07
TEMPLATES: identical token ids 20/20 | empty think block 20/20
SCORING: primary = lenient (strict answer, else last number), truncated = wrong ; strict = secondary
TIMING (min): S1 30.96 | S1rc 8.95 | S2 5.5 | S3 2.67 | S4 0.72 | S5 0.0 | S6 0.0 | S7 0.01 | S8 15.7 | total 65.19
HEADROOM primary/strict: student p2 (n=200) 30.5%/20.5% / main (n=40) 82.5%/62.5% / p1 (n=40) 55.0%/37.5% / rc (n=100) 13.0%/10.0% ; teacher p2 (n=200) 82.0%/81.0% / main (n=40) 97.5%/97.5% / p1 (n=40) 87.5%/87.5% / rc (n=100) 21.0%/23.0% ; gap p2 (primary) 51.5 ± 3.7 pts (paired SE) ; max_new_tokens tried [640, 800] -> 800 ; truncation 640 tokens: student_p2 2.0%, student_main 2.5%, student_p1 2.5%, student_rc 35.0%, teacher_p2 6.0%, teacher_main 0.0%, teacher_p1 0.0%, teacher_rc 53.0% ; 800 tokens: student_p2 1.5%, student_main 2.5%, student_p1 0.0%, student_rc 17.0%, teacher_p2 3.0%, teacher_main 0.0%, teacher_p1 0.0%, teacher_rc 29.0%
RC CALIBRATION (60 per level, 800 tokens, primary): 3 ops: student 31.7% (trunc 3.3%), teacher 58.3% (trunc 3.3%) ; 4 ops: student 33.3% (trunc 8.3%), teacher 43.3% (trunc 8.3%) ; 5 ops: student 6.7% (trunc 16.7%), teacher 21.7% (trunc 11.7%) ; choice for FULL eval and train prompts: none qualifies ops (rule: largest level with teacher >= 60% and truncation <= 5%)
ROLLOUTS: n 600 | primary acc greedy 64.7% / sampled 64.0% (strict greedy 54.0%) | mean tokens 191.4 | mean boundaries 8.64 | truncation 0.0%
PROBE: layers 21 | C 0.01 | AUROC held-out 0.747 (cv 0.770) | tau 0.9650424006910396 | FPR 9.8% | recall 29.8% | restart share 0.0% | first-flag histogram {2: 4, 4: 2, 5: 6, 6: 2, 7: 1, 8: 2, 9: 2, 11: 1} | held-out traces 108 (47 wrong) | features fp32 prefill, batch 8
PROBE AUROC TABLE (5-fold CV grouped by original_id on train ids; held-out at each set's best C):
  21: C=0.0001 cv 0.733 ; C=0.001 cv 0.757 ; C=0.01 cv 0.770 held-out 0.747 ; C=0.1 cv 0.742 ; C=1 cv 0.710
  28: C=0.0001 cv 0.726 ; C=0.001 cv 0.728 held-out 0.623 ; C=0.01 cv 0.714 ; C=0.1 cv 0.682 ; C=1 cv 0.666
  21+28: C=0.0001 cv 0.739 ; C=0.001 cv 0.759 held-out 0.663 ; C=0.01 cv 0.757 ; C=0.1 cv 0.735 ; C=1 cv 0.709
IDENTITY: A3-untrained == A2 y (n 20, max token diff 0)
STATE TOLERANCE (informational; max rel L2 vs generation states at 28 boundaries): batch-1 unpadded bf16 prefill max 0.02488 (layers 7: 0.00998, 14: 0.01869, 21: 0.02346, 28: 0.02460) ; fp32 batch-8 prefill max 0.02118 (layers 7: 0.00973, 14: 0.01339, 21: 0.01931, 28: 0.02083)
REPAIR DATA: skipped (probe gate failed)
TRAIN: skipped (probe gate failed)
EVAL (380) — adapter untrained (S7 skipped): A3 ~ A2, A6 ~ A0:
  A0: primary/strict p2 30.5%/20.5%, main 82.5%/62.5%, p1 55.0%/37.5%, rc 13.0%/10.0% ; recovered 0 ; broken 0 ; net 0 ; interv. rate 0.0% ; FPR 0.0% ; extra tokens 0.0 ; truncation 5.5%
  A2: primary/strict p2 32.0%/22.0%, main 82.5%/65.0%, p1 52.5%/37.5%, rc 13.0%/10.0% ; recovered 7 ; broken 5 ; net 2 ; interv. rate 30.3% ; FPR 18.6% ; extra tokens -1.0 ; truncation 4.5%
  A3: primary/strict p2 32.0%/22.0%, main 82.5%/65.0%, p1 52.5%/37.5%, rc 13.0%/10.0% ; recovered 7 ; broken 5 ; net 2 ; interv. rate 30.3% ; FPR 18.6% ; extra tokens -1.0 ; truncation 4.5%
  A6: primary/strict p2 30.5%/20.5%, main 82.5%/62.5%, p1 55.0%/37.5%, rc 13.0%/10.0% ; recovered 0 ; broken 0 ; net 0 ; interv. rate 0.0% ; FPR 0.0% ; extra tokens 0.0 ; truncation 5.5%
THROUGHPUT (HF generate, tok/s): S1 student 399.3 / teacher 258.1 | S2 greedy 358.1 / sampled 340.2 | S8 A6 245.6 | teacher prefill 4425.3 (SMOKE-1) | train steps/s 0.4729 (SMOKE-1)
PROJECTION: FULL 7.4 h and $6.10 per seed on this tier (rough; rollouts 2.1 h, features 0.4 h, continuations 1.4 h, teacher_scoring 0.3 h, training 0.3 h, headroom 0.6 h, arms 2.4 h; ceiling 15 h/pod)
GATES: headroom (gap >= 8 pts) pass ; base rate in [20%, 85%] pass ; probe AUROC >= 0.80 and recall >= 0.40 at FPR 10% fail ; identity pass ; resume n/a ; RC calibrated fail ; timing (<= 900 min/pod) pass
BUDGET: max_minutes 180 x $0.82/h = $2.46 <= $4
BLOCKERS: RandomCalculation: no ops level with teacher >= 60% and truncation <= 5%; probe gate failed: FULL not cleared (decide probe-mc labels vs Qwen3-4B student)
