RUN: SMOKE-20261030-1807-6462834 | SMOKE | 6462834 | kq1nmq2mma7deb | NVIDIA L40 | 44.4 GB | US-KS-2 | $0.82/h | 55.1 min | $0.75
IMAGE PULL: 0.1 min (pod create -> bootstrap start) | cost of pull $0.00 | billed total est. 55.1 min $0.75
SSH: public port assigned yes (18076)
ENV: runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04 | torch 2.8.0.dev20250319+cu128 (cuda 12.8) | transformers 4.57.1 | peft 0.17.1 | Qwen3-1.7B / Qwen3-8B | cache hits 3 / downloads 0 | download MB/s n/a (all cache hits)
DEVICE: capability 8.9 | bf16 matmul ok (rel err 0.00166) | tier 48 | driver 595.91.07
TEMPLATES: identical token ids 20/20 | empty think block 20/20
TIMING (min): S1 31.86 | S2 5.89 | S3 0.38 | S4 0.65 | S5 1.03 | S6 1.52 | S7 0.22 | S8 12.61 | total 55.06
HEADROOM strict/lenient: student p2 (n=200) 20.5%/30.5% / main (n=40) 62.5%/82.5% / p1 (n=40) 37.5%/55.0% / rc (n=100) 10.0%/14.0% ; teacher p2 (n=200) 81.0%/82.0% / main (n=40) 97.5%/97.5% / p1 (n=40) 87.5%/87.5% / rc (n=100) 23.0%/24.0% ; gap p2 60.5 ± 3.5 pts (paired SE) ; max_new_tokens tried [640, 800] -> 800 ; truncation 640 tokens: student_p2 2.0%, student_main 2.5%, student_p1 2.5%, student_rc 35.0%, teacher_p2 6.0%, teacher_main 0.0%, teacher_p1 0.0%, teacher_rc 53.0% ; 800 tokens: student_p2 1.5%, student_main 2.5%, student_p1 0.0%, student_rc 17.0%, teacher_p2 3.0%, teacher_main 0.0%, teacher_p1 0.0%, teacher_rc 29.0%
ROLLOUTS: n 600 | strict acc greedy 54.0% / sampled 50.3% | mean tokens 191.5 | mean boundaries 8.57 | truncation 0.0%
PROBE: layer 28 | AUROC 0.746 (by layer {'14': 0.569, '21': 0.717, '28': 0.746}) | tau 1.0000 | FPR 10.0% | recall 24.1% | restart share 0.0% | first-flag histogram {2: 14, 3: 3, 7: 1, 8: 1} | held-out traces 108
IDENTITY: A3-untrained == A2 y (n 20, max token diff 0) ; prefill-vs-generation state tolerance 0.03683 (max rel L2, limit 1e-2)
REPAIR DATA: flagged-wrong 22 (used 22) | continuations 88 (strict 6.8%, truncation 0.0%) | teacher-also-wrong 36.4%
TRAIN: steps 6 | loss start/end -0.0688/-0.7930 | resume check y (kill at 3, checkpoint every 2, max rel diff 0.000000) | adapter params 17432576
EVAL (380):
  A0: strict p2 20.5%, main 62.5%, p1 37.5%, rc 10.0% ; recovered 0 ; broken 0 ; net 0 ; interv. rate 0.0% ; FPR 0.0% ; extra tokens 0.0 ; truncation 5.5%
  A2: strict p2 22.0%, main 62.5%, p1 37.5%, rc 10.0% ; recovered 4 ; broken 1 ; net 3 ; interv. rate 9.5% ; FPR 4.4% ; extra tokens 0.1 ; truncation 5.5%
  A3: strict p2 22.5%, main 65.0%, p1 40.0%, rc 11.0% ; recovered 7 ; broken 0 ; net 7 ; interv. rate 9.5% ; FPR 4.4% ; extra tokens -0.8 ; truncation 5.0%
  A6: strict p2 37.0%, main 82.5%, p1 57.5%, rc 13.0% ; recovered 65 ; broken 13 ; net 52 ; interv. rate 0.0% ; FPR 0.0% ; extra tokens 7.6 ; truncation 6.3%
THROUGHPUT: gen tok/s 324.9 (S2, batch incl. sampling) | teacher prefill tok/s 4425.3 | train steps/s 0.4729
PROJECTION: FULL 12.6 h and $10.37 per seed on this tier (rough linear estimate; ceiling 15 h/pod)
GATES (informational in SMOKE): headroom (gap >= 8 pts) pass ; base rate in [20%, 85%] pass ; probe AUROC >= 0.80 and recall >= 0.40 fail ; identity pass ; resume pass ; timing (<= 900 min/pod) pass
BUDGET: max_minutes 180 x $0.82/h = $2.46 <= $4
BLOCKERS: prefill-vs-generation state tolerance 0.0368 > 1e-2
