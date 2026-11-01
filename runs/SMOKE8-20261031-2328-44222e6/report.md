RUN: SMOKE8-20261031-2328-44222e6 | SMOKE8 | 44222e6 | g4p5clifxbxozq | NVIDIA A100-SXM4-80GB | 79.3 GB | US-KS-2 | $1.79/h | 78.0 min | $2.33
IMAGE PULL: 0.0 min (pod create -> bootstrap start) | cost of pull $0.00 | billed total est. 78.0 min $2.33
SSH: public port assigned yes (15002)
ENV: runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04 | torch 2.8.0.dev20250319+cu128 (cuda 12.8) | transformers 5.22.1 | peft 0.21.2 | Qwen3.5-9B / Qwen3.5-9B + reference | cache hits 1 / downloads 1 | download MB/s 131.7
DEVICE: capability 8.0 | bf16 matmul ok (rel err 0.00166) | tier 80 | driver 595.91.07
TEMPLATES: identical token ids n/a/n/a | empty think block n/a/n/a
METHOD: Qwen3.5-9B student; self-teacher = same weights with a 'Reference solution' system line (dataset answer / RC evaluation steps); training units = contested train problems, restart from the prompt, 4 continuations with the adapter on over the whole sequence; rkl + PPO clip 0.2, lr 5e-5, dev 30 traces every 10 steps, <= 3 rounds; arms E0 / E2 (hint) / E6 (adapter) / E6h (adapter + hint); A6 = adapter always on, greedy
TIMING (min): Z0 0.88 | Z1 4.19 | Z2 28.26 | Z3 2.77 | Z4 31.99 | Z5 3.47 | Z6 2.19 | Z7 0.0 | total 77.99
Z0 ENV: transformers 5.22.1 (>= 5: y) | fla 0.6.0 | causal_conv1d 1.7.0 | triton 3.3.0 | peft 0.21.2 | class Qwen3_5ForConditionalGeneration | params 9.41 B | load 4 s | VRAM after load 17.5 GB | fast-path warning none | module flags {} | load warnings []
  TEMPLATES (20 prompts, student vs teacher variant, enable_thinking=False): common suffix >= user turn 20/20 (min common suffix 81 tokens, user tokens mean 69.8) | think block in render 20/20 | hint round-trips True | prompt tail ' or bowls?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n' | smoke generation 32 tokens in 32.9 s
Z1 CALIBRATION (student greedy, 60 per set; keep accuracy in [35%, 70%]): p1 88.3% (strict 88.3%, trunc 0.0%, n 60, 284 tok) ; p2 83.3% (strict 83.3%, trunc 3.3%, n 60, 419 tok) ; rc3 63.3% (strict 63.3%, trunc 35.0%, n 60, 650 tok) ; rc4 18.3% (strict 18.3%, trunc 80.0%, n 60, 766 tok) ; rc5 6.7% (strict 6.7%, trunc 90.0%, n 60, 781 tok) ; rc6 6.7% (strict 6.7%, trunc 93.3%, n 60, 790 tok) | chosen ['rc3'] | tok/s 881.5
Z2 DATA (4 samples per problem, T 0.8, cap 800; greedy on test): problems 640/640 (sizes train {'rc3': 400}, test {'rc3': 240}) | tok/s 1031.4 | truncation 38.6% | mean tokens 621.9 | test units (wrong greedy of contested test) 47 | harm traces 60 | cache hits/misses z1 0/360 ; z2 0/2800
  test_all: problems 240 | #correct of 4 [64, 29, 30, 44, 73] | accuracy 53.4% | contested 42.9% | greedy acc 56.2% | wrong-greedy contested 47
  test_rc3: problems 240 | #correct of 4 [64, 29, 30, 44, 73] | accuracy 53.4% | contested 42.9% | greedy acc 56.2% | wrong-greedy contested 47
  train_all: problems 400 | #correct of 4 [78, 59, 70, 63, 130] | accuracy 56.8% | contested 48.0%
  train_rc3: problems 400 | #correct of 4 [78, 59, 70, 63, 130] | accuracy 56.8% | contested 48.0%
Z3 SELF-TEACHER: teacher greedy accuracy with the reference 96.9% (n 192; rc3 96.9%) | teacher tok/s 286.0 | mean per-token log-prob gap teacher - student: correct traces -0.048 (n 100), wrong traces -0.050 (n 100) | student mean log-prob correct/wrong -0.071/-0.076 | prefill tok/s 5742.7
Z4 TRAIN: units 162 {'rc3': 162} | dev 30 {'rc3': 30} | rounds 2 | executed steps 86 | best round 2 step 10 dev E6 recovery 90.0% | early stopping fired y | resume y (kill at 15, ckpt every 10, max rel diff 0.000061) | train 1914 s (dev evals 343 s, resampling + scoring 1168 s) | steps/s pure 0.2129 (overall 0.0449) | adapter params 43278336 | micro 4
  round 1: continuations 648 (primary 48.9%, trunc 43.8%, teacher-also-wrong 3.7%, mean adv -0.043) | steps 41 | loss start/end 0.0403/0.0260 | dev per eval (step: rate) 0: 46.7% ; 10: 26.7% ; 20: 50.0% ; 30: 50.0% ; 40: 46.7% ; 41: 50.0% | best step 20 | stopped early n
  round 2: continuations 648 (primary 46.5%, trunc 0.9%, teacher-also-wrong 3.7%, mean adv -0.277) | steps 41 | loss start/end 0.0765/0.0791 | dev per eval (step: rate) 0: 50.0% ; 10: 90.0% ; 20: 73.3% ; 30: 76.7% ; 40: 70.0% | best step 10 | stopped early y
  round 3: stage time budget
Z5 ARMS (test units = wrong greedy traces of contested test problems; one sampled continuation per arm, identical batches and seeds; adapter trained; arms done ['E0', 'E2', 'E6', 'E6h']):
  pooled (n 47, problems 47): E0 31.9% (658 tok, trunc 55.3%) ; E2 4.3% (794 tok, trunc 95.7%) ; E6 78.7% (219 tok, trunc 0.0%) ; E6h 59.6% (590 tok, trunc 25.5%)
  rc3 (n 47, problems 47): E0 31.9% (658 tok, trunc 55.3%) ; E2 4.3% (794 tok, trunc 95.7%) ; E6 78.7% (219 tok, trunc 0.0%) ; E6h 59.6% (590 tok, trunc 25.5%)
  HARM (correct greedy test traces resampled from the prompt, n 60): adapter 21.7% vs plain 21.7%
  DELTAS (points; 95% CI by cluster bootstrap on original_id, 10,000 resamples):
    pooled (n 47): E6-E0 +46.8 [+27.7, +63.8] | E2-E0 -27.7 [-42.6, -14.9] | E6h-E0 +27.7 [+10.6, +44.7] | E6h-E6 -19.1 [-36.2, -2.1] | E6-E2 +74.5 [+59.6, +87.2]
    rc3 (n 47): E6-E0 +46.8 [+27.7, +63.8] | E2-E0 -27.7 [-42.6, -14.9] | E6h-E0 +27.7 [+10.6, +44.7] | E6h-E6 -19.1 [-36.2, -2.1] | E6-E2 +74.5 [+59.6, +87.2]
Z6 PASS@1 (greedy on all test problems, base vs adapter always on): pooled base 56.2% -> A6 74.2% (+17.9 [+9.6, +26.2], n 240) ; rc3 base 56.2% -> A6 74.2% (+17.9 [+9.6, +26.2], n 240) | A6 tok/s 356.6 | A6 truncation 0.4%
THROUGHPUT: sampling tok/s 1031.4 | adapter gen tok/s 513.4 | teacher prefill tok/s 8559.1 | train steps/s pure 0.2129
PROJECTION: FULL 14.2 h and $25.34 per seed on this tier (rough; continuations_3_rounds 6.1 h, teacher_scoring 1.6 h, training 1.5 h, dev_evals 0.9 h, arms_and_harm 0.8 h, a6 0.5 h, sampling_4x 2.8 h)
GATES: env (transformers >= 5, fast path, template suffix 20/20) pass ; calibration (>= 1 set in band) pass ; pipeline (resume, early stop fired) pass ; teacher (accuracy with reference >= 90%) pass ; signal (E6-E0 pooled CI lower bound > 0) pass ; harm (adapter <= 1.5 x plain) pass ; pass@1 (A6 >= base pooled) pass ; timing (FULL <= 12 h per seed) fail
BUDGET: max_minutes 150 x $1.79/h = $4.47 <= $6
BLOCKERS: none
