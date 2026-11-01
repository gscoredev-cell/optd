RUN: SMOKE9-20261101-0143-4e80d6d | SMOKE9 | 4e80d6d | es0lklklkopf52 | NVIDIA A100-SXM4-80GB | 79.3 GB | US-KS-2 | $1.79/h | 64.6 min | $1.93
IMAGE PULL: 78.9 min (pod create -> bootstrap start) | cost of pull $2.35 | billed total est. 143.5 min $4.28
SSH: public port assigned yes (32842)
ENV: runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04 | torch 2.8.0.dev20250319+cu128 (cuda 12.8) | transformers 5.22.1 | peft 0.21.2 | Qwen3.5-9B /  SMOKE-8 adapter re-evaluated | cache hits 4 / downloads 0 | download MB/s n/a (all cache hits)
DEVICE: capability 8.0 | bf16 matmul ok (rel err 0.00166) | tier 80 | driver 595.91.07
TEMPLATES: identical token ids n/a/n/a | empty think block n/a/n/a
METHOD: cap 3200 everywhere; SMOKE-8 run SMOKE8-20261031-2328-44222e6 re-evaluated (rkl adapter, no training) and a GRPO adapter (reward = primary correctness, groups of 4, DAPO dynamic sampling, no KL, clip 0.2, lr 5e-5) on the same units; no hint arms
TIMING (min): Y0 1.13 | Y1 21.56 | Y2 0.0 | Y3 45.77 | Y4 17.86 | Y5 0.0 | total 64.62
Y0 ENV: transformers 5.22.1 | class Qwen3_5ForConditionalGeneration | load 68 s | VRAM 17.5 GB | fast-path warning none | log-prob batch 4 | micro 2 | cache hits/misses grpo_base_rc3 240/0 ; rkl_base_rc3 145/95 ; y1_rc4cal 12/48
SMOKE-8 AT CAP 800 (for comparison): pass@1 pooled base 56.2% -> A6 74.2% (+17.9 [+9.6, +26.2], n 240) | E0 31.9% (658 tok, trunc 55.3%) vs E6 78.7% (219 tok, trunc 0.0%), E6-E0 +46.8 [+27.7, +63.8] (n 47) | harm adapter 21.7% vs plain 21.7% | sampled-trace truncation 38.6%, mean tokens 622
Y1 SMOKE-8 rkl ADAPTER AT CAP 3200:
  pass@1 rc3 (n 240): base 84.6% [79.6%, 89.2%] -> A6 75.4% [69.6%, 80.4%] | delta -9.2 [-16.2, -2.1] | tokens base/A6 987/196 | truncation base/A6 7.5%/0.0%
  pass@1 pooled (n 240): base 84.6% [79.6%, 89.2%] -> A6 75.4% [69.6%, 80.4%] | delta -9.2 [-16.2, -2.1] | tokens base/A6 987/196 | truncation base/A6 7.5%/0.0%
  E0 vs E6 rc3 (n 47): E0 72.3% (896 tok, trunc 6.4%) ; E6 85.1% (221 tok, trunc 0.0%) | E6-E0 +12.8 [-4.3, +29.8]
  E0 vs E6 pooled (n 47): E0 72.3% (896 tok, trunc 6.4%) ; E6 85.1% (221 tok, trunc 0.0%) | E6-E0 +12.8 [-4.3, +29.8]
  HARM (n 60): adapter 20.0% vs plain 11.7% | per set rc3 20.0% vs 11.7%
  RC4 CALIBRATION at 3200 (student greedy, n 60): accuracy 75.0% | truncation 18.3% | tokens 1375 | in [35%, 70%]: no (at 800: 18.3%)
Y2 RC4 DATA: not needed: RC4 not in the accuracy band at 3200
Y3 GRPO: units 162 {'rc3': 162} | dev 30 {'rc3': 30} | rounds 1 | executed steps 6 | best round 1 step 5 dev E6 recovery 86.7% | early stopping fired n | resume y (exact) (max rel diff 0.000000) | train 2662 s (dev evals 296 s, sampling + log-probs 2322 s) | steps/s pure 0.1352 | adapter params 43278336
  round 1: reused SMOKE-8 continuations 364 (dropped 284 truncated at 800) | groups 147 kept 26 (dropped all-correct 85, all-wrong 4, incomplete 32) -> examples 80 | steps 5 | loss start/end -0.2281/0.1849 | dev per eval (step: rate) 0: 76.7% ; 5: 86.7% | best step 5 | stopped early n
  round 2: stage time budget (sampled, training skipped)
Y4 GRPO ADAPTER AT CAP 3200:
  pass@1 rc3 (n 240): base 84.6% [79.6%, 89.2%] -> A6 81.7% [76.7%, 86.7%] | delta -2.9 [-7.9, +2.1] | tokens base/A6 987/928 | truncation base/A6 7.5%/7.9%
  pass@1 pooled (n 240): base 84.6% [79.6%, 89.2%] -> A6 81.7% [76.7%, 86.7%] | delta -2.9 [-7.9, +2.1] | tokens base/A6 987/928 | truncation base/A6 7.5%/7.9%
  arms: deadline before E0 on rc3
  HARM (n 0): adapter nan vs plain nan | per set rc3 nan vs nan
SUMMARY AT CAP 3200 (pass@1 base / A6 rkl / A6 grpo ; E6-E0 rkl / grpo ; harm rkl / grpo):
  rc3: pass@1 base 84.6% | A6 rkl 75.4% (-9.2 [-16.2, -2.1]) | A6 grpo 81.7% (-2.9 [-7.9, +2.1]) ; E6-E0 rkl +12.8 [-4.3, +29.8] | grpo nan [nan, nan] ; harm rkl 20.0% / grpo nan (plain 11.7%)
  pooled: pass@1 base 84.6% | A6 rkl 75.4% (-9.2 [-16.2, -2.1]) | A6 grpo 81.7% (-2.9 [-7.9, +2.1]) ; E6-E0 rkl +12.8 [-4.3, +29.8] | grpo nan [nan, nan] ; harm rkl 20.0% / grpo nan (plain 11.7%)
BREVITY: of SMOKE-8's gain at cap 800, the part that disappears at 3200 with the same adapter = E6-E0 +46.8 - +12.8 = +34.0 points ; A6 pass@1 +17.9 - -9.2 = +27.1 points
THROUGHPUT: base greedy tok/s 360.5 | A6 tok/s 325.0 | GRPO sampling tok/s 194.1 | train steps/s pure 0.1352
PROJECTION: FULL at cap 3200 39.7 h and $71.12 per seed on this tier (rough; sampling_4x 10.0 h, continuations_3_rounds 21.5 h, logprobs 1.2 h, training 1.4 h, dev_evals 0.3 h, arms_and_harm 1.7 h, a6_and_base 3.6 h)
GATES: env (fast path, model loaded) pass ; re-evaluation complete (rkl pass@1, arms, harm) pass ; pipeline (resume, early stop fired) fail ; rkl A6 > base at 3200 (CI lower bound > 0) fail ; grpo A6 > base at 3200 (CI lower bound > 0) fail ; harm (each adapter <= 1.5 x plain) fail ; timing (FULL <= 12 h per seed at 3200) fail
BUDGET: max_minutes 150 x $1.79/h = $4.47 <= $6
BLOCKERS: none
