RUN: ENVCHECK-20261030-1716-8283630 | ENVCHECK | 8283630 | yscfrcl7sd4hmp | NVIDIA RTX PRO 4500 Blackwell Server Edition | 31.4 GB | US-KS-2 | $0.72/h | 3.9 min | $0.05
IMAGE PULL: 3.2 min (pod create -> bootstrap start) | cost of pull $0.04 | billed total est. 7.1 min $0.09
SSH: public port assigned yes (18925)
ENV: runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04 | torch 2.8.0.dev20250319+cu128 (cuda 12.8) | transformers 4.57.1 | peft 0.17.1 | Qwen3-1.7B / Qwen3-8B | cache hits 3 / downloads 0 | download MB/s n/a (all cache hits)
DEVICE: capability 12.0 | bf16 matmul ok (rel err 0.00166) | tier 32 | driver 595.91.07
TEMPLATES: identical token ids 20/20 | empty think block 20/20
TIMING (min): bootstrap 3.3 | env 0.39 | templates 0.01 | gen 0.17 | teacher_prefill 0.06 | total 3.93
THROUGHPUT: gen tok/s 963.3 (batch 24, 8359 tokens, truncated 8/24, peak 4.7 GB) | teacher prefill tok/s 5635.7 (batch 12, 5732 tokens, peak 18.0 GB, mean logp -0.333)
DOWNLOADS: Qwen/Qwen3-1.7B cache hit ; Qwen/Qwen3-8B cache hit ; apple/GSM-Symbolic cache hit
BUDGET: max_minutes 30 x $0.72/h = $0.36 <= $1
BLOCKERS: none
