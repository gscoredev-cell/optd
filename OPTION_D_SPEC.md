# Option D: detector-gated repair — specification for agents

Version 1.5 · 2026-10-30 · audience: Claude Code running locally on the human's Windows machine (Git Bash + pwsh 7.6); no human reader assumed.
Changes from 1.4 (after verifying `runpodctl 3.0.1 pod create --help` and `gpu list` on the human's machine): there is **no `--terminate-after`** flag, so the hard spending cap moves into the pod (deadline watchdog with the pod-scoped key) plus an independent local Windows scheduled task; `--wait` is used to block until SSH is reachable; GPU list is filtered by the volume's datacenter (`US-KS-2`), where the current stock is the 32 GB RTX PRO 4500 Blackwell SE at $0.72/h → new tier 32; prices updated from the live list. Volume `5l0v4bx88l`, endpoint `https://s3api-us-ks-2.runpod.io`.
Changes from 1.3: Qwen3 dense is the primary pair (Llama optional second pair); offline-equivalent rollback (no KV cropping); adapter-off prefixes; results and polling through the network volume's S3 API (pod terminates at DONE); splits by `original_id`; A6 plain-distillation control; budget-derived pod lifetime; integer minutes; restart-loop guard; pod-scoped API key; template-clustered statistics; tightened Claude Code permissions.

## 0. Purpose

Test one hypothesis: **a small model can learn to recover from its own reasoning slips if an external detector decides *when* and a trained adapter learns only *how*.**
Prior work (2026) shows the detector exists (hidden-state probe, ~0.95 AUROC) and that inference-time fixes without training fail. Training the repair conditioned on the detector is untested.

Phases: **ENVCHECK** (≤ 20 min, validates the cloud path and throughput), **SMOKE** (~1 h, pipeline check only — its accuracy numbers carry no evidential weight), **FULL** (only after a human approves the SMOKE report). Never start FULL unasked.

## 1. Roles and places

| Role | Where | Reaches |
| --- | --- | --- |
| Orchestrator | Claude Code, local Windows, commands via Git Bash; scripts in pwsh 7.3+ | RunPod REST/`runpodctl`; the volume's S3 API (`aws` CLI, profile `runpod`); pods over SSH only for debugging |
| Worker | RunPod Secure Cloud GPU pod, pinned `runpod/pytorch` image, Linux | Internet; reads/writes the volume at `/workspace`; terminates itself at DONE |
| Storage | Network volume `optd-vol` in one US datacenter with S3 API (`US-KS-2` or `US-CA-2`) | `/workspace/models`, `/workspace/hf`, `/workspace/venv`, `/workspace/runs/<RUN_ID>`; readable without a pod via `https://s3api-<dc>.runpod.io/` |
| Results | `runs/<RUN_ID>/` on the volume → `aws s3 sync` to `runs\<RUN_ID>\` locally | Survive pod termination and local sleep |

Secrets: `RUNPOD_API_KEY` (account; local env var only) is used by the orchestrator to create/list/delete pods. `HF_TOKEN` is a RunPod Secret referenced in pod env as `{{ RUNPOD_SECRET_HF_TOKEN }}`, so `pod get` shows the reference, not the value. The pod self-terminates with the **pod-scoped** `RUNPOD_API_KEY` RunPod injects; the account key never enters a pod. S3 access key/secret live only in the local `aws` profile `runpod`.

## 2. Human pre-flight (once)

- Windows: Git for Windows (Claude Code's Bash), **pwsh 7.3+** (`winget install Microsoft.PowerShell`), built-in OpenSSH, **AWS CLI v2** (`winget install Amazon.AWSCLI`), `runpodctl` (GitHub `runpod/runpodctl` releases) on PATH; `runpodctl config --apiKey $env:RUNPOD_API_KEY`.
- RunPod console: API key (read/write) → `setx RUNPOD_API_KEY ...`. Settings → SSH Public Keys ← `ssh-keygen -t ed25519` public key. Settings → Secrets: `HF_TOKEN` (and `GIT_READ_TOKEN` if the repo is private). Settings → S3 API Keys: create one; locally `aws configure --profile runpod` with its access key (`user_...`) and secret (`rps_...`), region `US-KS-2` (or `US-CA-2`), output json.
- Storage: network volume `optd-vol` = `5l0v4bx88l`, 60 GB, `US-KS-2`, S3 endpoint `https://s3api-us-ks-2.runpod.io` (bucket = volume id, region `us-ks-2`). Already created. Billed ≈ $4.20/month while it exists; delete it when the project ends.
- Local watchdog (independent of Claude and of the pod): after Claude writes `orchestrator/watchdog.ps1` and `orchestrator/register_watchdog.ps1`, the human runs the latter once in pwsh; it registers a Windows scheduled task that runs `watchdog.ps1` every 15 minutes while the PC is on. The human can inspect it in Task Scheduler (`OptD-RunPod-Watchdog`) and delete it when the project ends.
- Credits ≥ $40 (phase budgets total ≈ $35).
- Hugging Face: nothing to accept for Qwen3. Only if the Llama second pair is used: accept both Llama licenses.

## 3. Repository layout

```
OPTION_D_SPEC.md
CLAUDE.md                         §15
.claude/settings.json             §15
orchestrator/gputypes.ps1         candidate GPUs for the volume's datacenter + current Secure $/h (from `runpodctl gpu list`)
orchestrator/launch.ps1           create pod (§4.1) with --wait; writes runs/<RUN_ID>/pod.json
orchestrator/watch.ps1            S3 polling; prints only on state change; exits on DONE/FAILED/timeout
orchestrator/pull.ps1             aws s3 sync runs/<RUN_ID> → local
orchestrator/kill.ps1             delete pod by id or all optd-*
orchestrator/watchdog.ps1         deletes any optd-* pod whose deadline (encoded in its name) has passed; run by a Windows scheduled task
orchestrator/register_watchdog.ps1  one-time: registers the scheduled task (human runs it)
pod/bootstrap.sh                  §5
pod/requirements.txt, pod/constraints.txt (pins torch to the image's version)
pipeline/run_phase.py             one long-lived process per phase; loads each model once (§13)
pipeline/*.py                     stage modules imported by run_phase.py
config/gpu_tiers.yaml, config/run_defaults.yaml
runs/<RUN_ID>/                    local copies (git-ignored except report.md, metrics.json, JUDGEMENT.md)
```

## 4. RunPod orchestration (local, pwsh 7.3+)

### 4.1 Create (`launch.ps1 -Phase <P> [-Seed n]`)

1. `RUN_ID = <P>-<yyyymmdd-hhmm>-<sha7>[-s<seed>]`; `SHA` = current commit (pushed). Refuse to launch with uncommitted changes.
2. `gputypes.ps1`: parse `runpodctl gpu list` (JSON). Keep entries with `secureCloud: true` and `memoryInGb ≥ 20`. Prefer entries whose `dataCenterAvailability` includes the volume's datacenter; entries with no `dataCenterAvailability` field are tried after those (availability unknown). Order within: 48 GB (A40, RTX A6000, L40, L40S) → 32 GB (RTX PRO 4500 Blackwell SE) → 24 GB (RTX 4090, L4, PRO 6000 MIG 24GB, RTX 3090) → 80 GB (A100 PCIe, A100 SXM). Print the ordered table with `securePricePerHr`. Snapshot on 2026-10-30: only `NVIDIA RTX PRO 4500 Blackwell Server Edition` (32 GB, $0.72/h) listed `US-KS-2`; stock changes hourly, so the launcher always re-queries.
3. Lifetime: `MAX_MINUTES = min(phase_ceiling, floor(phase_budget / securePricePerHr × 60))`, ceilings 30 / 180 / 900, budgets $1 / $4 / $10 per pod (FULL per seed). `DEADLINE_EPOCH = now + (MAX_MINUTES + 20) × 60`. Encode it in the pod name: `optd-<RUN_ID>-d<DEADLINE_EPOCH>`. Pass `MAX_MINUTES` and `DEADLINE_EPOCH` in env (integers).
4. `runpodctl pod create --name <name> --cloud-type SECURE --gpu-id "<id>" --network-volume-id 5l0v4bx88l --image <pinned tag> --min-cuda-version 12.8 --container-disk-in-gb 30 --ports "22/tcp" --ssh --env '<json>' --docker-args '<start cmd>' --wait --wait-timeout 10m -o json`. Never combine with `--template-id`. Build the env JSON with `ConvertTo-Json -Compress` and pass it as one argument (pwsh 7.6 quotes native args correctly). On a capacity error, try the next GPU in the list; after the list is exhausted, wait 15 min and retry the list up to 4×, then report `NO_CAPACITY`. If `--wait` times out, the pod is kept and the error carries its id: run `kill.ps1 <id>` and treat as `FAILED: no ssh`.
5. Env: `RUN_ID`, `PHASE`, `SEED`, `GIT_REPO`, `GIT_SHA`, `MAX_MINUTES`, `DEADLINE_EPOCH`, `GPU_TIER_OVERRIDE` (empty), `HF_TOKEN={{ RUNPOD_SECRET_HF_TOKEN }}`, `GIT_READ_TOKEN={{ RUNPOD_SECRET_GIT_READ_TOKEN }}` (private repo only), `PUBLIC_KEY` and `SSH_PUBLIC_KEY` = your public key.
6. Start command (public repo): `bash -lc "curl -fsSL https://raw.githubusercontent.com/<owner>/<repo>/$GIT_SHA/pod/bootstrap.sh | bash"` — SHA in the URL, never a branch (branch heads are cached). Private repo: `BOOTSTRAP_B64` in env, start `bash -lc 'echo "$BOOTSTRAP_B64" | base64 -d | bash'` (single quotes so nothing expands locally).
7. `--wait` returns the pod as `pod get` does once SSH answers; record `pod.json` {id, name, gpu, price, dc, ip, port, created, max_minutes, deadline_epoch}. Print GPU and $/h and the deadline in local time.

### 4.2 Watch (`watch.ps1 -RunId <id> -TimeoutMinutes <n>`)

- Every 5 min: `aws --profile runpod s3 cp s3://<volume_id>/runs/<RUN_ID>/status.json - --endpoint-url https://s3api-<dc>.runpod.io/ --region <dc>`; print only when `state`/`step` changes; on change also print the last 10 lines of `log.txt` (S3 `cp` to temp). Set `AWS_MAX_ATTEMPTS=10` (S3 API returns occasional 502).
- Exits 0 on `DONE`, 2 on `FAILED`, 3 on timeout. Claude runs it in the foreground with a long Bash timeout (§15 sets `BASH_MAX_TIMEOUT_MS`), then acts. No `/loop`; one wake-up per state change.
- Unreachable rule: if `status.json` has not changed for 30 min while `runpodctl pod get` says RUNNING → `kill.ps1 <id>`, state `FAILED: stalled`.
- SSH (`ssh -p <port> -o UserKnownHostsFile=runs/<RUN_ID>/known_hosts -o StrictHostKeyChecking=accept-new root@<ip>`) is for debugging only.

### 4.3 Pull and verify (`pull.ps1`)

`aws s3 sync s3://<volume_id>/runs/<RUN_ID> runs/<RUN_ID>` (report.md, metrics.json, status.json, log.txt, env.json, pod.json; adapters stay on the volume). Then `runpodctl pod list`; if the pod still exists → `kill.ps1 <id>` and record `SELF_TERMINATE_FAILED`.

### 4.4 Kill switch (`kill.ps1`) and watchdog (`watchdog.ps1`)

- `kill.ps1 <id>` → `runpodctl pod delete <id>` (volume-attached pods can only be terminated, not stopped). `kill.ps1 --all` deletes every pod named `optd-*`. Run `--all` at session start and end.
- `watchdog.ps1`: `runpodctl pod list -o json`; for each pod whose name matches `optd-.*-d(\d+)$`, delete it if `now ≥ deadline` or if `now ≥ created + 24h` (malformed name). Also delete any `optd-*` pod whose `runs/<RUN_ID>/status.json` on S3 says DONE/FAILED for more than 15 min. Logs to `orchestrator/watchdog.log`. Needs `RUNPOD_API_KEY` from the user environment and the `runpod` aws profile; no Claude session involved.
- `register_watchdog.ps1` (human runs once): `Register-ScheduledTask -TaskName OptD-RunPod-Watchdog` with a 15-minute repeating trigger running `pwsh -ExecutionPolicy Bypass -File <abs path>\watchdog.ps1`, run whether the user is logged on or not is **not** required (user session is fine); prints the task it created.

### 4.5 Cost guard

Secure Cloud $/h from `runpodctl gpu list` on 2026-10-30: RTX PRO 4500 Blackwell SE (32 GB) $0.72, A40 $0.59, RTX A6000 $0.59, L40 $0.82, L40S $1.09, RTX 4090 $0.89, L4 $0.59, RTX 3090 $0.50, A100 PCIe/SXM $1.79. Volume ≈ $4.20/month. Four independent stops, none of which is `--terminate-after` (absent in runpodctl 3.0.1): (1) in-pod deadline watchdog using the pod-scoped key (§5 step 0), (2) pod `timeout` + self-terminate at DONE/FAILED, (3) `watch.ps1` stall rule while Claude is active, (4) the local Windows scheduled `watchdog.ps1` every 15 min. Every report states price × elapsed. Expected on the 32 GB card: ENVCHECK ≈ $0.25, SMOKE ≈ $1.2, FULL ≈ $7 per seed.

## 5. Pod bootstrap (`pod/bootstrap.sh`)

```
0  /start.sh &                      # keep the image's sshd + key setup alive
   set -euo pipefail; R=/workspace/runs/$RUN_ID; mkdir -p $R
   deadline watchdog (background, first thing, uses the pod-scoped RUNPOD_API_KEY and the preinstalled runpodctl):
      while [ $(date +%s) -lt $DEADLINE_EPOCH ]; do sleep 60; done; runpodctl pod delete $RUNPOD_POD_ID
      (DEADLINE_EPOCH is absolute, so a container restart does not reset the clock)
   if status.json exists with state DONE|FAILED: retry self-terminate 5×; sleep infinity   # restart-loop guard
1  heartbeat loop (background): every 5 min write status.json atomically (tmp + mv)
   trap EXIT: write final status (DONE if rc==0 else FAILED, first error line); sync; self-terminate:
      runpodctl pod delete $RUNPOD_POD_ID  ||  curl -X DELETE https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID -H "Authorization: Bearer $RUNPOD_API_KEY"
      (pod-scoped key injected by RunPod); retry 5× with 30 s sleep; then sleep infinity (terminate-after ends it)
2  GPU: nvidia-smi name,memory.total → GPU_TIER = largest tier ≤ VRAM (§6) unless GPU_TIER_OVERRIDE; write env.json
3  code to container disk, never the volume: mkdir /root/repo && cd /root/repo && git init && git remote add origin $GIT_REPO
   && git fetch --depth 1 origin $GIT_SHA && git checkout FETCH_HEAD   (private: token in the remote URL from GIT_READ_TOKEN)
4  venv on the volume keyed by image tag: /workspace/venv/<image_tag_slug>; create with --system-site-packages if missing;
   pip install -r pod/requirements.txt -c pod/constraints.txt   (constraints pin torch==<image torch>)
   export HF_HOME=/workspace/hf
5  models: flock /workspace/.dl.lock; for each repo: skip if /workspace/models/<name>/.complete; else delete partial dir,
   hf download, verify org + size, write .complete; log "cache hit"/"downloaded MB/s". Dataset via HF cache (same lock).
6  timeout $((MAX_MINUTES*60 - 600)) python pipeline/run_phase.py --phase $PHASE --run-id $RUN_ID --seed $SEED --tier $GPU_TIER
7  write report.md, metrics.json; state DONE; exit 0 → trap terminates the pod
```
Secrets are masked in `log.txt` (`HF_TOKEN`, `RUNPOD_API_KEY`, `GIT_READ_TOKEN`). Never write secrets to the volume.

## 6. GPU tiers (`config/gpu_tiers.yaml`)

Tier = largest tier whose VRAM bound ≤ measured VRAM.

| Tier | VRAM ≥ | Typical | Gen batch | Teacher prefill batch | Co-resident teacher + student |
| --- | --- | --- | --- | --- | --- |
| 24 | 20 GB | RTX 3090, RTX 4090, L4, PRO 6000 MIG 24GB | 16 | 8 | No |
| 32 | 30 GB | RTX PRO 4500 Blackwell SE, RTX 5090 | 24 | 12 | Inference only (S1, S6); teacher unloaded for training |
| 48 | 40 GB | A40, RTX A6000, L40S, L40 | 32 | 16 | Yes |
| 80 | 70 GB | A100 80 GB, H100 | 64 | 32 | Yes |

Image: pin one exact tag in `config/run_defaults.yaml`. Default `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04` with `--min-cuda-version 12.8`; if that tag no longer exists, use the current CUDA 12.8 `runpod/pytorch` tag shown in `runpodctl pod create --help` examples (e.g. `runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404`) and record it. Blackwell cards (RTX PRO 4500/6000, RTX 5090) need a CUDA 12.8+ build of torch, which these images provide; ENVCHECK must log `torch.cuda.get_device_capability()` and a successful bf16 matmul. Changing the tag invalidates the venv (keyed by tag).

## 7. Models

| Role | Primary | Step-up if student base rate < 20% | Optional second pair (FULL) |
| --- | --- | --- | --- |
| Student | `Qwen/Qwen3-1.7B` (~4 GB bf16) | `Qwen/Qwen3-4B` | `meta-llama/Llama-3.2-3B-Instruct` |
| Teacher | `Qwen/Qwen3-8B` (~16.4 GB bf16) | same | `meta-llama/Llama-3.1-8B-Instruct` |

- Qwen3: `enable_thinking=False` in every `apply_chat_template` call (teacher, student, rollouts, eval). Assert on 20 prompts that teacher and student render **identical token ids**; assert the empty think block is present and identical.
- Llama pair: pass a fixed `date_string="2026-01-01"` to both templates; same rendered-id assertion.
- Contamination note: Qwen has likely seen GSM8K test, from which GSM-Symbolic templates derive. RandomCalculation is therefore a co-primary metric (§8.2). The decisive comparisons (A3 vs A2 vs A6) are within-model on identical problems.
- bf16, `attn_implementation="sdpa"`, no quantized weights, no custom kernels.

## 8. Data

### 8.1 GSM-Symbolic
- HF `apple/GSM-Symbolic`, configs `main`, `p1`, `p2`, split `test`. Fields: `id` (template within config), `instance`, `question`, `answer` (`#### <number>`), `original_id` (GSM8K test question), `original_question`, `original_answer`. CC BY-NC-ND 4.0: research only; never copy question text into results or git.
- **Split by `original_id`** across all three configs (seed 0): shuffle the union of `original_id`s; first half → `train_oids`, second half → `test_oids`. Train prompts = all instances (`main`+`p1`) whose `original_id` ∈ `train_oids`. Test = instances from `main`, `p1`, `p2` with `original_id` ∈ `test_oids`. Save `data/split.json`.

### 8.2 RandomCalculation (co-primary)
- https://github.com/wumingqi/LLM-Math-Evaluation (arXiv 2507.10532). Generate with seed 0: 300 eval problems (4–8 operations) and, for FULL, 2,000 train prompts (disjoint seeds). Fully synthetic; post-dates the models.

### 8.3 Prompt and scoring
- System: "Solve the problem step by step. Write each step on its own line. End with exactly one line: `The final answer is <number>`." User: question.
- Greedy first attempts; retries `temperature=0.8, top_p=0.95`; `max_new_tokens=400`; left padding; batch per tier.
- Scoring: **strict** = number after "The final answer is" (missing → wrong); **lenient** = strict, else last number. Report both plus the truncation rate (hit 400 tokens). Gates use strict.

## 9. Method

### 9.1 Boundaries
Decode each generated token; a boundary is the first token whose decoded text contains `\n`, with boundaries ≥ 8 tokens apart, max 12. Hidden states for probe features come from an **adapter-off** prefill of the finished trace (causal, so identical to incremental states up to numerics); layers {L/2, 3L/4, L}. Verify once on 5 traces that prefill states match generation states within tolerance 1e-2 (bf16).

### 9.2 Detector (probe)
Logistic regression on one layer's boundary state. Labels: SMOKE and FULL default = trace-level (1 if final answer wrong) on every boundary. FULL option `probe-mc`: step label = 1 if Monte Carlo success from that prefix (4 sampled completions) < 0.25 while the previous boundary's ≥ 0.75. Hold out 20% **by `original_id`**. Choose layer by held-out AUROC. Trace score = max over boundaries. `τ` = smallest value with FPR ≤ 10% on correct held-out traces. Report the share of first flags at boundary 1 (restart share); if > 60%, say so in the report: the detector is acting as a difficulty estimator.

### 9.3 Rollback (offline-equivalent)
First attempts are greedy, so: generate the full trace, score all boundaries, cut at the first flagged boundary `s` → prefix = tokens before boundary `s−1` (restart from the prompt if `s = 1`), then **re-prefill the prefix** and generate the continuation. This equals online rollback exactly, needs no cache cropping, and lets A1–A4 branch from A0's single first attempt. One intervention per problem in SMOKE, two in FULL (second uses adapter-off features of the continuation).

### 9.4 Repair adapter
- LoRA on the student, r = 16, all linear layers, B = 0 at init. Prefix KV is always built with the adapter **disabled** (`no_grad`); only continuation tokens run with the adapter enabled — identically at training and inference. Untrained A3 must equal A2 **in identical batches**.
- Training data per round: train problems whose first attempt is wrong and flagged → 4 continuations from the flagged prefix (adapter on, T=0.8). Shared prefix is prefilled once per problem for teacher scoring and for training.
- Loss `rkl` (default): teacher prefill gives `logp_T(y_t)`; `A_t = −(logp_S_old(y_t) − logp_T(y_t))`, clipped to [−5, 5]; loss = −mean_t A_t · logp_S_θ(y_t) over continuation tokens. Teacher-also-wrong problems kept; count logged. `rft` ablation: SFT on continuations with correct strict answer.
- AdamW, lr 1e-4, cosine, 3% warmup, effective batch 16, 1 epoch/round. SMOKE 1 round; FULL 3 rounds. Checkpoint every 10 steps in SMOKE (every 100 in FULL); SMOKE kills training at step 15 and resumes once to prove `--resume`.
- Seeds: `int(hashlib.sha256(f"{RUN_ID}:{arm}:{problem_id}".encode()).hexdigest(),16) % 2**31`. Retry batches: same composition (sorted problem ids, tier batch size) and the same seed for A2 and A3.

### 9.5 Arms

| Arm | What | SMOKE | FULL |
| --- | --- | --- | --- |
| A0 | Greedy, no intervention | ✓ | ✓ |
| A1 | Flag → append "Check your work and continue." (no rollback, no adapter) | – | ✓ |
| A2 | Flag → rollback → sampled retry, no adapter | ✓ | ✓ |
| A3 | Flag → rollback → retry with repair adapter | ✓ | ✓ |
| A4 | No probe: same number of interventions as A3, on random problems, flag steps drawn from A3's distribution | – | ✓ |
| A5 | Majority vote at k = 1, 3, 5; accuracy-vs-tokens curve with A3 placed on it | – | ✓ |
| A6 | Adapter on from the first token, no probe (plain distillation control) | ✓ | ✓ |

Per arm vs A0: accuracy (strict, lenient); recovered; broken; net = recovered − broken; intervention rate; FPR; extra tokens; truncation rate.

## 10. Phases

### ENVCHECK
Steps 0–5 of bootstrap; tokenizer/template identity; **one full tier batch** of greedy generation and one teacher prefill batch (tokens/s); self-terminate; verify via S3 that `report.md` landed and via `pod list` that the pod is gone. Second ENVCHECK on the warm volume must show all "cache hit".

### SMOKE (pipeline check; fixed sizes)

| Step | Job | Size |
| --- | --- | --- |
| S1 | Headroom: A0 student and teacher, greedy | **200 `p2`** + 40 `main` + 40 `p1` + 100 RandomCalculation |
| S2 | Student rollouts on train prompts, greedy + 1 sample | 300 problems |
| S3 | Probe fit/select/τ; restart share | from S2 |
| S4 | Identity: A3-untrained == A2, identical batches | 20 flagged |
| S5 | Continuations ×4 | ≤ 150 traces |
| S6 | Teacher scoring (shared-prefix prefill) | from S5 |
| S7 | Train 1 round `rkl`; kill at step 15; resume | from S6 |
| S8 | Eval A0, A2, A3, A6 on S1's test problems | 380 |
| S9 | Report; FULL projection per tier and $ | – |

### FULL (after approval; ceilings)
Train prompts ≤ 5,000 GSM-Symbolic + 2,000 RandomCalculation; 3 rounds; 4 continuations per flagged wrong trace. Test 500 per config + 300 RandomCalculation. Seeds 0,1,2, one pod per seed (parallel pods share the volume safely: code on container disk, downloads under `flock`, unique `runs/<RUN_ID>`). Projection from SMOKE; if over the per-pod ceiling, cut train prompts, then continuations to 2, then seeds to 2; never below 300 test per config. Spot pods only after `--resume` passed in SMOKE.

## 11. Gates and decision

| Gate | Condition | If it fails |
| --- | --- | --- |
| Cloud path | ENVCHECK DONE, report via S3, pod gone, warm rerun all cache hits | Fix orchestration first |
| Headroom | teacher − student ≥ 8 points strict on `p2` (n=200, SE ≈ 3.3) | Step-up student / second pair; rerun S1; stop |
| Student base rate | train-rollout strict accuracy in [20%, 85%] | Step up to Qwen3-4B or shift metric |
| Probe | AUROC ≥ 0.80, recall ≥ 0.40 at FPR ≤ 10%, held out by `original_id`; restart share reported | Other layers / 2-layer concat once; else invalid |
| Identity | A3-untrained == A2 exactly | Fix wiring |
| Resume | killed-and-resumed training matches loss trajectory | Fix before FULL |
| Timing | FULL projection within per-pod ceiling | Cut per §10 |

FULL decision (`p2` strict and RandomCalculation): outcomes averaged over seeds per problem; **template-clustered bootstrap** (cluster = `original_id`, 10,000 resamples) for A3 − A2, A3 − A4, A3 − A6 net recovery; **Holm** correction across the three. **Pass** = all three lower CI bounds > 0 and A3 lies above the A5 accuracy-vs-tokens curve. **Kill** = A3 − A2 ≤ 0 or A3 − A6 ≤ 0. Otherwise inconclusive → one more seed, then decide.

## 12. Report template (`report.md`; every field numeric)

```
RUN: id | phase | sha | pod | GPU | VRAM | dc | $/h | elapsed min | cost
ENV: image tag | torch | transformers | peft | pair | cache hits/downloads | download MB/s
TIMING (min): S1 | S2 | S5 | S6 | S7 | S8 | total
HEADROOM strict/lenient: student p2 (n=200) / main / p1 / RC ; teacher same ; gap p2 ± SE ; truncation rates
ROLLOUTS: n | strict acc | mean tokens | mean boundaries
PROBE: layer | AUROC | tau | FPR | recall | restart share (flags at s=1) | first-flag histogram
IDENTITY: A3-untrained == A2 (y/n, max token diff) ; prefill-vs-generation state tolerance
REPAIR DATA: flagged-wrong | continuations | teacher-also-wrong %
TRAIN: steps | loss start/end | resume check (y/n) | adapter params
EVAL (380): per arm A0/A2/A3/A6: strict acc per set ; recovered ; broken ; net ; interv. rate ; FPR ; extra tokens
THROUGHPUT: gen tok/s | teacher prefill tok/s | train steps/s
PROJECTION: FULL hours and $ per seed on this tier
BLOCKERS
```

## 13. Pipeline design

- `pipeline/run_phase.py` runs all stages in one process: loads the student once; on tiers 48/80 also keeps the teacher loaded; on tier 24 loads the teacher only for S1 and S6 and frees it. Stage modules: `data.py`, `generate.py` (plain greedy, sampled continuations, prefill features), `probe.py`, `continuations.py`, `teacher_score.py`, `train_repair.py` (`--loss rkl|rft`, `--resume`), `eval.py`, `report.py`.
- Every stage writes its artifacts to `runs/<RUN_ID>/stage_<name>/` and is skipped on restart if its `.done` marker exists.
- Optional FULL speed-up: vLLM for plain generation, with probe features from an HF prefill checked against tolerance.

Pitfalls: boundaries by decoded text, not token id; prefill features adapter-off; identical batches for identity; `hashlib` not `hash`; atomic `status.json`; `original_id` for every split; strict vs lenient scoring; never copy question text out of the pod; mask secrets in logs; code on container disk, data on volume; `flock` around downloads.

## 14. Validation checklist before SMOKE (orchestrator ticks each)

ENVCHECK DONE via S3 · pod self-terminated · warm rerun shows cache hits · tokenizer/template identity passed · first-batch tokens/s and teacher prefill tok/s recorded · `kill.ps1 --all` leaves zero pods · `watch.ps1` exited 0 · budget math printed (price × MAX_MINUTES ≤ phase budget) · in-pod deadline watchdog proven once (ENVCHECK run with `MAX_MINUTES=3` and `DEADLINE_EPOCH = now + 4 min`, confirm the pod disappears without the pipeline finishing) · Windows scheduled task `OptD-RunPod-Watchdog` registered and its log shows a run.

## 15. Agent configuration

`CLAUDE.md`:
```markdown
# Option D — detector-gated repair (RunPod from local Claude Code)
Spec: OPTION_D_SPEC.md. Follow it exactly. Phases: ENVCHECK → SMOKE → (human approval) → FULL.
- Create pods only with `pwsh -ExecutionPolicy Bypass -File orchestrator/launch.ps1`. Never call `runpodctl pod create` directly.
- Wait with `pwsh -ExecutionPolicy Bypass -File orchestrator/watch.ps1` (foreground, long timeout). No /loop.
- Pull with pull.ps1; verify the pod is gone; print report.md verbatim; stop.
- Run `kill.ps1 --all` at session start and end. Never enlarge phase sizes. Never start FULL unasked. Never print secrets or env dumps.
```

`.claude/settings.json`:
```json
{
  "env": { "BASH_DEFAULT_TIMEOUT_MS": "600000", "BASH_MAX_TIMEOUT_MS": "7200000" },
  "permissions": {
    "allow": [
      "Bash(pwsh -ExecutionPolicy Bypass -File orchestrator/*)",
      "Bash(runpodctl pod list*)", "Bash(runpodctl pod get*)", "Bash(runpodctl pod delete*)",
      "Bash(runpodctl gpu list*)", "Bash(runpodctl datacenter list*)",
      "Bash(aws --profile runpod s3 *)",
      "Bash(git status*)", "Bash(git add*)", "Bash(git commit*)", "Bash(git log*)", "Bash(git diff*)", "Bash(git push origin main)",
      "Bash(python pipeline/*)", "Bash(python -m pytest*)"
    ],
    "deny": [
      "Bash(runpodctl pod create*)", "Bash(runpodctl pod start*)", "Bash(runpodctl pod stop*)",
      "Bash(git push -f*)", "Bash(git push --force*)", "Bash(git push * +*)",
      "Bash(env*)", "Bash(set*)", "Bash(printenv*)", "Bash(python -c*)",
      "Read(./.env)", "Read(./.env.*)", "Read(~/.aws/**)", "Read(~/.ssh/**)"
    ]
  }
}
```
Paths use forward slashes (Git Bash). Bash rules are prefix matches; keep orchestrator scripts small and auditable.

## 16. References

- Probe and failed inference-time fixes: arXiv 2605.09502 (ICML 2026); arXiv 2605.31561. SCoRe: arXiv 2409.12917. GKD: arXiv 2306.13649.
- GSM-Symbolic: arXiv 2410.05229; HF `apple/GSM-Symbolic`. RandomCalculation: arXiv 2507.10532.
- RunPod (verified on runpodctl 3.0.1, 2026-10-30): `pod create` flags `--cloud-type`, `--gpu-id`, `--network-volume-id`, `--image`, `--min-cuda-version`, `--container-disk-in-gb`, `--ports`, `--ssh`, `--env` (JSON), `--docker-args`, `--data-center-ids`, `--wait`, `--wait-timeout`; **no** `--terminate-after`/`--stop-after`/`--public-ip`. `gpu list` JSON fields: `gpuId`, `memoryInGb`, `secureCloud`, `securePricePerHr`, `dataCenterAvailability[].dataCenterId`. Pods ship with `runpodctl` and a pod-scoped `RUNPOD_API_KEY`. REST `POST/GET/DELETE /v1/pods`; Secrets `{{ RUNPOD_SECRET_name }}`; network volumes (Secure Cloud only, $0.07/GB/month); S3 API `https://s3api-us-ks-2.runpod.io`, bucket `5l0v4bx88l`, region `us-ks-2`.
- Claude Code: permission rules (deny → ask → allow), `BASH_MAX_TIMEOUT_MS`, settings precedence.
