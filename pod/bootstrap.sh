#!/usr/bin/env bash
# Pod bootstrap (OPTION_D_SPEC.md §5). Fetched by commit SHA and piped into bash, so the whole body lives in main():
# bash parses it completely before running anything, and no command can swallow the rest of the script from stdin.

R=""
step()  { echo "$1" > "$R/.step"; write_status RUNNING "$1" ""; echo "== $1 $(date -u +%FT%TZ)"; }
write_status() {   # state step msg   (atomic: tmp + mv)
  local tmp; tmp=$(mktemp -p "$R" .status.XXXXXX)
  printf '{"run_id":"%s","state":"%s","step":"%s","msg":"%s","ts":%s}\n' \
    "$RUN_ID" "$1" "$2" "$(printf '%s' "$3" | tr -d '"\\\n' | cut -c1-300)" "$(date +%s)" > "$tmp" && mv "$tmp" "$R/status.json"
}
mask() { sed -u -e "s|${HF_TOKEN:-@none@}|***|g" -e "s|${RUNPOD_API_KEY:-@none@}|***|g" -e "s|${GIT_READ_TOKEN:-@none@}|***|g"; }
terminate() {   # pod-scoped RUNPOD_API_KEY injected by RunPod
  for _ in 1 2 3 4 5; do
    { runpodctl remove pod "$RUNPOD_POD_ID" || runpodctl pod delete "$RUNPOD_POD_ID" \
      || curl -fsS -X DELETE "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID" -H "Authorization: Bearer $RUNPOD_API_KEY"; } \
      && { echo "== pod delete accepted ($RUNPOD_POD_ID)"; return 0; }   # the image's older runpodctl has only `remove pod`
    sleep 30
  done
  return 1
}
on_exit() {
  local rc=$? msg=""
  kill "$HB" 2>/dev/null || true
  if [ "$rc" -eq 0 ]; then write_status DONE "$(cat "$R/.step")" ""
  else
    [ -s "$R/.err" ] && msg=$(head -1 "$R/.err")
    write_status FAILED "$(cat "$R/.step")" "rc=$rc ${msg}"
  fi
  sleep 2; sync
  if [ "$PHASE" = WATCHDOGTEST ]; then echo "== exit rc=$rc; WATCHDOGTEST: self-terminate skipped"; sleep infinity; fi
  echo "== exit rc=$rc; self-terminating"
  terminate || echo "self-terminate failed; deadline watchdog will delete the pod"
  sleep infinity
}
fetch() {   # kind repo min_gb   (models -> /workspace/models/<name>; dataset -> HF cache). Caller holds the download lock.
  local kind=$1 repo=$2 min_gb=$3 dest marker t0 t1 bytes mbps
  if [ "$kind" = model ]; then dest=/workspace/models/${repo##*/}; marker=$dest/.complete
  else dest=""; marker=/workspace/hf/.complete-${repo//\//__}; fi
  if [ -f "$marker" ]; then
    echo "cache hit $repo"
    printf '{"repo":"%s","kind":"%s","cache":"hit"}\n' "$repo" "$kind" >> "$R/downloads.jsonl"; return 0
  fi
  local author
  author=$(curl -fsS "https://huggingface.co/api/${kind}s/$repo" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("author",""))')
  [ "$author" = "${repo%%/*}" ] || { echo "org mismatch for $repo: '$author'" > "$R/.err"; return 1; }
  t0=$(date +%s.%N)
  if [ "$kind" = model ]; then
    rm -rf "$dest"
    hf download "$repo" --local-dir "$dest" --exclude "original/*" < /dev/null > /dev/null
    bytes=$(du -sb "$dest" | cut -f1)
  else
    dest=$(hf download "$repo" --repo-type dataset < /dev/null | tail -1)
    bytes=$(du -sbL "$dest" | cut -f1)
  fi
  t1=$(date +%s.%N)
  python3 -c "import sys; sys.exit(0 if $bytes >= $min_gb * 1e9 else 1)" \
    || { echo "size check failed for $repo: $bytes bytes < $min_gb GB" > "$R/.err"; return 1; }
  mbps=$(python3 -c "print(round($bytes / 1e6 / max($t1 - $t0, 1e-3), 1))")
  echo "downloaded $repo $((bytes / 1000000)) MB at $mbps MB/s"
  printf '{"repo":"%s","kind":"%s","cache":"download","mb":%s,"sec":%s,"mb_per_s":%s}\n' \
    "$repo" "$kind" "$((bytes / 1000000))" "$(python3 -c "print(round($t1 - $t0, 1))")" "$mbps" >> "$R/downloads.jsonl"
  touch "$marker"
}

main() {
  # 0. Keep the image's sshd + PUBLIC_KEY setup alive.
  /start.sh > /tmp/start.log 2>&1 &
  set -eEuo pipefail
  export BOOT_EPOCH; BOOT_EPOCH=$(date +%s)
  R=/workspace/runs/$RUN_ID; mkdir -p "$R"
  exec > >(mask | tee -a "$R/log.txt") 2>&1
  echo "== bootstrap $RUN_ID phase=$PHASE sha=$GIT_SHA max_minutes=$MAX_MINUTES deadline_epoch=$DEADLINE_EPOCH"

  # Deadline watchdog: absolute epoch, so a container restart does not reset the clock.
  ( set +e; while [ "$(date +%s)" -lt "$DEADLINE_EPOCH" ]; do sleep 60; done
    echo "== deadline watchdog: DEADLINE_EPOCH=$DEADLINE_EPOCH reached at $(date +%s); deleting pod"
    sync; terminate ) &

  # Restart-loop guard: a finished run never runs again.
  if [ -f "$R/status.json" ] && grep -Eq '"state":"(DONE|FAILED)"' "$R/status.json"; then
    echo "== restart guard: final status present; terminating"
    terminate || true; sleep infinity
  fi

  # 1. Heartbeat + final status/self-terminate.
  rm -f "$R/.err"; echo boot > "$R/.step"; write_status RUNNING boot ""
  ( set +e; while true; do sleep 300; write_status RUNNING "$(cat "$R/.step")" ""; done ) & HB=$!
  trap on_exit EXIT
  trap '[ -s "$R/.err" ] || echo "line $LINENO: $BASH_COMMAND" > "$R/.err"' ERR

  # 2. GPU tier (thresholds mirror config/gpu_tiers.yaml).
  step gpu
  local gpu vram_mib gib tier
  gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
  vram_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
  gib=$((vram_mib / 1024))
  if [ -n "${GPU_TIER_OVERRIDE:-}" ]; then tier=$GPU_TIER_OVERRIDE
  elif [ "$gib" -ge 70 ]; then tier=80; elif [ "$gib" -ge 40 ]; then tier=48
  elif [ "$gib" -ge 30 ]; then tier=32; elif [ "$gib" -ge 20 ]; then tier=24
  else echo "VRAM ${gib} GiB below tier 24" > "$R/.err"; exit 1; fi
  echo "gpu=$gpu vram_mib=$vram_mib tier=$tier"
  # image pull = pod create (LAUNCH_EPOCH, local clock) -> bootstrap start; RUNPOD_TCP_PORT_22 is set only if a public port exists
  local pull=$((BOOT_EPOCH - ${LAUNCH_EPOCH:-$BOOT_EPOCH}))
  echo "image pull ${pull} s; public ssh port: ${RUNPOD_TCP_PORT_22:-none}"
  printf '{"run_id":"%s","phase":"%s","seed":"%s","git_sha":"%s","pod_id":"%s","gpu":"%s","vram_mib":%s,"tier":%s,"driver":"%s","max_minutes":%s,"deadline_epoch":%s,"price_per_hr":%s,"launch_epoch":%s,"boot_epoch":%s,"image_pull_sec":%s,"ssh_public_port":"%s"}\n' \
    "$RUN_ID" "$PHASE" "${SEED:-0}" "$GIT_SHA" "${RUNPOD_POD_ID:-}" "$gpu" "$vram_mib" "$tier" \
    "$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)" "$MAX_MINUTES" "$DEADLINE_EPOCH" "${PRICE_PER_HR:-0}" \
    "${LAUNCH_EPOCH:-0}" "$BOOT_EPOCH" "$pull" "${RUNPOD_TCP_PORT_22:-}" > "$R/env.json"

  # 3. Code on the container disk, never the volume.
  step code
  rm -rf /root/repo && mkdir /root/repo && cd /root/repo
  git init -q && git remote add origin "$GIT_REPO"
  git fetch -q --depth 1 origin "$GIT_SHA" && git checkout -q FETCH_HEAD

  # WATCHDOGTEST: no venv, no downloads (shared volume untouched), no timeout; only DEADLINE_EPOCH can end the pod.
  if [ "$PHASE" = WATCHDOGTEST ]; then
    step pipeline
    python3 pipeline/run_phase.py --phase "$PHASE" --run-id "$RUN_ID" --seed "${SEED:-0}" --tier "$tier" < /dev/null
    exit 0
  fi

  # 4. venv on the volume keyed by image tag + requirements hash. torch is pinned to the image's exact build (a dev
  #    build in this image), so pip never pulls a second torch into the venv.
  step venv
  local image venv req reqfile
  image=$(sed -nE 's/^image: *"(.*)"$/\1/p' config/run_defaults.yaml)
  reqfile=pod/requirements.txt; [ -f "pod/requirements_$PHASE.txt" ] && reqfile="pod/requirements_$PHASE.txt"   # per-phase stack
  { cat pod/constraints.txt; echo "torch==$(python3 -c 'import torch; print(torch.__version__)')"; } > /tmp/constraints.txt
  req=$(cat "$reqfile" /tmp/constraints.txt | sha256sum | cut -c1-12)
  venv=/workspace/venv/$(printf '%s' "$image" | tr -c 'A-Za-z0-9._-' '_')-$req
  echo "requirements: $reqfile ; constraints: $(grep -v '^#' /tmp/constraints.txt | tr '\n' ' ')"
  (
    flock 9
    if [ -f "$venv/.complete" ]; then echo "venv cache hit $venv"
    else
      rm -rf "$venv"; python3 -m venv --system-site-packages "$venv"
      "$venv/bin/pip" install -q --disable-pip-version-check -r "$reqfile" -c /tmp/constraints.txt < /dev/null
      touch "$venv/.complete"; echo "venv installed $venv"
      grep -q '^flash-linear-attention' "$reqfile" && touch "$venv/.no_fastpath"   # Qwen3.5 fast path needs causal-conv1d too
    fi
    if [ -f "$venv/.no_fastpath" ]; then   # retried on every pod until it succeeds (prebuilt wheel, else a build)
      local t0; t0=$(date +%s)
      if MAX_JOBS=8 "$venv/bin/pip" install -q --disable-pip-version-check --no-build-isolation causal-conv1d==1.7.0 < /dev/null; then
        rm -f "$venv/.no_fastpath"; echo "causal-conv1d installed in $(( $(date +%s) - t0 )) s"
      else echo "WARN causal-conv1d install failed after $(( $(date +%s) - t0 )) s (slow path this run)"; fi
    fi
  ) 9> /workspace/.venv.lock
  # shellcheck disable=SC1091
  set +u; source "$venv/bin/activate"; set -u
  export HF_HOME=/workspace/hf HF_HUB_DISABLE_PROGRESS_BARS=1 PYTHONUNBUFFERED=1

  # 5. Models and dataset under one lock (parallel pods share the volume).
  step models
  mkdir -p /workspace/models /workspace/hf
  local line
  exec 8> /workspace/.dl.lock; flock 8
  # phase_models lists the repos a phase needs (default: every entry of models_min_gb). Before a large download on a short
  # volume (free < 25 GB) the unused cached Qwen3-4B is evicted (re-downloadable).
  mapfile -t lines < <(python -c '
import os, yaml
c = yaml.safe_load(open("config/run_defaults.yaml"))
want = (c.get("phase_models") or {}).get(os.environ["PHASE"]) or list(c["models_min_gb"])
[print(r, c["models_min_gb"][r]) for r in want]')
  # space: the missing wanted models must fit with 5 GB to spare; otherwise evict cached models this phase does not use
  local need=0 free dir name
  for line in "${lines[@]}"; do
    name=${line%% *}; [ -f "/workspace/models/${name##*/}/.complete" ] || need=$(python3 -c "print(int($need + ${line##* } + 1))")
  done
  # df sees the backing store, not the volume's quota: free = volume_gb - du. Evict unused cached models first, then venvs
  # other than this run's (all re-creatable).
  local vol; vol=$(sed -nE 's/^volume_gb: *([0-9]+).*$/\1/p' config/run_defaults.yaml); vol=${vol:-60}
  volfree() { echo $(( vol - $(du -sxB1G /workspace 2>/dev/null | cut -f1) )); }
  free=$(volfree)
  for dir in /workspace/models/*/ /workspace/venv/*/; do
    [ -d "$dir" ] || continue
    [ "$free" -lt $((need + 5)) ] || break
    name=$(basename "$dir")
    case "$dir" in
      /workspace/models/*) printf '%s\n' "${lines[@]}" | grep -q "/$name " && continue ;;
      *) [ "$dir" = "$venv/" ] && continue ;;
    esac
    echo "volume short on space (free ${free} GB, need ${need} GB): removing cached $dir (re-creatable)"; rm -rf "$dir"
    free=$(volfree)
  done
  echo "volume free: ${free} GB of ${vol}; models to fetch need ${need} GB"
  [ "$free" -ge "$need" ] || { echo "volume still short: free ${free} GB < need ${need} GB" > "$R/.err"; exit 1; }
  for line in "${lines[@]}"; do fetch model ${line}; done
  fetch dataset "$(sed -nE 's/^dataset: *"(.*)"$/\1/p' config/run_defaults.yaml)" 0
  flock -u 8

  # 6. Pipeline, bounded by the pod budget (floor 60 s so tiny test budgets still start).
  step pipeline
  local t=$((MAX_MINUTES * 60 - 600)); [ "$t" -ge 60 ] || t=60
  timeout "$t" python pipeline/run_phase.py --phase "$PHASE" --run-id "$RUN_ID" --seed "${SEED:-0}" --tier "$tier" < /dev/null

  # 7. run_phase.py wrote report.md + metrics.json; exit 0 -> trap writes DONE and terminates the pod.
  step report
  exit 0
}

main "$@"
