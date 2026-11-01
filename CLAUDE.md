# Option E — trained repair under an external trigger (RunPod from local Claude Code)
Method spec: OPTION_E_SPEC.md (v1.6). Infrastructure spec: OPTION_D_SPEC.md §1–§6, §13–§15 (still in force). Follow both exactly.
Phases: SMOKE-6/7 (pointer, closed) → SMOKE-8 (`-Phase SMOKE8`, Qwen3.5-9B self-teacher, spec §13) → (human approval) → FULL. Option D (detector) is closed: docs/RESULTS.md; do not redo it.
- Session start: OPTION_E_SPEC.md §10 checklist; `kill.ps1 --All` at session start and end.
- Create pods only with `pwsh -ExecutionPolicy Bypass -File orchestrator/launch.ps1`. Never call `runpodctl pod create` directly.
- Wait with `pwsh -ExecutionPolicy Bypass -File orchestrator/watch.ps1` (foreground, long timeout). No /loop.
- Pull with pull.ps1; verify the pod is gone; print report.md verbatim; stop.
- Never enlarge phase sizes. Never start FULL unasked. Never print secrets or env dumps. Never copy question text out of the pod.
- Reusable artifacts live on the volume (SMOKE-4 A1/A3, SMOKE-2, /workspace/gencache): import, do not regenerate.
