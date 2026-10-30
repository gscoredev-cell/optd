# Option D — detector-gated repair (RunPod from local Claude Code)
Spec: OPTION_D_SPEC.md. Follow it exactly. Phases: ENVCHECK → SMOKE → (human approval) → FULL.
- Create pods only with `pwsh -ExecutionPolicy Bypass -File orchestrator/launch.ps1`. Never call `runpodctl pod create` directly.
- Wait with `pwsh -ExecutionPolicy Bypass -File orchestrator/watch.ps1` (foreground, long timeout). No /loop.
- Pull with pull.ps1; verify the pod is gone; print report.md verbatim; stop.
- Run `kill.ps1 --all` at session start and end. Never enlarge phase sizes. Never start FULL unasked. Never print secrets or env dumps.
