// PreToolUse hook (Bash|PowerShell): block commands that reference the account API key or dump the environment.
// Permission rules only match prefixes, so substring checks live here. Exit 2 = block, stderr goes back to Claude.
let s = '';
process.stdin.on('data', d => (s += d)).on('end', () => {
  const c = (JSON.parse(s).tool_input || {}).command || '';
  const bad = /RUNPOD_API_KEY|GetEnvironmentVariables?\b|\bprintenv\b|(^|[\s;|&(])(Get-ChildItem|gci|ls|dir|gi|Get-Item|Get-Content|cat)\s+env:/i;
  if (bad.test(c)) {
    process.stderr.write('blocked by .claude/hooks/deny_secrets.js: command references RUNPOD_API_KEY or dumps the environment');
    process.exit(2);
  }
});
