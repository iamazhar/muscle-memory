# Release Checklist

## v1 Release Gate

1. Run the full test suite:
   `.venv/bin/python -m pytest tests/ -v`
2. Run the behavioral proof on the Claude Code-first surface:
   `CLAUDE_TESTS=1 .venv/bin/python -m pytest tests/test_behavioral.py -v`
   Manual reproductions should use the same project-scoped Claude settings isolation as the suite.
3. Verify the release preflight for the target version:
   `uv run python scripts/release_preflight.py <version>`
4. Run the benchmark gate and confirm it reports `"thresholds_passed": true`:
   `uv run mm eval run --json > benchmark-run.json`
   Then inspect `benchmark-run.json` for `failed_thresholds` if the gate fails.

## Recovery

If a release candidate or live Claude Code session needs operator cleanup,
pause the hooks, inspect the state, and resume only after the problem is clear:

- `mm maint pause`
- `mm doctor`
- `mm review list`
- `mm jobs retry-failed`
- `mm maint resume`
