# Release Checklist

## v1 Release Gate

1. Run the full test suite:
   `.venv/bin/python -m pytest tests/ -v`
2. Run the behavioral proof on the Claude Code-first surface:
   `CLAUDE_TESTS=1 .venv/bin/python -m pytest tests/test_behavioral.py -v`
   Manual reproductions should use the same project-scoped Claude settings isolation, via `--setting-sources project`, as the suite.
3. Verify the release preflight for the target version:
   `uv run python scripts/release_preflight.py <version>`
4. Run the benchmark gate and confirm it reports `"thresholds_passed": true`:
   `uv run python scripts/release_benchmark_gate.py > benchmark-run.json`
   Then inspect `benchmark-run.json` for `failed_thresholds` if the gate fails.
5. Build and smoke-test the standalone release binaries:
   `uv run python scripts/build_release_binaries.py <version> dist --target linux-x86_64`
   `uv run python scripts/check_release_binaries.py <version> dist --target linux-x86_64`
   `uv run python scripts/build_release_binaries.py <version> dist --target darwin-arm64`
   `uv run python scripts/check_release_binaries.py <version> dist --target darwin-arm64`
6. Verify the GitHub Release asset bundle contains the curl-install payload:
   `install.sh`, `mm-darwin-arm64`, `mm-linux-x86_64`, `SHA256SUMS`,
   plus the wheel and sdist.
7. If `publish_to_pypi` is enabled for the workflow dispatch, treat that as an
   optional follow-up publish after the GitHub Release is live rather than a
   blocking release gate.

## Recovery

If a release candidate or live Claude Code session needs operator cleanup,
start with the core diagnostics, then use advanced repair commands only when
the problem is clear:

- `mm status`
- `mm doctor`
- `mm maint pause`
- `mm review list`
- `mm jobs retry-failed`
- `mm maint resume`
