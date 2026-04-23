# Curl Binary Install Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make GitHub Releases the primary install surface by shipping standalone `mm` binaries for `darwin-arm64` and `linux-x86_64`, plus a checksum-verifying `curl | sh` installer.

**Architecture:** Keep wheel/sdist validation intact, but add a parallel binary distribution path. Build binaries with PyInstaller through Python helper code, verify them directly in CI/release jobs, and publish an `install.sh` asset that downloads the matching binary from GitHub Releases and verifies it against `SHA256SUMS`. Local release preflight remains focused on package metadata and source-state checks; cross-platform binary validation lives in CI and the release workflow.

**Tech Stack:** Python 3.11+, Typer, PyInstaller, GitHub Actions, POSIX shell, pytest.

---

## File Map

**Create**

- `src/muscle_memory/release_binaries.py`
  Responsibility: define supported binary targets, build a PyInstaller executable for one target, and smoke-test a standalone `mm` binary by checking `--version`.
- `scripts/build_release_binaries.py`
  Responsibility: CLI wrapper for building one or more standalone release binaries into a target directory.
- `scripts/check_release_binaries.py`
  Responsibility: CLI wrapper for smoke-testing built standalone binaries.
- `scripts/mm_binary_entrypoint.py`
  Responsibility: minimal Python entrypoint file for PyInstaller to bundle into the standalone `mm` executable.
- `scripts/install.sh`
  Responsibility: release installer that detects platform, downloads the right asset from GitHub Releases, verifies the checksum, installs to `${MM_INSTALL_DIR:-$HOME/.local/bin}`, and prints PATH guidance.
- `tests/test_release_binaries.py`
  Responsibility: unit tests for binary target naming, release asset paths, and standalone binary smoke-check helpers.
- `tests/test_install_script.py`
  Responsibility: installer smoke tests that invoke `sh scripts/install.sh` against a temporary local HTTP release layout.

**Modify**

- `pyproject.toml`
  Responsibility: add the build dependency set required for binary packaging, preferably via the existing dev install path used by CI/release jobs.
- `src/muscle_memory/release_artifacts.py`
  Responsibility: keep wheel/sdist verification, add checksum-manifest support for arbitrary extra release assets, and avoid conflating package smoke tests with binary smoke tests.
- `scripts/generate_release_checksums.py`
  Responsibility: stay as the package-only checksum wrapper unless the helper API changes require a small wrapper update.
- `.github/workflows/ci.yml`
  Responsibility: add one PR-time binary/install smoke path, ideally Linux-only for cost control.
- `.github/workflows/release.yml`
  Responsibility: split verification from binary builds, upload/download artifacts across jobs, publish binaries plus `install.sh`, and make PyPI non-blocking or off by default.
- `README.md`
  Responsibility: make `curl | sh` the primary install path and keep package install instructions secondary.
- `docs/development.md`
  Responsibility: explain local binary-building and binary-release verification commands.
- `docs/release.md`
  Responsibility: extend the release checklist with binary build, installer, and GitHub Release asset verification.
- `tests/test_release_artifacts.py`
  Responsibility: cover generic checksum manifest behavior with non-package assets.
- `tests/test_release_workflow.py`
  Responsibility: assert the release workflow builds binaries, uploads artifacts between jobs, publishes `install.sh`, and does not require PyPI for the core release path.
- `tests/test_ci_workflow.py`
  Responsibility: assert CI validates the binary/install path and the docs reflect the new primary install method.

### Task 1: Add Binary Build Helpers

**Files:**
- Create: `src/muscle_memory/release_binaries.py`
- Create: `scripts/build_release_binaries.py`
- Create: `scripts/check_release_binaries.py`
- Create: `scripts/mm_binary_entrypoint.py`
- Modify: `pyproject.toml`
- Test: `tests/test_release_binaries.py`

- [ ] **Step 1: Write the failing helper tests**

```python
from pathlib import Path

import pytest

from muscle_memory.release_binaries import (
    BinaryTarget,
    binary_asset_name,
    release_binary_targets,
    verify_binary_version_output,
)


def test_release_binary_targets_match_supported_matrix() -> None:
    assert release_binary_targets() == [
        BinaryTarget(key="darwin-arm64", runner="macos-latest", asset_name="mm-darwin-arm64"),
        BinaryTarget(key="linux-x86_64", runner="ubuntu-latest", asset_name="mm-linux-x86_64"),
    ]


def test_binary_asset_name_rejects_unknown_target() -> None:
    with pytest.raises(ValueError, match="Unsupported binary target"):
        binary_asset_name("windows-x86_64")


def test_verify_binary_version_output_accepts_mm_prefix() -> None:
    verify_binary_version_output("muscle-memory 0.11.0\n", "0.11.0")


def test_verify_binary_version_output_rejects_mismatch() -> None:
    with pytest.raises(ValueError, match="Expected exact version '0.11.0'"):
        verify_binary_version_output("muscle-memory 0.11.1\n", "0.11.0")
```

- [ ] **Step 2: Run the new tests to verify RED**

Run: `uv run pytest tests/test_release_binaries.py -q`

Expected: FAIL with import errors because `muscle_memory.release_binaries` does not exist yet.

- [ ] **Step 3: Add the minimal binary helper implementation**

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BinaryTarget:
    key: str
    runner: str
    asset_name: str


_TARGETS = {
    "darwin-arm64": BinaryTarget(
        key="darwin-arm64",
        runner="macos-latest",
        asset_name="mm-darwin-arm64",
    ),
    "linux-x86_64": BinaryTarget(
        key="linux-x86_64",
        runner="ubuntu-latest",
        asset_name="mm-linux-x86_64",
    ),
}


def release_binary_targets() -> list[BinaryTarget]:
    return [_TARGETS["darwin-arm64"], _TARGETS["linux-x86_64"]]


def binary_asset_name(target_key: str) -> str:
    try:
        return _TARGETS[target_key].asset_name
    except KeyError as exc:
        raise ValueError(f"Unsupported binary target: {target_key}") from exc


def verify_binary_version_output(output: str, version: str) -> None:
    actual = output.strip()
    expected = {version, f"muscle-memory {version}"}
    if actual not in expected:
        raise ValueError(f"Expected exact version {version!r} in CLI output; got {actual!r}")
```

- [ ] **Step 4: Add the build/smoke wrappers and binary dependency**

```python
# scripts/build_release_binaries.py
from muscle_memory.release_binaries import build_binaries_main

if __name__ == "__main__":
    raise SystemExit(build_binaries_main())
```

```python
# scripts/check_release_binaries.py
from muscle_memory.release_binaries import smoke_main

if __name__ == "__main__":
    raise SystemExit(smoke_main())
```

```python
# scripts/mm_binary_entrypoint.py
from muscle_memory.cli import app

if __name__ == "__main__":
    app()
```

```toml
# pyproject.toml
[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "ruff>=0.6",
    "mypy>=1.10",
    "pyinstaller>=6.0",
]
```

- [ ] **Step 5: Run the helper tests to verify GREEN**

Run: `uv sync --extra dev && uv run pytest tests/test_release_binaries.py -q`

Expected: PASS

- [ ] **Step 6: Commit the helper layer**

```bash
git add pyproject.toml src/muscle_memory/release_binaries.py scripts/build_release_binaries.py scripts/check_release_binaries.py scripts/mm_binary_entrypoint.py tests/test_release_binaries.py
git commit -m "feat: add release binary helpers"
```

### Task 2: Implement The Curl Installer

**Files:**
- Create: `scripts/install.sh`
- Create: `tests/test_install_script.py`
- Verify: `src/muscle_memory/release_binaries.py`

- [ ] **Step 1: Write the failing installer smoke tests**

```python
from __future__ import annotations

import http.server
import os
import socketserver
import subprocess
import threading
from pathlib import Path


def _serve(directory: Path) -> tuple[str, socketserver.TCPServer]:
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(directory), **kwargs)

    server = socketserver.TCPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return f"http://127.0.0.1:{server.server_address[1]}", server


def test_install_script_downloads_and_installs_binary(tmp_path: Path) -> None:
    release_dir = tmp_path / "download" / "v0.11.0"
    release_dir.mkdir(parents=True)
    binary = release_dir / "mm-linux-x86_64"
    binary.write_text("#!/bin/sh\necho 'muscle-memory 0.11.0'\n", encoding="utf-8")
    binary.chmod(0o755)
    import hashlib
    digest = hashlib.sha256(binary.read_bytes()).hexdigest()
    (release_dir / "SHA256SUMS").write_text(
        f"{digest}  mm-linux-x86_64\n",
        encoding="utf-8",
    )

    base_url, server = _serve(tmp_path)
    install_dir = tmp_path / "bin"
    try:
        result = subprocess.run(
            ["sh", "scripts/install.sh"],
            cwd=Path.cwd(),
            env=os.environ
            | {
                "MM_VERSION": "0.11.0",
                "MM_RELEASE_BASE_URL": base_url,
                "MM_INSTALL_DIR": str(install_dir),
            },
            check=False,
            capture_output=True,
            text=True,
        )
    finally:
        server.shutdown()

    assert result.returncode == 0
    assert (install_dir / "mm").exists()


def test_install_script_rejects_unsupported_platform(tmp_path: Path) -> None:
    result = subprocess.run(
        ["sh", "scripts/install.sh"],
        cwd=Path.cwd(),
        env=os.environ | {"MM_UNAME_S": "Darwin", "MM_UNAME_M": "x86_64"},
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "Unsupported platform" in result.stderr
```

- [ ] **Step 2: Run the installer tests to verify RED**

Run: `uv run pytest tests/test_install_script.py -q`

Expected: FAIL because `scripts/install.sh` does not exist.

- [ ] **Step 3: Implement the installer script**

```sh
#!/bin/sh
set -eu

UNAME_S="${MM_UNAME_S:-$(uname -s)}"
UNAME_M="${MM_UNAME_M:-$(uname -m)}"
INSTALL_DIR="${MM_INSTALL_DIR:-$HOME/.local/bin}"
VERSION="${MM_VERSION:-latest}"
BASE_URL="${MM_RELEASE_BASE_URL:-https://github.com/iamazhar/muscle-memory/releases}"

case "${UNAME_S}:${UNAME_M}" in
  Darwin:arm64) ASSET="mm-darwin-arm64" ;;
  Linux:x86_64) ASSET="mm-linux-x86_64" ;;
  *)
    echo "Unsupported platform: ${UNAME_S} ${UNAME_M}" >&2
    exit 1
    ;;
esac

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required to install muscle-memory" >&2
  exit 1
fi

if command -v shasum >/dev/null 2>&1; then
  SHA_CMD="shasum -a 256"
elif command -v sha256sum >/dev/null 2>&1; then
  SHA_CMD="sha256sum"
else
  echo "shasum or sha256sum is required to verify release checksums" >&2
  exit 1
fi

if [ "${VERSION}" = "latest" ]; then
  DOWNLOAD_ROOT="${BASE_URL}/latest/download"
else
  DOWNLOAD_ROOT="${BASE_URL}/download/v${VERSION}"
fi

TMPDIR="$(mktemp -d)"
trap 'rm -rf "${TMPDIR}"' EXIT

curl -fsSL "${DOWNLOAD_ROOT}/${ASSET}" -o "${TMPDIR}/${ASSET}"
curl -fsSL "${DOWNLOAD_ROOT}/SHA256SUMS" -o "${TMPDIR}/SHA256SUMS"

EXPECTED="$(grep "  ${ASSET}\$" "${TMPDIR}/SHA256SUMS" | awk '{print $1}')"
ACTUAL="$(${SHA_CMD} "${TMPDIR}/${ASSET}" | awk '{print $1}')"

if [ -z "${EXPECTED}" ] || [ "${EXPECTED}" != "${ACTUAL}" ]; then
  echo "Checksum verification failed for ${ASSET}" >&2
  exit 1
fi

mkdir -p "${INSTALL_DIR}"
install -m 0755 "${TMPDIR}/${ASSET}" "${INSTALL_DIR}/mm"

echo "Installed mm to ${INSTALL_DIR}/mm"
case ":${PATH}:" in
  *":${INSTALL_DIR}:"*) ;;
  *) echo "Add ${INSTALL_DIR} to PATH to run mm from a new shell" ;;
esac
```

- [ ] **Step 4: Run the installer tests to verify GREEN**

Run: `uv run pytest tests/test_install_script.py -q`

Expected: PASS

- [ ] **Step 5: Commit the installer**

```bash
git add scripts/install.sh tests/test_install_script.py
git commit -m "feat: add curl installer"
```

### Task 3: Wire Binaries Into CI And Releases

**Files:**
- Modify: `src/muscle_memory/release_artifacts.py`
- Modify: `.github/workflows/ci.yml`
- Modify: `.github/workflows/release.yml`
- Modify: `tests/test_release_artifacts.py`
- Modify: `tests/test_release_workflow.py`
- Modify: `tests/test_ci_workflow.py`

- [ ] **Step 1: Write the failing release-helper and workflow tests**

```python
def test_build_checksum_manifest_supports_binary_and_installer_assets(tmp_path: Path) -> None:
    wheel = tmp_path / "muscle_memory-0.11.0-py3-none-any.whl"
    sdist = tmp_path / "muscle_memory-0.11.0.tar.gz"
    binary = tmp_path / "mm-linux-x86_64"
    installer = tmp_path / "install.sh"
    for path, text in (
        (wheel, "wheel"),
        (sdist, "sdist"),
        (binary, "binary"),
        (installer, "installer"),
    ):
        path.write_text(text, encoding="utf-8")

    manifest = build_checksum_manifest(
        [
            ArtifactSpec(kind="wheel", path=wheel),
            ArtifactSpec(kind="sdist", path=sdist),
            ArtifactSpec(kind="binary", path=binary),
            ArtifactSpec(kind="installer", path=installer),
        ],
        tmp_path,
    )

    assert "mm-linux-x86_64" in manifest.read_text(encoding="utf-8")
    assert "install.sh" in manifest.read_text(encoding="utf-8")
```

```python
def test_release_workflow_builds_and_uploads_binaries() -> None:
    text = RELEASE_WORKFLOW.read_text(encoding="utf-8")
    assert "build-binaries" in text
    assert "macos-latest" in text
    assert "ubuntu-latest" in text
    assert "scripts/build_release_binaries.py" in text
    assert "scripts/check_release_binaries.py" in text
    assert "actions/upload-artifact" in text
    assert "actions/download-artifact" in text
    assert "install.sh" in text
```

```python
def test_ci_workflow_smokes_linux_binary_install_path() -> None:
    text = CI_WORKFLOW.read_text(encoding="utf-8")
    assert "scripts/build_release_binaries.py" in text
    assert "tests/test_install_script.py" in text
```

- [ ] **Step 2: Run the targeted tests to verify RED**

Run: `uv run pytest tests/test_release_artifacts.py tests/test_release_workflow.py tests/test_ci_workflow.py -q`

Expected: FAIL because the workflow files and helper code do not mention binaries or installer smoke yet.

- [ ] **Step 3: Implement the release-asset helper changes**

```python
def build_checksum_manifest_for_paths(paths: list[Path], output_dir: Path) -> Path:
    specs = [ArtifactSpec(kind="generic", path=path) for path in paths]
    return build_checksum_manifest(specs, output_dir)
```

```python
def smoke_check_binary(binary_path: Path, version: str) -> None:
    result = _run([str(binary_path), "--version"])
    assert_version_output(result.stdout, version)
```

- [ ] **Step 4: Implement the workflow split**

```yaml
jobs:
  verify:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v6
      - uses: astral-sh/setup-uv@v7
      - run: uv sync --extra dev
      - run: uv run pytest tests/ --ignore=tests/test_behavioral.py -q
      - run: uv build
      - run: uvx --from twine twine check dist/*
      - run: uv run python scripts/check_release_artifacts.py "${{ inputs.version }}"
      - uses: actions/upload-artifact@v4
        with:
          name: python-dist
          path: dist/*

  build-binaries:
    strategy:
      matrix:
        include:
          - target: darwin-arm64
            runner: macos-latest
          - target: linux-x86_64
            runner: ubuntu-latest
    runs-on: ${{ matrix.runner }}
    steps:
      - uses: actions/checkout@v6
      - uses: astral-sh/setup-uv@v7
      - run: uv sync --extra dev --extra openai --extra voyage
      - run: uv run python scripts/build_release_binaries.py "${{ inputs.version }}" dist --target "${{ matrix.target }}"
      - run: uv run python scripts/check_release_binaries.py "${{ inputs.version }}" dist --target "${{ matrix.target }}"
      - uses: actions/upload-artifact@v4
        with:
          name: binary-${{ matrix.target }}
          path: dist/mm-*
```

- [ ] **Step 5: Implement the final release job asset assembly**

```yaml
  release:
    needs: [verify, build-binaries]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v6
      - uses: actions/download-artifact@v4
        with:
          pattern: "*"
          path: release-assets
          merge-multiple: true
      - run: cp scripts/install.sh release-assets/install.sh
      - run: |
          python - <<'PY'
          from pathlib import Path
          from muscle_memory.release_artifacts import build_checksum_manifest_for_paths

          root = Path("release-assets")
          paths = sorted(path for path in root.iterdir() if path.is_file() and path.name != "SHA256SUMS")
          build_checksum_manifest_for_paths(paths, root)
          PY
      - uses: softprops/action-gh-release@v2
        with:
          files: release-assets/*
```

- [ ] **Step 6: Make PyPI non-blocking in the release workflow**

```yaml
on:
  workflow_dispatch:
    inputs:
      publish_to_pypi:
        default: false
```

```yaml
      - name: Publish to PyPI
        if: ${{ inputs.publish_to_pypi }}
        uses: pypa/gh-action-pypi-publish@release/v1
```

- [ ] **Step 7: Re-run the targeted workflow/helper tests**

Run: `uv run pytest tests/test_release_artifacts.py tests/test_release_workflow.py tests/test_ci_workflow.py -q`

Expected: PASS

- [ ] **Step 8: Commit the workflow wiring**

```bash
git add src/muscle_memory/release_artifacts.py .github/workflows/ci.yml .github/workflows/release.yml tests/test_release_artifacts.py tests/test_release_workflow.py tests/test_ci_workflow.py
git commit -m "feat: publish release binaries"
```

### Task 4: Update Public Install And Release Docs

**Files:**
- Modify: `README.md`
- Modify: `docs/development.md`
- Modify: `docs/release.md`
- Verify: `tests/test_ci_workflow.py`
- Verify: `tests/test_release_workflow.py`

- [ ] **Step 1: Write the failing docs assertions**

```python
def test_readme_prefers_curl_install() -> None:
    text = README.read_text(encoding="utf-8")
    assert "curl -fsSL https://github.com/iamazhar/muscle-memory/releases/latest/download/install.sh | sh" in text
    assert "GitHub Releases" in text
```

```python
def test_release_docs_include_binary_steps() -> None:
    text = RELEASE_DOC.read_text(encoding="utf-8")
    assert "build_release_binaries.py" in text
    assert "install.sh" in text
    assert "SHA256SUMS" in text
```

- [ ] **Step 2: Run the docs tests to verify RED**

Run: `uv run pytest tests/test_ci_workflow.py tests/test_release_workflow.py -q`

Expected: FAIL because the docs still lead with `uv tool install`.

- [ ] **Step 3: Update the README install path**

Suggested README block:

```bash
curl -fsSL https://github.com/iamazhar/muscle-memory/releases/latest/download/install.sh | sh
mm --version
```

Pinned install example:

```bash
curl -fsSL https://github.com/iamazhar/muscle-memory/releases/latest/download/install.sh | MM_VERSION=0.12.0 sh
```

- [ ] **Step 4: Update development and release docs**

Suggested `docs/development.md` commands:

```bash
uv sync --extra dev --extra openai --extra voyage
uv run python scripts/build_release_binaries.py 0.12.0 dist --target darwin-arm64
uv run python scripts/check_release_binaries.py 0.12.0 dist --target darwin-arm64
```

Suggested `docs/release.md` additions:

```text
5. Build and smoke-test standalone binaries:
   uv run python scripts/build_release_binaries.py <version> dist --target linux-x86_64
   uv run python scripts/check_release_binaries.py <version> dist --target linux-x86_64
6. Verify installer and checksum assets are present in the GitHub Release payload:
   install.sh, mm-darwin-arm64, mm-linux-x86_64, SHA256SUMS
```

- [ ] **Step 5: Re-run the docs tests to verify GREEN**

Run: `uv run pytest tests/test_ci_workflow.py tests/test_release_workflow.py -q`

Expected: PASS

- [ ] **Step 6: Commit the docs update**

```bash
git add README.md docs/development.md docs/release.md tests/test_ci_workflow.py tests/test_release_workflow.py
git commit -m "docs: switch install docs to curl binary flow"
```

### Task 5: End-To-End Verification

**Files:**
- Verify: `src/muscle_memory/release_binaries.py`
- Verify: `scripts/install.sh`
- Verify: `.github/workflows/ci.yml`
- Verify: `.github/workflows/release.yml`

- [ ] **Step 1: Run the focused release/install test set**

Run: `uv run pytest tests/test_release_binaries.py tests/test_install_script.py tests/test_release_artifacts.py tests/test_release_workflow.py tests/test_ci_workflow.py -q`

Expected: PASS

- [ ] **Step 2: Run the full test suite**

Run: `uv run pytest -q`

Expected: PASS

- [ ] **Step 3: Build and smoke-test the current-platform binary locally**

Run: `uv sync --extra dev --extra openai --extra voyage`

Expected: dependencies installed with PyInstaller available.

Run: `uv run python scripts/build_release_binaries.py 0.12.0 dist --target darwin-arm64`

Expected: creates `dist/mm-darwin-arm64` on Apple Silicon macOS or the local platform target if adjusted for your machine.

Run: `uv run python scripts/check_release_binaries.py 0.12.0 dist --target darwin-arm64`

Expected: PASS with exact version output from the binary.

- [ ] **Step 4: Smoke-test the shell installer against a local release layout**

Run: `uv run pytest tests/test_install_script.py::test_install_script_downloads_and_installs_binary -q`

Expected: PASS

- [ ] **Step 5: Review the final diff**

Run: `git status --short`

Expected: only the curl-install/binary-distribution files described in this plan are changed.

- [ ] **Step 6: Commit the verification snapshot if the team wants a final rollup commit**

```bash
git log --oneline --max-count=5
```

Expected: helper, installer, workflow, and docs commits are present and the branch is ready for review.
