"""Helpers for verifying built release artifacts."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ArtifactSpec:
    kind: str
    path: Path


def _require_single(matches: list[Path], dist_dir: Path, kind: str, version: str) -> Path:
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one {kind} for version {version} in {dist_dir}; found {len(matches)}"
        )
    return matches[0]


def discover_release_artifacts(dist_dir: Path, version: str) -> list[ArtifactSpec]:
    wheel_matches = sorted(path for path in dist_dir.glob("*.whl") if f"-{version}-" in path.name)
    sdist_matches = sorted(path for path in dist_dir.glob("*.tar.gz") if path.name.endswith(f"-{version}.tar.gz"))

    wheel = _require_single(wheel_matches, dist_dir, "wheel", version)
    sdist = _require_single(sdist_matches, dist_dir, "sdist", version)
    return [
        ArtifactSpec(kind="wheel", path=wheel),
        ArtifactSpec(kind="sdist", path=sdist),
    ]


def assert_version_output(output: str, version: str) -> None:
    actual = output.strip()
    expected_outputs = {version, f"muscle-memory {version}"}
    if actual not in expected_outputs:
        raise ValueError(
            f"Expected exact version {version!r} in CLI output; got {actual!r}"
        )


def _bin_dir(venv_dir: Path) -> Path:
    return venv_dir / ("Scripts" if sys.platform == "win32" else "bin")


def _run(command: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, capture_output=True, text=True, cwd=cwd)


def _smoke_check_artifact(artifact: ArtifactSpec, version: str) -> None:
    with tempfile.TemporaryDirectory(prefix=f"mm-{artifact.kind}-") as temp_dir:
        venv_dir = Path(temp_dir) / "venv"
        _run([sys.executable, "-m", "venv", str(venv_dir)])

        bin_dir = _bin_dir(venv_dir)
        python_bin = bin_dir / ("python.exe" if sys.platform == "win32" else "python")
        mm_bin = bin_dir / ("mm.exe" if sys.platform == "win32" else "mm")

        _run([str(python_bin), "-m", "pip", "install", str(artifact.path.resolve())])

        version_result = _run([str(mm_bin), "--version"])
        assert_version_output(version_result.stdout, version)

        import_result = _run(
            [
                str(python_bin),
                "-c",
                "import muscle_memory; print(muscle_memory.__version__)",
            ]
        )
        assert_version_output(import_result.stdout, version)


def verify_release_artifacts(version: str, dist_dir: Path) -> None:
    for artifact in discover_release_artifacts(dist_dir, version):
        _smoke_check_artifact(artifact, version)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or len(args) > 2:
        print("usage: check_release_artifacts.py <version> [dist_dir]", file=sys.stderr)
        return 1

    version = args[0]
    dist_dir = Path(args[1]) if len(args) == 2 else Path("dist")

    try:
        verify_release_artifacts(version, dist_dir)
    except (OSError, shutil.Error, subprocess.CalledProcessError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"Verified wheel and sdist install cleanly for {version}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
