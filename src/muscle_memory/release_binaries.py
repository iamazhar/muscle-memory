"""Helpers for building and smoke-testing standalone release binaries."""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BinaryTarget:
    key: str
    runner: str
    asset_name: str


_TARGETS = (
    BinaryTarget(key="darwin-arm64", runner="macos-latest", asset_name="mm-darwin-arm64"),
    BinaryTarget(key="linux-x86_64", runner="ubuntu-latest", asset_name="mm-linux-x86_64"),
)
_TARGETS_BY_KEY = {target.key: target for target in _TARGETS}
_MACHINE_ALIASES = {
    "amd64": "x86_64",
    "arm64e": "arm64",
    "aarch64": "arm64",
}
_HOST_TARGETS = {
    ("Darwin", "arm64"): "darwin-arm64",
    ("Linux", "x86_64"): "linux-x86_64",
}


def release_binary_targets() -> list[BinaryTarget]:
    return list(_TARGETS)


def binary_asset_name(target_key: str) -> str:
    return _target_for_key(target_key).asset_name


def verify_binary_version_output(output: str, version: str) -> None:
    actual = output.strip()
    expected_outputs = {version, f"muscle-memory {version}"}
    if actual not in expected_outputs:
        raise ValueError(f"Expected exact version {version!r} in CLI output; got {actual!r}")


def detect_host_binary_target(
    system: str | None = None, machine: str | None = None
) -> BinaryTarget:
    normalized_machine = _normalize_machine(platform.machine() if machine is None else machine)
    system_name = platform.system() if system is None else system
    target_key = _HOST_TARGETS.get((system_name, normalized_machine))
    if target_key is None:
        raise ValueError(f"Unsupported binary target host: {system_name} {normalized_machine}")
    return _target_for_key(target_key)


def pyinstaller_command(
    target_key: str,
    *,
    dist_dir: Path,
    repo_root: Path | None = None,
) -> list[str]:
    target = _target_for_key(target_key)
    root = _resolve_repo_root(repo_root)
    build_root = dist_dir / ".pyinstaller" / target.key
    return [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onefile",
        "--name",
        target.asset_name,
        "--distpath",
        str(dist_dir),
        "--workpath",
        str(build_root / "build"),
        "--specpath",
        str(build_root / "spec"),
        "--paths",
        str(root / "src"),
        # Prompt templates are loaded via importlib.resources at runtime, so
        # PyInstaller needs both the package import and its bundled data files.
        "--hidden-import",
        "muscle_memory.prompts",
        "--collect-data",
        "muscle_memory.prompts",
        str(root / "scripts" / "mm_binary_entrypoint.py"),
    ]


def build_release_binary(
    version: str,
    dist_dir: Path,
    *,
    target_key: str,
    repo_root: Path | None = None,
) -> Path:
    target = _target_for_key(target_key)
    host_target = detect_host_binary_target()
    if target != host_target:
        raise ValueError(
            f"Cannot build target {target.key} on this host; current platform supports {host_target.key}"
        )

    root = _resolve_repo_root(repo_root)
    dist_dir.mkdir(parents=True, exist_ok=True)
    command = pyinstaller_command(target.key, dist_dir=dist_dir, repo_root=root)
    _run(command, cwd=root)

    binary_path = dist_dir / target.asset_name
    if not binary_path.exists():
        raise ValueError(f"PyInstaller did not produce expected binary at {binary_path}")

    smoke_check_binary(binary_path, version)
    return binary_path


def smoke_check_binary(binary_path: Path, version: str) -> None:
    result = _run([str(binary_path), "--version"])
    verify_binary_version_output(result.stdout, version)


def verify_release_binaries(version: str, dist_dir: Path, target_keys: list[str]) -> None:
    for target_key in target_keys:
        smoke_check_binary(dist_dir / binary_asset_name(target_key), version)


def build_binaries_main(argv: list[str] | None = None) -> int:
    parser = _build_parser("build_release_binaries.py")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    dist_dir = Path(args.dist_dir)

    try:
        for target_key in _resolve_requested_target_keys(args.targets):
            print(build_release_binary(args.version, dist_dir, target_key=target_key))
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        print(_cli_error_message(exc), file=sys.stderr)
        return 1

    return 0


def smoke_main(argv: list[str] | None = None) -> int:
    parser = _build_parser("check_release_binaries.py")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    target_keys = _resolve_requested_target_keys(args.targets)

    try:
        verify_release_binaries(args.version, Path(args.dist_dir), target_keys)
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        print(_cli_error_message(exc), file=sys.stderr)
        return 1

    print(f"Verified standalone release binaries for {args.version}: {', '.join(target_keys)}")
    return 0


def _resolve_requested_target_keys(target_keys: list[str] | None) -> list[str]:
    if target_keys:
        return target_keys
    return [detect_host_binary_target().key]


def _target_for_key(target_key: str) -> BinaryTarget:
    try:
        return _TARGETS_BY_KEY[target_key]
    except KeyError as exc:
        raise ValueError(f"Unsupported binary target: {target_key}") from exc


def _normalize_machine(machine: str) -> str:
    normalized = machine.strip()
    return _MACHINE_ALIASES.get(normalized, normalized)


def _resolve_repo_root(repo_root: Path | None) -> Path:
    return Path(__file__).resolve().parents[2] if repo_root is None else repo_root.resolve()


def _run(command: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, capture_output=True, text=True, cwd=cwd)


def _build_parser(prog: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog)
    parser.add_argument("version")
    parser.add_argument("dist_dir", nargs="?", default="dist")
    parser.add_argument(
        "--target", dest="targets", action="append", choices=sorted(_TARGETS_BY_KEY)
    )
    return parser


def _cli_error_message(exc: OSError | subprocess.CalledProcessError | ValueError) -> str:
    if isinstance(exc, subprocess.CalledProcessError):
        if isinstance(exc.stderr, str) and exc.stderr.strip():
            return exc.stderr.strip()
        if isinstance(exc.stdout, str) and exc.stdout.strip():
            return exc.stdout.strip()
    return str(exc)


__all__ = [
    "BinaryTarget",
    "binary_asset_name",
    "build_binaries_main",
    "build_release_binary",
    "detect_host_binary_target",
    "pyinstaller_command",
    "release_binary_targets",
    "smoke_check_binary",
    "smoke_main",
    "verify_binary_version_output",
    "verify_release_binaries",
]
