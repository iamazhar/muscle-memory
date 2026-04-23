# Curl Binary Install Design

## Goal

Make GitHub Releases the primary distribution channel for `muscle-memory` so users install with `curl` instead of PyPI. The default public install flow should download a standalone `mm` binary from GitHub Releases, verify its checksum, and place it on `PATH` without requiring local Python or `uv`.

## Scope

This design covers:

- release artifacts for standalone binaries
- a `curl | sh` install path backed by GitHub Releases
- release workflow changes required to publish and verify those assets
- documentation and smoke-test updates for the new install path

This design does not cover:

- Windows support
- auto-update behavior
- Homebrew tap changes
- removal of wheel/sdist artifacts from the repository entirely

## Product Decision

GitHub Releases become the primary install surface.

Each tagged release should publish:

- `mm-darwin-arm64`
- `mm-linux-x86_64`
- `SHA256SUMS`
- `install.sh`
- the existing wheel and sdist artifacts as secondary artifacts for development and internal packaging workflows

The default install command becomes:

```bash
curl -fsSL https://github.com/iamazhar/muscle-memory/releases/latest/download/install.sh | sh
```

Pinned installs should also be supported by an environment override or versioned release URL, but `latest` is the default user-facing path.

## Recommended Approach

Use GitHub Release binaries plus a small shell installer.

The installer downloads the matching prebuilt binary from the release rather than invoking `uv`, `pip`, or PyPI. This makes installation independent of Python packaging infrastructure and keeps PyPI out of the critical path. Wheel and sdist outputs remain available as secondary artifacts so the existing Python package validation path is preserved while the public install story changes.

## Binary Build Strategy

The first supported targets are:

- `darwin-arm64`
- `linux-x86_64`

Binary builds should run on native GitHub-hosted runners:

- `macos-latest` for `darwin-arm64`
- `ubuntu-latest` for `linux-x86_64`

The binary packaging implementation should produce a single executable named `mm` and then rename the release asset to a stable platform-qualified filename. PyInstaller is the preferred implementation for the first pass because the project already has historical precedent for standalone binary distribution and it produces a self-contained executable without requiring Python on the target machine.

The binary build path should live in a dedicated script so release jobs and local verification use the same commands.

## Release Workflow Changes

The release workflow should be updated so GitHub Release publication is complete without PyPI.

Required release flow:

1. Validate version metadata and changelog.
2. Run lint, typecheck, tests, and existing release preflight checks.
3. Build wheel and sdist artifacts.
4. Build `darwin-arm64` and `linux-x86_64` standalone binaries.
5. Generate `SHA256SUMS` for all published assets.
6. Publish `install.sh`, binaries, checksums, wheel, and sdist to the GitHub Release.
7. Create the tag and GitHub Release as today.

PyPI publication should no longer be part of the success path for a release. The simplest first-pass rule is:

- keep the workflow input for PyPI only if needed for compatibility
- default it to `false`
- do not treat PyPI publish failure as a blocker for the release

This ensures the release process stays usable even while PyPI trusted publishing remains misconfigured.

## Installer Behavior

`install.sh` should be intentionally small and predictable.

Behavior:

- detect platform from `uname -s` and `uname -m`
- map only the supported targets:
  - `Darwin arm64` -> `mm-darwin-arm64`
  - `Linux x86_64` -> `mm-linux-x86_64`
- determine the release URL from either:
  - `latest`
  - an explicit version override such as `MM_VERSION=0.11.0`
- download the target binary and `SHA256SUMS`
- verify the binary checksum before install
- install to `${MM_INSTALL_DIR:-$HOME/.local/bin}`
- mark the binary executable
- print a PATH hint when the install directory is not already on `PATH`

Failure rules:

- unsupported platform -> fail with a precise unsupported-platform message
- checksum mismatch -> fail hard
- missing `curl` -> fail with prerequisite guidance
- missing checksum tool (`shasum -a 256` or `sha256sum`) -> fail with prerequisite guidance
- no silent fallback to Python install

## User-Facing Documentation Changes

The README and install-facing docs should move from `uv tool install muscle-memory` to the `curl` installer as the primary path.

Required documentation updates:

- README quickstart uses `curl | sh`
- include a pinned-version example
- include PATH guidance for `~/.local/bin`
- keep developer-oriented package install instructions in development docs, not as the main user install path
- explain that GitHub Releases are now the canonical distribution source

The current PyPI-oriented install story should be downgraded to optional/internal status or removed from the main documentation flow.

## Verification Strategy

The change should be considered complete only if distribution is tested end to end.

Required verification:

- existing Python test suite remains green
- release workflow smoke-tests the standalone binary directly on both supported platforms
- installer script gets a shell-level smoke test in CI
- release workflow verifies wheel/sdist metadata and binary checksums together
- at least one CI path validates that `curl` installer logic can fetch and install the expected asset layout

The package job in CI should continue validating wheel and sdist outputs so secondary packaging remains healthy.

## Risks And Constraints

- Linux binary portability will be limited to the runtime characteristics of the build environment in the first pass. This is acceptable for the initial release as long as the target is documented as `linux-x86_64` on modern distributions.
- `curl | sh` increases the importance of checksum verification; the installer must verify before install.
- Release complexity increases because platform-specific binary builds are added, so the workflow should keep artifact naming and smoke tests simple.

## Migration Plan

Phase 1:

- add binary build script(s)
- add `install.sh`
- extend release workflow to publish binaries and checksums
- update docs to prefer `curl`

Future follow-up work outside this spec:

- revisit Homebrew integration
- add more binary targets
- decide whether wheel/sdist should remain public-facing or become secondary-only

## Success Criteria

This shift is successful when:

- a user on supported macOS arm64 or Linux x86_64 can install `mm` with `curl | sh`
- the installer does not require Python, `uv`, or PyPI
- the release workflow publishes binaries plus checksums on every tagged release
- GitHub Releases, not PyPI, are sufficient to distribute the tool
- existing repo verification remains green after the workflow change
