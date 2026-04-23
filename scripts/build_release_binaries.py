"""CLI wrapper for standalone release binary builds."""

from muscle_memory.release_binaries import build_binaries_main

if __name__ == "__main__":
    raise SystemExit(build_binaries_main())
