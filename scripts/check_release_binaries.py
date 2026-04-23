"""CLI wrapper for standalone release binary smoke checks."""

from muscle_memory.release_binaries import smoke_main

if __name__ == "__main__":
    raise SystemExit(smoke_main())
