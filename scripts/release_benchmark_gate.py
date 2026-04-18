"""CLI wrapper for the reproducible release benchmark gate."""

from muscle_memory.release_preflight import benchmark_gate_main

if __name__ == "__main__":
    raise SystemExit(benchmark_gate_main())
