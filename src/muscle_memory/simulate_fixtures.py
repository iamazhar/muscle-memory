"""Hand-written skill + scenario fixtures for `mm simulate`.

Kept separate from `simulate.py` so that users can `--import-fixtures`
their own YAML / JSON libraries without touching the engine. The
fixtures here are intentionally thin — they exist to exercise the
scoring pipeline, not to be production-quality playbooks.
"""

from __future__ import annotations

from muscle_memory.models import Maturity, Scope, Skill
from muscle_memory.simulate import Scenario


def demo_skills() -> list[Skill]:
    """Three fixed-id CANDIDATE skills matched to the demo scenarios.

    Fixed ids let scenarios reference skills by a stable handle and keep
    the DB reproducible across runs — important for `mm list` / `mm show`
    inspection after `mm simulate run`.
    """
    return [
        Skill(
            id="sim-pytest-runner",
            activation=(
                "When pytest fails with ModuleNotFoundError and a repo-local test runner exists"
            ),
            execution=(
                "1. ls tools/ for a test-runner.sh\n"
                "2. If present, invoke ./tools/test-runner.sh instead of bare pytest"
            ),
            termination="Tests pass or the runner is confirmed absent.",
            tool_hints=["Bash: ./tools/test-runner.sh"],
            tags=["python", "testing", "demo"],
            scope=Scope.PROJECT,
            maturity=Maturity.CANDIDATE,
        ),
        Skill(
            id="sim-venv-activate",
            activation=("When `python` or `pip` fails because the virtualenv is not active"),
            execution=(
                "1. Check for .venv/ or venv/ in the project root\n"
                "2. Re-run with ./.venv/bin/python (or source the activate script)"
            ),
            termination="Import succeeds or no venv is found.",
            tool_hints=["Bash: ./.venv/bin/python"],
            tags=["python", "env", "demo"],
            scope=Scope.PROJECT,
            maturity=Maturity.CANDIDATE,
        ),
        Skill(
            id="sim-flaky-retry",
            activation="When a flaky test fails intermittently in CI",
            execution=(
                "1. Re-run the single test with pytest -x --lf\n"
                "2. If still failing, the test is deterministic — debug it"
            ),
            termination="Test passes twice in a row or is filed as a real bug.",
            tags=["testing", "flaky", "demo"],
            scope=Scope.PROJECT,
            maturity=Maturity.CANDIDATE,
        ),
    ]


def demo_scenarios() -> list[Scenario]:
    """Three scenarios that exercise distinct points on the score curve.

    Expected end-state after `mm simulate run`:
      * sim-pytest-runner → PROVEN  (100%, score 1.0, >=10 successes —
        deterministic).
      * sim-venv-activate → CANDIDATE (mean ~0.6 score, but stays
        CANDIDATE in Layer 1). LIVE promotion needs >=2 distinct
        source_episode_ids, which only the extractor can populate —
        Layer 1 drives usage, not provenance.
      * sim-flaky-retry   → pruned on `--prune` (score <= 0.2 at 5+
        invocations triggers the loser rule in Scorer.prune).
    """
    return [
        Scenario(
            name="golden-path",
            prompt="pytest fails with ModuleNotFoundError",
            activated_skills=["sim-pytest-runner"],
            success_rate=1.0,
            n=20,
            tags=["demo", "promote"],
        ),
        Scenario(
            name="mixed-signal",
            prompt="ModuleNotFoundError when running python",
            activated_skills=["sim-venv-activate"],
            success_rate=0.6,
            n=20,
            tags=["demo", "borderline"],
        ),
        Scenario(
            name="loser",
            prompt="flaky test fails sometimes",
            activated_skills=["sim-flaky-retry"],
            success_rate=0.2,
            n=20,
            tags=["demo", "prune"],
        ),
    ]


__all__ = ["demo_skills", "demo_scenarios"]
