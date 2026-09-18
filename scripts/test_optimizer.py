#!/usr/bin/env python3
"""Offline check of the LP optimizer, with the LLM taken out of the picture.

Feeds each public sample case the *reference* directives, so any failure here is
an optimizer or validator bug rather than an interpretation bug.

    python scripts/test_optimizer.py

Every case must MATCH the reference optimal cost and pass the replay validator.
"""

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from app.models import OptimizeRequest  # noqa: E402
from app.optimizer import build_optimal_plan  # noqa: E402
from app.planner import totals_from_plan  # noqa: E402
from app.validator import TOL, replay  # noqa: E402

GREEN, RED, DIM, RESET = "\033[32m", "\033[31m", "\033[2m", "\033[0m"

SAMPLES = (
    REPO
    / "BUP_CSE_FEST_2026_Participant_Docs"
    / "BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json"
)


def main() -> int:
    cases = json.loads(SAMPLES.read_text(encoding="utf-8"))["cases"]

    header = f"{'case':<11}{'ref cost':>12}{'our cost':>12}{'delta':>10}{'valid':>9}{'route':>18}{'ms':>8}"
    print()
    print(header)
    print("-" * len(header))

    failures = 0
    for case in cases:
        req = OptimizeRequest.model_validate(case["input"])
        expected = case["expected_output"]
        directives = [
            entry for entry in expected["directive_interpretation"] if entry["applies"]
        ]

        started = time.perf_counter()
        rows, route = build_optimal_plan(req, directives)
        ms = (time.perf_counter() - started) * 1000

        total_grid, total_cost, peak = totals_from_plan(rows, req.hours_by_hour())
        response = {
            "scenario_id": req.scenario_id,
            "directive_interpretation": expected["directive_interpretation"],
            "hourly_plan": [row.model_dump() for row in rows],
            "total_grid_kwh": total_grid,
            "total_cost_bdt": total_cost,
            "peak_grid_kwh": peak,
            "plan_summary": "test",
        }

        # Replay against organizer ground truth, exactly as the judge does.
        problems = replay(case["input"], response, directives)
        ref = expected["total_cost_bdt"]
        delta = total_cost - ref
        cost_ok = abs(delta) <= TOL
        valid = not problems
        if not (cost_ok and valid):
            failures += 1

        print(
            f"{case['id']:<11}{ref:>12,.2f}{total_cost:>12,.2f}"
            f"{(GREEN if cost_ok else RED)}{delta:>10.4f}{RESET}"
            f"{(GREEN + 'pass' if valid else RED + 'FAIL') + RESET:>18}"
            f"{route:>18}{ms:>8.1f}"
        )
        for problem in problems[:5]:
            print(f"    {RED}- {problem}{RESET}")

    print("-" * len(header))
    if failures:
        print(f"\n{RED}{failures} case(s) failed.{RESET}\n")
        return 1
    print(f"\n{GREEN}10/10 exact optimal, 10/10 valid against ground-truth directives.{RESET}")
    print(f"{DIM}Optimization Quality and Directive Application are now covered.{RESET}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
