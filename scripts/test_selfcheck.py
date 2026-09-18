#!/usr/bin/env python3
"""The judge-replica gate and its fallbacks.

    python scripts/test_selfcheck.py

Covers the paths that only fire when something has gone wrong, and which
therefore never get exercised by the happy-path suites:

  1. the safe plan is valid by construction, on every public case
  2. the safe plan honours solar_reduction (it uses effective solar, not raw)
  3. an infeasible directive is dropped individually, not wholesale
  4. the validator actually catches a tampered response
  5. validation runs on serialized JSON, not on internal Python floats
"""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from app.models import OptimizeRequest, OptimizeResponse  # noqa: E402
from app.optimizer import build_optimal_plan  # noqa: E402
from app.planner import build_safe_plan, totals_from_plan  # noqa: E402
from app.validator import directive_effects, replay  # noqa: E402

GREEN, RED, DIM, RESET = "\033[32m", "\033[31m", "\033[2m", "\033[0m"

SAMPLES = (
    REPO
    / "BUP_CSE_FEST_2026_Participant_Docs"
    / "BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json"
)


def response_for(req, rows, interpretation):
    total_grid, total_cost, peak = totals_from_plan(rows, req.hours_by_hour())
    return OptimizeResponse(
        scenario_id=req.scenario_id,
        directive_interpretation=interpretation,
        hourly_plan=rows,
        total_grid_kwh=total_grid,
        total_cost_bdt=total_cost,
        peak_grid_kwh=peak,
        plan_summary="test",
    )


def serialized(result):
    return json.loads(result.model_dump_json())


def main() -> int:
    cases = json.loads(SAMPLES.read_text(encoding="utf-8"))["cases"]
    failures = 0
    print()

    # 1 + 2. Safe plan valid everywhere, and it respects effective solar.
    worst_solar = 0.0
    for case in cases:
        req = OptimizeRequest.model_validate(case["input"])
        interpretation = case["expected_output"]["directive_interpretation"]
        directives = [d for d in interpretation if d["applies"]]
        effective_solar, _, _, _, _ = directive_effects(req.model_dump(), directives)

        rows = build_safe_plan(req, effective_solar)
        result = response_for(req, rows, interpretation)
        problems = replay(case["input"], serialized(result), directives)

        # physics must always hold; a reserve/grid-cap directive may not be
        # satisfiable while standing still, so only count the structural rules
        structural = [p for p in problems if "reserve" not in p and "max_grid_window" not in p
                      and "below the required floor" not in p]
        if structural:
            failures += 1
            print(f"  {RED}BAD {RESET} safe plan invalid for {case['id']}")
            for p in structural[:3]:
                print(f"        {RED}- {p}{RESET}")
        for row in rows:
            worst_solar = max(worst_solar, row.solar_used_kwh - effective_solar[row.hour])

    ok = worst_solar <= 0.01
    failures += not ok
    print(f"  {(GREEN + 'ok  ' if not failures else RED + 'BAD ') + RESET} safe plan is structurally valid on all 10 cases")
    print(f"  {(GREEN + 'ok  ' if ok else RED + 'BAD ') + RESET} safe plan never exceeds effective solar "
          f"(worst overshoot {worst_solar:.4f} kWh)")

    # 3. An impossible directive is dropped on its own, not wholesale.
    case = cases[0]
    req = OptimizeRequest.model_validate(case["input"])
    impossible = [
        # hour 18 needs 205 kWh with no solar; capping grid at 10 while forbidding
        # discharge cannot be satisfied by any schedule
        {"directive_type": "max_grid_window", "applies": True,
         "structured_adjustment": {"hours": [18], "max_grid_kwh": 10}},
        {"directive_type": "no_discharge_window", "applies": True,
         "structured_adjustment": {"hours": [18]}},
        # a perfectly satisfiable directive that must survive
        {"directive_type": "no_charge_window", "applies": True,
         "structured_adjustment": {"hours": [2, 3]}},
    ]
    rows, route = build_optimal_plan(req, impossible)
    kept_no_charge = all(
        not (row.battery_action == "charge" and row.hour in (2, 3)) for row in rows
    )
    ok = route.startswith("lp-dropped") and kept_no_charge
    failures += not ok
    print(f"  {(GREEN + 'ok  ' if ok else RED + 'BAD ') + RESET} infeasible directive dropped individually "
          f"-> route '{route}', unrelated no_charge_window kept: {kept_no_charge}")

    # 4. The validator catches a tampered response.
    req = OptimizeRequest.model_validate(cases[0]["input"])
    rows, _ = build_optimal_plan(req, [])
    result = response_for(req, rows, cases[0]["expected_output"]["directive_interpretation"])
    tampered = serialized(result)
    tampered["hourly_plan"][5]["grid_kwh"] += 50.0          # breaks energy balance
    caught = replay(cases[0]["input"], tampered, [])
    ok = bool(caught)
    failures += not ok
    print(f"  {(GREEN + 'ok  ' if ok else RED + 'BAD ') + RESET} validator catches a tampered row "
          f"({len(caught)} problem(s) found)")

    # 5. Serialized JSON is what gets validated.
    clean = serialized(result)
    ok = all(not isinstance(v, float) or v == v for v in
             [clean["total_cost_bdt"], clean["total_grid_kwh"], clean["peak_grid_kwh"]])
    ok &= not replay(cases[0]["input"], clean, [])
    failures += not ok
    print(f"  {(GREEN + 'ok  ' if ok else RED + 'BAD ') + RESET} serialized JSON round-trip passes the validator")

    print()
    if failures:
        print(f"{RED}{failures} check(s) failed.{RESET}\n")
        return 1
    print(f"{GREEN}Judge-replica gate and every fallback path hold.{RESET}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
