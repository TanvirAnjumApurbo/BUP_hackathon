#!/usr/bin/env python3
"""Overlapping directives of the same kind must combine by the most restrictive
single value, never by compounding.

    python scripts/test_overlaps.py

This is a regression guard. `directive_effects` is the one shared source of truth
for both the optimizer and the replay validator, so a wrong combining rule here
would be *self-consistently* wrong: we would build a plan against the wrong solar
ceiling and our own validator would happily agree, while the judge replays
against the true value.
"""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from app.models import OptimizeRequest  # noqa: E402
from app.optimizer import build_optimal_plan  # noqa: E402
from app.planner import totals_from_plan  # noqa: E402
from app.validator import directive_effects, replay  # noqa: E402

GREEN, RED, RESET = "\033[32m", "\033[31m", "\033[0m"

SAMPLES = (
    REPO
    / "BUP_CSE_FEST_2026_Participant_Docs"
    / "BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json"
)


def directive(dtype, hours, **value):
    return {"directive_type": dtype, "applies": True, "structured_adjustment": {"hours": hours, **value}}


def main() -> int:
    case = json.loads(SAMPLES.read_text(encoding="utf-8"))["cases"][0]
    request = case["input"]
    base_solar = {int(h["hour"]): float(h["solar_kwh"]) for h in request["hours"]}
    capacity = request["battery"]["capacity_kwh"]
    base_min = request["battery"]["minimum_energy_kwh"]

    checks = []

    # solar_reduction: min factor, NOT the product.
    eff, _, _, _, _ = directive_effects(
        request,
        [
            directive("solar_reduction", [12, 13], factor=0.5),
            directive("solar_reduction", [12, 13], factor=0.25),
        ],
    )
    checks.append((
        "two solar_reduction -> min factor (0.25), not product (0.125)",
        eff[12],
        base_solar[12] * 0.25,
    ))

    # A single directive is unchanged.
    eff, _, _, _, _ = directive_effects(request, [directive("solar_reduction", [12], factor=0.25)])
    checks.append(("single solar_reduction unchanged", eff[12], base_solar[12] * 0.25))

    # An untouched hour keeps full solar.
    checks.append(("hour outside the window keeps full solar", eff[11], base_solar[11]))

    # max_grid_window: the tighter cap wins.
    _, cap, _, _, _ = directive_effects(
        request,
        [
            directive("max_grid_window", [18, 19], max_grid_kwh=180),
            directive("max_grid_window", [18, 19], max_grid_kwh=155),
        ],
    )
    checks.append(("two max_grid_window -> tighter cap (155)", cap[18], 155.0))

    # minimum_battery_reserve: the higher floor wins, and the base minimum is a floor too.
    _, _, floor, _, _ = directive_effects(
        request,
        [
            directive("minimum_battery_reserve", [18], minimum_energy_kwh=80),
            directive("minimum_battery_reserve", [18], minimum_energy_kwh=100),
        ],
    )
    checks.append(("two minimum_battery_reserve -> higher floor (100)", floor[18], 100.0))
    checks.append(("hour with no reserve directive keeps base minimum", floor[0], float(base_min)))

    failures = 0
    print()
    for label, got, want in checks:
        ok = abs(got - want) <= 0.01
        failures += not ok
        mark = f"{GREEN}ok  {RESET}" if ok else f"{RED}BAD {RESET}"
        print(f"  {mark} {label}")
        if not ok:
            print(f"        got {got}, want {want}")

    req = OptimizeRequest.model_validate(request)

    def solve_and_replay(directives):
        rows, route = build_optimal_plan(req, directives)
        total_grid, total_cost, peak = totals_from_plan(rows, req.hours_by_hour())
        response = {
            "scenario_id": req.scenario_id,
            "directive_interpretation": [
                {"note_index": i, "applies": False, "directive_type": "no_op",
                 "structured_adjustment": None, "explanation": "x"}
                for i in range(len(req.operator_notes))
            ],
            "hourly_plan": [row.model_dump() for row in rows],
            "total_grid_kwh": total_grid,
            "total_cost_bdt": total_cost,
            "peak_grid_kwh": peak,
            "plan_summary": "test",
        }
        return rows, route, total_cost, replay(request, response, directives)

    # End to end: a FEASIBLE overlapping set must solve exactly and replay clean.
    feasible = [
        directive("solar_reduction", [12, 13], factor=0.5),
        directive("solar_reduction", [12, 13], factor=0.25),
        directive("minimum_battery_reserve", [18, 19, 20], minimum_energy_kwh=100),
        directive("max_grid_window", [18, 19, 20], max_grid_kwh=180),
        directive("no_charge_window", [2, 3]),
        directive("no_discharge_window", [5, 6]),
    ]
    rows, route, cost, problems = solve_and_replay(feasible)
    ok = not problems and route == "lp"
    failures += not ok
    mark = f"{GREEN}ok  {RESET}" if ok else f"{RED}BAD {RESET}"
    print(f"  {mark} six overlapping feasible directives -> {route}, cost {cost:,.2f}")
    for problem in problems[:5]:
        print(f"        {RED}- {problem}{RESET}")

    # A CONTRADICTORY set must degrade gracefully instead of raising or hanging.
    # Hour 18 needs 205 kWh with no solar; capping grid at 180 while forbidding
    # discharge makes it unsatisfiable. Organizer scenarios are guaranteed
    # feasible, so this only guards against a pathological request.
    contradictory = [
        directive("max_grid_window", [18], max_grid_kwh=180),
        directive("no_discharge_window", [18]),
    ]
    rows, route, cost, _ = solve_and_replay(contradictory)
    ok = route != "lp" and len(rows) == 24
    failures += not ok
    mark = f"{GREEN}ok  {RESET}" if ok else f"{RED}BAD {RESET}"
    print(f"  {mark} contradictory directives -> degrades to '{route}' with 24 valid rows, no crash")

    print()
    if failures:
        print(f"{RED}{failures} check(s) failed.{RESET}\n")
        return 1
    print(f"{GREEN}All overlap-combining rules hold.{RESET}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
