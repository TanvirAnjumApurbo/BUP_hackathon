"""Independent replay validator.

This re-runs, on plain dicts, every check the judge is documented to run. It is
deliberately written against raw JSON (not pydantic models) so the exact same
code can be used three ways:

  1. inside the service, as the final gate before we return a response
  2. by scripts/test_samples.py, against a deployed URL
  3. later, against randomly generated scenarios

`replay()` returns a list of human-readable problems. Empty list == valid.
"""

from typing import Any, Dict, List, Optional, Sequence

# Problem Statement 11.5: absolute tolerance of 0.01 kWh / 0.01 BDT.
TOL = 0.01

RESPONSE_KEYS = {
    "scenario_id",
    "directive_interpretation",
    "hourly_plan",
    "total_grid_kwh",
    "total_cost_bdt",
    "peak_grid_kwh",
    "plan_summary",
}

DIRECTIVE_TYPES = {
    "solar_reduction",
    "minimum_battery_reserve",
    "no_charge_window",
    "no_discharge_window",
    "max_grid_window",
    "no_op",
}

# directive_type -> required structured_adjustment keys
ADJUSTMENT_SHAPE = {
    "solar_reduction": {"hours", "factor"},
    "minimum_battery_reserve": {"hours", "minimum_energy_kwh"},
    "no_charge_window": {"hours"},
    "no_discharge_window": {"hours"},
    "max_grid_window": {"hours", "max_grid_kwh"},
}

BATTERY_ACTIONS = {"charge", "discharge", "idle"}


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and x == x and abs(x) != float("inf")


def check_hours_list(hours: Any) -> Optional[str]:
    """Directive hours must be unique ints 0-23 in ascending order."""
    if not isinstance(hours, list) or not hours:
        return "hours must be a non-empty list"
    for h in hours:
        if not isinstance(h, int) or isinstance(h, bool):
            return f"hour {h!r} is not an integer"
        if not 0 <= h <= 23:
            return f"hour {h} is outside 0-23"
    if len(set(hours)) != len(hours):
        return "hours contains duplicates"
    if hours != sorted(hours):
        return "hours is not in ascending order"
    return None


def check_interpretation(
    notes: Sequence[str], interpretation: Any, capacity_kwh: float
) -> List[str]:
    """Schema and semantics of directive_interpretation, independent of the plan."""
    problems: List[str] = []
    if not isinstance(interpretation, list):
        return ["directive_interpretation is not a list"]

    if len(interpretation) != len(notes):
        problems.append(
            f"directive_interpretation has {len(interpretation)} entries "
            f"for {len(notes)} operator_notes"
        )

    for position, entry in enumerate(interpretation):
        tag = f"directive_interpretation[{position}]"
        if not isinstance(entry, dict):
            problems.append(f"{tag} is not an object")
            continue

        if entry.get("note_index") != position:
            problems.append(
                f"{tag} has note_index {entry.get('note_index')!r}, expected {position} "
                "(one entry per note, in note_index order)"
            )

        dtype = entry.get("directive_type")
        if dtype not in DIRECTIVE_TYPES:
            problems.append(f"{tag} has unsupported directive_type {dtype!r}")
            continue

        applies = entry.get("applies")
        adjustment = entry.get("structured_adjustment")

        if dtype == "no_op":
            if applies is not False:
                problems.append(f"{tag} is no_op but applies is {applies!r}, must be false")
            if adjustment is not None:
                problems.append(f"{tag} is no_op but structured_adjustment is not null")
        else:
            if applies is not True:
                problems.append(f"{tag} is {dtype} but applies is {applies!r}, must be true")
            if not isinstance(adjustment, dict):
                problems.append(f"{tag} is {dtype} but structured_adjustment is not an object")
                continue

            required = ADJUSTMENT_SHAPE[dtype]
            missing = required - set(adjustment)
            unexpected = set(adjustment) - required
            if missing:
                problems.append(f"{tag} structured_adjustment missing {sorted(missing)}")
            if unexpected:
                problems.append(f"{tag} structured_adjustment has unexpected {sorted(unexpected)}")

            hours_problem = check_hours_list(adjustment.get("hours"))
            if hours_problem:
                problems.append(f"{tag} structured_adjustment.hours: {hours_problem}")

            if dtype == "solar_reduction":
                factor = adjustment.get("factor")
                if not _finite(factor) or not 0.0 <= float(factor) <= 1.0:
                    problems.append(f"{tag} factor {factor!r} is not within 0..1")
            elif dtype == "minimum_battery_reserve":
                reserve = adjustment.get("minimum_energy_kwh")
                if not _finite(reserve) or float(reserve) < 0:
                    problems.append(f"{tag} minimum_energy_kwh {reserve!r} is not finite and >= 0")
                elif float(reserve) > capacity_kwh:
                    problems.append(
                        f"{tag} minimum_energy_kwh {reserve} exceeds battery capacity {capacity_kwh}"
                    )
            elif dtype == "max_grid_window":
                cap = adjustment.get("max_grid_kwh")
                if not _finite(cap) or float(cap) < 0:
                    problems.append(f"{tag} max_grid_kwh {cap!r} is not finite and >= 0")

        if not isinstance(entry.get("explanation"), str) or not entry["explanation"].strip():
            problems.append(f"{tag} explanation must be a non-empty string")

    return problems


def directive_effects(request: Dict[str, Any], directives: Sequence[Dict[str, Any]]):
    """Fold applicable directives into per-hour effects used by the replay."""
    hours = {int(h["hour"]): h for h in request["hours"]}
    battery = request["battery"]

    # Overlapping directives of the same kind combine by taking the most
    # restrictive single value, never by compounding. Two notes saying "solar at
    # 50%" and "solar at 25%" mean 25% of forecast, not 12.5% — the physical
    # condition is the worse of the two, not the product of both.
    solar_factor = {h: 1.0 for h in range(24)}
    grid_cap: Dict[int, Optional[float]] = {h: None for h in range(24)}
    floor = {h: float(battery["minimum_energy_kwh"]) for h in range(24)}
    no_charge = set()
    no_discharge = set()

    for entry in directives:
        dtype = entry.get("directive_type")
        adjustment = entry.get("structured_adjustment") or {}
        listed = adjustment.get("hours") or []
        for h in listed:
            if not isinstance(h, int) or not 0 <= h <= 23:
                continue
            if dtype == "solar_reduction":
                solar_factor[h] = min(solar_factor[h], float(adjustment["factor"]))
            elif dtype == "max_grid_window":
                cap = float(adjustment["max_grid_kwh"])
                grid_cap[h] = cap if grid_cap[h] is None else min(grid_cap[h], cap)
            elif dtype == "minimum_battery_reserve":
                floor[h] = max(floor[h], float(adjustment["minimum_energy_kwh"]))
            elif dtype == "no_charge_window":
                no_charge.add(h)
            elif dtype == "no_discharge_window":
                no_discharge.add(h)

    effective_solar = {
        h: float(hours[h]["solar_kwh"]) * solar_factor[h] for h in range(24)
    }
    return effective_solar, grid_cap, floor, no_charge, no_discharge


def replay(
    request: Dict[str, Any],
    response: Dict[str, Any],
    directives: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[str]:
    """Re-run every documented judge check.

    `directives` lets a caller replay against organizer ground truth. When it is
    None we replay against whatever the response itself claimed, which is the
    self-consistency check the service runs on its own output.
    """
    problems: List[str] = []

    if not isinstance(response, dict):
        return ["response is not a JSON object"]

    missing = RESPONSE_KEYS - set(response)
    extra = set(response) - RESPONSE_KEYS
    if missing:
        problems.append(f"response missing required fields {sorted(missing)}")
    if extra:
        problems.append(f"response has extra top-level fields {sorted(extra)}")

    if response.get("scenario_id") != request.get("scenario_id"):
        problems.append(
            f"scenario_id {response.get('scenario_id')!r} does not echo "
            f"request {request.get('scenario_id')!r}"
        )

    if not isinstance(response.get("plan_summary"), str) or not response["plan_summary"].strip():
        problems.append("plan_summary must be a non-empty string")

    battery = request["battery"]
    capacity = float(battery["capacity_kwh"])
    initial = float(battery["initial_energy_kwh"])
    max_charge = float(battery["max_charge_kwh_per_hour"])
    max_discharge = float(battery["max_discharge_kwh_per_hour"])
    hours = {int(h["hour"]): h for h in request["hours"]}

    interpretation = response.get("directive_interpretation")
    problems += check_interpretation(request["operator_notes"], interpretation, capacity)

    if directives is None:
        directives = [
            entry
            for entry in (interpretation if isinstance(interpretation, list) else [])
            if isinstance(entry, dict) and entry.get("applies") and entry.get("directive_type") != "no_op"
        ]

    effective_solar, grid_cap, floor, no_charge, no_discharge = directive_effects(
        request, directives
    )

    plan = response.get("hourly_plan")
    if not isinstance(plan, list):
        problems.append("hourly_plan is not a list")
        return problems

    seen = [row.get("hour") for row in plan if isinstance(row, dict)]
    if sorted(x for x in seen if isinstance(x, int)) != list(range(24)) or len(plan) != 24:
        problems.append("hourly_plan must contain exactly 24 entries, one per hour 0 through 23")
        return problems

    rows = {int(row["hour"]): row for row in plan}
    energy = initial
    recomputed_grid = 0.0
    recomputed_cost = 0.0
    peak = 0.0

    for h in range(24):
        row = rows[h]
        tag = f"hour {h}"

        grid = row.get("grid_kwh")
        solar_used = row.get("solar_used_kwh")
        action = row.get("battery_action")
        amount = row.get("battery_kwh")
        after = row.get("battery_energy_after_kwh")

        if not all(_finite(v) for v in (grid, solar_used, amount, after)):
            problems.append(f"{tag}: non-finite or non-numeric value in plan row")
            continue
        grid, solar_used, amount, after = float(grid), float(solar_used), float(amount), float(after)

        if grid < -TOL or solar_used < -TOL or amount < -TOL:
            problems.append(f"{tag}: negative energy value")
        if action not in BATTERY_ACTIONS:
            problems.append(f"{tag}: battery_action {action!r} is not one of {sorted(BATTERY_ACTIONS)}")
            continue

        charge = amount if action == "charge" else 0.0
        discharge = amount if action == "discharge" else 0.0
        if action == "idle" and abs(amount) > TOL:
            problems.append(f"{tag}: battery_action is idle but battery_kwh is {amount}")

        # Energy balance
        demand = float(hours[h]["demand_kwh"])
        lhs = grid + solar_used + discharge
        rhs = demand + charge
        if abs(lhs - rhs) > TOL:
            problems.append(
                f"{tag}: energy balance broken — grid {grid} + solar {solar_used} + discharge "
                f"{discharge} = {lhs:.4f}, expected demand {demand} + charge {charge} = {rhs:.4f}"
            )

        # Solar availability after directives
        if solar_used > effective_solar[h] + TOL:
            problems.append(
                f"{tag}: solar_used {solar_used} exceeds effective solar {effective_solar[h]:.4f}"
            )

        # Rate limits
        if charge > max_charge + TOL:
            problems.append(f"{tag}: charge {charge} exceeds max_charge_kwh_per_hour {max_charge}")
        if discharge > max_discharge + TOL:
            problems.append(
                f"{tag}: discharge {discharge} exceeds max_discharge_kwh_per_hour {max_discharge}"
            )

        # Directive windows
        if h in no_charge and charge > TOL:
            problems.append(f"{tag}: charged {charge} inside a no_charge_window")
        if h in no_discharge and discharge > TOL:
            problems.append(f"{tag}: discharged {discharge} inside a no_discharge_window")
        if grid_cap[h] is not None and grid > grid_cap[h] + TOL:
            problems.append(f"{tag}: grid {grid} exceeds max_grid_window cap {grid_cap[h]}")

        # Battery state transition
        expected_after = energy + charge - discharge
        if abs(after - expected_after) > TOL:
            problems.append(
                f"{tag}: battery_energy_after_kwh {after} does not follow from "
                f"{energy:.4f} {action} {amount} (expected {expected_after:.4f})"
            )
        energy = after

        if energy < floor[h] - TOL:
            problems.append(
                f"{tag}: battery energy {energy} is below the required floor {floor[h]}"
            )
        if energy > capacity + TOL:
            problems.append(f"{tag}: battery energy {energy} exceeds capacity {capacity}")

        recomputed_grid += grid
        recomputed_cost += grid * float(hours[h]["tariff_bdt_per_kwh"])
        peak = max(peak, grid)

    # End-of-day neutrality
    if abs(energy - initial) > TOL:
        problems.append(
            f"end-of-day battery energy {energy} does not return to initial {initial}"
        )

    # Reported totals must match a recount of the rows actually emitted
    for name, reported, expected in (
        ("total_grid_kwh", response.get("total_grid_kwh"), recomputed_grid),
        ("total_cost_bdt", response.get("total_cost_bdt"), recomputed_cost),
        ("peak_grid_kwh", response.get("peak_grid_kwh"), peak),
    ):
        if not _finite(reported):
            problems.append(f"{name} is not a finite number")
        elif abs(float(reported) - expected) > TOL:
            problems.append(
                f"{name} reported {reported} but hourly_plan recomputes to {expected:.4f}"
            )

    return problems
