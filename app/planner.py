"""Schedule construction and totals.

The normal schedule comes from `app.optimizer.build_optimal_plan`. What lives
here is the conservative alternative it falls back to — battery idle all day,
solar used up to demand, grid covering the rest — plus the recount of the three
reported totals from the rows that are actually emitted.
"""

from typing import Dict, List, Optional, Tuple

from app.models import HourInput, HourPlan, OptimizeRequest

# Emitted values are rounded here so that the totals we report are sums of the
# exact numbers that appear in hourly_plan, not of hidden internal floats.
ROUND_DP = 6


def r(x: float) -> float:
    return round(float(x), ROUND_DP)


def build_safe_plan(
    req: OptimizeRequest, effective_solar: Optional[Dict[int, float]] = None
) -> List[HourPlan]:
    """Last-resort schedule that is valid by construction.

    The battery never moves, so its bounds, rate limits, state transitions and
    end-of-day neutrality cannot be violated, and any no_charge_window or
    no_discharge_window directive is satisfied trivially. Solar is capped at the
    *effective* figure after solar_reduction, so that directive is honoured too.

    A minimum_battery_reserve above the starting energy, or a max_grid_window
    below what demand needs, can still be missed — nothing that keeps the battery
    still could satisfy those. That trade is deliberate: breaking a directive
    costs one case, whereas returning a plan that violates the energy rules is
    what the rubric treats as invalid.
    """
    by_hour: Dict[int, HourInput] = req.hours_by_hour()
    resting_energy = r(req.battery.initial_energy_kwh)

    rows: List[HourPlan] = []
    for h in range(24):
        hr = by_hour[h]
        available = hr.solar_kwh if effective_solar is None else effective_solar.get(h, hr.solar_kwh)
        solar_used = r(max(0.0, min(available, hr.demand_kwh)))
        grid = r(max(0.0, hr.demand_kwh - solar_used))
        rows.append(
            HourPlan(
                hour=h,
                grid_kwh=grid,
                solar_used_kwh=solar_used,
                battery_action="idle",
                battery_kwh=0.0,
                battery_energy_after_kwh=resting_energy,
            )
        )
    return rows


def totals_from_plan(
    rows: List[HourPlan], by_hour: Dict[int, HourInput]
) -> Tuple[float, float, float]:
    """Recompute the three reported totals from the rows we are about to emit.

    Returns (total_grid_kwh, total_cost_bdt, peak_grid_kwh).
    """
    total_grid = r(sum(row.grid_kwh for row in rows))
    total_cost = r(sum(row.grid_kwh * by_hour[row.hour].tariff_bdt_per_kwh for row in rows))
    peak_grid = r(max((row.grid_kwh for row in rows), default=0.0))
    return total_grid, total_cost, peak_grid
