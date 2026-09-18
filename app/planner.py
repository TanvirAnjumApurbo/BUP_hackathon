"""Schedule construction.

STAGE 1 (now): a trivially valid baseline — battery idle all day, use whatever
solar the demand can absorb, buy the rest from the grid. It is not cheap, but it
satisfies every GridWise energy rule, which is what earns points first.

STAGE 2 (next): `build_optimal_plan` replaces the baseline with the LP.
"""

from typing import Dict, List, Tuple

from app.models import HourInput, HourPlan, OptimizeRequest

# Emitted values are rounded here so that the totals we report are sums of the
# exact numbers that appear in hourly_plan, not of hidden internal floats.
ROUND_DP = 6


def r(x: float) -> float:
    return round(float(x), ROUND_DP)


def build_baseline_plan(req: OptimizeRequest) -> List[HourPlan]:
    """Battery idle every hour. Solar used up to demand. Grid covers the rest.

    Valid by construction: energy balance holds, the battery never moves so its
    bounds and rate limits cannot be broken, and end-of-day energy equals the
    initial energy.
    """
    by_hour: Dict[int, HourInput] = req.hours_by_hour()
    resting_energy = r(req.battery.initial_energy_kwh)

    rows: List[HourPlan] = []
    for h in range(24):
        hr = by_hour[h]
        solar_used = r(min(hr.solar_kwh, hr.demand_kwh))
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
