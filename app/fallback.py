"""LAST-RESORT keyword interpreter — degraded mode only.

This is NOT the primary interpretation path and must never become one. It runs
only when every configured language-model provider has failed (outage, quota
exhaustion, repeated malformed output). Its purpose is to keep the service
answering with something defensible instead of collapsing mid-evaluation.

The LLM requirement is satisfied by `app/llm.py`, which is always attempted
first and which produces the interpretation on the normal path. See the README
section "Where the LLM sits".
"""

import logging
import re
from typing import Any, Dict, List, Sequence

from app.timewindows import parse_window

logger = logging.getLogger("gridwise.fallback")

_NEGATION = re.compile(
    r"\b(no|not|never|cannot|can't|don't|do not|must not|may not|without|"
    r"disabled|unavailable|isolated|prohibited|prevented|suspend\w*|halt\w*|stop\w*)\b",
    re.I,
)

_SOLAR = re.compile(r"\b(solar|pv|photovoltaic|panel|inverter|rooftop|array)\b", re.I)
_CHARGE = re.compile(r"\b(charg\w*)\b", re.I)
_DISCHARGE = re.compile(r"\b(discharg\w*|drawn from the battery|draw from the battery)\b", re.I)
_RESERVE = re.compile(
    r"\b(reserve|at least|no lower than|not fall below|not drop below|minimum|"
    r"remain in the battery|stored in the battery|keep)\b",
    re.I,
)
_GRID = re.compile(r"\b(grid|import|intake|feeder|substation|transformer|mains)\b", re.I)
_CAP = re.compile(
    r"\b(not exceed|no more than|at or below|capped|cap|limit\w*|maximum|max|up to|ceiling)\b", re.I
)
_BATTERY = re.compile(r"\b(batter\w*)\b", re.I)


def _classify(note: str) -> str:
    """Best-effort directive type from keywords alone."""
    negated = bool(_NEGATION.search(note))

    # Grid caps first: "grid import must not exceed 155 kWh" also matches negation.
    if _GRID.search(note) and _CAP.search(note):
        return "max_grid_window"

    if _SOLAR.search(note):
        return "solar_reduction"

    if _DISCHARGE.search(note) and negated:
        return "no_discharge_window"

    if _CHARGE.search(note) and negated:
        return "no_charge_window"

    if _BATTERY.search(note) and _RESERVE.search(note):
        return "minimum_battery_reserve"

    return "no_op"


def interpret(notes: Sequence[str], battery: Any) -> List[Dict[str, Any]]:
    """Flat entries in the same shape the model would return.

    Numeric values are deliberately left as None: `app.guardrails` resolves every
    number from the note text anyway, so there is nothing useful to guess here.
    """
    logger.warning("DEGRADED MODE: keyword fallback interpreting %d note(s)", len(notes))

    entries: List[Dict[str, Any]] = []
    for index, note in enumerate(notes):
        directive_type = _classify(note)
        window = parse_window(note)
        if directive_type != "no_op" and not window:
            directive_type = "no_op"

        entries.append(
            {
                "note_index": index,
                "directive_type": directive_type,
                "hours": list(window.hours) if window and directive_type != "no_op" else [],
                "factor": None,
                "minimum_energy_kwh": None,
                "max_grid_kwh": None,
                "explanation": (
                    "This note does not affect today's 24-hour energy schedule."
                    if directive_type == "no_op"
                    else "Interpreted from the operator note."
                ),
            }
        )
    return entries
