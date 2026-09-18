"""Turn untrusted model output into a validated, repaired interpretation.

The model returns a FLAT object per note — directive_type, hours, and at most one
numeric value. It never produces the nested `structured_adjustment`; this module
assembles that, so the model cannot emit a wrong adjustment shape for a given
directive type.

On top of shape safety, four repairs are applied. These are the four mistakes
LLMs actually make on this task, each confirmed against a real provider:

  1. hours off by one at the window end   -> deterministic parse wins (timewindows.py)
  2. solar factor polarity flipped        -> "80% reduction" is 0.2, not 0.8
  3. percent reserve left as a percent    -> "50% of capacity" becomes kWh
  4. a distractor turned into a directive -> relevance re-check forces no_op
"""

import logging
import re
from typing import Any, Dict, List, Optional, Sequence

from app.models import BatteryInput, DirectiveInterpretation
from app.timewindows import parse_window  # noqa: F401

logger = logging.getLogger("gridwise.guardrails")

VALID_TYPES = {
    "solar_reduction",
    "minimum_battery_reserve",
    "no_charge_window",
    "no_discharge_window",
    "max_grid_window",
    "no_op",
}

NO_OP_EXPLANATION = "This note does not affect today's 24-hour energy schedule."

# Words that mean the stated share is what is LOST, so the usable fraction is 1 - p.
_REDUCTION_PATTERNS = (
    re.compile(r"(\d+(?:\.\d+)?)\s*%\s*(?:reduction|cut|decrease|drop|loss|less|lower)", re.I),
    re.compile(
        r"(?:reduc\w*|cut|decreas\w*|drop\w*|fall\w*|declin\w*|lower\w*|down)\s+by\s+"
        r"(?:about\s+|roughly\s+|around\s+|approximately\s+)?(\d+(?:\.\d+)?)\s*%",
        re.I,
    ),
)

_PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s*(?:%|percent)", re.I)
_KWH = re.compile(r"(\d+(?:\.\d+)?)\s*(?:kwh|kw-h|kilowatt[- ]hours?)", re.I)

# Fraction words, which in these notes always describe what REMAINS
# ("will leave roughly one-fifth of normal output").
_FRACTIONS = (
    (re.compile(r"\b(?:one[- ]half|a half|half)\b", re.I), 0.5),
    (re.compile(r"\b(?:one[- ]fifth|a fifth)\b", re.I), 0.2),
    (re.compile(r"\b(?:one[- ]quarter|a quarter|one[- ]fourth)\b", re.I), 0.25),
    (re.compile(r"\b(?:two[- ]thirds)\b", re.I), 2.0 / 3.0),
    (re.compile(r"\b(?:one[- ]third|a third)\b", re.I), 1.0 / 3.0),
    (re.compile(r"\b(?:three[- ]quarters)\b", re.I), 0.75),
)

_SOLAR_OFF = re.compile(
    r"\b(?:offline|off ?line|out of service|no output|zero output|shut down|"
    r"shutdown|completely down|fully down|disconnected)\b",
    re.I,
)

# A note only earns a directive if it talks about today's electricity operation.
# Every noun here takes \w* so plurals match: an earlier version used `panel\b`
# and `inverter\b`, which silently failed on "panels" and "Inverters" and forced
# real directives to no_op.
_ENERGY_WORDS = re.compile(
    r"\b(solar|pv|panel\w*|inverter\w*|photovoltaic\w*|rooftop|array\w*|module\w*|"
    r"batter\w*|charg\w*|discharg\w*|storage|"
    r"grid|import\w*|intake|feeder\w*|substation\w*|transformer\w*|mains|meter\w*|"
    r"kwh|kw|reserve\w*|load\w*|demand\w*|tariff\w*|"
    r"electric\w*|power\w*|energy|generator\w*|genset|ups)\b",
    re.I,
)

# Solar hardware, used to recognise an outage note that the model mislabelled.
_SOLAR_EQUIPMENT = re.compile(
    r"\b(solar|pv|photovoltaic\w*|panel\w*|inverter\w*|rooftop|array\w*|module\w*)\b",
    re.I,
)


def validate_model_output(
    notes: Sequence[str], battery: BatteryInput, raw: Any
) -> List[str]:
    """Pure function. Returns the list of things wrong with the model's output.

    Empty list means the output is acceptable. Anything else is fed back to the
    model verbatim as the retry prompt, so the wording here is written to be
    useful to a model, not just to a human.
    """
    problems: List[str] = []

    if not isinstance(raw, list):
        return ["Output must be a JSON array of interpretation objects."]

    if len(raw) != len(notes):
        problems.append(
            f"Expected exactly {len(notes)} interpretation objects, one per operator note, "
            f"but received {len(raw)}."
        )

    seen_indexes = []
    for position, entry in enumerate(raw):
        tag = f"interpretations[{position}]"
        if not isinstance(entry, dict):
            problems.append(f"{tag} is not an object.")
            continue

        index = entry.get("note_index")
        if not isinstance(index, int) or isinstance(index, bool):
            problems.append(f"{tag}.note_index must be an integer.")
        else:
            seen_indexes.append(index)
            if not 0 <= index < len(notes):
                problems.append(
                    f"{tag}.note_index is {index}, but valid note indexes are 0 to {len(notes) - 1}."
                )

        directive_type = entry.get("directive_type")
        if directive_type not in VALID_TYPES:
            problems.append(
                f"{tag}.directive_type is {directive_type!r}. It must be one of: "
                f"{', '.join(sorted(VALID_TYPES))}."
            )
            continue

        hours = entry.get("hours")
        if directive_type == "no_op":
            if hours:
                problems.append(f"{tag} is no_op, so hours must be an empty array.")
            for field in ("factor", "minimum_energy_kwh", "max_grid_kwh"):
                if entry.get(field) is not None:
                    problems.append(f"{tag} is no_op, so {field} must be null.")
            continue

        if not isinstance(hours, list) or not hours:
            problems.append(f"{tag}.hours must be a non-empty array of integers for {directive_type}.")
        else:
            bad = [h for h in hours if not isinstance(h, int) or isinstance(h, bool) or not 0 <= h <= 23]
            if bad:
                problems.append(f"{tag}.hours contains {bad!r}; every hour must be an integer from 0 to 23.")
            elif len(set(hours)) != len(hours):
                problems.append(f"{tag}.hours contains duplicates: {hours!r}.")
            elif hours != sorted(hours):
                problems.append(f"{tag}.hours must be in ascending order, got {hours!r}.")

        if directive_type == "solar_reduction":
            factor = _number(entry.get("factor"))
            if factor is None:
                problems.append(f"{tag}.factor is required for solar_reduction.")
            elif not 0.0 <= factor <= 1.0:
                problems.append(
                    f"{tag}.factor is {factor}. It must be between 0 and 1 and represents the "
                    f"fraction of solar that REMAINS (an 80% reduction is 0.2)."
                )
        elif directive_type == "minimum_battery_reserve":
            reserve = _number(entry.get("minimum_energy_kwh"))
            if reserve is None:
                problems.append(f"{tag}.minimum_energy_kwh is required for minimum_battery_reserve.")
            elif not 0.0 <= reserve <= battery.capacity_kwh:
                problems.append(
                    f"{tag}.minimum_energy_kwh is {reserve}. It must be in kWh, between 0 and the "
                    f"battery capacity of {battery.capacity_kwh} kWh. If the note gives a percentage, "
                    f"multiply it by {battery.capacity_kwh}."
                )
        elif directive_type == "max_grid_window":
            cap = _number(entry.get("max_grid_kwh"))
            if cap is None:
                problems.append(f"{tag}.max_grid_kwh is required for max_grid_window.")
            elif cap < 0:
                problems.append(f"{tag}.max_grid_kwh is {cap}; it must be zero or greater.")

    expected_indexes = list(range(len(notes)))
    if seen_indexes and sorted(seen_indexes) != expected_indexes:
        problems.append(
            f"note_index values must be exactly {expected_indexes}, each appearing once, "
            f"in ascending order. Got {seen_indexes}."
        )

    return problems


def _clean_hours(raw: Any) -> List[int]:
    """Unique integers 0-23 in ascending order, silently dropping the rest."""
    if not isinstance(raw, (list, tuple)):
        return []
    out = set()
    for item in raw:
        if isinstance(item, bool):
            continue
        if isinstance(item, int):
            hour = item
        elif isinstance(item, float) and item.is_integer():
            hour = int(item)
        else:
            continue
        if 0 <= hour <= 23:
            out.add(hour)
    return sorted(out)


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if number == number and abs(number) != float("inf") else None
    return None


def resolve_hours(note: str, model_hours: Any) -> tuple[List[int], str]:
    """Deterministic parse wins whenever the note states the window unambiguously.

    Returns (hours, provenance) where provenance records who decided, so the
    disagreement rate between model and parser is visible in the logs.
    """
    cleaned = _clean_hours(model_hours)
    window = parse_window(note)

    if window and window.confidence == "high":
        if cleaned and cleaned != window.hours:
            logger.info(
                "hours override: model said %s, parsed %s from %r",
                cleaned, window.hours, window.source,
            )
            return window.hours, "parser-override"
        return window.hours, "parser"

    # Ambiguous wording such as "from one until three" — the model reads context
    # better than a regex can, so it keeps the decision.
    if cleaned:
        return cleaned, "model"
    if window:
        return window.hours, "parser-low-confidence"
    return [], "none"


def resolve_solar_factor(note: str, model_factor: Any) -> float:
    """Usable fraction of solar that REMAINS. Text wins over the model."""
    if _SOLAR_OFF.search(note):
        return 0.0

    for pattern in _REDUCTION_PATTERNS:
        match = pattern.search(note)
        if match:
            return max(0.0, min(1.0, 1.0 - float(match.group(1)) / 100.0))

    for pattern, value in _FRACTIONS:
        if pattern.search(note):
            return value

    match = _PERCENT.search(note)
    if match:
        return max(0.0, min(1.0, float(match.group(1)) / 100.0))

    factor = _number(model_factor)
    if factor is None:
        return 1.0
    return max(0.0, min(1.0, factor))


def resolve_reserve(note: str, model_value: Any, capacity_kwh: float) -> Optional[float]:
    """Reserve in kWh. A percentage in the text is resolved against capacity."""
    match = _KWH.search(note)
    if match:
        return max(0.0, min(capacity_kwh, float(match.group(1))))

    match = _PERCENT.search(note)
    if match:
        return max(0.0, min(capacity_kwh, float(match.group(1)) / 100.0 * capacity_kwh))

    for pattern, fraction in _FRACTIONS:
        if pattern.search(note):
            return max(0.0, min(capacity_kwh, fraction * capacity_kwh))

    value = _number(model_value)
    if value is None:
        return None
    # The model answered in percent when it should have answered in kWh.
    if 0 < value <= 100 and value < capacity_kwh and _PERCENT.search(note):
        value = value / 100.0 * capacity_kwh
    return max(0.0, min(capacity_kwh, value))


def resolve_grid_cap(note: str, model_value: Any) -> Optional[float]:
    match = _KWH.search(note)
    if match:
        return float(match.group(1))
    value = _number(model_value)
    return value if value is not None and value >= 0 else None


def looks_irrelevant(note: str) -> bool:
    """Final defence against a distractor being promoted into a real directive.

    Deliberately conservative. Forcing a genuine directive to no_op is the most
    expensive mistake available — it loses the interpretation point *and* makes
    the schedule violate the true directive, which costs the optimization credit
    for that case too. Letting a distractor through only costs the first.

    So a note is only overridden when it mentions nothing electrical AND states
    no time window. A real distractor ("the seminar room booking moved to next
    week") has neither.
    """
    if _ENERGY_WORDS.search(note or ""):
        return False
    return parse_window(note or "") is None


def reclassify_solar_outage(note: str, directive_type: str) -> str:
    """Solar hardware plus an outage phrase is a solar_reduction, whatever the model said.

    Observed on a real provider, from the same note on different runs:
      "Inverters are fully offline from 1 PM until 4 PM"
        -> no_discharge_window  (an offline inverter was read as blocking the battery)
        -> no_op                (the note was read as not affecting the schedule)
    Both are wrong; the note is about generation, not storage.

    Overriding no_op needs all three signals — named solar hardware, an explicit
    outage phrase, and a parseable time window. A distractor has none of them, so
    this cannot promote one into a directive.
    """
    if directive_type == "solar_reduction":
        return directive_type
    if not (_SOLAR_EQUIPMENT.search(note) and _SOLAR_OFF.search(note)):
        return directive_type
    if directive_type == "no_op" and parse_window(note) is None:
        return directive_type
    logger.info("reclassified %s -> solar_reduction (solar hardware outage)", directive_type)
    return "solar_reduction"


def _no_op(index: int, explanation: str = NO_OP_EXPLANATION) -> DirectiveInterpretation:
    return DirectiveInterpretation(
        note_index=index,
        applies=False,
        directive_type="no_op",
        structured_adjustment=None,
        explanation=explanation,
    )


def build_interpretations(
    notes: Sequence[str],
    battery: BatteryInput,
    model_output: Optional[Sequence[Dict[str, Any]]],
) -> List[DirectiveInterpretation]:
    """One validated entry per note, in note_index order. Never raises."""
    by_index: Dict[int, Dict[str, Any]] = {}
    for position, entry in enumerate(model_output or []):
        if not isinstance(entry, dict):
            continue
        index = entry.get("note_index")
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(notes):
            index = position
        by_index.setdefault(index, entry)

    results: List[DirectiveInterpretation] = []
    for index, note in enumerate(notes):
        entry = by_index.get(index) or {}
        directive_type = entry.get("directive_type")
        explanation = entry.get("explanation")
        if not isinstance(explanation, str) or not explanation.strip():
            explanation = "Interpreted from the operator note."

        if directive_type not in VALID_TYPES:
            directive_type = "no_op"

        # Runs before the no_op short-circuit: an unmistakable solar outage must
        # survive the model having called it no_op.
        directive_type = reclassify_solar_outage(note, directive_type)

        if directive_type == "no_op":
            results.append(_no_op(index))
            continue

        if looks_irrelevant(note):
            logger.info("relevance guard forced no_op on note %d", index)
            results.append(_no_op(index))
            continue

        hours, provenance = resolve_hours(note, entry.get("hours"))
        if not hours:
            logger.info("no hours for note %d (%s) — falling back to no_op", index, directive_type)
            results.append(_no_op(index))
            continue

        adjustment: Dict[str, Any] = {"hours": hours}

        if directive_type == "solar_reduction":
            adjustment["factor"] = round(resolve_solar_factor(note, entry.get("factor")), 6)
        elif directive_type == "minimum_battery_reserve":
            reserve = resolve_reserve(note, entry.get("minimum_energy_kwh"), battery.capacity_kwh)
            if reserve is None:
                results.append(_no_op(index))
                continue
            adjustment["minimum_energy_kwh"] = round(reserve, 6)
        elif directive_type == "max_grid_window":
            cap = resolve_grid_cap(note, entry.get("max_grid_kwh"))
            if cap is None:
                results.append(_no_op(index))
                continue
            adjustment["max_grid_kwh"] = round(cap, 6)

        logger.debug("note %d -> %s %s (hours via %s)", index, directive_type, adjustment, provenance)
        results.append(
            DirectiveInterpretation(
                note_index=index,
                applies=True,
                directive_type=directive_type,
                structured_adjustment=adjustment,
                explanation=explanation.strip()[:300],
            )
        )

    return results
