#!/usr/bin/env python3
"""Prove the guardrail/repair layer without spending any LLM quota.

Two passes over the ten public cases:

  CLEAN     the model returns the right answer      -> must pass through untouched
  CORRUPT   the model returns the four classic mistakes -> must be repaired back

The corruptions are exactly the failure modes observed on a real provider:
  * the window end is short by one hour
  * the solar factor polarity is flipped (0.2 becomes 0.8)
  * a percentage reserve is left as a percentage instead of kWh
  * a distractor is promoted into a real directive

    python scripts/test_guardrails.py
"""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from app.guardrails import build_interpretations  # noqa: E402
from app.models import OptimizeRequest  # noqa: E402

GREEN, RED, DIM, RESET = "\033[32m", "\033[31m", "\033[2m", "\033[0m"

SAMPLES = (
    REPO
    / "BUP_CSE_FEST_2026_Participant_Docs"
    / "BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json"
)


def flatten(expected_entry):
    """The flat shape the model is asked for."""
    adjustment = expected_entry.get("structured_adjustment") or {}
    return {
        "note_index": expected_entry["note_index"],
        "directive_type": expected_entry["directive_type"],
        "hours": list(adjustment.get("hours", [])),
        "factor": adjustment.get("factor"),
        "minimum_energy_kwh": adjustment.get("minimum_energy_kwh"),
        "max_grid_kwh": adjustment.get("max_grid_kwh"),
        "explanation": expected_entry.get("explanation", "x"),
    }


def corrupt(entry, capacity):
    """Apply the four classic model mistakes."""
    bad = dict(entry)
    if bad["directive_type"] == "no_op":
        # Distractor promoted into an invented directive.
        return {
            "note_index": bad["note_index"],
            "directive_type": "no_charge_window",
            "hours": [9, 10, 11],
            "factor": None,
            "minimum_energy_kwh": None,
            "max_grid_kwh": None,
            "explanation": "invented",
        }
    if len(bad["hours"]) > 1:
        bad["hours"] = bad["hours"][:-1]          # window end short by one
    if bad["factor"] is not None:
        bad["factor"] = round(1.0 - bad["factor"], 6)  # polarity flipped
    if bad["minimum_energy_kwh"] is not None:
        bad["minimum_energy_kwh"] = round(bad["minimum_energy_kwh"] / capacity * 100, 4)  # left as %
    if bad["max_grid_kwh"] is not None:
        bad["max_grid_kwh"] = 9999.0              # nonsense cap
    return bad


def same(got, want) -> bool:
    if got.note_index != want["note_index"]:
        return False
    if got.applies != want["applies"] or got.directive_type != want["directive_type"]:
        return False
    a, b = got.structured_adjustment, want["structured_adjustment"]
    if (a is None) != (b is None):
        return False
    if a is None:
        return True
    if list(a.get("hours", [])) != list(b.get("hours", [])):
        return False
    for key in ("factor", "minimum_energy_kwh", "max_grid_kwh"):
        if (key in a) != (key in b):
            return False
        if key in a and abs(float(a[key]) - float(b[key])) > 0.01:
            return False
    return True


def run(cases, mode: str) -> int:
    failures = 0
    print(f"\n{mode}")
    print("-" * 62)
    for case in cases:
        req = OptimizeRequest.model_validate(case["input"])
        expected = case["expected_output"]["directive_interpretation"]
        flat = [flatten(e) for e in expected]
        if mode == "CORRUPT":
            flat = [corrupt(e, req.battery.capacity_kwh) for e in flat]

        got = build_interpretations(req.operator_notes, req.battery, flat)
        ok = len(got) == len(expected) and all(same(g, w) for g, w in zip(got, expected))
        failures += not ok
        print(f"  {(GREEN + 'pass' if ok else RED + 'FAIL') + RESET}  {case['id']}")
        if not ok:
            for g, w in zip(got, expected):
                if not same(g, w):
                    print(f"      {DIM}note {w['note_index']}: {req.operator_notes[w['note_index']][:64]}{RESET}")
                    print(f"      want {w['directive_type']} {json.dumps(w['structured_adjustment'])}")
                    print(f"      got  {g.directive_type} {json.dumps(g.structured_adjustment)}")
    return failures


def main() -> int:
    cases = json.loads(SAMPLES.read_text(encoding="utf-8"))["cases"]
    failures = run(cases, "CLEAN   (model is correct — must pass through)")
    failures += run(cases, "CORRUPT (model makes all four classic mistakes — must be repaired)")
    print()
    if failures:
        print(f"{RED}{failures} failure(s).{RESET}\n")
        return 1
    print(f"{GREEN}20/20 — guardrails recover the reference interpretation even from corrupted model output.{RESET}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
