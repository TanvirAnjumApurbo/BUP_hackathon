#!/usr/bin/env python3
"""Adversarial paraphrase suite for operator-note interpretation.

    python scripts/test_paraphrases.py              # full pipeline: LLM + guardrails
    python scripts/test_paraphrases.py --offline    # deterministic layers only, no quota
    python scripts/test_paraphrases.py --group solar_reduction
    python scripts/test_paraphrases.py --verbose

Why this exists: the ten public cases are not a test set. A plural-noun bug in
the relevance guard once survived with all ten public cases green, because no
public note says "inverters" or "panels". Every case in tests/paraphrases.json is
written to break something the public set cannot reach.

`--offline` checks only what `app/timewindows.py` can decide by itself, which is
free and instant and catches most window regressions. The default run exercises
the real path — model, guardrails, repairs — and is what actually predicts the
interpretation score.
"""

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from app import llm  # noqa: E402
from app.config import load_dotenv  # noqa: E402
from app.guardrails import build_interpretations  # noqa: E402
from app.models import BatteryInput  # noqa: E402
from app.timewindows import parse_window  # noqa: E402

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"

CASES_FILE = REPO / "tests" / "paraphrases.json"
CONCURRENCY = 4


def expected_value(spec, capacity):
    """The single numeric the directive should carry, or None."""
    if "factor" in spec:
        return spec["factor"]
    if "reserve" in spec:
        return spec["reserve"]
    if "reserve_pct" in spec:
        return spec["reserve_pct"] * capacity
    if "cap" in spec:
        return spec["cap"]
    return None


def actual_value(adjustment):
    if not adjustment:
        return None
    for key in ("factor", "minimum_energy_kwh", "max_grid_kwh"):
        if key in adjustment:
            return adjustment[key]
    return None


def compare(result, spec, capacity):
    """Returns (ok, reason)."""
    if result.directive_type != spec["type"]:
        return False, f"type {result.directive_type} != {spec['type']}"
    if spec["type"] == "no_op":
        return True, ""

    adjustment = result.structured_adjustment or {}
    hours = list(adjustment.get("hours") or [])
    if hours != spec["hours"]:
        return False, f"hours {hours} != {spec['hours']}"

    want = expected_value(spec, capacity)
    if want is not None:
        got = actual_value(adjustment)
        if got is None or abs(float(got) - float(want)) > 0.01:
            return False, f"value {got} != {want}"
    return True, ""


async def run_case(case, battery, capacity, semaphore, offline):
    notes = case["notes"]
    if offline:
        # Deterministic layer only: feed guardrails the expected type with no
        # hours, so only app/timewindows.py decides the window.
        raw = [
            {"note_index": i, "directive_type": spec["type"], "hours": [],
             "factor": None, "minimum_energy_kwh": None, "max_grid_kwh": None,
             "explanation": "offline"}
            for i, spec in enumerate(case["expect"])
        ]
    else:
        async with semaphore:
            raw, _label = await llm.interpret(notes, battery)
        if raw is None:
            from app import fallback
            raw = fallback.interpret(notes, battery)

    built = build_interpretations(notes, battery, raw)
    results = []
    for i, spec in enumerate(case["expect"]):
        if i >= len(built):
            results.append((False, "missing entry", None))
            continue
        ok, reason = compare(built[i], spec, capacity)
        results.append((ok, reason, built[i]))
    return case, results


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true",
                        help="deterministic layers only; no model calls")
    parser.add_argument("--group", help="only run one group")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    load_dotenv()
    data = json.loads(CASES_FILE.read_text(encoding="utf-8"))
    meta = data["_meta"]
    capacity = meta["capacity_kwh"]
    battery = BatteryInput(**meta["battery"])
    cases = [c for c in data["cases"] if not args.group or c["group"] == args.group]

    mode = "OFFLINE (deterministic layers only)" if args.offline else "FULL PIPELINE (model + guardrails)"
    note_count = sum(len(c["notes"]) for c in cases)
    print(f"\n{mode}")
    print(f"{len(cases)} scenarios, {note_count} notes\n")

    semaphore = asyncio.Semaphore(CONCURRENCY)
    outcomes = await asyncio.gather(
        *(run_case(c, battery, capacity, semaphore, args.offline) for c in cases)
    )

    by_group = defaultdict(lambda: [0, 0])
    failures = []
    for case, results in outcomes:
        passed = all(ok for ok, _, _ in results)
        by_group[case["group"]][1] += 1
        by_group[case["group"]][0] += passed
        if not passed:
            failures.append((case, results))
        if args.verbose or not passed:
            mark = f"{GREEN}pass{RESET}" if passed else f"{RED}FAIL{RESET}"
            trap = f"  {DIM}[{case['trap']}]{RESET}" if case.get("trap") else ""
            print(f"  {mark}  {case['id']}{trap}")
            if not passed:
                for i, (ok, reason, built) in enumerate(results):
                    if ok:
                        continue
                    print(f"        {DIM}note: {case['notes'][i][:78]}{RESET}")
                    print(f"        {RED}{reason}{RESET}")
                    if built is not None:
                        print(f"        got  {built.directive_type} {json.dumps(built.structured_adjustment)}")

    print(f"\n{'group':<26}{'pass':>8}{'total':>8}")
    print("-" * 42)
    total_pass = total = 0
    for group in sorted(by_group):
        got, count = by_group[group]
        total_pass += got
        total += count
        colour = GREEN if got == count else RED
        print(f"{group:<26}{colour}{got:>8}{RESET}{count:>8}")
    print("-" * 42)
    colour = GREEN if total_pass == total else RED
    print(f"{'TOTAL':<26}{colour}{total_pass:>8}{RESET}{total:>8}   "
          f"{colour}{100.0 * total_pass / max(total, 1):.1f}%{RESET}")

    if not args.offline:
        await llm.aclose()

    print()
    if failures:
        print(f"{RED}{len(failures)} scenario(s) failed.{RESET}\n")
        return 1
    print(f"{GREEN}All paraphrase scenarios interpreted correctly.{RESET}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
