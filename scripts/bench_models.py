#!/usr/bin/env python3
"""Benchmark candidate models on this task: accuracy and latency.

    python scripts/bench_models.py

Two accuracy numbers per model:

  raw   what the model returned by itself
  final what the service actually emits, after app/guardrails.py has repaired it

`final` is the one that scores. `raw` is only useful for seeing how much work the
guardrails are doing.

Latency matters as much as accuracy here: the rubric pays 3/3 only for p95 <= 5s,
so a model that is more accurate but slower can be a net loss.
"""

import asyncio
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from app import llm  # noqa: E402
from app.config import ProviderConfig, load_dotenv, configured_providers  # noqa: E402
from app.guardrails import build_interpretations  # noqa: E402
from app.models import BatteryInput  # noqa: E402

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"

BATTERY = BatteryInput(
    capacity_kwh=200, initial_energy_kwh=120, minimum_energy_kwh=40,
    max_charge_kwh_per_hour=50, max_discharge_kwh_per_hour=50,
)

# Deliberately paraphrased away from the public wording, covering every type
# plus the traps: end-exclusive windows, factor polarity, percent-of-capacity,
# solar-hardware outage, and a distractor.
CASES = [
    ("Facilities will wash the rooftop solar panels from noon until 2 PM. During cleaning, "
     "usable solar should be treated as roughly 25% of the forecast.",
     "solar_reduction", [12, 13], 0.25),
    ("Expect an 80% reduction in rooftop solar between 11 AM and 2 PM because of inverter work.",
     "solar_reduction", [11, 12, 13], 0.2),
    ("Keep at least 50% of the battery capacity stored from 6 PM until 9 PM for emergency operations.",
     "minimum_battery_reserve", [18, 19, 20], 100.0),
    ("The battery charger will be isolated from 2 AM until 5 AM for electrical maintenance.",
     "no_charge_window", [2, 3, 4], None),
    ("For protection testing, the battery must not discharge from 6 PM until 8 PM.",
     "no_discharge_window", [18, 19], None),
    ("From 6 PM until 9 PM, campus grid import must not exceed 155 kWh in any hour.",
     "max_grid_window", [18, 19, 20], 155.0),
    ("Inverters are fully offline from 1 PM until 4 PM.",
     "solar_reduction", [13, 14, 15], 0.0),
    ("The sports office moved next month's registration deadline.",
     "no_op", None, None),
]

CANDIDATES = [
    ("openai", "gpt-4o-mini"),
    ("openai", "gpt-4.1-mini"),
    ("openai", "gpt-5-mini"),
    ("openai", "gpt-5.4-mini"),
    ("groq", "openai/gpt-oss-120b"),
    ("gemini", "gemini-flash-lite-latest"),
]


def raw_ok(entry, want_type, want_hours, want_value) -> bool:
    if not isinstance(entry, dict):
        return False
    if entry.get("directive_type") != want_type:
        return False
    if want_hours is not None and list(entry.get("hours") or []) != want_hours:
        return False
    if want_value is not None:
        got = entry.get("factor")
        if got is None:
            got = entry.get("minimum_energy_kwh")
        if got is None:
            got = entry.get("max_grid_kwh")
        if got is None or abs(float(got) - want_value) > 0.01:
            return False
    return True


def final_ok(result, want_type, want_hours, want_value) -> bool:
    if result.directive_type != want_type:
        return False
    adj = result.structured_adjustment or {}
    if want_hours is not None and list(adj.get("hours") or []) != want_hours:
        return False
    if want_value is not None:
        got = adj.get("factor", adj.get("minimum_energy_kwh", adj.get("max_grid_kwh")))
        if got is None or abs(float(got) - want_value) > 0.01:
            return False
    return True


async def bench(cfg: ProviderConfig):
    caller = llm._call_openai_compatible if cfg.kind == "openai" else llm._call_gemini
    raw_score = final_score = 0
    latencies = []
    errors = []

    for note, want_type, want_hours, want_value in CASES:
        started = time.perf_counter()
        try:
            payload = await asyncio.wait_for(caller(cfg, [note], BATTERY, None), timeout=45)
        except Exception as exc:  # noqa: BLE001
            message = str(exc)
            if hasattr(exc, "response"):
                try:
                    message = f"HTTP {exc.response.status_code}: {exc.response.text[:90]}"
                except Exception:  # noqa: BLE001
                    pass
            errors.append(message[:110])
            continue
        latencies.append((time.perf_counter() - started) * 1000)

        entries = llm._extract(payload)
        entry = entries[0] if isinstance(entries, list) and entries else {}
        raw_score += raw_ok(entry, want_type, want_hours, want_value)
        built = build_interpretations([note], BATTERY, entries)[0]
        final_score += final_ok(built, want_type, want_hours, want_value)

    return raw_score, final_score, latencies, errors


async def main() -> int:
    load_dotenv()
    available = {c.name: c for c in configured_providers()}

    header = f"{'model':<30}{'raw':>7}{'final':>8}{'median':>10}{'p95':>9}{'max':>9}"
    print(f"\n{header}\n{'-' * len(header)}")

    for provider, model in CANDIDATES:
        base = available.get(provider)
        if base is None:
            print(f"{model:<30}{DIM}  no {provider} key configured{RESET}")
            continue
        cfg = ProviderConfig(
            name=base.name, kind=base.kind, base_url=base.base_url,
            api_key=base.api_key, model=model,
        )
        raw_score, final_score, latencies, errors = await bench(cfg)
        if not latencies:
            print(f"{model:<30}{RED}  FAILED: {errors[0] if errors else 'no response'}{RESET}")
            continue
        latencies.sort()
        p95 = latencies[max(0, int(round(len(latencies) * 0.95)) - 1)]
        colour = GREEN if final_score == len(CASES) else (YELLOW if final_score >= len(CASES) - 1 else RED)
        lat_colour = GREEN if p95 <= 5000 else YELLOW
        print(f"{model:<30}{raw_score:>5}/{len(CASES)}{colour}{final_score:>6}/{len(CASES)}{RESET}"
              f"{statistics.median(latencies):>10.0f}{lat_colour}{p95:>9.0f}{RESET}{max(latencies):>9.0f}")
        for err in errors[:1]:
            print(f"  {DIM}{len(errors)} error(s), first: {err}{RESET}")

    print(f"\n{DIM}raw   = the model on its own{RESET}")
    print(f"{DIM}final = what the service emits after guardrail repair — this is what scores{RESET}")
    print(f"{DIM}p95 <= 5000 ms keeps the full 3/3 latency points{RESET}\n")
    await llm.aclose()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
