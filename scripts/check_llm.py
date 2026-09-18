#!/usr/bin/env python3
"""Verify every configured LLM provider actually works from this machine.

Run this the moment you paste a key into .env:

    python scripts/check_llm.py

For each provider with a key it:
  1. confirms the credential is accepted
  2. sends one real operator note and asks for structured output
  3. reports latency and whether the extraction was correct

Standard library only. Nothing here is imported by the service.

NOTE ON THE SCHEMA
------------------
The model is asked for a FLAT object per note, never the nested
`structured_adjustment`. Our own code assembles the nested shape afterwards.
That way the model literally cannot emit a wrong adjustment shape for a given
directive_type, which removes a whole class of scoring failure.
"""

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from app.config import ProviderConfig, configured_providers, load_dotenv  # noqa: E402

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"

DIRECTIVE_TYPES = [
    "solar_reduction",
    "minimum_battery_reserve",
    "no_charge_window",
    "no_discharge_window",
    "max_grid_window",
    "no_op",
]

SYSTEM_PROMPT = """You convert campus operator notes into structured energy directives.

For EACH note return exactly one object, in order, with these rules:

directive_type is one of:
  solar_reduction          - usable solar is reduced during specific hours
  minimum_battery_reserve  - battery energy must stay at or above a level
  no_charge_window         - the battery cannot charge
  no_discharge_window      - the battery cannot discharge
  max_grid_window          - grid import is capped
  no_op                    - the note does not affect today's 24-hour schedule

TIME WINDOWS. Work this out in two steps and do not skip either one.

Step 1 - read the START hour and the END hour as plain 24-hour numbers.
         midnight = 0, noon = 12, 1 PM = 13, 2 PM = 14, 9 PM = 21, 10 PM = 22.
         Write down the end hour EXACTLY as stated. Never subtract from it yourself.
Step 2 - list every whole hour h where start <= h < end.

  "1 PM to 3 PM"           start 13, end 15 -> [13, 14]
  "between 11 AM and 2 PM" start 11, end 14 -> [11, 12, 13]
  "6 PM until 9 PM"        start 18, end 21 -> [18, 19, 20]
  "noon until 2 PM"        start 12, end 14 -> [12, 13]
  "2 AM until 5 AM"        start 2,  end 5  -> [2, 3, 4]
  "from 10 AM until noon"  start 10, end 12 -> [10, 11]
  "7 PM to 10 PM"          start 19, end 22 -> [19, 20, 21]
  "between 13:00 and 15:00" start 13, end 15 -> [13, 14]

factor (solar_reduction only) is the fraction of solar that REMAINS, not the loss.
  "80% reduction"          -> 0.2
  "drops to about 20%"     -> 0.2
  "about half the forecast"-> 0.5
  "one-fifth of normal"    -> 0.2
  "25% of the forecast"    -> 0.25
  "panels offline"         -> 0.0

minimum_energy_kwh (minimum_battery_reserve only) is in kWh. If the note gives a
percentage, multiply by the battery capacity given to you.

max_grid_kwh (max_grid_window only) is the hourly kWh cap.

For no_op: hours is [] and every numeric field is null.
Set unused numeric fields to null."""

NOTE_SCHEMA = {
    "type": "object",
    "properties": {
        "interpretations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "note_index": {"type": "integer"},
                    "directive_type": {"type": "string", "enum": DIRECTIVE_TYPES},
                    "hours": {"type": "array", "items": {"type": "integer"}},
                    "factor": {"type": ["number", "null"]},
                    "minimum_energy_kwh": {"type": ["number", "null"]},
                    "max_grid_kwh": {"type": ["number", "null"]},
                    "explanation": {"type": "string"},
                },
                "required": [
                    "note_index",
                    "directive_type",
                    "hours",
                    "factor",
                    "minimum_energy_kwh",
                    "max_grid_kwh",
                    "explanation",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["interpretations"],
    "additionalProperties": False,
}

# One case per directive type, deliberately paraphrased away from the public wording.
CASES = [
    {
        "notes": [
            "Facilities will wash the rooftop solar panels from noon until 2 PM. "
            "During cleaning, usable solar should be treated as roughly 25% of the forecast.",
            "The sports office moved next month's registration deadline.",
        ],
        "capacity": 220,
        "want": [
            {"directive_type": "solar_reduction", "hours": [12, 13], "factor": 0.25},
            {"directive_type": "no_op", "hours": []},
        ],
    },
    {
        "notes": ["Expect an 80% reduction in rooftop solar between 11 AM and 2 PM because of inverter work."],
        "capacity": 240,
        "want": [{"directive_type": "solar_reduction", "hours": [11, 12, 13], "factor": 0.2}],
    },
    {
        "notes": ["Keep at least 50% of the battery capacity stored from 6 PM until 9 PM for emergency operations."],
        "capacity": 200,
        "want": [
            {"directive_type": "minimum_battery_reserve", "hours": [18, 19, 20], "minimum_energy_kwh": 100}
        ],
    },
    {
        "notes": ["The battery charger will be isolated from 2 AM until 5 AM for electrical maintenance."],
        "capacity": 200,
        "want": [{"directive_type": "no_charge_window", "hours": [2, 3, 4]}],
    },
    {
        "notes": ["For protection testing, the battery must not discharge from 6 PM until 8 PM."],
        "capacity": 230,
        "want": [{"directive_type": "no_discharge_window", "hours": [18, 19]}],
    },
    {
        "notes": ["From 6 PM until 9 PM, campus grid import must not exceed 155 kWh in any hour."],
        "capacity": 240,
        "want": [{"directive_type": "max_grid_window", "hours": [18, 19, 20], "max_grid_kwh": 155}],
    },
]


def build_user_prompt(notes, capacity_kwh) -> str:
    listed = "\n".join(f"[{i}] {note}" for i, note in enumerate(notes))
    return (
        f"Battery capacity: {capacity_kwh} kWh\n"
        f"Planning horizon: hours 0 through 23 of today.\n\n"
        f"Operator notes:\n{listed}\n\n"
        f"Return exactly {len(notes)} interpretation(s), one per note, in note_index order."
    )


def call_openai(cfg: ProviderConfig, notes, capacity, timeout: float):
    body = {
        "model": cfg.model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(notes, capacity)},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "gridwise_directives", "strict": True, "schema": NOTE_SCHEMA},
        },
    }
    req = urllib.request.Request(
        f"{cfg.base_url}/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {cfg.api_key}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode())
    return json.loads(payload["choices"][0]["message"]["content"])["interpretations"]


def call_gemini(cfg: ProviderConfig, notes, capacity, timeout: float):
    schema = json.loads(json.dumps(NOTE_SCHEMA))  # Gemini rejects additionalProperties
    def strip(node):
        if isinstance(node, dict):
            node.pop("additionalProperties", None)
            for value in node.values():
                strip(value)
        elif isinstance(node, list):
            for value in node:
                strip(value)
    strip(schema)

    body = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": build_user_prompt(notes, capacity)}]}],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
            "responseSchema": schema,
        },
    }
    url = f"{cfg.base_url}/models/{cfg.model}:generateContent?key={cfg.api_key}"
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode())
    text = payload["candidates"][0]["content"]["parts"][0]["text"]
    return json.loads(text)["interpretations"]


def matches(got, want) -> bool:
    if got.get("directive_type") != want["directive_type"]:
        return False
    if list(got.get("hours") or []) != want["hours"]:
        return False
    for field in ("factor", "minimum_energy_kwh", "max_grid_kwh"):
        if field in want and abs(float(got.get(field) or 0) - want[field]) > 0.01:
            return False
    return True


def check(cfg: ProviderConfig, timeout: float) -> None:
    print(f"\n{'=' * 70}\n{cfg.redacted()}\n{'=' * 70}")
    caller = call_openai if cfg.kind == "openai" else call_gemini
    latencies, passed = [], 0

    for case in CASES:
        label = case["notes"][0][:52] + "..."
        started = time.perf_counter()
        try:
            got = caller(cfg, case["notes"], case["capacity"], timeout)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode()[:220]
            print(f"  {RED}HTTP {exc.code}{RESET}  {label}\n    {DIM}{detail}{RESET}")
            return
        except Exception as exc:  # noqa: BLE001
            print(f"  {RED}ERROR{RESET}  {label}\n    {DIM}{type(exc).__name__}: {exc}{RESET}")
            return

        ms = (time.perf_counter() - started) * 1000
        latencies.append(ms)

        ok = len(got) == len(case["want"]) and all(
            matches(g, w) for g, w in zip(got, case["want"])
        )
        passed += bool(ok)
        mark = f"{GREEN}pass{RESET}" if ok else f"{RED}FAIL{RESET}"
        print(f"  [{mark}] {ms:6.0f} ms  {label}")
        if not ok:
            for g, w in zip(got, case["want"]):
                if not matches(g, w):
                    print(f"       {YELLOW}want{RESET} {json.dumps(w)}")
                    print(f"       {YELLOW}got {RESET} {json.dumps({k: g.get(k) for k in ('directive_type','hours','factor','minimum_energy_kwh','max_grid_kwh')})}")

    latencies.sort()
    p95 = latencies[int(len(latencies) * 0.95) - 1] if latencies else 0
    verdict = GREEN if passed == len(CASES) else (YELLOW if passed else RED)
    print(f"\n  {verdict}{passed}/{len(CASES)} correct{RESET}   "
          f"median {latencies[len(latencies)//2]:.0f} ms   p95 {p95:.0f} ms   max {max(latencies):.0f} ms")
    if p95 > 4000:
        print(f"  {YELLOW}p95 above 4s leaves little headroom for the 5s latency band.{RESET}")


def main() -> int:
    load_dotenv()
    providers = configured_providers()
    if not providers:
        print(f"{RED}No provider has a key. Paste one into .env (see .env.example).{RESET}")
        return 1
    timeout = 30.0
    for cfg in providers:
        check(cfg, timeout)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
