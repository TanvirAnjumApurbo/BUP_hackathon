"""LLM interpretation of operator notes — the primary and mandatory path.

One call per request. Every note goes in together with the battery context, and
the model returns one flat object per note. The nested `structured_adjustment`
is assembled afterwards by `app.guardrails`, so the model cannot emit a wrong
adjustment shape for a given directive type.

Flow per request:

    cache hit? -> return
    for each configured provider, in failover order:
        attempt 1
        validate with guardrails.validate_model_output
        if invalid: attempt 2, with the validator's own error text fed back
        if valid: cache and return
    every provider failed -> caller falls back to app.fallback (degraded mode)
"""

import asyncio
import hashlib
import json
import logging
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import httpx

from app.config import (
    ProviderConfig,
    configured_providers,
    llm_timeout_seconds,
    llm_total_budget_seconds,
)
from app.guardrails import validate_model_output
from app.models import BatteryInput

logger = logging.getLogger("gridwise.llm")

DIRECTIVE_TYPES = [
    "solar_reduction",
    "minimum_battery_reserve",
    "no_charge_window",
    "no_discharge_window",
    "max_grid_window",
    "no_op",
]

SYSTEM_PROMPT = """You convert campus energy operator notes into structured directives.

Return ONE object for every operator note, in the same order the notes are given.

DIRECTIVE TYPES — choose exactly one per note:
  solar_reduction          usable solar output is reduced during specific hours
  minimum_battery_reserve  battery energy must stay at or above a level during hours
  no_charge_window         the battery cannot be charged during hours
  no_discharge_window      the battery cannot be discharged during hours
  max_grid_window          grid import is capped during hours
  no_op                    the note does not affect today's 24-hour energy schedule

Anything about staffing, bookings, menus, deadlines, notices or next week is no_op.

TIME WINDOWS — start-inclusive, end-exclusive. Do this in two steps:
  Step 1: read the START hour and the END hour as 24-hour numbers, exactly as written.
          midnight=0, noon=12, 1 PM=13, 2 PM=14, 6 PM=18, 9 PM=21, 10 PM=22.
          Never subtract anything from the end hour yourself.
  Step 2: list every whole hour h where start <= h < end.

  "noon until 2 PM"         start 12, end 14 -> [12, 13]
  "1 PM to 3 PM"            start 13, end 15 -> [13, 14]
  "between 11 AM and 2 PM"  start 11, end 14 -> [11, 12, 13]
  "6 PM until 9 PM"         start 18, end 21 -> [18, 19, 20]
  "from 6 PM until 10 PM"   start 18, end 22 -> [18, 19, 20, 21]
  "2 AM until 5 AM"         start 2,  end 5  -> [2, 3, 4]
  "from 10 AM until noon"   start 10, end 12 -> [10, 11]
  "between 13:00 and 15:00" start 13, end 15 -> [13, 14]

FACTOR (solar_reduction only) is the fraction of solar that REMAINS, never the loss.
  "an 80% reduction"              -> 0.2
  "will drop to about 20%"        -> 0.2
  "roughly 25% of the forecast"   -> 0.25
  "about half of the forecast"    -> 0.5
  "leave one-fifth of normal"     -> 0.2
  "panels offline / no output"    -> 0.0

MINIMUM_ENERGY_KWH (minimum_battery_reserve only) must be in kWh, never a percentage.
If the note gives a percentage, multiply it by the battery capacity you are given.
  capacity 200 kWh, "keep at least 50% of capacity"  -> 100
  capacity 250 kWh, "keep at least 90 kWh"           -> 90

MAX_GRID_KWH (max_grid_window only) is the hourly kWh import cap stated in the note.

For no_op: hours must be [] and factor, minimum_energy_kwh and max_grid_kwh must all be null.
For every other type: set only the one numeric field that applies; leave the others null.

Respond with JSON of exactly this shape:
{"interpretations": [{"note_index": 0, "directive_type": "...", "hours": [...],
 "factor": null, "minimum_energy_kwh": null, "max_grid_kwh": null, "explanation": "..."}]}"""

RESPONSE_SCHEMA = {
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
                    "note_index", "directive_type", "hours",
                    "factor", "minimum_energy_kwh", "max_grid_kwh", "explanation",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["interpretations"],
    "additionalProperties": False,
}

_CACHE_LIMIT = 512
_cache: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()
_client: Optional[httpx.AsyncClient] = None


def cache_key(notes: Sequence[str], battery: BatteryInput) -> str:
    payload = json.dumps(
        {
            "notes": [n.strip().lower() for n in notes],
            "capacity": battery.capacity_kwh,
            "initial": battery.initial_energy_kwh,
            "minimum": battery.minimum_energy_kwh,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def cache_get(key: str) -> Optional[List[Dict[str, Any]]]:
    entry = _cache.get(key)
    if entry is not None:
        _cache.move_to_end(key)
    return entry


def cache_put(key: str, value: List[Dict[str, Any]]) -> None:
    _cache[key] = value
    _cache.move_to_end(key)
    while len(_cache) > _CACHE_LIMIT:
        _cache.popitem(last=False)


async def get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(timeout=llm_timeout_seconds())
    return _client


async def aclose() -> None:
    global _client
    if _client is not None and not _client.is_closed:
        await _client.aclose()
    _client = None


def build_user_prompt(notes: Sequence[str], battery: BatteryInput) -> str:
    listed = "\n".join(f"[{i}] {note.strip()}" for i, note in enumerate(notes))
    return (
        "Battery context for this scenario:\n"
        f"  capacity_kwh        = {battery.capacity_kwh}\n"
        f"  initial_energy_kwh  = {battery.initial_energy_kwh}\n"
        f"  minimum_energy_kwh  = {battery.minimum_energy_kwh}\n"
        "Planning horizon: hours 0 through 23 of today.\n\n"
        f"Operator notes ({len(notes)} total):\n{listed}\n\n"
        f"Return exactly {len(notes)} object(s) in the interpretations array, "
        "one per note, in note_index order 0"
        f"{'' if len(notes) == 1 else f' to {len(notes) - 1}'}."
    )


def _messages(
    notes: Sequence[str], battery: BatteryInput, correction: Optional[str]
) -> List[Dict[str, str]]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(notes, battery)},
    ]
    if correction:
        messages.append(
            {
                "role": "user",
                "content": (
                    "Your previous answer was rejected by a strict validator:\n"
                    f"{correction}\n\n"
                    "Re-read the rules, especially the two-step time-window method and the "
                    "factor/kWh conventions, and return a corrected JSON object."
                ),
            }
        )
    return messages


async def _call_openai_compatible(
    cfg: ProviderConfig, notes, battery, correction: Optional[str]
) -> Any:
    body: Dict[str, Any] = {
        "model": cfg.model,
        "temperature": 0,
        "messages": _messages(notes, battery, correction),
    }
    if cfg.name == "openai":
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "gridwise_directives",
                "strict": True,
                "schema": RESPONSE_SCHEMA,
            },
        }
    else:
        # Universally supported across OpenAI-compatible providers.
        body["response_format"] = {"type": "json_object"}

    client = await get_client()
    response = await client.post(
        f"{cfg.base_url}/chat/completions",
        json=body,
        headers={"Authorization": f"Bearer {cfg.api_key}"},
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    return json.loads(content)


async def _call_gemini(cfg: ProviderConfig, notes, battery, correction: Optional[str]) -> Any:
    schema = json.loads(json.dumps(RESPONSE_SCHEMA))

    def strip(node):  # Gemini rejects additionalProperties
        if isinstance(node, dict):
            node.pop("additionalProperties", None)
            for value in node.values():
                strip(value)
        elif isinstance(node, list):
            for value in node:
                strip(value)

    strip(schema)

    parts = [{"text": build_user_prompt(notes, battery)}]
    if correction:
        parts.append({"text": f"Your previous answer was rejected:\n{correction}\nReturn a corrected object."})

    body = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
            "responseSchema": schema,
        },
    }
    client = await get_client()
    response = await client.post(
        f"{cfg.base_url}/models/{cfg.model}:generateContent",
        params={"key": cfg.api_key},
        json=body,
    )
    response.raise_for_status()
    text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
    return json.loads(text)


def _extract(payload: Any) -> Any:
    """Accept {"interpretations": [...]} or a bare array."""
    if isinstance(payload, dict):
        for key in ("interpretations", "directive_interpretation", "results", "data"):
            if isinstance(payload.get(key), list):
                return payload[key]
        return payload
    return payload


async def interpret(
    notes: Sequence[str], battery: BatteryInput
) -> Tuple[Optional[List[Dict[str, Any]]], str]:
    """Returns (flat entries, provenance label). None means every provider failed."""
    key = cache_key(notes, battery)
    cached = cache_get(key)
    if cached is not None:
        return cached, "cache"

    providers = configured_providers()
    if not providers:
        logger.error("no LLM provider is configured")
        return None, "unconfigured"

    # One wall-clock budget for the entire phase, so adding a third provider can
    # never push a request past the judge's 30s ceiling.
    deadline = time.monotonic() + llm_total_budget_seconds()

    for cfg in providers:
        caller = _call_openai_compatible if cfg.kind == "openai" else _call_gemini
        correction: Optional[str] = None

        for attempt in (1, 2):
            remaining = deadline - time.monotonic()
            if remaining < 1.0:
                logger.warning(
                    "interpretation budget exhausted before %s attempt %d", cfg.name, attempt
                )
                return None, "budget-exhausted"

            try:
                payload = await asyncio.wait_for(
                    caller(cfg, notes, battery, correction),
                    timeout=min(llm_timeout_seconds() + 2.0, remaining),
                )
            except Exception as exc:  # noqa: BLE001 — provider failure must never escape
                logger.warning(
                    "%s attempt %d failed: %s: %s", cfg.name, attempt, type(exc).__name__, exc
                )
                break  # a transport/HTTP failure will not be fixed by a reworded retry

            entries = _extract(payload)
            problems = validate_model_output(notes, battery, entries)
            if not problems:
                label = f"llm:{cfg.name}/{cfg.model}" + ("" if attempt == 1 else "+retry")
                cache_put(key, entries)
                return entries, label

            logger.info("%s attempt %d rejected: %s", cfg.name, attempt, problems[:3])
            if attempt == 2:
                # Second failure on this provider: the repair layer downstream can
                # still rescue a partially-correct answer, so keep it rather than
                # discarding everything.
                if isinstance(entries, list) and entries:
                    label = f"llm:{cfg.name}/{cfg.model}+unvalidated"
                    return entries, label
                break
            correction = "\n".join(f"- {problem}" for problem in problems[:8])

    logger.error("every provider failed for %d note(s)", len(notes))
    return None, "all-providers-failed"
