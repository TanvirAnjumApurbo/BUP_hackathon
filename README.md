# GridWise — LLM-Assisted Smart Campus Energy Optimization

BUP CSE Fest 2026 Hackathon · Online Preliminary · `POST /optimize-energy`

An HTTP service that reads free-text campus operator notes with a language model,
converts them into machine-checkable directives, validates those directives
deterministically, and then solves the 24-hour energy schedule as a linear program.

> **Status: complete.** LLM interpretation, deterministic guardrails, the LP optimizer and
> the replay validator are all wired in. All ten public sample cases pass on
> directive interpretation, plan validity and exact optimal cost.

---

## 1. Quickstart from a clean environment

Requires Python 3.11+ and nothing else.

```bash
git clone https://github.com/TanvirAnjumApurbo/BUP_hackathon.git
cd BUP_hackathon

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env               # then fill in your key (see §2)

uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The service is ready as soon as `/health` answers, typically under two seconds:

```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

---

## 2. Configuration

All configuration is by environment variable. **No secret is ever committed to this
repository or baked into the Docker image.** See `.env.example` for the full list
of names with empty values.

| Variable | Required | Purpose |
|---|---|---|
| `PORT` | no (default `8000`) | Port the service binds on. Always binds `0.0.0.0`. |
| `LLM_PROVIDER` | yes | Which provider to use first: `openai`, `groq`, or `gemini`. |
| `OPENAI_API_KEY` | yes* | Key for the OpenAI provider. |
| `GROQ_API_KEY` | no | Key for the Groq fallback provider. |
| `GEMINI_API_KEY` | no | Key for the Google AI Studio fallback provider. |
| `LLM_MODEL` | yes | Model identifier. Recorded here at submission time. |

*At least one provider key is required. Providers are tried in failover order.

**Model / provider actually used: `openai` / `gpt-4o-mini`** (OpenAI Chat Completions with strict
`json_schema` structured output, `temperature=0`).

---

## 3. Endpoints

### `GET /health`

```bash
curl http://localhost:8000/health
```
```json
{"status":"ok"}
```

### `POST /optimize-energy`

```bash
curl -s -X POST http://localhost:8000/optimize-energy \
  -H 'Content-Type: application/json' \
  -d @- <<'JSON'
{
  "scenario_id": "GRID-101",
  "operator_notes": [
    "Facilities will wash the rooftop solar panels from noon until 2 PM. During cleaning, usable solar should be treated as roughly 25% of the forecast.",
    "The sports office moved next month's registration deadline."
  ],
  "hours": [
    {"hour": 0, "demand_kwh": 90, "solar_kwh": 0, "tariff_bdt_per_kwh": 6}
    /* ... 23 more entries, one per hour 0 through 23 ... */
  ],
  "battery": {
    "capacity_kwh": 220,
    "initial_energy_kwh": 110,
    "minimum_energy_kwh": 40,
    "max_charge_kwh_per_hour": 50,
    "max_discharge_kwh_per_hour": 50
  }
}
JSON
```

Response — exactly seven top-level fields, never more:

```json
{
  "scenario_id": "GRID-101",
  "directive_interpretation": [
    {
      "note_index": 0,
      "applies": true,
      "directive_type": "solar_reduction",
      "structured_adjustment": {"hours": [12, 13], "factor": 0.25},
      "explanation": "Solar availability is reduced to 25% during the panel-cleaning window."
    },
    {
      "note_index": 1,
      "applies": false,
      "directive_type": "no_op",
      "structured_adjustment": null,
      "explanation": "This note does not affect today's 24-hour energy schedule."
    }
  ],
  "hourly_plan": [
    {"hour": 0, "grid_kwh": 90.0, "solar_used_kwh": 0.0, "battery_action": "idle", "battery_kwh": 0.0, "battery_energy_after_kwh": 110.0}
  ],
  "total_grid_kwh": 2692.5,
  "total_cost_bdt": 38365.0,
  "peak_grid_kwh": 175.0,
  "plan_summary": "..."
}
```

Diagnostics are returned as response **headers**, so the JSON body stays exactly
on contract:

| Header | Meaning |
|---|---|
| `X-Interpreter` | Which interpretation path produced the directives |
| `X-Optimizer` | Which solve path produced the plan (`lp` on the normal path) |
| `X-Latency-Ms` | Server-side handling time |
| `X-Self-Check` | Whether our own replay validator passed on this response |

### HTTP status codes

| Code | When |
|---|---|
| `200` | Success |
| `400` | Malformed JSON or a structurally invalid request |
| `500` | Controlled internal error. Never a stack trace, never a secret |

---

## 4. Architecture — where the LLM sits

```
POST /optimize-energy
  |
  |-- app/models.py       pydantic schema validation  -> 400 on anything malformed
  |
  |-- app/llm.py          LLM INTERPRETATION  <-- mandatory primary path
  |                         one call per request: every note + battery capacity,
  |                         initial and minimum energy, temperature 0, JSON mode,
  |                         ~10s budget, cache keyed on note-text hash.
  |                         Invalid output -> retry once with the validator's own
  |                         error text -> then fail over to the next provider.
  |
  |-- app/guardrails.py   deterministic validation + repair of model output
  |-- app/timewindows.py  deterministic hour extraction from the note text
  |-- app/fallback.py     keyword interpreter -- LAST RESORT ONLY, never primary
  |
  |-- app/optimizer.py    LP optimizer (scipy HiGHS), provably optimal
  |-- app/validator.py    replay validator, re-runs every judge rule on our output
  +-- response
```

### Division of labour

The model is asked for a **flat** object per note — `directive_type`, `hours`, and at
most one numeric value. It never produces the nested `structured_adjustment`;
`app/guardrails.py` assembles that. The model therefore *cannot* emit a wrong
adjustment shape for a given directive type.

The response array is built from **our** note list, indexed by our own position.
The model's `note_index` is validated but never trusted for ordering, so a
missing, duplicated or out-of-order entry cannot shift the mapping.

Measured on a real provider, hour extraction was the weakest link: the same phrase
`"6 PM until 9 PM"` came back as `[18,19]` inside a battery-reserve note and
`[18,19,20]` inside a grid-cap note. So hours are extracted deterministically by
`app/timewindows.py` whenever the note states the window unambiguously, and the
model keeps the decision only when the wording is genuinely ambiguous
(`"from one until three"`), where reading context beats any regex.

Four repairs are applied to model output, each one a failure mode confirmed in testing:

| Failure | Repair |
|---|---|
| Window end off by one | Deterministic parse wins on high confidence |
| Solar factor polarity flipped | `"80% reduction"` resolves to `0.2`, not `0.8` |
| Percent reserve left as a percent | `"50% of capacity"` becomes kWh against `capacity_kwh` |
| Distractor promoted to a directive | Relevance re-check forces `no_op` |

The language model is the **only** component that reads the operator notes.
Its structured output is what the optimizer's constraints are built from — it is
not used for `plan_summary` or documentation. Everything downstream of it is
deterministic: the model's output is treated as untrusted data and must survive
`app/guardrails.py` before a single constraint is created.

**The optimizer is a linear program, not a heuristic.** Minimise
`SUM(tariff[h] * grid[h])` subject to hourly energy balance, effective-solar
limits, battery bounds and rate limits, directive windows, and end-of-day battery
neutrality. This returns the provably cheapest valid schedule rather than an
approximation.

**`app/validator.py` is our own independent judge.** Before any response is
returned it replays the schedule hour by hour against every documented rule —
energy balance, effective solar, battery transitions and bounds, charge and
discharge windows, grid caps, end-of-day neutrality, and a recount of all three
reported totals. The result is reported in the `X-Self-Check` header.

---

## 5. Testing against the public sample cases

With the service running:

```bash
python scripts/test_samples.py http://localhost:8000
```

The script POSTs all ten published cases and reports, per case: HTTP status,
whether the returned plan survives the full replay validator, whether
`directive_interpretation` matches the published reference, and
`total_cost_bdt` against the reference optimal cost.

**Expected result: ten `pass` in every column**, with `total_cost_bdt` matching the
reference optimal cost exactly.

Two further suites need no network and no LLM quota:

```bash
python scripts/test_optimizer.py    # LP vs the reference optima, plus the rubric's quality_ratio
python scripts/test_guardrails.py   # repair layer, on clean AND deliberately corrupted model output
python scripts/test_overlaps.py     # how overlapping directives combine, plus infeasibility fallback
python scripts/check_llm.py         # provider reachability, accuracy and latency
```

`test_optimizer.py` reports the rubric's own Optimization Quality formula,
`min(1, organizer_optimal_cost / our_cost)`, per case and as a mean:

```
mean quality_ratio                     1.0000
projected Optimization Quality          10.00 / 10
```

### How overlapping directives combine

Two directives of the same kind on the same hour resolve to the single most
restrictive value — they never compound:

| Kind | Rule |
|---|---|
| `solar_reduction` | **minimum** factor. 0.5 and 0.25 give 0.25, not 0.125 |
| `max_grid_window` | minimum cap |
| `minimum_battery_reserve` | maximum floor, and never below `battery.minimum_energy_kwh` |
| `no_charge_window` / `no_discharge_window` | union of hours |

This matters because `directive_effects` in `app/validator.py` is the single
shared source of truth for both the optimizer and the replay validator. A wrong
rule here would be self-consistently wrong — we would plan against the wrong
ceiling and our own validator would agree, while the judge replays against the
true value. `scripts/test_overlaps.py` locks all four rules in.

---

## 6. Docker

The published image is the same artifact that is deployed, built and smoke-tested
by `.github/workflows/docker.yml` on every push.

```bash
docker pull ghcr.io/tanviranjumapurbo/bup_hackathon:latest
docker run --rm -p 8000:8000 \
  -e PORT=8000 \
  -e LLM_PROVIDER=openai \
  -e OPENAI_API_KEY=... \
  ghcr.io/tanviranjumapurbo/bup_hackathon:latest

curl http://localhost:8000/health
# {"status":"ok"}
```

**Exact digest for submission: _(paste from the CI run summary before submitting)_**

The image runs as a non-root user, binds `0.0.0.0`, honours `$PORT`, and contains
no credentials of any kind.

---

## 7. Dependencies and credits

| Dependency | Role |
|---|---|
| [FastAPI](https://fastapi.tiangolo.com/) + [uvicorn](https://www.uvicorn.org/) | HTTP service and ASGI server |
| [pydantic](https://docs.pydantic.dev/) | Request and response schema validation |
| [SciPy](https://scipy.org/) (`linprog`, HiGHS) | Linear-programming optimizer |
| [NumPy](https://numpy.org/) | Constraint matrix construction |
| [httpx](https://www.python-httpx.org/) | Async client for the model provider |

The problem statement, participant guide and public sample cases are provided by
the BUP CSE Fest 2026 organizers. Architecture and implementation are our own work;
an AI coding assistant was used during development, which the official rulebook permits.

---

## 8. Secret handling

- Every credential is read from an environment variable at run time.
- `.env` is gitignored. `.env.example` lists variable **names only**, never values.
- No key, token, raw prompt or stack trace is written to logs or returned in any
  API response. The generic exception handler returns a fixed message and logs the
  detail server-side only.
- The Docker image contains no baked-in credentials.

---

## 9. Known limitations

- Hour extraction defers to the model only when a window has no meridiem anywhere
  (for example "from one until three"); everywhere else it is deterministic.
- The keyword fallback in `app/fallback.py` runs only when every configured provider
  has failed. It is an availability measure, not an interpretation strategy.
- The in-process interpretation cache is per-instance and is not shared across
  replicas.
