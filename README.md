# GridWise — LLM-Assisted Smart Campus Energy Optimization

BUP CSE Fest 2026 Hackathon · Online Preliminary · `POST /optimize-energy`

An HTTP service that reads free-text campus operator notes with a language model,
converts them into machine-checkable directives, validates those directives
deterministically, and then solves the 24-hour energy schedule as a linear program.

> **Status: Stage 1** — API contract, schema validation and a valid baseline schedule.
> The LLM interpreter and the LP optimizer land in stage 2. Sections marked
> **(stage 2)** are not wired up yet.

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
| `LLM_PROVIDER` | stage 2 | Which provider to use first: `openai`, `groq`, or `gemini`. |
| `OPENAI_API_KEY` | stage 2 | Key for the OpenAI provider. |
| `GROQ_API_KEY` | stage 2 | Key for the Groq fallback provider. |
| `GEMINI_API_KEY` | stage 2 | Key for the Google AI Studio fallback provider. |
| `LLM_MODEL` | stage 2 | Model identifier. Recorded here at submission time. |

**Model / provider actually used: _(stage 2 — record the exact model id here before submitting)_**

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
  |-- app/models.py      pydantic schema validation   -> 400 on anything malformed
  |-- app/llm.py         LLM INTERPRETATION           (stage 2) notes -> directives
  |-- app/guardrails.py  deterministic validation     (stage 2) reject bad model output
  |-- app/normalizer.py  deterministic repair         (stage 2) window/factor/percent fixes
  |-- app/planner.py     LP optimizer                 (stage 2) scipy HiGHS, exact optimum
  |-- app/validator.py   replay validator             re-runs every judge rule on our output
  +-- response
```

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

**Expected result at stage 1:** `/health` passes and all ten cases return HTTP 200
with a structurally valid plan. `interp` and `cost` are shown in amber and are
*expected to fail* — the baseline schedule ignores directives and does not
optimize, which is exactly what stages 2 and 3 fix.

**Expected result at stage 3:** ten `pass` in every column.

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

- **Stage 1:** the schedule is a valid baseline, not an optimized one, and every
  operator note is reported as `no_op`. The LLM interpreter and the LP optimizer
  are not yet wired in.
- The in-process interpretation cache is per-instance and is not shared across
  replicas.
