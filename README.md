# GridWise — LLM-Assisted Smart Campus Energy Optimization

**BUP CSE Fest 2026 Hackathon · Online Preliminary**

| | |
|---|---|
| **Live endpoint** | `https://buphackathon-production-7134.up.railway.app` |
| **Docker image** | `ghcr.io/tanviranjumapurbo/bup_hackathon:0466c8f` |
| **Image digest** | `sha256:711dbc4b689dc0a02bb2e1c20a7c6479b4c00f5e94ed885d9f044879869bf2ea` |
| **Model / provider** | OpenAI `gpt-5.4-mini` (Chat Completions, strict `json_schema`, `temperature=0`) |
| **Runtime** | Python 3.11, FastAPI + uvicorn, SciPy HiGHS |

```bash
curl https://buphackathon-production-7134.up.railway.app/health
# {"status":"ok"}
```

### What this service does

A campus draws electricity from the grid, generates rooftop solar, and has a battery.
`POST /optimize-energy` receives 24 hours of demand, solar forecast and tariff, **plus one to
three free-text notes written by a human operator**, and returns two things:

1. a machine-checkable interpretation of every note — one of six directive types, or `no_op`
   for a note that does not affect today's schedule;
2. a valid, cheapest-possible 24-hour operating plan that honours those directives.

A language model reads the notes. Deterministic code does everything else: it validates and
repairs the model's output, builds the optimization constraints, solves the schedule as a
linear program, and replays the finished plan against every judge rule before responding.

### Jump to

| Section | Covers |
|---|---|
| [1. Quickstart](#1-quickstart-from-a-clean-machine) | Clean-machine install and exact run command |
| [2. Configuration](#2-configuration-and-model-provider) | Env var names, model/provider |
| [3. Endpoints](#3-endpoints-and-worked-example) | `/health` and a full copy-pasteable sample |
| [4. Testing](#4-testing-against-the-public-sample-cases) | Test procedure and expected output |
| [5. Architecture](#5-architecture--llm--guardrails--lp-optimizer--replay-validator) | LLM → guardrails → optimizer → validator |
| [6. Docker](#6-docker-pull--run-fallback) | Pull/run fallback |
| [7. Dependencies, limitations, secrets](#7-dependencies-limitations-and-secret-handling) | Credits, known limits, secret handling |

---

## 1. Quickstart from a clean machine

Requires **Python 3.11 or newer** and `git`. Nothing else — no database, no build step, no
training job.

```bash
git clone https://github.com/TanvirAnjumApurbo/BUP_hackathon.git
cd BUP_hackathon

python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1

pip install -r requirements.txt

cp .env.example .env                # then set OPENAI_API_KEY inside .env — see section 2
```

**Exact run command:**

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The service is ready as soon as `/health` answers, normally in **under 3 seconds** and always
well inside the 60-second readiness window:

```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

Instead of a `.env` file the variables can be exported directly:

```bash
export OPENAI_API_KEY="<your key>"
export LLM_PROVIDER=openai
export LLM_MODEL=gpt-5.4-mini
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

There are no other setup steps. If any command above fails on a clean machine, that is a bug in
this README — nothing is assumed to be pre-installed beyond Python and git.

---

## 2. Configuration and model / provider

### Model actually used

| | |
|---|---|
| Provider | **OpenAI** (`https://api.openai.com/v1`, Chat Completions) |
| Model id | **`gpt-5.4-mini`** |
| Output mode | strict `json_schema` structured output |
| Temperature | `0` |
| Calls per request | **one**, covering all operator notes together |
| Retry | one retry on the same provider, with the validator's error text fed back |
| Failover | next configured provider, then a keyword interpreter (degraded mode) |

Two failover providers are configured behind it, tried in order if OpenAI fails:

| Order | Provider | Model |
|---|---|---|
| 1 | OpenAI | `gpt-5.4-mini` |
| 2 | Groq | `openai/gpt-oss-120b` |
| 3 | Google AI Studio | `gemini-flash-lite-latest` |

Model choice was made by measurement, not preference — `scripts/bench_models.py`
benchmarks candidates on this exact task and reports accuracy and latency. `gpt-5.4-mini`
scored 8/8 with the lowest p95 of every candidate tried. Anything OpenAI-compatible can be
substituted by pointing `OPENAI_BASE_URL` elsewhere.

### Environment variables

Names only. **No values appear in this repository.** `.env.example` is the committed template
and contains empty values; `.env` is gitignored.

| Variable | Required | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | **yes\*** | Credential for the OpenAI provider |
| `LLM_PROVIDER` | no (default `openai`) | Which provider to try first: `openai`, `groq`, `gemini` |
| `LLM_MODEL` | no (default `gpt-5.4-mini`) | Model identifier for the OpenAI provider |
| `OPENAI_BASE_URL` | no (default OpenAI) | Override for an OpenAI-compatible endpoint |
| `GROQ_API_KEY` | no | Credential for the Groq failover provider |
| `GROQ_MODEL` | no | Model identifier for Groq |
| `GEMINI_API_KEY` | no | Credential for the Google AI Studio failover provider |
| `GEMINI_MODEL` | no | Model identifier for Gemini |
| `LLM_TIMEOUT_SECONDS` | no (default `8`) | Budget for a single provider call |
| `LLM_TOTAL_BUDGET_SECONDS` | no (default `20`) | Wall-clock cap for the whole interpretation phase, across every provider and retry |
| `PORT` | no (default `8000`) | Port to bind. The service always binds `0.0.0.0` |

\* At least one provider key is required for the language model to run. Without any key the
service still answers and still returns valid plans, but it degrades to the keyword fallback
interpreter — the `X-Interpreter` response header states which path was used.

---

## 3. Endpoints and worked example

| Endpoint | Purpose |
|---|---|
| `GET /health` | Readiness. Returns exactly `{"status":"ok"}` |
| `POST /optimize-energy` | Interpretation plus the 24-hour schedule |

| Status | When |
|---|---|
| `200` | Success |
| `400` | Malformed JSON or a structurally invalid request |
| `500` | Controlled internal error — a fixed message, never a stack trace or a secret |

### `GET /health`

```bash
curl http://localhost:8000/health
```
```json
{"status":"ok"}
```

### `POST /optimize-energy` — complete, copy-pasteable request

This is public sample case `SAMPLE-01` in full. Paste it as-is.

```bash
curl -s -X POST http://localhost:8000/optimize-energy \
  -H 'Content-Type: application/json' \
  -d @- <<'JSON'
{
  "scenario_id": "SAMPLE-01",
  "operator_notes": [
    "Facilities will wash the rooftop solar panels from noon until 2 PM. During cleaning, usable solar should be treated as roughly 25% of the forecast.",
    "The sports office moved next month's registration deadline."
  ],
  "hours": [
    {"hour": 0, "demand_kwh": 90, "solar_kwh": 0, "tariff_bdt_per_kwh": 6},
    {"hour": 1, "demand_kwh": 85, "solar_kwh": 0, "tariff_bdt_per_kwh": 6},
    {"hour": 2, "demand_kwh": 80, "solar_kwh": 0, "tariff_bdt_per_kwh": 5},
    {"hour": 3, "demand_kwh": 80, "solar_kwh": 0, "tariff_bdt_per_kwh": 5},
    {"hour": 4, "demand_kwh": 85, "solar_kwh": 0, "tariff_bdt_per_kwh": 5},
    {"hour": 5, "demand_kwh": 95, "solar_kwh": 0, "tariff_bdt_per_kwh": 6},
    {"hour": 6, "demand_kwh": 110, "solar_kwh": 5, "tariff_bdt_per_kwh": 8},
    {"hour": 7, "demand_kwh": 130, "solar_kwh": 20, "tariff_bdt_per_kwh": 10},
    {"hour": 8, "demand_kwh": 150, "solar_kwh": 50, "tariff_bdt_per_kwh": 12},
    {"hour": 9, "demand_kwh": 165, "solar_kwh": 90, "tariff_bdt_per_kwh": 14},
    {"hour": 10, "demand_kwh": 175, "solar_kwh": 130, "tariff_bdt_per_kwh": 16},
    {"hour": 11, "demand_kwh": 180, "solar_kwh": 160, "tariff_bdt_per_kwh": 16},
    {"hour": 12, "demand_kwh": 185, "solar_kwh": 180, "tariff_bdt_per_kwh": 15},
    {"hour": 13, "demand_kwh": 180, "solar_kwh": 170, "tariff_bdt_per_kwh": 14},
    {"hour": 14, "demand_kwh": 170, "solar_kwh": 140, "tariff_bdt_per_kwh": 13},
    {"hour": 15, "demand_kwh": 165, "solar_kwh": 90, "tariff_bdt_per_kwh": 14},
    {"hour": 16, "demand_kwh": 170, "solar_kwh": 45, "tariff_bdt_per_kwh": 18},
    {"hour": 17, "demand_kwh": 185, "solar_kwh": 10, "tariff_bdt_per_kwh": 22},
    {"hour": 18, "demand_kwh": 205, "solar_kwh": 0, "tariff_bdt_per_kwh": 28},
    {"hour": 19, "demand_kwh": 215, "solar_kwh": 0, "tariff_bdt_per_kwh": 30},
    {"hour": 20, "demand_kwh": 205, "solar_kwh": 0, "tariff_bdt_per_kwh": 26},
    {"hour": 21, "demand_kwh": 175, "solar_kwh": 0, "tariff_bdt_per_kwh": 18},
    {"hour": 22, "demand_kwh": 135, "solar_kwh": 0, "tariff_bdt_per_kwh": 10},
    {"hour": 23, "demand_kwh": 105, "solar_kwh": 0, "tariff_bdt_per_kwh": 7}
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

Actual response, abridged to the first three plan rows (the real response contains all 24):

```json
{
  "scenario_id": "SAMPLE-01",
  "directive_interpretation": [
    {
      "note_index": 0,
      "applies": true,
      "directive_type": "solar_reduction",
      "structured_adjustment": {"hours": [12, 13], "factor": 0.25},
      "explanation": "Usable solar output is reduced to roughly 25% of the forecast during the cleaning of the rooftop solar panels."
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
    {"hour": 0, "grid_kwh": 50.0,  "solar_used_kwh": 0.0, "battery_action": "discharge", "battery_kwh": 40.0, "battery_energy_after_kwh": 70.0},
    {"hour": 1, "grid_kwh": 85.0,  "solar_used_kwh": 0.0, "battery_action": "idle",      "battery_kwh": 0.0,  "battery_energy_after_kwh": 70.0},
    {"hour": 2, "grid_kwh": 130.0, "solar_used_kwh": 0.0, "battery_action": "charge",    "battery_kwh": 50.0, "battery_energy_after_kwh": 120.0}
  ],
  "total_grid_kwh": 2692.5,
  "total_cost_bdt": 38365.0,
  "peak_grid_kwh": 187.5,
  "plan_summary": "Applied 1 operator directive(s): solar reduction over hours [12, 13]. Ignored 1 unrelated note(s). ..."
}
```

`total_cost_bdt` is **38365.0**, matching the published reference optimum exactly. The hourly
actions differ from the reference schedule, which the rules permit — equivalent optimal
schedules are accepted, and only the recalculated cost and validity are judged.

The response body carries exactly the seven required top-level fields and nothing else.
Diagnostics are returned as **headers** so the JSON stays on contract:

| Header | Meaning |
|---|---|
| `X-Interpreter` | Which path produced the directives, e.g. `llm:openai/gpt-5.4-mini` |
| `X-Optimizer` | Which solve path produced the plan, e.g. `lp` |
| `X-Self-Check` | Result of our own replay validation on this response |
| `X-Latency-Ms` | Server-side handling time |

---

## 4. Testing against the public sample cases

With the service running (locally or deployed), point the script at the base URL:

```bash
python scripts/test_samples.py http://localhost:8000
# or against the live endpoint:
python scripts/test_samples.py https://buphackathon-production-7134.up.railway.app
```

The script is **standard-library only** — it needs no extra installation. It POSTs all ten
published cases, replays every returned plan against organizer ground-truth directives, then
repeats the whole sweep twice more to report cold and warm latency, and finally sends 20
consecutive valid requests to confirm no server errors.

### Expected output

```
GET /health -> 200 {"status": "ok"}  [ok]

case         http   valid   interp         cost    vs ref       ms
------------------------------------------------------------------
SAMPLE-01     200    pass     pass       38,365      pass     2902
SAMPLE-02     200    pass     pass       42,885      pass     1231
SAMPLE-03     200    pass     pass       35,480      pass     2317
SAMPLE-04     200    pass     pass       40,495      pass     1760
SAMPLE-05     200    pass     pass       33,950      pass     1563
SAMPLE-06     200    pass     pass       34,090      pass     2717
SAMPLE-07     200    pass     pass       38,550      pass     2353
SAMPLE-08     200    pass     pass       37,665      pass     2106
SAMPLE-09     200    pass     pass       34,873      pass     1872
SAMPLE-10     200    pass     pass       41,620      pass     2611
------------------------------------------------------------------
run 2: p95 14 ms   median 4 ms
run 3: p95 14 ms   median 5 ms

                     p95    median       max
run 1 (cold)        2902      2106      2902
runs 2-3              14         5        14
all 30              2611         5      2902

p95 over 3 runs: 2611 ms -> latency score 3/3   (<=5s is 3/3; hard ceiling 30s)
cache speedup on repeats: 453.3x

ok   hammer 20x valid payloads: 20/20 returned 200, 0 server errors, p95 13 ms

All 10 cases returned a structurally valid plan.
```

Column meanings: **valid** = the plan survives the full replay validator; **interp** =
`directive_type`, affected hours and numeric values match the published reference; **cost** =
`total_cost_bdt` equals the reference optimum within the 0.01 tolerance. All ten must read
`pass`. Exit code is `0` on success, `1` on any failure.

### Other suites

All are offline except the last two and consume no model quota:

```bash
python scripts/test_optimizer.py      # LP against the reference optima, plus quality_ratio
python scripts/test_guardrails.py     # repair layer, on clean AND deliberately corrupted output
python scripts/test_overlaps.py       # how overlapping directives combine; infeasibility fallback
python scripts/test_selfcheck.py      # replay gate, safe fallback plan, directive dropping
python scripts/check_llm.py           # provider reachability, extraction accuracy, latency
python scripts/bench_models.py        # compare candidate models on accuracy and latency
python scripts/verify_deployment.py <base-url>   # full check of a live deployment
```

`test_optimizer.py` reports the rubric's own optimization formula,
`min(1, organizer_optimal_cost / our_cost)`:

```
mean quality_ratio                     1.0000
projected Optimization Quality          10.00 / 10
```

---

## 5. Architecture — LLM → guardrails → LP optimizer → replay validator

```
POST /optimize-energy
  │
  ├─ app/models.py       pydantic schema validation → 400 on anything malformed
  │
  ├─ app/llm.py          LLM INTERPRETATION  ← mandatory primary path
  │                        one call per request carrying every operator note plus
  │                        battery capacity / initial / minimum energy.
  │                        temperature 0, strict json_schema, ~10 s per call,
  │                        20 s wall-clock budget for the whole phase,
  │                        cache keyed on a hash of the note text.
  │                        Invalid output → retry once with the validator's own
  │                        error text → next provider → keyword fallback.
  │
  ├─ app/guardrails.py   deterministic validation and repair of model output
  ├─ app/timewindows.py  deterministic hour extraction from the note text
  ├─ app/fallback.py     keyword interpreter — LAST RESORT ONLY, never primary
  │
  ├─ app/optimizer.py    LP optimizer (SciPy HiGHS), provably optimal
  ├─ app/validator.py    replay validator — re-runs every judge rule on our output
  │
  └─ response (exactly the seven required fields)
```

### The language model's role

The model is the **only** component that reads the operator notes, and its structured output is
what the optimizer's constraints are built from. It is not used for `plan_summary` or for
documentation. Everything downstream is deterministic, and the model's output is treated as
untrusted data until it has passed `app/guardrails.py`.

It is asked for a **flat** object per note — `directive_type`, `hours`, and at most one numeric
value. It never produces the nested `structured_adjustment`; `app/guardrails.py` assembles that.
The model therefore cannot emit a wrong adjustment shape for a given directive type. The
response array is built from **our** note list by our own index, so a missing, duplicated or
out-of-order `note_index` from the model cannot shift the mapping.

### Guardrails and repairs

`validate_model_output` is a pure function checking directive type, one entry per note in
`note_index` order, hours unique and ascending within 0–23, `factor` in [0,1], reserve within
battery capacity, non-negative grid cap, and the `no_op ⇔ applies=false ⇔ null` rule. Its error
text is what gets fed back to the model on the retry.

On top of that, five repairs are applied, each one a failure mode observed against a real
provider during development:

| Failure observed | Repair |
|---|---|
| Window end off by one | `app/timewindows.py` parses the window from the note text and wins whenever the wording is unambiguous |
| Solar factor polarity flipped | `"80% reduction"` resolves to `0.2`, not `0.8` |
| Percent reserve left as a percent | `"50% of capacity"` is converted to kWh against `capacity_kwh` |
| Distractor promoted to a directive | Relevance re-check forces `no_op` |
| Solar outage mislabelled | Named solar hardware plus an outage phrase plus a time window resolves to `solar_reduction`, even if the model said `no_op` or `no_discharge_window` |

Hour extraction was the weakest link measured on a real provider: the same phrase
`"6 PM until 9 PM"` came back as `[18,19]` inside a battery-reserve note and `[18,19,20]` inside
a grid-cap note. Hours are therefore extracted deterministically whenever the note states the
window unambiguously, and the model keeps the decision only when the wording is genuinely
ambiguous, such as `"from one until three"`, where reading context beats any regular expression.

### The optimizer is a linear program, not a heuristic

Four non-negative variables per hour — `grid`, `solar_used`, `charge`, `discharge` — minimising
`Σ tariff[h] × grid[h]` subject to hourly energy balance, effective solar after any
`solar_reduction`, battery bounds and rate limits, directive windows and grid caps, and
end-of-day battery neutrality. Solved with `scipy.optimize.linprog(method="highs")` in roughly
1.5 ms. Battery round-trip efficiency is 100%, so charge and discharge in the same hour is
cost-neutral and is netted into a single action afterwards; no integer variables are needed and
the continuous solution is exactly optimal.

Overlapping directives of the same kind resolve to the single most restrictive value and never
compound: minimum factor for `solar_reduction`, minimum cap for `max_grid_window`, maximum floor
for `minimum_battery_reserve`, union of hours for the charge and discharge windows.

If the model is infeasible — typically a misparsed directive — the single offending directive is
dropped and the problem re-solved, rather than discarding all of them. The route taken is
reported in `X-Optimizer`: `lp`, `lp-dropped:<type>`, `lp-relaxed:NofM`, `lp-no-directives`,
or `safe-plan`.

### Replay validator

Before any response leaves the process it is replayed against every documented judge rule: 24
unique hours, finite non-negative values, hourly energy balance, `solar_used` within effective
solar, battery transitions, bounds and rate limits, every applied directive, end-of-day
neutrality, and a recount of all three reported totals — at the published 0.01 tolerance.

Two details make this a real replica rather than a formality. It validates the **serialized
JSON**, so the numbers under test are the exact ones on the wire, not internal Python floats.
And a failure **changes the response**: the plan is rebuilt with an idle-battery schedule capped
at effective solar, re-validated, and whichever of the two has fewer violations is the one that
ships. The reason is logged internally and never appears in the response body.

---

## 6. Docker pull / run fallback

The published image is the same artifact that is deployed. It is built, smoke-tested against all
ten public cases, and pushed by GitHub Actions on every commit to `main`
(`.github/workflows/docker.yml`). The package is public, so no login is required.

```bash
docker pull ghcr.io/tanviranjumapurbo/bup_hackathon:0466c8f

docker run --rm -p 8000:8000 \
  -e PORT=8000 \
  -e LLM_PROVIDER=openai \
  -e LLM_MODEL=gpt-5.4-mini \
  -e OPENAI_API_KEY="<your key>" \
  ghcr.io/tanviranjumapurbo/bup_hackathon:0466c8f
```

```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

Pin by digest instead of tag if preferred:

```
ghcr.io/tanviranjumapurbo/bup_hackathon@sha256:711dbc4b689dc0a02bb2e1c20a7c6479b4c00f5e94ed885d9f044879869bf2ea
```

The container binds `0.0.0.0`, honours `$PORT` (so the same image runs on any host), exposes
`8000`, runs as non-root uid 1000, becomes ready in about 2 seconds, and contains **no
credentials of any kind** — every key is supplied at run time.

---

## 7. Dependencies, limitations and secret handling

### Dependencies and credits

| Dependency | Role |
|---|---|
| [FastAPI](https://fastapi.tiangolo.com/) | HTTP framework |
| [uvicorn](https://www.uvicorn.org/) | ASGI server |
| [pydantic](https://docs.pydantic.dev/) | Request and response schema validation |
| [SciPy](https://scipy.org/) — `linprog`, HiGHS | Linear-programming optimizer |
| [NumPy](https://numpy.org/) | Constraint matrix construction |
| [httpx](https://www.python-httpx.org/) | Async HTTP client for the model provider |
| [OpenAI API](https://platform.openai.com/) | `gpt-5.4-mini`, operator-note interpretation |

Exact pins are in `requirements.txt`. Verified on Python 3.11 (container) and 3.12 (development)
with FastAPI 0.141, pydantic 2.13, SciPy 1.13, NumPy 1.26, httpx 0.28.

The problem statement, participant guide and public sample cases are provided by the BUP CSE
Fest 2026 organizers. Architecture and implementation are our own work. An AI coding assistant
was used during development, which the official rulebook permits.

### Known limitations

- **Hosted-model dependency.** Operator-note interpretation depends on a reachable model
  provider. If every configured provider fails, the service stays available but degrades to the
  keyword interpreter in `app/fallback.py` and accuracy drops. `X-Interpreter` always states
  which path ran. This fallback is a last resort and is never the primary path.
- **Cache scope.** The interpretation and response caches are in-process and per-instance. They
  are not shared across replicas and do not survive a restart.
- **Ambiguous time wording.** When a window carries no meridiem anywhere — for example
  `"from one until three"` — the deterministic parser defers to the model, because context
  resolves it and a regular expression cannot.
- **Unsatisfiable directives.** If a directive cannot be met by any schedule, the optimizer drops
  the offending one and reports that in `X-Optimizer`. Organizer scenarios are documented as
  feasible, so this only guards against a misparse.
- **Latency floor.** A request that misses the cache is dominated by the provider round trip,
  typically 1.0–2.0 s. Measured p95 over three full sweeps is about 1.7 s, well inside the 5 s
  band, but it moves with provider and network conditions.

### Secret handling

- Every credential is read from an environment variable at run time. Nothing is hard-coded.
- `.env` is gitignored. `.env.example` is committed and contains **variable names only, with
  empty values**.
- No key, token, raw prompt, stack trace or internal hostname is written to logs or returned in
  any API response. The generic exception handler returns a fixed message and logs detail
  server-side only. This was verified by forcing a provider failure and grepping both the
  response body and the server log.
- The Docker image contains no baked-in credentials — confirmed with
  `docker inspect --format='{{json .Config.Env}}'`, which shows only Python and `PORT` variables.
- Repository history has been swept for key-shaped strings; there are none.
