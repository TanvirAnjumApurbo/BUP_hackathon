"""GridWise API — BUP CSE Fest 2026 preliminary.

Endpoints are exactly as specified by the Problem Statement:
    GET  /health           -> {"status": "ok"}
    POST /optimize-energy  -> interpretation + 24-hour schedule

STAGE 1: schema, contract and a valid baseline schedule. No LLM, no optimizer.
"""

import logging
import time
from typing import Any, Dict, List

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app import fallback, llm
from app.config import load_dotenv
from app.guardrails import build_interpretations
from app.models import OptimizeRequest, OptimizeResponse
from app.optimizer import build_optimal_plan
from app.planner import totals_from_plan
from app.validator import replay

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("gridwise")

app = FastAPI(title="GridWise", version="1.0.0")


@app.on_event("shutdown")
async def _shutdown() -> None:
    await llm.aclose()


def summarize(interpretation, route: str) -> str:
    applied = [entry for entry in interpretation if entry.applies]
    ignored = len(interpretation) - len(applied)

    if applied:
        described = "; ".join(
            f"{entry.directive_type.replace('_', ' ')} over hours "
            f"{entry.structured_adjustment['hours']}"
            for entry in applied
        )
        head = f"Applied {len(applied)} operator directive(s): {described}."
    else:
        head = "No operator note changed today's schedule."

    if ignored:
        head += f" Ignored {ignored} unrelated note(s)."

    if route == "lp":
        tail = (
            " The 24-hour schedule then minimises grid cost by charging the battery during "
            "cheap hours and discharging it into expensive ones, using all available solar "
            "first and returning the battery to its starting level by the end of the day."
        )
    else:
        tail = " A conservative valid schedule was produced for this scenario."
    return head + tail


# --------------------------------------------------------------------------
# Error handling — the service must never return an uncontrolled 5xx, and must
# never leak a stack trace or a secret to the caller.
# --------------------------------------------------------------------------


@app.exception_handler(RequestValidationError)
async def on_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Malformed JSON or a structurally invalid request is a 400, not a 422.

    FastAPI defaults to 422; the Problem Statement reserves 422 for well-formed
    but semantically invalid bodies, and documents 400 for this case.
    """
    details: List[Dict[str, Any]] = []
    for err in exc.errors()[:10]:
        location = ".".join(str(part) for part in err.get("loc", ()) if part != "body")
        details.append({"field": location or "body", "error": err.get("msg", "invalid")})
    return JSONResponse(
        status_code=400,
        content={"error": "invalid_request", "detail": details},
    )


@app.exception_handler(StarletteHTTPException)
async def on_http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": "http_error", "detail": str(exc.detail)},
    )


@app.exception_handler(Exception)
async def on_unhandled_error(request: Request, exc: Exception) -> JSONResponse:
    # Full detail to our logs only. The caller gets a controlled message.
    logger.exception("unhandled error on %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={"error": "internal_error", "detail": "An internal error occurred."},
    )


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/optimize-energy", response_model=OptimizeResponse)
async def optimize_energy(req: OptimizeRequest, response: Response) -> OptimizeResponse:
    started = time.perf_counter()

    # 1. The language model reads the notes. This is the mandatory primary path.
    raw, interpreter = await llm.interpret(req.operator_notes, req.battery)

    # 2. Only if every provider failed do we degrade to keywords, so that an
    #    outage costs accuracy rather than availability.
    if raw is None:
        raw = fallback.interpret(req.operator_notes, req.battery)
        interpreter = f"fallback-keyword ({interpreter})"

    # 3. Validate and repair. The nested adjustment shape is built here, from our
    #    own note list — the model's note_index is never trusted for ordering.
    interpretation = build_interpretations(req.operator_notes, req.battery, raw)

    # 4. Exact optimum under the resulting constraints.
    directives = [
        entry.model_dump() for entry in interpretation if entry.applies
    ]
    rows, route = build_optimal_plan(req, directives)

    by_hour = req.hours_by_hour()
    total_grid, total_cost, peak_grid = totals_from_plan(rows, by_hour)

    result = OptimizeResponse(
        scenario_id=req.scenario_id,
        directive_interpretation=interpretation,
        hourly_plan=rows,
        total_grid_kwh=total_grid,
        total_cost_bdt=total_cost,
        peak_grid_kwh=peak_grid,
        plan_summary=summarize(interpretation, route),
    )

    # 5. Final gate: replay our own output through the judge's own rules before
    #    shipping it. Anything found here is a genuine bug worth shouting about.
    problems = replay(req.model_dump(), result.model_dump())
    if problems:
        logger.error("self-check failed for %s: %s", req.scenario_id, problems[:5])

    elapsed_ms = (time.perf_counter() - started) * 1000
    response.headers["X-Interpreter"] = interpreter
    response.headers["X-Optimizer"] = route
    response.headers["X-Latency-Ms"] = f"{elapsed_ms:.1f}"
    response.headers["X-Self-Check"] = "pass" if not problems else "fail"
    logger.info(
        "%s: %s via %s, plan via %s, %.0f ms, self-check %s",
        req.scenario_id, len(directives), interpreter, route, elapsed_ms,
        "pass" if not problems else "FAIL",
    )

    return result
