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

from app.models import DirectiveInterpretation, OptimizeRequest, OptimizeResponse
from app.planner import build_baseline_plan, totals_from_plan
from app.validator import replay

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("gridwise")

app = FastAPI(title="GridWise", version="0.1.0")

NO_OP_EXPLANATION = "This note does not affect today's 24-hour energy schedule."

BASELINE_SUMMARY = (
    "Baseline schedule: campus demand is met from available solar first and the remaining "
    "energy is purchased from the grid. The battery is held at its initial state of charge, "
    "so end-of-day battery neutrality is preserved."
)


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

    # STAGE 2 will replace this with real LLM interpretation.
    interpretation = [
        DirectiveInterpretation(
            note_index=index,
            applies=False,
            directive_type="no_op",
            structured_adjustment=None,
            explanation=NO_OP_EXPLANATION,
        )
        for index in range(len(req.operator_notes))
    ]

    # STAGE 2 will replace this with the LP optimizer.
    rows = build_baseline_plan(req)
    by_hour = req.hours_by_hour()
    total_grid, total_cost, peak_grid = totals_from_plan(rows, by_hour)

    result = OptimizeResponse(
        scenario_id=req.scenario_id,
        directive_interpretation=interpretation,
        hourly_plan=rows,
        total_grid_kwh=total_grid,
        total_cost_bdt=total_cost,
        peak_grid_kwh=peak_grid,
        plan_summary=BASELINE_SUMMARY,
    )

    # Final gate: replay our own output through the judge's rules before shipping
    # it. At this stage the baseline is valid by construction, so anything found
    # here is a genuine bug worth shouting about.
    problems = replay(req.model_dump(), result.model_dump())
    if problems:
        logger.error("self-check failed for %s: %s", req.scenario_id, problems[:5])

    elapsed_ms = (time.perf_counter() - started) * 1000
    response.headers["X-Interpreter"] = "stage1:baseline-no-llm"
    response.headers["X-Latency-Ms"] = f"{elapsed_ms:.1f}"
    response.headers["X-Self-Check"] = "pass" if not problems else "fail"
    logger.info("%s handled in %.1f ms (self-check %s)", req.scenario_id, elapsed_ms,
                "pass" if not problems else "FAIL")

    return result
