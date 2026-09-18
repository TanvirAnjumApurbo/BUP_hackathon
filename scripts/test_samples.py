#!/usr/bin/env python3
"""POST every public sample case at a running GridWise service and report.

Standard library only, so it runs anywhere with no install step.

    python scripts/test_samples.py                        # http://localhost:8000
    python scripts/test_samples.py https://your.koyeb.app

Checks per case:
  HTTP    200 with a JSON body
  VALID   the returned plan survives the full replay validator
  INTERP  directive_interpretation matches the published reference
  COST    total_cost_bdt versus the reference optimal cost

At stage 1 (no LLM, no optimizer) INTERP and COST are expected to fail on cases
that carry a real directive. HTTP and VALID must pass on all ten.
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from app.validator import TOL, replay  # noqa: E402

SAMPLES = REPO / "BUP_CSE_FEST_2026_Participant_Docs" / "BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json"

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"


def post(url: str, payload: dict, timeout: float = 35.0):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode())
            return resp.status, body, (time.perf_counter() - started) * 1000
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode()
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            body = {"raw": raw[:400]}
        return exc.code, body, (time.perf_counter() - started) * 1000


def get(url: str, timeout: float = 20.0):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode())
    except Exception as exc:  # noqa: BLE001
        return None, {"error": str(exc)}


def norm_adjustment(adj):
    """Compare numbers loosely; wording is never compared."""
    if adj is None:
        return None
    out = {}
    for key, value in adj.items():
        out[key] = list(value) if isinstance(value, list) else round(float(value), 4)
    return out


def interpretation_matches(actual, expected) -> bool:
    if not isinstance(actual, list) or len(actual) != len(expected):
        return False
    for got, want in zip(actual, expected):
        if got.get("note_index") != want["note_index"]:
            return False
        if got.get("applies") != want["applies"]:
            return False
        if got.get("directive_type") != want["directive_type"]:
            return False
        if norm_adjustment(got.get("structured_adjustment")) != norm_adjustment(
            want["structured_adjustment"]
        ):
            return False
    return True


def main() -> int:
    base = (sys.argv[1] if len(sys.argv) > 1 else os.environ.get("GRIDWISE_URL", "http://localhost:8000")).rstrip("/")
    cases = json.loads(SAMPLES.read_text(encoding="utf-8"))["cases"]

    print(f"\ntarget: {base}\n")

    status, body = get(f"{base}/health")
    health_ok = status == 200 and body == {"status": "ok"}
    mark = f"{GREEN}ok{RESET}" if health_ok else f"{RED}FAIL{RESET}"
    print(f"GET /health -> {status} {json.dumps(body)}  [{mark}]")
    if not health_ok:
        print(f"{RED}health must return exactly {{\"status\":\"ok\"}} — stopping{RESET}")
        return 1

    print()
    header = f"{'case':<11}{'http':>6}{'valid':>8}{'interp':>9}{'cost':>13}{'vs ref':>10}{'ms':>9}"
    print(header)
    print("-" * len(header))

    failures = []
    latencies = []
    hard_fail = 0

    for case in cases:
        cid = case["id"]
        payload = case["input"]
        expected = case["expected_output"]

        status, body, ms = post(f"{base}/optimize-energy", payload)
        latencies.append(ms)

        if status != 200:
            print(f"{cid:<11}{status:>6}{RED}{'-':>8}{'-':>9}{'-':>13}{'-':>10}{RESET}{ms:>9.0f}")
            failures.append((cid, [f"HTTP {status}: {json.dumps(body)[:200]}"]))
            hard_fail += 1
            continue

        problems = replay(payload, body)
        valid = not problems
        if not valid:
            hard_fail += 1
            failures.append((cid, problems))

        interp_ok = interpretation_matches(body.get("directive_interpretation"), expected["directive_interpretation"])
        cost = body.get("total_cost_bdt")
        ref = expected["total_cost_bdt"]
        cost_ok = isinstance(cost, (int, float)) and abs(cost - ref) <= TOL
        ratio = (ref / cost) if isinstance(cost, (int, float)) and cost > 0 else 0.0

        def tick(flag, good=GREEN, bad=RED):
            return f"{good}pass{RESET}" if flag else f"{bad}FAIL{RESET}"

        # interp/cost are expected to fail at stage 1 -> show them amber
        print(
            f"{cid:<11}{status:>6}"
            f"{tick(valid):>17}"
            f"{tick(interp_ok, bad=YELLOW):>18}"
            f"{(f'{cost:,.0f}' if isinstance(cost, (int, float)) else '-'):>13}"
            f"{tick(cost_ok, bad=YELLOW):>19}"
            f"{ms:>9.0f}"
        )

    latencies.sort()
    p95 = latencies[int(len(latencies) * 0.95) - 1] if latencies else 0.0
    print("-" * len(header))
    print(f"{DIM}p95 latency {p95:.0f} ms   max {max(latencies, default=0):.0f} ms{RESET}")

    if failures:
        print(f"\n{RED}{'=' * 60}\nHARD FAILURES (these must be zero)\n{'=' * 60}{RESET}")
        for cid, problems in failures:
            print(f"\n{RED}{cid}{RESET}")
            for problem in problems[:8]:
                print(f"  - {problem}")
            if len(problems) > 8:
                print(f"  ... {len(problems) - 8} more")

    print()
    if hard_fail:
        print(f"{RED}{hard_fail} case(s) returned an invalid plan.{RESET}")
        return 1
    print(f"{GREEN}All 10 cases returned a structurally valid plan.{RESET}")
    print(f"{DIM}interp = directive_type + hours + numeric values vs reference. cost = exact optimal.{RESET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
