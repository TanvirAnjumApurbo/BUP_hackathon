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


def percentile(sorted_values, pct: float) -> float:
    if not sorted_values:
        return 0.0
    index = max(0, min(len(sorted_values) - 1, int(round(len(sorted_values) * pct / 100.0)) - 1))
    return sorted_values[index]


def latency_band(p95_ms: float):
    """The rubric's latency scoring bands."""
    if p95_ms <= 5000:
        return "3/3", GREEN
    if p95_ms <= 15000:
        return "2/3", YELLOW
    if p95_ms <= 30000:
        return "1/3", YELLOW
    return "0/3", RED


def hammer(base: str, payload: dict, times: int = 20):
    """Repeated valid requests must never produce a 5xx."""
    statuses, latencies = [], []
    for _ in range(times):
        status, _, ms = post(f"{base}/optimize-energy", payload)
        statuses.append(status)
        latencies.append(ms)
    return statuses, latencies


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

        # Replay exactly as the judge does: against organizer ground-truth
        # directives, not against whatever the response claimed for itself.
        truth = [d for d in expected["directive_interpretation"] if d["applies"]]
        problems = replay(payload, body, truth)
        # Also self-consistency, in case our own interpretation disagrees.
        problems += [p for p in replay(payload, body) if p not in problems]
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

    print("-" * len(header))

    # Two further passes over every case. Run 1 is cold; runs 2 and 3 exercise
    # the caches, which is what the judge sees when it replays a scenario.
    run_latencies = [list(latencies)]
    for run in (2, 3):
        this_run = []
        for case in cases:
            _, _, ms = post(f"{base}/optimize-energy", case["input"])
            this_run.append(ms)
        run_latencies.append(this_run)
        this_run_sorted = sorted(this_run)
        print(f"{DIM}run {run}: p95 {percentile(this_run_sorted, 95):.0f} ms   "
              f"median {percentile(this_run_sorted, 50):.0f} ms{RESET}")
        latencies.extend(this_run)

    cold = sorted(run_latencies[0])
    warm = sorted(run_latencies[1] + run_latencies[2])
    combined = sorted(latencies)
    p95 = percentile(combined, 95)

    print()
    print(f"{'':<14}{'p95':>10}{'median':>10}{'max':>10}")
    print(f"{'run 1 (cold)':<14}{percentile(cold, 95):>10.0f}{percentile(cold, 50):>10.0f}{max(cold):>10.0f}")
    print(f"{'runs 2-3':<14}{percentile(warm, 95):>10.0f}{percentile(warm, 50):>10.0f}{max(warm):>10.0f}")
    print(f"{'all 30':<14}{p95:>10.0f}{percentile(combined, 50):>10.0f}{max(combined):>10.0f}")

    band, colour = latency_band(p95)
    print(f"\np95 over 3 runs: {colour}{p95:.0f} ms{RESET} -> latency score {colour}{band}{RESET}"
          f"   {DIM}(<=5s is 3/3; hard ceiling 30s){RESET}")
    if max(combined) > 30000:
        print(f"{RED}A request exceeded the 30s hard ceiling.{RESET}")
        hard_fail += 1
    speedup = percentile(cold, 50) / max(percentile(warm, 50), 0.001)
    print(f"{DIM}cache speedup on repeats: {speedup:.1f}x{RESET}")

    if failures:
        print(f"\n{RED}{'=' * 60}\nHARD FAILURES (these must be zero)\n{'=' * 60}{RESET}")
        for cid, problems in failures:
            print(f"\n{RED}{cid}{RESET}")
            for problem in problems[:8]:
                print(f"  - {problem}")
            if len(problems) > 8:
                print(f"  ... {len(problems) - 8} more")

    # 20 valid requests in a row: zero 5xx is the requirement.
    print()
    statuses, hammer_ms = hammer(base, cases[0]["input"], 20)
    server_errors = [s for s in statuses if s is None or s >= 500]
    non_200 = [s for s in statuses if s != 200]
    hammer_sorted = sorted(hammer_ms)
    ok = not server_errors and not non_200
    print(f"{(GREEN + 'ok  ' if ok else RED + 'FAIL') + RESET} hammer 20x valid payloads: "
          f"{len(statuses) - len(non_200)}/20 returned 200, {len(server_errors)} server errors, "
          f"p95 {percentile(hammer_sorted, 95):.0f} ms")
    if not ok:
        hard_fail += 1
        print(f"      {RED}statuses: {statuses}{RESET}")

    print()
    if hard_fail:
        print(f"{RED}{hard_fail} failure(s).{RESET}")
        return 1
    print(f"{GREEN}All 10 cases returned a structurally valid plan.{RESET}")
    print(f"{DIM}interp = directive_type + hours + numeric values vs reference. cost = exact optimal.{RESET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
