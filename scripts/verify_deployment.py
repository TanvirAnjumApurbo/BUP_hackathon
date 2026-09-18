#!/usr/bin/env python3
"""One command to confirm a live deployment is actually judge-ready.

    python scripts/verify_deployment.py https://your-app.up.railway.app

Goes beyond scripts/test_samples.py by checking the things that only go wrong on
a real deployment:

  * /health returns exactly {"status":"ok"} and is reachable from outside
  * the LLM is genuinely in the path (X-Interpreter), not the keyword fallback
  * outbound network to the model provider works from inside the container
  * p95 latency against the scored bands
  * malformed input still returns 4xx rather than 5xx
  * cold-start readiness

The LLM check matters most. The keyword fallback scores 10/10 on the public
cases, so a deployment with a missing or blocked API key looks perfect while
silently failing the mandatory-LLM requirement.
"""

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from app.validator import TOL, replay  # noqa: E402

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"

SAMPLES = (
    REPO
    / "BUP_CSE_FEST_2026_Participant_Docs"
    / "BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json"
)


def request(url, payload=None, timeout=35.0):
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    req = urllib.request.Request(url, data=data, headers=headers,
                                 method="POST" if data else "GET")
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, dict(resp.headers), resp.read().decode(), (time.perf_counter() - started) * 1000
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers), exc.read().decode(), (time.perf_counter() - started) * 1000
    except Exception as exc:  # noqa: BLE001
        return None, {}, str(exc), (time.perf_counter() - started) * 1000


def header(headers, name, default=""):
    for key, value in headers.items():
        if key.lower() == name.lower():
            return value
    return default


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python scripts/verify_deployment.py https://your-app-url")
        return 2
    base = sys.argv[1].rstrip("/")
    cases = json.loads(SAMPLES.read_text(encoding="utf-8"))["cases"]
    failures = []

    print(f"\ntarget: {base}\n{'=' * 72}")

    # 1. health
    status, _, body, ms = request(f"{base}/health")
    ok = status == 200 and body.strip() == '{"status":"ok"}'
    print(f"{(GREEN + 'ok  ' if ok else RED + 'FAIL') + RESET} /health -> {status} {body.strip()[:60]} ({ms:.0f} ms)")
    if not ok:
        failures.append("health endpoint")
        print(f"{RED}cannot continue without a healthy endpoint{RESET}\n")
        return 1

    # 2. full case sweep
    print()
    head = f"{'case':<11}{'http':>6}{'valid':>8}{'interp':>9}{'cost':>12}{'vs ref':>9}{'ms':>8}  interpreter"
    print(head)
    print("-" * len(head))
    latencies, interpreters = [], set()
    passed = 0

    for case in cases:
        expected = case["expected_output"]
        status, headers, raw, ms = request(f"{base}/optimize-energy", case["input"])
        latencies.append(ms)
        interpreter = header(headers, "X-Interpreter", "?")
        interpreters.add(interpreter)

        if status != 200:
            print(f"{case['id']:<11}{status:>6}{RED}{'  -':>8}{'  -':>9}{'  -':>12}{'  -':>9}{RESET}{ms:>8.0f}  {interpreter}")
            failures.append(f"{case['id']} HTTP {status}")
            continue

        body = json.loads(raw)
        problems = replay(case["input"], body)
        valid = not problems
        cost = body.get("total_cost_bdt")
        ref = expected["total_cost_bdt"]
        cost_ok = isinstance(cost, (int, float)) and abs(cost - ref) <= TOL
        interp_ok = all(
            g.get("directive_type") == w["directive_type"]
            and (g.get("structured_adjustment") or {}).get("hours")
            == (w["structured_adjustment"] or {}).get("hours")
            for g, w in zip(body.get("directive_interpretation", []), expected["directive_interpretation"])
        ) and len(body.get("directive_interpretation", [])) == len(expected["directive_interpretation"])

        if not (valid and cost_ok and interp_ok):
            failures.append(f"{case['id']} valid={valid} interp={interp_ok} cost={cost_ok}")
        else:
            passed += 1

        def tick(flag):
            return f"{GREEN}pass{RESET}" if flag else f"{RED}FAIL{RESET}"

        print(f"{case['id']:<11}{status:>6}{tick(valid):>17}{tick(interp_ok):>18}"
              f"{cost:>12,.0f}{tick(cost_ok):>18}{ms:>8.0f}  {interpreter}")
        for problem in problems[:3]:
            print(f"    {RED}- {problem}{RESET}")

    print("-" * len(head))
    latencies.sort()
    p95 = latencies[int(len(latencies) * 0.95) - 1]
    band = ("3/3" if p95 <= 5000 else "2/3" if p95 <= 15000 else "1/3" if p95 <= 30000 else "0/3")
    colour = GREEN if p95 <= 5000 else YELLOW if p95 <= 15000 else RED
    print(f"{passed}/10 fully correct   p95 {colour}{p95:.0f} ms{RESET} "
          f"(latency score {band})   max {max(latencies):.0f} ms")

    # 3. the check that silently fails
    print()
    llm_live = any(i.startswith("llm:") or i.startswith("cache") for i in interpreters)
    if llm_live and not any("fallback" in i for i in interpreters):
        print(f"{GREEN}ok  {RESET} LLM is in the interpretation path: {', '.join(sorted(interpreters))}")
    else:
        print(f"{RED}FAIL{RESET} LLM NOT in the path — saw: {', '.join(sorted(interpreters))}")
        print(f"      {YELLOW}The keyword fallback scores 10/10 on public cases, so this looks fine{RESET}")
        print(f"      {YELLOW}but fails the mandatory-LLM requirement. Check OPENAI_API_KEY and{RESET}")
        print(f"      {YELLOW}whether the host allows outbound HTTPS to api.openai.com.{RESET}")
        failures.append("LLM not in interpretation path")

    # 4. malformed input must not 5xx
    print()
    bad_ok = True
    for label, payload in (("malformed json", None), ("empty object", {}), ("wrong shape", {"scenario_id": 5})):
        if payload is None:
            req = urllib.request.Request(f"{base}/optimize-energy", data=b'{"bad',
                                         headers={"Content-Type": "application/json"}, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=20) as resp:
                    status = resp.status
            except urllib.error.HTTPError as exc:
                status = exc.code
            except Exception:  # noqa: BLE001
                status = None
        else:
            status, _, _, _ = request(f"{base}/optimize-energy", payload)
        good = status is not None and 400 <= status < 500
        bad_ok &= good
        print(f"{(GREEN + 'ok  ' if good else RED + 'FAIL') + RESET} {label:<16} -> {status}")
    if not bad_ok:
        failures.append("malformed input handling")

    print(f"\n{'=' * 72}")
    if failures:
        print(f"{RED}NOT READY — {len(failures)} problem(s):{RESET}")
        for item in failures:
            print(f"  - {item}")
        print()
        return 1
    print(f"{GREEN}DEPLOYMENT READY — 10/10 correct, LLM in path, p95 {p95:.0f} ms, no 5xx.{RESET}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
