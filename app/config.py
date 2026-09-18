"""Environment-driven configuration.

Every credential arrives as an environment variable. `.env` is a local developer
convenience only and is gitignored; on a deployment host the platform supplies
the variables directly and no `.env` file exists in the image.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


def load_dotenv(path: str = ".env") -> None:
    """Populate os.environ from a local .env, without ever overriding a real one.

    setdefault matters: on a deployed host the platform's variables must win.
    """
    candidate = Path(path)
    if not candidate.exists():
        return
    for raw in candidate.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    kind: str  # "openai" for any OpenAI-compatible API, or "gemini"
    base_url: str
    api_key: str
    model: str

    def redacted(self) -> str:
        """Safe for logs: proves a key is present without revealing it."""
        return f"{self.name}:{self.model} (key {'set' if self.api_key else 'MISSING'})"


_PLACEHOLDER_MARKERS = ("paste", "your-key", "your_key", "xxx", "changeme", "<", "...")


def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def _secret(name: str) -> str:
    """Read a credential, treating an unfilled placeholder as absent.

    A host that injects an unset variable as literal placeholder text would
    otherwise send us into a 401 loop instead of failing over to the next
    provider.
    """
    value = _env(name)
    lowered = value.lower()
    if any(marker in lowered for marker in _PLACEHOLDER_MARKERS):
        return ""
    return value


def configured_providers() -> List[ProviderConfig]:
    """Providers that actually have a key, ordered with LLM_PROVIDER first.

    The order is the failover cascade: we race/try them in this sequence.
    """
    catalogue = {
        "openai": ProviderConfig(
            name="openai",
            kind="openai",
            base_url=_env("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            api_key=_secret("OPENAI_API_KEY"),
            model=_env("LLM_MODEL", "gpt-5.4-mini"),
        ),
        "groq": ProviderConfig(
            name="groq",
            kind="openai",
            base_url=_env("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
            api_key=_secret("GROQ_API_KEY"),
            model=_env("GROQ_MODEL", "openai/gpt-oss-120b"),
        ),
        "gemini": ProviderConfig(
            name="gemini",
            kind="gemini",
            base_url=_env(
                "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta"
            ),
            api_key=_secret("GEMINI_API_KEY"),
            model=_env("GEMINI_MODEL", "gemini-flash-lite-latest"),
        ),
    }

    preferred = _env("LLM_PROVIDER", "openai").lower()
    order = [preferred] + [n for n in catalogue if n != preferred]
    return [catalogue[n] for n in order if n in catalogue and catalogue[n].api_key]


def llm_total_budget_seconds() -> float:
    """Wall-clock ceiling for the whole interpretation phase, across every
    provider and retry.

    Without this the worst case is providers x attempts x per-call timeout, which
    with three providers configured would be 60s — double the 30s per-request
    hard ceiling the judge enforces. The budget leaves ample headroom for the
    optimizer (about 2 ms) and serialization.
    """
    try:
        return float(_env("LLM_TOTAL_BUDGET_SECONDS", "20"))
    except ValueError:
        return 20.0


def llm_timeout_seconds() -> float:
    try:
        return float(_env("LLM_TIMEOUT_SECONDS", "8"))
    except ValueError:
        return 8.0


def port() -> int:
    try:
        return int(_env("PORT", "8000"))
    except ValueError:
        return 8000


def first_provider() -> Optional[ProviderConfig]:
    providers = configured_providers()
    return providers[0] if providers else None
