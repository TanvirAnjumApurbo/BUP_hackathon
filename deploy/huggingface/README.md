---
title: GridWise Energy Optimizer
emoji: ⚡
colorFrom: yellow
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# GridWise — fallback deployment

Backup endpoint for the BUP CSE Fest 2026 preliminary. Runs the exact container
image published by CI at `ghcr.io/tanviranjumapurbo/bup_hackathon`, so this
fallback and the primary deployment are the same artifact.

- `GET /health` → `{"status":"ok"}`
- `POST /optimize-energy` → directive interpretation + 24-hour schedule

Source and full documentation: https://github.com/TanvirAnjumApurbo/BUP_hackathon

## Required secrets

Set these in **Settings → Variables and secrets** (as *secrets*, not variables):

| Name | Value |
|---|---|
| `OPENAI_API_KEY` | your key |
| `LLM_PROVIDER` | `openai` |
| `LLM_MODEL` | `gpt-4o-mini` |

Without `OPENAI_API_KEY` the service still answers, but degrades to the keyword
fallback interpreter — check the `X-Interpreter` response header to confirm which
path is live.
