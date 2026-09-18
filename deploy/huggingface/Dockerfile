# Fallback deployment: Hugging Face Spaces.
#
# This pulls the exact image CI already built, smoke-tested and published, so the
# fallback and the primary deployment are the same artifact — there is no second
# build to keep in sync.
#
# Requires the GHCR package to be PUBLIC (which the rubric needs anyway).
FROM ghcr.io/tanviranjumapurbo/bup_hackathon:latest

# Spaces route traffic to 7860 by default; app_port in README.md pins it anyway.
ENV PORT=7860
EXPOSE 7860
