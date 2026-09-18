FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

WORKDIR /srv

# Dependencies first so code edits do not invalidate the pip layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Non-root. No secrets are baked in — every key arrives as an environment
# variable at run time.
# uid 1000: some hosts (Hugging Face Spaces among them) require the container to
# run as uid 1000, and every other platform is happy with it.
RUN useradd --create-home --uid 1000 gridwise
USER gridwise

EXPOSE 8000

# Honour $PORT so the same image runs on Koyeb, Hugging Face Spaces (7860),
# Railway, or locally without a rebuild.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
