# Multi-stage build to keep the final image lean - build deps (pip cache,
# compilers pulled in transitively) don't end up in the runtime layer.
FROM python:3.11-slim AS builder

WORKDIR /build
RUN pip install --no-cache-dir --target=/deps torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir --target=/deps -r requirements.txt

FROM python:3.11-slim

RUN useradd --create-home --shell /bin/bash appuser
WORKDIR /app

COPY --from=builder /deps /usr/local/lib/python3.11/site-packages
COPY api/ ./api/

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface

# Bake all 11 language models into the image at build time so a fresh
# container never needs network access on first request (matches the
# "never reload per request, load once" requirement, extended to "load
# once at build time" for reproducible cold starts).
RUN python -c "\
from transformers import VitsModel, AutoTokenizer; \
langs = ['tam','hin','tel','mal','kan','mar','guj','ben','pan','ory','asm']; \
[(VitsModel.from_pretrained(f'facebook/mms-tts-{l}'), AutoTokenizer.from_pretrained(f'facebook/mms-tts-{l}')) for l in langs]"

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8503

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8503/health', timeout=3)" || exit 1

WORKDIR /app/api
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8503"]
