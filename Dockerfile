FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app
COPY . /app
RUN python -m pip install --upgrade pip \
    && python -m pip install "uv==0.9.28" \
    && python -m uv sync --frozen --no-dev --extra space

RUN useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 7860
CMD ["/app/.venv/bin/python", "app.py"]
