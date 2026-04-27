# Build Stage
FROM python:3.12-slim AS builder

WORKDIR /app

COPY ./requirements ./requirements

RUN pip install -r requirements/base.txt && pip install -r requirements/serving.txt

COPY ./src ./src
COPY ./config/ ./config


# Runtime stage
FROM python:3.12-slim

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY ./src ./src
COPY ./config/ ./config

RUN useradd --create-home appuser
RUN chown -R appuser:appuser /app
USER appuser

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/v1/health')"

EXPOSE 8000
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]