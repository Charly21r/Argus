from contextlib import asynccontextmanager
from time import perf_counter

from fastapi import FastAPI
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from src.serving.metrics import (
    INFERENCE_LATENCY,
    MODEL_CONFIDENCE,
    PREDICTION_DISTRIBUTION,
    REQUEST_COUNT,
    REQUEST_LATENCY,
)
from src.serving.model_manager import get_model_info, is_loaded, load_model, predict, predict_batch
from src.serving.schemas import BatchTextRequest, BatchTextResponse, ModerationResult, TextInput
from src.serving.tracing import setup_tracing, tracer


@asynccontextmanager
async def lifespan(app):
    setup_tracing()
    load_model()
    yield


app = FastAPI(
    title="Argus",
    description="Multimodal content moderation - toxicity and hate-speech detection",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/v1/health")
def health():
    return {"is_loaded": is_loaded()}


@app.get("/v1/model/info")
def model_info():
    return get_model_info()


@app.post("/v1/moderate/text")
def moderate(text: TextInput) -> ModerationResult:
    status_code = "500"
    try:
        with tracer.start_as_current_span("moderate_text") as span:
            span.set_attribute("input.length", len(text.content))

            with REQUEST_LATENCY.labels(endpoint="/v1/moderate/text").time():
                with tracer.start_as_current_span("model_inference"):
                    start = perf_counter()
                    result = predict(text.content)
                    processing_time_ms = perf_counter() - start
                    INFERENCE_LATENCY.observe(processing_time_ms)

                safe = not (result[0].flagged or result[1].flagged)

                with tracer.start_as_current_span("post_processing"):
                    if result[0].flagged:
                        PREDICTION_DISTRIBUTION.labels(label="toxic").inc()
                    if result[1].flagged:
                        PREDICTION_DISTRIBUTION.labels(label="hate").inc()
                    if safe:
                        PREDICTION_DISTRIBUTION.labels(label="safe").inc()

                    MODEL_CONFIDENCE.labels(label="toxicity").observe(result[0].prob)
                    MODEL_CONFIDENCE.labels(label="hate").observe(result[1].prob)

            span.set_attribute("result.safe", safe)
            span.set_attribute("result.toxicity_score", result[0].prob)
            span.set_attribute("result.hate_score", result[1].prob)

        status_code = "200"
        return ModerationResult(
            id=text.id,
            text=text.content,
            toxicity=result[0],
            hate=result[1],
            safe=safe,
            processing_time_ms=processing_time_ms,
        )
    finally:
        REQUEST_COUNT.labels(endpoint="/v1/moderate/text", status_code=status_code).inc()


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/v1/moderate/text/batch")
def moderate_batch(batch: BatchTextRequest) -> BatchTextResponse:
    status_code = "500"
    try:
        with tracer.start_as_current_span("moderate_text_batch") as span:
            span.set_attribute("batch.size", len(batch.items))

            with REQUEST_LATENCY.labels(endpoint="/v1/moderate/text/batch").time():
                with tracer.start_as_current_span("model_inference"):
                    start = perf_counter()
                    results = predict_batch([item.content for item in batch.items])
                    total_processing_time_ms = perf_counter() - start
                    INFERENCE_LATENCY.observe(total_processing_time_ms)

                with tracer.start_as_current_span("post_processing"):
                    moderation_results = []
                    for item, (toxicity, hate) in zip(batch.items, results, strict=True):
                        safe = not (toxicity.flagged or hate.flagged)

                        if toxicity.flagged:
                            PREDICTION_DISTRIBUTION.labels(label="toxic").inc()
                        if hate.flagged:
                            PREDICTION_DISTRIBUTION.labels(label="hate").inc()
                        if safe:
                            PREDICTION_DISTRIBUTION.labels(label="safe").inc()

                        MODEL_CONFIDENCE.labels(label="toxicity").observe(toxicity.prob)
                        MODEL_CONFIDENCE.labels(label="hate").observe(hate.prob)

                        moderation_results.append(
                            ModerationResult(
                                id=item.id,
                                text=item.content,
                                toxicity=toxicity,
                                hate=hate,
                                safe=safe,
                                processing_time_ms=total_processing_time_ms / len(batch.items),
                            )
                        )

        status_code = "200"
        return BatchTextResponse(
            items=moderation_results,
            total_processing_time_ms=total_processing_time_ms,
        )
    finally:
        REQUEST_COUNT.labels(endpoint="/v1/moderate/text/batch", status_code=status_code).inc()
