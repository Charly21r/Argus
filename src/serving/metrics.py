from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter(
    name="moderation_requests", documentation="Total HTTP requests", labelnames=["endpoint", "status_code"]
)

REQUEST_LATENCY = Histogram(
    name="moderation_request_duration_seconds", documentation="End-to-end HTTP request latency", labelnames=["endpoint"]
)

INFERENCE_LATENCY = Histogram(
    name="moderation_inference_request_duration_seconds",
    documentation="Model inference request latency (excludes HTTP overhead)",
)

PREDICTION_DISTRIBUTION = Counter(
    "moderation_predictions_total",
    "Prediction outcomes",
    ["label"],  # values: "toxic", "hate", "safe"
)

MODEL_CONFIDENCE = Histogram(
    "moderation_confidence_score",
    "Raw model probability scores",
    ["label"],  # "toxicity", "hate"
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99],
)
