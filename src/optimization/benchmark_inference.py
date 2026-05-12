import logging
import time
from pathlib import Path

import numpy as np
import torch
from src.config import get_settings
from src.utils.loading import load_model_and_tokenizer

logger = logging.getLogger(__name__)

_settings = get_settings()

ROOT = Path(__file__).resolve().parents[2]

ONNX_DIR = ROOT / _settings.optimization.onnx_dir
PT_MODEL_DIR = ROOT / _settings.paths.model_dir / "model"
QUANTIZED_DIR = ROOT / _settings.optimization.quantized_dir
LABEL_COLS = _settings.model.label_cols
MAX_LENGTH = _settings.model.max_length


def benchmark_model(model, tokenizer, device: torch.device, n_iter: int = 100) -> dict[str, float]:
    from optimum.onnxruntime import ORTModel

    is_ort = isinstance(model, ORTModel)

    sample = tokenizer("This is a warmup input.", return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
    inputs = {
        k: (v if is_ort else v.to(device)) for k, v in sample.items() if k in set(model.forward.__code__.co_varnames)
    }

    # Warm-up for 10 iterations
    for _ in range(10):
        with torch.no_grad():
            model(**inputs)

    # Measure
    latencies = []
    for _ in range(n_iter):
        start = time.perf_counter()
        with torch.no_grad():
            model(**inputs)
        latencies.append(time.perf_counter() - start)

    # Compute percentiles
    prc = np.array(latencies)
    p50 = np.percentile(prc, 50)
    p95 = np.percentile(prc, 95)
    p99 = np.percentile(prc, 99)

    return {"p50": p50, "p95": p95, "p99": p99}


def main():
    # Force CPU for all variants — ORT always runs on CPU
    device = torch.device("cpu")

    pt_model, pt_tokenizer = load_model_and_tokenizer(PT_MODEL_DIR, device)
    onnx_model, onnx_tokenizer = load_model_and_tokenizer(ONNX_DIR, device)
    quant_model, quant_tokenizer = load_model_and_tokenizer(QUANTIZED_DIR, device)

    results = {
        "pytorch_fp32": benchmark_model(pt_model, pt_tokenizer, device),
        "onnx_fp32": benchmark_model(onnx_model, onnx_tokenizer, device),
        "onnx_int8": benchmark_model(quant_model, quant_tokenizer, device),
    }

    pt_p50 = results["pytorch_fp32"]["p50"]
    for _name, metrics in results.items():
        metrics["speedup_vs_pytorch"] = round(pt_p50 / metrics["p50"], 2)

    # Log table
    header = f"{'Model':<20} {'p50 (ms)':>10} {'p95 (ms)':>10} {'p99 (ms)':>10} {'speedup':>10}"
    logger.info(header)
    logger.info("-" * len(header))
    for name, m in results.items():
        logger.info(
            "%-20s %10.1f %10.1f %10.1f %10.2fx",
            name,
            m["p50"] * 1000,
            m["p95"] * 1000,
            m["p99"] * 1000,
            m["speedup_vs_pytorch"],
        )

    # Write report
    report_path = ROOT / "reports" / "benchmark_inference.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    import json

    report_path.write_text(json.dumps({"device": str(device), "results": results}, indent=2))
    logger.info("Report written to %s", report_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
