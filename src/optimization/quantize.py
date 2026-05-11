"""Quantize the ONNX FP32 model to INT8 and compare bias metrics before/after.

Usage:
    python -m src.optimization.quantize

Requires export_onnx.py to have been run first (produces the FP32 ONNX model).
"""

import json
import logging
import platform
import shutil
from pathlib import Path

from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from scripts.run_bias_eval import run_bias_eval
from src.config import get_settings

logger = logging.getLogger(__name__)

_settings = get_settings()

ROOT = Path(__file__).resolve().parents[2]

ONNX_DIR = ROOT / _settings.optimization.onnx_dir
QUANTIZED_DIR = ROOT / _settings.optimization.quantized_dir
LABEL_COLS = _settings.model.label_cols


def _quantization_config():
    """Pick a quantization config based on CPU architecture."""
    arch = platform.machine().lower()
    if "arm" in arch or "aarch64" in arch:
        return AutoQuantizationConfig.arm64(is_static=False, per_channel=False)
    return AutoQuantizationConfig.avx2(is_static=False, per_channel=False)


def quantize(onnx_dir: Path, quantized_dir: Path) -> Path:
    """Apply dynamic INT8 quantization to the ONNX model at onnx_dir."""
    if not (onnx_dir / "model.onnx").exists():
        raise FileNotFoundError(f"No model.onnx found at {onnx_dir}. Run export_onnx.py first.")

    onnx_model = ORTModelForSequenceClassification.from_pretrained(onnx_dir)
    quantizer = ORTQuantizer.from_pretrained(onnx_model)

    qconfig = _quantization_config()
    logger.info("Quantizing with config: %s", qconfig)

    quantizer.quantize(save_dir=quantized_dir, quantization_config=qconfig)

    # Copy tokenizer and thresholds so the quantized dir is self-contained
    for fname in ["tokenizer.json", "tokenizer_config.json", "vocab.txt", "special_tokens_map.json", "thresholds.json"]:
        src = onnx_dir / fname
        if src.exists():
            shutil.copy2(src, quantized_dir / fname)

    logger.info("Quantized model saved to %s", quantized_dir)
    return quantized_dir


def compare_reports(fp32_report: dict, int8_report: dict, output_path: Path) -> None:
    """Write a side-by-side comparison of FP32 vs INT8 bias metrics."""
    rows = {}
    for label in LABEL_COLS:
        key = f"val_{label}_f1"
        rows[key] = {
            "fp32": fp32_report.get("val", {}).get(key),
            "int8": int8_report.get("val", {}).get(key),
        }
        fpr_key = f"val_lex_group_{label}_FPR_delta"
        rows[fpr_key] = {
            "fp32": fp32_report.get("val", {}).get(fpr_key),
            "int8": int8_report.get("val", {}).get(fpr_key),
        }

    comparison = {
        "summary": rows,
        "fp32_full": fp32_report,
        "int8_full": int8_report,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(comparison, indent=2))
    logger.info("Comparison report written to %s", output_path)

    logger.info("--- FP32 vs INT8 comparison ---")
    for metric, values in rows.items():
        logger.info("  %-45s  fp32=%.4f  int8=%.4f", metric, values["fp32"] or 0, values["int8"] or 0)


def main():
    QUANTIZED_DIR.mkdir(parents=True, exist_ok=True)

    quantize(ONNX_DIR, QUANTIZED_DIR)

    logger.info("Running bias eval on FP32 ONNX model...")
    fp32_report = run_bias_eval(
        model_path=ONNX_DIR,
        thresholds_path=ONNX_DIR / "thresholds.json",
        report_path=ONNX_DIR / "bias_report.json",
    )

    logger.info("Running bias eval on INT8 quantized model...")
    int8_report = run_bias_eval(
        model_path=QUANTIZED_DIR,
        thresholds_path=QUANTIZED_DIR / "thresholds.json",
        report_path=QUANTIZED_DIR / "bias_report.json",
    )

    compare_reports(
        fp32_report=fp32_report,
        int8_report=int8_report,
        output_path=ROOT / "reports" / "quantization_fairness_comparison.json",
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
