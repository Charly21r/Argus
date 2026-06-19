"""Export the registered content-moderation model to ONNX and validate numerical equivalence.

Usage:
    python -m src.optimization.export_onnx

Environment variables (optional):
    MLFLOW_TRACKING_URI            — MLflow tracking server ("databricks" or "file:./mlruns")
    DATABRICKS_HOST                — required when MLFLOW_TRACKING_URI is "databricks"
    DATABRICKS_TOKEN               — required when MLFLOW_TRACKING_URI is "databricks"
    CMS_OPTIMIZATION__MODEL_ALIAS  — override registry alias (default: Production)
    MODEL_VERSION                  — export a specific version instead of the alias
"""

import json
import logging
import os
from pathlib import Path

import mlflow
import numpy as np
import torch
from optimum.onnxruntime import ORTModelForSequenceClassification
from src.config import get_settings
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)

_settings = get_settings()

ROOT = Path(__file__).resolve().parents[2]

REGISTERED_MODEL_NAME = _settings.optimization.registered_model_name
ONNX_REGISTERED_MODEL_NAME = _settings.optimization.onnx_registered_model_name
MODEL_ALIAS = _settings.optimization.model_alias
ONNX_DIR = ROOT / _settings.optimization.onnx_dir
VALIDATION_TOLERANCE = _settings.optimization.validation_tolerance
MAX_LENGTH = _settings.model.max_length
LABEL_COLS = _settings.model.label_cols


def _find_dir_containing(root: Path, filename: str) -> Path:
    """Recursively find the first directory under root that contains `filename`."""
    for path in root.rglob(filename):
        return path.parent
    raise FileNotFoundError(f"Could not find {filename} under {root}")


def download_model(version: str | None = None) -> tuple[Path, dict]:
    """Download the registered model, materialize it as a flat HuggingFace dir, return (hf_dir, thresholds)."""
    client = mlflow.tracking.MlflowClient()

    if version:
        mv = client.get_model_version(REGISTERED_MODEL_NAME, version)
    else:
        mv = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, MODEL_ALIAS)

    logger.info("Exporting %s v%s (run %s)", REGISTERED_MODEL_NAME, mv.version, (mv.run_id or "")[:8])

    model_uri = f"models:/{REGISTERED_MODEL_NAME}/{mv.version}"

    thresholds: dict = {}
    try:
        model_info = mlflow.models.get_model_info(model_uri)
        thresholds = (model_info.metadata or {}).get("thresholds", {})
    except Exception as e:
        logger.warning("Could not read model metadata: %s", e)

    artifact_path = Path(mlflow.artifacts.download_artifacts(model_uri))
    logger.info("Downloaded artifacts to %s", artifact_path)

    # mlflow.transformers stores files in nested subdirs (model/, components/tokenizer/).
    # Locate them by looking for the canonical HuggingFace marker files,
    # load with the standard Auto classes, then re-save as a flat HF directory.
    model_subdir = _find_dir_containing(artifact_path, "config.json")
    tokenizer_subdir = _find_dir_containing(artifact_path, "tokenizer_config.json")
    logger.info("Found model in %s, tokenizer in %s", model_subdir, tokenizer_subdir)

    model = AutoModelForSequenceClassification.from_pretrained(model_subdir)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_subdir, use_fast=True)

    hf_dir = ONNX_DIR.parent / "hf_export"
    hf_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(hf_dir)
    tokenizer.save_pretrained(hf_dir)
    logger.info("Materialized flat HuggingFace artifacts at %s", hf_dir)

    return hf_dir, thresholds


def export_to_onnx(local_path: Path, onnx_dir: Path) -> None:
    """Convert the HuggingFace model at local_path to ONNX at onnx_dir."""
    logger.info("Exporting to ONNX at %s ...", onnx_dir)
    ort_model = ORTModelForSequenceClassification.from_pretrained(local_path, export=True)
    tokenizer = AutoTokenizer.from_pretrained(local_path)

    ort_model.save_pretrained(onnx_dir)
    tokenizer.save_pretrained(onnx_dir)
    logger.info("ONNX model saved")


def validate(local_path: Path, onnx_dir: Path) -> float:
    """Compare PyTorch and ONNX logits. Raises if max abs diff > tolerance."""
    logger.info("Validating numerical equivalence...")

    tokenizer = AutoTokenizer.from_pretrained(local_path)

    pt_model = AutoModelForSequenceClassification.from_pretrained(local_path)
    pt_model.eval()

    # Only pass keys the model's forward signature accepts (DistilBERT skips token_type_ids).
    forward_keys = set(pt_model.forward.__code__.co_varnames)
    raw_inputs = tokenizer(
        "This is a test input for numerical validation.",
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )
    inputs = {k: v for k, v in raw_inputs.items() if k in forward_keys}

    with torch.no_grad():
        pt_logits = pt_model(**inputs).logits.numpy()

    ort_model = ORTModelForSequenceClassification.from_pretrained(onnx_dir)
    ort_logits = ort_model(**inputs).logits.detach().numpy()

    max_diff = float(np.max(np.abs(pt_logits - ort_logits)))
    if max_diff > VALIDATION_TOLERANCE:
        raise ValueError(
            f"Numerical mismatch: max abs diff={max_diff:.2e} exceeds tolerance {VALIDATION_TOLERANCE:.2e}"
        )

    logger.info("Validation passed — max abs diff: %.2e (tolerance %.2e)", max_diff, VALIDATION_TOLERANCE)
    for i, label in enumerate(LABEL_COLS):
        logger.info("  %s — PT: %.4f  ORT: %.4f", label, pt_logits[0][i], ort_logits[0][i])

    return max_diff


def save_thresholds(thresholds: dict, onnx_dir: Path) -> None:
    if not thresholds:
        logger.warning("No thresholds found in model metadata — skipping thresholds.json")
        return
    thresholds_path = onnx_dir / "thresholds.json"
    with thresholds_path.open("w") as f:
        json.dump(thresholds, f, indent=2)
    logger.info("Thresholds saved to %s", thresholds_path)


def register_onnx_model(onnx_dir: Path, source_version: str) -> None:
    """Log the ONNX model + tokenizer + thresholds to MLflow and register as a new version."""
    import numpy as np
    import onnx as onnx_lib
    from mlflow.models import ModelSignature
    from mlflow.types import Schema, TensorSpec

    onnx_model = onnx_lib.load(str(onnx_dir / "model.onnx"))
    run_name = f"onnx-export-v{source_version}"
    artifact_path = "model"

    # Unity Catalog requires a signature — describe the tokenizer output / logits shape
    signature = ModelSignature(
        inputs=Schema(
            [
                TensorSpec(np.dtype("int64"), (-1, -1), name="input_ids"),
                TensorSpec(np.dtype("int64"), (-1, -1), name="attention_mask"),
            ]
        ),
        outputs=Schema(
            [
                TensorSpec(np.dtype("float32"), (-1, len(LABEL_COLS)), name="logits"),
            ]
        ),
    )

    # Tokenizer + threshold files to bundle alongside the ONNX weights
    sidecar_names = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "thresholds.json",
    ]
    extra_files = [str(onnx_dir / n) for n in sidecar_names if (onnx_dir / n).exists()]

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("source_pt_version", source_version)
        logged = mlflow.onnx.log_model(
            onnx_model,
            name=artifact_path,
            signature=signature,
            extra_files=extra_files,
        )

    mv = mlflow.register_model(logged.model_uri, ONNX_REGISTERED_MODEL_NAME)
    logger.info(
        "Registered ONNX model as '%s' version %s",
        ONNX_REGISTERED_MODEL_NAME,
        mv.version,
    )


def main() -> None:
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    logger.info("MLflow tracking URI: %s", tracking_uri)

    ONNX_DIR.mkdir(parents=True, exist_ok=True)

    version = os.environ.get("MODEL_VERSION", None)
    local_path, thresholds = download_model(version)

    export_to_onnx(local_path, ONNX_DIR)
    max_diff = validate(local_path, ONNX_DIR)
    save_thresholds(thresholds, ONNX_DIR)

    # Get the source version string for the run name
    client = mlflow.tracking.MlflowClient()
    if version:
        mv = client.get_model_version(REGISTERED_MODEL_NAME, version)
    else:
        mv = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, MODEL_ALIAS)

    logger.info("Registering ONNX model in MLflow (max_diff=%.2e)...", max_diff)
    register_onnx_model(ONNX_DIR, mv.version)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
