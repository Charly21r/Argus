import json
import logging
from pathlib import Path

import torch
from src.config import get_settings
from src.serving.schemas import LabelResult
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

# Module-level singleton
_tokenizer: PreTrainedTokenizerBase | None = None
_model: PreTrainedModel | None = None
_model_path: Path | None = None
_device: str = "cpu"
_thresholds: dict = {}
_label_cols = ["toxicity", "hate"]


def _resolve_paths(uri: str) -> tuple[Path, Path]:
    """Return (model_path, tokenizer_path) for a local path or MLflow registry URI.

    MLflow's transformers flavor splits artifacts into:
      <root>/model/           — PyTorch weights + config.json
      <root>/components/tokenizer/ — tokenizer files

    For local paths the tokenizer lives alongside the weights in the same dir.
    """
    if uri.startswith("models:/"):
        import mlflow

        logger.info("Downloading model from MLflow registry: %s", uri)
        root = Path(mlflow.artifacts.download_artifacts(uri))
    else:
        root = Path(uri)
        if not root.exists():
            raise FileNotFoundError(
                f"Local model path '{root}' does not exist. "
                "Either restore the models/ directory or set CMS_SERVING__MODEL_URI to a registry URI."
            )

    # Flat ONNX layout: model.onnx lives directly in root (produced by export_onnx.py).
    # When downloaded from MLflow registry, sidecar files (tokenizer, thresholds) are
    # in an extra_files/ subdirectory created by mlflow.onnx.log_model(extra_files=...).
    if (root / "model.onnx").exists():
        extra = root / "extra_files"
        tokenizer_path = extra if extra.is_dir() else root
        return root, tokenizer_path

    # MLflow transformers flavor: model/ subdir + components/tokenizer/ subdir
    model_nested = root / "model"
    model_path = model_nested if (model_nested / "config.json").exists() else root

    tokenizer_nested = root / "components" / "tokenizer"
    tokenizer_path = tokenizer_nested if tokenizer_nested.exists() else model_path

    return model_path, tokenizer_path


def _load_thresholds(uri: str, local_thresholds_path: Path, label_cols: list[str]) -> dict:
    """Load thresholds from MLflow model metadata, local file, or 0.5 fallback (in that order)."""
    if uri.startswith("models:/"):
        try:
            import mlflow

            metadata = mlflow.models.get_model_info(uri).metadata or {}
            thresholds = metadata.get("thresholds", {})
            if thresholds:
                logger.info("Loaded thresholds from MLflow model metadata: %s", thresholds)
                return {label: float(thresholds[label]) for label in label_cols if label in thresholds}
        except Exception as e:
            logger.warning("Could not read thresholds from MLflow metadata: %s", e)

    if local_thresholds_path.exists():
        with local_thresholds_path.open() as f:
            thresholds = json.load(f)
        logger.info("Loaded thresholds from %s: %s", local_thresholds_path, thresholds)
        return dict(thresholds)

    logger.warning("No thresholds found — using 0.5 for all labels.")
    return {label: 0.5 for label in label_cols}


def _get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    logger.warning("No GPU found — loading model on CPU.")
    return "cpu"


def load_model() -> None:
    """Load tokenizer and model. Call this from the FastAPI lifespan."""
    global _tokenizer, _model, _model_path, _device, _thresholds, _label_cols
    _settings = get_settings()
    backend = _settings.serving.backend

    # Determine source URI: explicit override > config default
    raw_uri = _settings.serving.model_uri or str(_settings.paths.model_dir)

    if backend == "pt":
        model_path, tokenizer_path = _resolve_paths(raw_uri)
        _model_path = model_path
        device = _get_device()
        _model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
        _model.eval()
        _device = device
        thresholds_path = model_path.parent / "thresholds.json"

    elif backend == "onnx":
        from optimum.onnxruntime import ORTModelForSequenceClassification

        model_path, tokenizer_path = _resolve_paths(raw_uri)
        _model_path = model_path
        provider = _settings.serving.onnx_provider
        _model = ORTModelForSequenceClassification.from_pretrained(model_path, provider=provider)
        _device = "cuda" if provider == "CUDAExecutionProvider" else "cpu"
        # Registry downloads place sidecar files in extra_files/; local exports keep them flat
        extra = model_path / "extra_files"
        thresholds_path = (extra / "thresholds.json") if extra.is_dir() else (model_path / "thresholds.json")

    else:
        raise ValueError(f"Unknown backend {backend!r}. Set CMS_SERVING__BACKEND to 'pt' or 'onnx'.")

    _tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    _thresholds = _load_thresholds(raw_uri, thresholds_path, _label_cols)

    logger.info("Model loaded from %s on %s | thresholds: %s", _model_path, _device, _thresholds)


def is_loaded() -> bool:
    """Return True if the model is ready to serve."""
    return _model is not None and _tokenizer is not None


def predict(text: str) -> tuple[LabelResult, LabelResult]:
    """Run inference on a single text. Returns (toxicity, hate) LabelResults."""
    if not is_loaded():
        raise RuntimeError("Model is not loaded")

    assert _tokenizer is not None
    assert _model is not None

    inputs = _tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    inputs = {k: v.to(_device) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = _model(**inputs)
        probs = torch.sigmoid(outputs.logits)

    toxicity = LabelResult(prob=probs[0][0].item(), flagged=(probs[0][0] > _thresholds["toxicity"]).item())
    hate = LabelResult(prob=probs[0][1].item(), flagged=(probs[0][1] > _thresholds["hate"]).item())
    return toxicity, hate


def predict_batch(texts: list[str]) -> list[tuple[LabelResult, LabelResult]]:
    """Run inference on multiple texts in a single forward pass."""
    if not is_loaded():
        raise RuntimeError("Model is not loaded")

    assert _tokenizer is not None
    assert _model is not None

    inputs = _tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    inputs = {k: v.to(_device) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = _model(**inputs)
        probs = torch.sigmoid(outputs.logits)

    results = []
    for row in probs:
        toxicity = LabelResult(prob=row[0].item(), flagged=(row[0] > _thresholds["toxicity"]).item())
        hate = LabelResult(prob=row[1].item(), flagged=(row[1] > _thresholds["hate"]).item())
        results.append((toxicity, hate))
    return results


def get_model_info() -> dict:
    """Return metadata for the /v1/model/info endpoint."""
    return {
        "model_path": str(_model_path),
        "device": _device,
        "is_loaded": is_loaded(),
        "thresholds": _thresholds,
        "label_cols": _label_cols,
    }
