import json
from pathlib import Path

import torch
from src.config import get_settings
from src.serving.schemas import LabelResult
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

# Module-level singleton
_tokenizer: PreTrainedTokenizerBase | None = None
_model: PreTrainedModel | None = None
_model_path: Path | None = None
_device: str = "cpu"
_thresholds: dict = {}
_label_cols = ["toxicity", "hate"]


def load_model() -> None:
    """Load tokenizer and model from local path. Call this from the FastAPI lifespan."""
    global _tokenizer, _model, _model_path, _device, _thresholds, _label_cols
    _settings = get_settings()
    backend = _settings.serving.backend

    if backend == "pt":
        _model_path = _settings.paths.model_dir

        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            print("Warning! The model and tokenizer are being loaded on the CPU.")
            device = "cpu"

        _model = AutoModelForSequenceClassification.from_pretrained(_model_path).to(device)
        _model.eval()
        _device = device
        thresholds_path = Path(_model_path).parent / "thresholds.json"

    elif backend == "onnx":
        from optimum.onnxruntime import ORTModelForSequenceClassification

        _model_path = _settings.optimization.onnx_dir
        provider = _settings.serving.onnx_provider
        _model = ORTModelForSequenceClassification.from_pretrained(_model_path, provider=provider)
        _device = "cuda" if provider == "CUDAExecutionProvider" else "cpu"
        thresholds_path = Path(_model_path) / "thresholds.json"

    else:
        raise ValueError(f"Unknown backend {backend!r}. Set CMS_SERVING__BACKEND to 'pt' or 'onnx'.")

    _tokenizer = AutoTokenizer.from_pretrained(_model_path)

    # Load thresholds per label
    with open(thresholds_path) as f:
        _thresholds = json.load(f)


def is_loaded() -> bool:
    """Return True if the model is ready to serve."""
    return _model is not None and _tokenizer is not None


def predict(text: str) -> tuple[LabelResult, LabelResult]:
    """Run inference on a single text. Returns (toxicity, hate) LabelResults."""
    if not is_loaded():
        raise RuntimeError("Model is not loaded")

    assert _tokenizer is not None
    assert _model is not None

    inputs = _tokenizer(
        text,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    inputs = {k: v.to(_device) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = _model(**inputs)
        probs = torch.sigmoid(outputs.logits)

    flag_toxicity = probs[0][0] > _thresholds["toxicity"]
    flag_hate = probs[0][1] > _thresholds["hate"]

    toxicity = LabelResult(prob=probs[0][0].item(), flagged=flag_toxicity.item())
    hate = LabelResult(prob=probs[0][1].item(), flagged=flag_hate.item())

    return toxicity, hate


def predict_batch(texts: list[str]) -> list[tuple[LabelResult, LabelResult]]:
    """Run inference on multiple texts in a single forward pass."""
    if not is_loaded():
        raise RuntimeError("Model is not loaded")

    assert _tokenizer is not None
    assert _model is not None

    inputs = _tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
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

    response = {
        "model_path": _model_path,
        "device": _device,
        "is_loaded": is_loaded(),
        "thresholds": _thresholds,
        "label_cols": _label_cols,
    }

    return response
