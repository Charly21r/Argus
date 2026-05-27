import json
from pathlib import Path

import torch
from src.serving.schemas import LabelResult
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

# Module-level singleton
_tokenizer: PreTrainedTokenizerBase | None = None
_model: PreTrainedModel | None = None
_model_path: str | None = None
_device: str = "cpu"
_thresholds: dict = {}
_label_cols = ["toxicity", "hate"]


def load_model(model_path: str) -> None:
    """Load tokenizer and model from local path. Call this from the FastAPI lifespan."""
    global _tokenizer, _model, _model_path, _device, _thresholds, _label_cols

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        print("Warning! The model and tokenizer are being loaded on the CPU.")
        device = "cpu"

    # _label_cols = [_model.config.id2label[i] for i in range(_model.config.num_labels)]
    _tokenizer = AutoTokenizer.from_pretrained(model_path)
    _model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    _model.eval()
    _model_path = model_path
    _device = device

    # Load thresholds per label
    thresholds_path = Path(_model_path).parent / "thresholds.json"
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
