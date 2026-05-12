"""Loading of the model and tokenizer"""

import json
from pathlib import Path

import torch
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from src.config import get_settings

_settings = get_settings()

LABEL_COLS = _settings.model.label_cols
MODEL_NAME = _settings.model.name
NUM_LABELS = _settings.model.num_labels


def load_thresholds(path: Path) -> dict[str, float]:
    if not path.exists():
        return {label: 0.5 for label in LABEL_COLS}
    with path.open("r") as f:
        th = json.load(f)
    return {label: float(th.get(label, 0.5)) for label in LABEL_COLS}


def load_model_and_tokenizer(model_path: Path, device: torch.device):
    """Load model and tokenizer from model_path. Auto-detects ONNX vs PyTorch."""
    is_ort = any(model_path.glob("*.onnx"))
    if is_ort:
        model = ORTModelForSequenceClassification.from_pretrained(model_path)
    else:
        model_source = model_path if model_path.exists() else MODEL_NAME
        config = AutoConfig.from_pretrained(
            model_source,
            num_labels=NUM_LABELS,
            problem_type="multi_label_classification",
        )
        model = AutoModelForSequenceClassification.from_pretrained(model_source, config=config).to(device)
        model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path if model_path.exists() else MODEL_NAME)
    return model, tokenizer
