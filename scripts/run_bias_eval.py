import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from src.config import get_settings
from src.training.train_text_model import TextClassificationDataset, compute_metrics
from src.utils.lexicon import load_group_terms
from src.utils.loading import load_model_and_tokenizer, load_thresholds
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

_settings = get_settings()

ROOT = Path(__file__).resolve().parents[1]

DATA_TEMPLATED_PATH = ROOT / _settings.bias_eval.templated_data_path
DATA_VAL_PATH = ROOT / _settings.data.preprocessed_dir / "val.csv"

MODEL_DIR = ROOT / _settings.paths.model_dir
MODEL_PATH = MODEL_DIR / "model"
THRESHOLDS_PATH = MODEL_DIR / "thresholds.json"
REPORT_PATH = MODEL_DIR / "bias_report.json"

LABEL_COLS = _settings.model.label_cols
MODEL_NAME = _settings.model.name
NUM_LABELS = _settings.model.num_labels
MAX_LENGTH = _settings.model.max_length
BATCH_SIZE = _settings.bias_eval.batch_size


def compute_identity_sensitivity(
    df: pd.DataFrame,
    probs: np.ndarray,
    thresholds: dict[str, float],
    pair_id_col: str = "pair_id",
) -> dict[str, float]:
    """Identity-agnostic counterfactual sensitivity on templated data."""
    if pair_id_col not in df.columns:
        raise ValueError(
            f"Missing '{pair_id_col}' column in templated CSV. "
            "Add pair_id to group counterfactual variants of the same template."
        )

    pair_ids = df[pair_id_col].to_numpy()
    unique_pairs = pd.unique(pair_ids)

    out: dict[str, float] = {"num_samples": int(len(df))}
    for j, label in enumerate(LABEL_COLS):
        t = float(thresholds.get(label, 0.5))
        pred = (probs[:, j] >= t).astype(int)

        deltas, flips = [], []
        for pid in unique_pairs:
            idx = np.where(pair_ids == pid)[0]
            if idx.size < 2:
                continue
            p = probs[idx, j]
            deltas.append(float(np.abs(p - p.mean()).mean()))
            flips.append(int(pred[idx].max() != pred[idx].min()))

        out[f"{label}_mean_abs_prob_delta"] = float(np.mean(deltas)) if deltas else float("nan")
        out[f"{label}_flip_rate"] = float(np.mean(flips)) if flips else float("nan")
        out[f"{label}_num_pairs"] = int(len(flips))

    return out


def predict_probs(model, dataloader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Returns (labels, probs) arrays of shape (N, num_labels)."""
    # Determine which input keys the model accepts to avoid passing unsupported ones
    # (e.g. DistilBERT does not accept token_type_ids).
    forward_keys = set(model.forward.__code__.co_varnames)

    # ORT models run on CPU and don't support tensor.to(device).
    from optimum.onnxruntime import ORTModel

    is_ort = isinstance(model, ORTModel)

    all_labels, all_probs = [], []
    with torch.no_grad():
        for batch in dataloader:
            labels = batch["labels"].cpu().numpy()
            inputs = {
                k: (v if is_ort else v.to(device))
                for k, v in batch.items()
                if k not in ("labels", "text") and k in forward_keys
            }
            probs = torch.sigmoid(model(**inputs).logits).cpu().numpy()
            all_labels.append(labels)
            all_probs.append(probs)

    return (
        np.concatenate(all_labels, axis=0).astype(np.float32),
        np.concatenate(all_probs, axis=0).astype(np.float32),
    )


def require_columns(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}. Found: {list(df.columns)}")


def run_bias_eval(
    model_path: Path,
    thresholds_path: Path,
    report_path: Path,
) -> dict:
    """Run the full bias evaluation for any model (PyTorch or ONNX) and write a report."""
    if not DATA_VAL_PATH.exists():
        raise FileNotFoundError(f"Validation dataset not found at {DATA_VAL_PATH}")
    if not DATA_TEMPLATED_PATH.exists():
        raise FileNotFoundError(f"Templated dataset not found at {DATA_TEMPLATED_PATH}")

    thresholds = load_thresholds(thresholds_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model_and_tokenizer(model_path, device)

    df_val = pd.read_csv(DATA_VAL_PATH)
    require_columns(df_val, ["text"] + LABEL_COLS, "val.csv")
    df_val["text"] = df_val["text"].astype(str)

    df_temp = pd.read_csv(DATA_TEMPLATED_PATH)
    require_columns(df_temp, ["text", "pair_id"], "templated_lexical_bias.csv")
    df_temp["text"] = df_temp["text"].astype(str)

    val_ds = TextClassificationDataset(df_val, tokenizer, label_cols=LABEL_COLS, max_length=MAX_LENGTH)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    temp_ds = TextClassificationDataset(df_temp, tokenizer, label_cols=LABEL_COLS, max_length=MAX_LENGTH)
    temp_loader = DataLoader(temp_ds, batch_size=BATCH_SIZE, shuffle=False)

    lexical_groups = load_group_terms(_settings.paths.sensitive_words_config)
    val_labels, val_probs = predict_probs(model, val_loader, device)
    val_metrics = compute_metrics(
        val_labels, val_probs, df_val["text"].tolist(), thresholds, LABEL_COLS, lexical_groups
    )

    _, temp_probs = predict_probs(model, temp_loader, device)
    templated_metrics = compute_identity_sensitivity(df_temp, temp_probs, thresholds)

    report = {
        "val": val_metrics,
        "templated_identity_sensitivity": templated_metrics,
        "meta": {
            "val_path": str(DATA_VAL_PATH),
            "templated_path": str(DATA_TEMPLATED_PATH),
            "thresholds": thresholds,
            "model_path": str(model_path),
            "device": str(device),
        },
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    logger.info("Wrote bias report to %s", report_path)

    return report


def main() -> None:
    run_bias_eval(
        model_path=MODEL_PATH,
        thresholds_path=THRESHOLDS_PATH,
        report_path=REPORT_PATH,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
