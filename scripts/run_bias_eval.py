import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from config import get_settings
from training.train_text_model import TextClassificationDataset, compute_metrics

logger = logging.getLogger(__name__)

_settings = get_settings()

ROOT = Path(__file__).resolve().parents[1]

DATA_TEMPLATED_PATH = ROOT / _settings.bias_eval.templated_data_path
DATA_VAL_PATH = ROOT / _settings.data.preprocessed_dir / "val.csv"

MODEL_DIR = ROOT / _settings.paths.model_dir
MODEL_PATH = MODEL_DIR / "model"
THRESHOLDS_PATH = MODEL_DIR / "thresholds.json"
REPORT_PATH = MODEL_DIR / "bias_report.json"

MODEL_NAME = _settings.model.name
NUM_LABELS = _settings.model.num_labels
MAX_LENGTH = _settings.model.max_length
BATCH_SIZE = _settings.bias_eval.batch_size


def load_thresholds(path: Path) -> dict[str, float]:
    if not path.exists():
        return {"toxicity": 0.5, "hate": 0.5}
    with path.open("r") as f:
        th = json.load(f)
    return {
        "toxicity": float(th.get("toxicity", 0.5)),
        "hate": float(th.get("hate", 0.5)),
    }


def load_model_and_tokenizer(device: torch.device):
    model_source = MODEL_PATH if MODEL_PATH.exists() else MODEL_NAME
    config = AutoConfig.from_pretrained(
        model_source,
        num_labels=NUM_LABELS,
        problem_type="multi_label_classification",
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_source,
        config=config,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_source)
    model.eval()
    return model, tokenizer


def compute_identity_sensitivity(
    df: pd.DataFrame,
    probs: np.ndarray,  # shape (N, 2): [toxicity_prob, hate_prob]
    thresholds: dict[str, float],
    pair_id_col: str = "pair_id",
) -> dict[str, float]:
    """
    Identity-agnostic counterfactual sensitivity on templated data.

    Metrics:
      - mean_abs_prob_delta: average within-pair |p - mean(p_pair)|
      - flip_rate: % pairs where thresholded predictions disagree
    """
    if pair_id_col not in df.columns:
        raise ValueError(
            f"Missing '{pair_id_col}' column in templated CSV. "
            "Add pair_id to group counterfactual variants of the same template."
        )

    pair_ids = df[pair_id_col].to_numpy()
    unique_pairs = pd.unique(pair_ids)

    out: dict[str, float] = {"num_samples": int(len(df))}
    for j, label in enumerate(["toxicity", "hate"]):
        t = float(thresholds.get(label, 0.5))
        pred = (probs[:, j] >= t).astype(int)

        deltas = []
        flips = []

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


def predict_probs(
    model,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      labels: (N, 2) float32
      probs:  (N, 2) float32  (sigmoid(logits))
    """
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in dataloader:
            labels = batch["labels"].cpu().numpy()

            inputs = {k: v.to(device) for k, v in batch.items() if k not in ["labels", "text"]}
            outputs = model(**inputs)

            probs = torch.sigmoid(outputs.logits).cpu().numpy()

            all_labels.append(labels)
            all_probs.append(probs)

    labels_arr = np.concatenate(all_labels, axis=0).astype(np.float32)
    probs_arr = np.concatenate(all_probs, axis=0).astype(np.float32)
    return labels_arr, probs_arr


def require_columns(df: pd.DataFrame, cols: list[str], name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}. Found: {list(df.columns)}")


def main():
    if not DATA_VAL_PATH.exists():
        raise FileNotFoundError(f"Validation dataset not found at {DATA_VAL_PATH}")

    if not DATA_TEMPLATED_PATH.exists():
        raise FileNotFoundError(f"Templated dataset not found at {DATA_TEMPLATED_PATH}")

    thresholds = load_thresholds(THRESHOLDS_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model_and_tokenizer(device)

    # Load data
    df_val = pd.read_csv(DATA_VAL_PATH)
    require_columns(df_val, ["text", "toxicity", "hate"], "val.csv")
    df_val["text"] = df_val["text"].astype(str)

    df_temp = pd.read_csv(DATA_TEMPLATED_PATH)
    require_columns(df_temp, ["text", "pair_id"], "templated_lexical_bias.csv")
    df_temp["text"] = df_temp["text"].astype(str)

    # Build datasets / loaders
    val_ds = TextClassificationDataset(df_val, tokenizer, max_length=MAX_LENGTH)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    temp_ds = TextClassificationDataset(df_temp, tokenizer, max_length=MAX_LENGTH)
    temp_loader = DataLoader(temp_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Predict on Val
    val_labels, val_probs = predict_probs(model, val_loader, device)
    val_metrics = compute_metrics(
        val_labels,
        val_probs,
        df_val["text"].tolist(),
        thresholds,
    )

    # Predict on templated
    _, temp_probs = predict_probs(model, temp_loader, device)
    templated_metrics = compute_identity_sensitivity(
        df=df_temp,
        probs=temp_probs,
        thresholds=thresholds,
        pair_id_col="pair_id",
    )

    # Write report
    report = {
        "val": val_metrics,
        "templated_identity_sensitivity": templated_metrics,
        "meta": {
            "val_path": str(DATA_VAL_PATH),
            "templated_path": str(DATA_TEMPLATED_PATH),
            "thresholds": thresholds,
            "model_source": str(MODEL_PATH) if MODEL_PATH.exists() else MODEL_NAME,
            "device": str(device),
        },
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    logger.info("Wrote bias report to %s", REPORT_PATH)


if __name__ == "__main__":
    main()
