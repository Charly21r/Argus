import json
import logging
import os
from collections.abc import Sequence
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel, get_linear_schedule_with_warmup

from src.config import get_settings
from src.utils.lexicon import load_group_terms

logger = logging.getLogger(__name__)


class JigsawDataset(Dataset):
    def __init__(self, df, tokenizer, label_cols: list[str], max_length: int):
        self.texts = df["text"].astype(str).tolist()
        self.labels = df[label_cols].values.astype("float32")
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        labels = self.labels[idx]

        enc = self.tokenizer(
            text, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        # BCEWithLogitsLoss expects float targets for multi-label
        item["labels"] = torch.tensor(labels, dtype=torch.float32)
        item["text"] = text  # raw text for lexical bias eval

        return item


def build_group_masks(
    texts: Sequence[str] | np.ndarray,
    keywords: Sequence[str],
) -> np.ndarray:
    """
    True if any keyword appears in the text
    """
    kw_lower = [k.lower() for k in keywords]
    mask = []

    for t in texts:
        t_low = t.lower()
        mask.append(any(kw in t_low for kw in kw_lower))

    return np.array(mask, dtype=bool)


def calculate_pos_weights(df: pd.DataFrame, labels) -> torch.Tensor:
    weights = []
    for lab in labels:
        pos = df[lab].sum()
        neg = len(df) - pos
        weight = neg / (pos + 1e-6)  # add the epsilon (1e-6) to avoid division by zero
        weights.append(weight)

    return torch.tensor(weights, dtype=torch.float32)


def find_optimal_thresholds(
    all_labels: np.ndarray,
    all_probs: np.ndarray,
    label_cols: list[str],
    search_space: np.ndarray | None = None,
) -> dict:
    """Calibrates the thresholds per label to maximize F1-score."""
    if search_space is None:
        search_space = np.linspace(0.01, 0.99, 99)

    thresholds = {}

    for i, label_name in enumerate(label_cols):
        y_true = all_labels[:, i]
        y_score = all_probs[:, i]

        best_f1 = -1
        best_t = 0.5

        for t in search_space:
            y_pred = (y_score >= t).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division="warn")

            if f1 > best_f1:
                best_f1 = f1
                best_t = t

        thresholds[label_name] = best_t

    return thresholds


def train_one_epoch(
    model: PreTrainedModel,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_scaler: torch.amp.GradScaler,
    epoch: int,
    mixed_precision: bool,
    amp_dtype: torch.dtype = torch.float16,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    log_every: int = 50,
):
    model.train()
    total_loss = 0.0
    global_step = epoch * len(dataloader)

    for step, batch in enumerate(tqdm(dataloader, desc=f"Training epoch {epoch}")):
        inputs = {k: v.to(device) for k, v in batch.items() if k not in ["labels", "text"]}
        labels = batch["labels"].to(device)

        with autocast(device.type, dtype=amp_dtype, enabled=mixed_precision):
            outputs = model(**inputs)
            loss = criterion(outputs.logits, labels)

        if grad_scaler.is_enabled():
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            loss.backward()
            optimizer.step()
        optimizer.zero_grad()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item()

        if step % log_every == 0:
            mlflow.log_metric("train_batch_loss", loss.item(), step=global_step + step)
            if scheduler is not None:
                mlflow.log_metric("lr", scheduler.get_last_lr()[0], step=global_step + step)

    return total_loss / len(dataloader)


def eval_model(
    model: PreTrainedModel,
    dataloader: DataLoader,
    device: torch.device,
    epoch: int,
    num_labels: int,
    mixed_precision: bool,
    amp_dtype: torch.dtype = torch.float16,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    model.eval()
    all_labels = []
    all_probs = []
    all_texts = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Val Epoch {epoch}"):
            labels = batch["labels"].cpu().numpy()
            texts = batch["text"]
            inputs = {k: v.to(device) for k, v in batch.items() if k not in ["labels", "text"]}

            with autocast(device.type, dtype=amp_dtype, enabled=mixed_precision):
                outputs = model(**inputs)
                logits = outputs.logits
            probs = torch.sigmoid(logits).float().cpu().numpy()
            all_labels.append(labels)
            all_probs.append(probs)
            all_texts.extend(texts)

    all_labels_arr = np.concatenate(all_labels, axis=0)
    all_probs_arr = np.concatenate(all_probs, axis=0)

    if all_labels_arr.ndim == 1:
        all_labels_arr = all_labels_arr.reshape(-1, num_labels)
    if all_probs_arr.ndim == 1:
        all_probs_arr = all_probs_arr.reshape(-1, num_labels)

    return all_labels_arr, all_probs_arr, all_texts


def _compute_labelwise_metrics_slice(
    labels: np.ndarray, probs: np.ndarray, thresholds: dict, prefix: str, label_cols: list[str]
) -> dict:
    metrics: dict[str, float | int] = {}

    num_samples = int(labels.shape[0])
    metrics[f"{prefix}num_samples"] = num_samples

    if num_samples == 0:
        return metrics

    for i, label_name in enumerate(label_cols):
        y_true = labels[:, i]
        y_score = probs[:, i]
        y_pred = (y_score >= thresholds[label_name]).astype(int)

        # Guard in case there are no positives in val for a label
        if len(np.unique(y_true)) == 1:
            roc_auc = float("nan")
            pr_auc = float("nan")
        else:
            roc_auc = roc_auc_score(y_true, y_score)
            precision, recall, _ = precision_recall_curve(y_true, y_score)
            pr_auc = auc(recall, precision)

        prec_val = precision_score(y_true, y_pred, zero_division="warn")
        rec_val = recall_score(y_true, y_pred, zero_division="warn")
        f1_val = f1_score(y_true, y_pred, zero_division="warn")
        acc = accuracy_score(y_true, y_pred)

        # Create a binary label confusion matrix for each label
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn + 1e-6)
        fnr = fn / (fn + tp + 1e-6)

        base = f"{prefix}{label_name}"
        metrics[f"{base}_roc_auc"] = float(roc_auc)
        metrics[f"{base}_pr_auc"] = float(pr_auc)
        metrics[f"{base}_precision"] = float(prec_val)
        metrics[f"{base}_recall"] = float(rec_val)
        metrics[f"{base}_f1"] = float(f1_val)
        metrics[f"{base}_accuracy"] = float(acc)
        metrics[f"{base}_TP"] = int(tp)
        metrics[f"{base}_FP"] = int(fp)
        metrics[f"{base}_FN"] = int(fn)
        metrics[f"{base}_TN"] = int(tn)
        metrics[f"{base}_FPR"] = float(fpr)
        metrics[f"{base}_FNR"] = float(fnr)

    return metrics


def compute_metrics(
    all_labels: np.ndarray,
    all_probs: np.ndarray,
    all_texts: Sequence[str] | np.ndarray,
    thresholds: dict,
    label_cols: list[str],
    lexical_groups: list[str],
) -> dict:
    metrics = {}

    overall_metrics = _compute_labelwise_metrics_slice(
        labels=all_labels, probs=all_probs, thresholds=thresholds, prefix="val_", label_cols=label_cols
    )
    metrics.update(overall_metrics)

    if not lexical_groups:
        return metrics

    mask = build_group_masks(all_texts, lexical_groups)

    group_prefix = "val_lex_group_"
    group_metrics = _compute_labelwise_metrics_slice(
        labels=all_labels[mask], probs=all_probs[mask], thresholds=thresholds, prefix=group_prefix, label_cols=label_cols
    )
    metrics.update(group_metrics)

    non_group_prefix = "val_non_lex_group_"
    non_group_metrics = _compute_labelwise_metrics_slice(
        labels=all_labels[~mask], probs=all_probs[~mask], thresholds=thresholds, prefix=non_group_prefix, label_cols=label_cols
    )
    metrics.update(non_group_metrics)

    for label_name in label_cols:
        g_fpr = group_metrics.get(f"{group_prefix}{label_name}_FPR")
        ng_fpr = non_group_metrics.get(f"{non_group_prefix}{label_name}_FPR")
        g_tpr = group_metrics.get(f"{group_prefix}{label_name}_recall")
        ng_tpr = non_group_metrics.get(f"{non_group_prefix}{label_name}_recall")

        if g_fpr is not None and ng_fpr is not None:
            metrics[f"{group_prefix}{label_name}_FPR_delta"] = g_fpr - ng_fpr
        if g_tpr is not None and ng_tpr is not None:
            metrics[f"{group_prefix}{label_name}_TPR_delta"] = g_tpr - ng_tpr

    return metrics


def main():
    cfg = get_settings()

    data_dir = Path(cfg.data.preprocessed_dir)
    model_dir = Path(cfg.paths.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    thresholds_path = model_dir / "thresholds.json"

    seed = cfg.training.seed
    model_name = cfg.model.name
    label_cols = cfg.model.label_cols
    num_labels = cfg.model.num_labels
    epochs = cfg.training.epochs
    batch_size = cfg.training.batch_size
    lr = cfg.training.learning_rate
    max_length = cfg.model.max_length
    mixed_precision = cfg.training.mixed_precision
    lexical_groups = load_group_terms(cfg.paths.sensitive_words_config)

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI") or cfg.mlflow_tracking_uri
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_NAME", "text_toxicity_moderation"))

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("Using device: %s", device)

    train_df = pd.read_csv(data_dir / "train.csv")[:3]
    val_df = pd.read_csv(data_dir / "val.csv")[:2]

    pos_weights = calculate_pos_weights(train_df, label_cols).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_ds = JigsawDataset(train_df, tokenizer, label_cols=label_cols, max_length=max_length)
    val_ds = JigsawDataset(val_df, tokenizer, label_cols=label_cols, max_length=max_length)

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=2, pin_memory=pin_memory)

    hf_config = AutoConfig.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type="multi_label_classification",
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=hf_config,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=lr)
    # GradScaler is CUDA-only; on MPS/CPU mixed precision uses bf16 (no scaling needed).
    amp_enabled = mixed_precision and device.type in {"cuda", "mps"}
    amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    grad_scaler = GradScaler(device=device.type, enabled=amp_enabled and device.type == "cuda")
    total_steps = epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps,
    )

    with mlflow.start_run():
        mlflow.log_params(
            {
                "model_name": model_name,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "max_length": max_length,
                "label_cols": ",".join(label_cols),
                "problem_type": "multi_label_classification",
            }
        )

        thresholds = {lab: 0.5 for lab in label_cols}

        for epoch in range(epochs):
            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, device, grad_scaler, epoch, amp_enabled, amp_dtype=amp_dtype, scheduler=scheduler
            )
            val_labels, val_probs, val_texts = eval_model(
                model, val_loader, device, epoch, num_labels, amp_enabled, amp_dtype=amp_dtype
            )
            val_metrics = compute_metrics(val_labels, val_probs, val_texts, thresholds, label_cols, lexical_groups)

            logger.info("Epoch %d: loss=%.4f, val_metrics=%s", epoch, train_loss, val_metrics)

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            for k, v in val_metrics.items():
                mlflow.log_metric(k, v, step=epoch)

        val_labels, val_probs, _ = eval_model(model, val_loader, device, epochs - 1, num_labels, amp_enabled, amp_dtype=amp_dtype)
        calibrated_thresholds = find_optimal_thresholds(val_labels, val_probs, label_cols)

        thresholds_path.parent.mkdir(parents=True, exist_ok=True)
        with thresholds_path.open("w") as f:
            json.dump(calibrated_thresholds, f)

        mlflow.log_params({f"threshold_{k}": v for k, v in calibrated_thresholds.items()})

        save_path = model_dir / "model"
        save_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        logger.info("Model and tokenizer saved to %s", save_path)

        mlflow.log_artifacts(str(save_path), artifact_path="full_model")
        mlflow.log_artifact(str(thresholds_path), artifact_path="full_model")


if __name__ == "__main__":
    main()
