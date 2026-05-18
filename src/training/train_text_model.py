import json
import logging
import os
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
import transformers
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
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    get_linear_schedule_with_warmup,
)

from src.config import get_settings
from src.training.losses import BinaryFocalLossWithLogits
from src.utils.lexicon import load_group_terms

logger = logging.getLogger(__name__)


class TextClassificationDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, label_cols: list[str], max_length: int):
        self.texts = df["text"].astype(str).tolist()
        self.labels = df[label_cols].values.astype("float32")
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        enc = self.tokenizer(
            text, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float32)
        item["text"] = text
        return item


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def build_group_masks(texts: Sequence[str] | np.ndarray, keywords: Sequence[str]) -> np.ndarray:
    """Returns True where any keyword appears in the text (case-insensitive)."""
    kw_lower = [k.lower() for k in keywords]
    return np.array([any(kw in t.lower() for kw in kw_lower) for t in texts], dtype=bool)


def calculate_pos_weights(df: pd.DataFrame, label_cols: list[str]) -> torch.Tensor:
    weights = [((len(df) - df[lab].sum()) / (df[lab].sum() + 1e-6)) for lab in label_cols]
    return torch.tensor(weights, dtype=torch.float32)


def find_optimal_thresholds(
    all_labels: np.ndarray,
    all_probs: np.ndarray,
    label_cols: list[str],
    search_space: np.ndarray | None = None,
) -> dict:
    """Searches per-label thresholds that maximize F1 on the provided data."""
    if search_space is None:
        search_space = np.linspace(0.01, 0.99, 99)

    thresholds = {}
    for i, label_name in enumerate(label_cols):
        y_true = all_labels[:, i]
        y_score = all_probs[:, i]
        best_f1, best_t = -1.0, 0.5
        for t in search_space:
            f = f1_score(y_true, (y_score >= t).astype(int), zero_division=0)
            if f > best_f1:
                best_f1, best_t = f, float(t)
        thresholds[label_name] = best_t

    return thresholds


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _compute_labelwise_metrics_slice(
    labels: np.ndarray, probs: np.ndarray, thresholds: dict, prefix: str, label_cols: list[str]
) -> dict:
    metrics: dict[str, float | int] = {f"{prefix}num_samples": int(labels.shape[0])}

    if labels.shape[0] == 0:
        return metrics

    for i, label_name in enumerate(label_cols):
        y_true = labels[:, i]
        y_score = probs[:, i]
        y_pred = (y_score >= thresholds[label_name]).astype(int)

        if len(np.unique(y_true)) == 1:
            roc_auc = float("nan")
            pr_auc = float("nan")
        else:
            roc_auc = roc_auc_score(y_true, y_score)
            precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_score)
            pr_auc = auc(recall_vals, precision_vals)

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        base = f"{prefix}{label_name}"
        metrics.update(
            {
                f"{base}_roc_auc": float(roc_auc),
                f"{base}_pr_auc": float(pr_auc),
                f"{base}_precision": float(precision_score(y_true, y_pred, zero_division=0)),
                f"{base}_recall": float(recall_score(y_true, y_pred, zero_division=0)),
                f"{base}_f1": float(f1_score(y_true, y_pred, zero_division=0)),
                f"{base}_accuracy": float(accuracy_score(y_true, y_pred)),
                f"{base}_TP": int(tp),
                f"{base}_FP": int(fp),
                f"{base}_FN": int(fn),
                f"{base}_TN": int(tn),
                f"{base}_FPR": float(fp / (fp + tn + 1e-6)),
                f"{base}_FNR": float(fn / (fn + tp + 1e-6)),
            }
        )

    return metrics


def compute_metrics(
    all_labels: np.ndarray,
    all_probs: np.ndarray,
    all_texts: Sequence[str] | np.ndarray,
    thresholds: dict,
    label_cols: list[str],
    lexical_groups: list[str],
) -> dict:
    metrics = _compute_labelwise_metrics_slice(
        labels=all_labels, probs=all_probs, thresholds=thresholds, prefix="val_", label_cols=label_cols
    )

    if not lexical_groups:
        return metrics

    mask = build_group_masks(all_texts, lexical_groups)
    group_prefix = "val_lex_group_"
    non_group_prefix = "val_non_lex_group_"

    group_metrics = _compute_labelwise_metrics_slice(
        labels=all_labels[mask],
        probs=all_probs[mask],
        thresholds=thresholds,
        prefix=group_prefix,
        label_cols=label_cols,
    )
    non_group_metrics = _compute_labelwise_metrics_slice(
        labels=all_labels[~mask],
        probs=all_probs[~mask],
        thresholds=thresholds,
        prefix=non_group_prefix,
        label_cols=label_cols,
    )
    metrics.update(group_metrics)
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


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: PreTrainedModel,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_scaler: GradScaler,
    epoch: int,
    amp_enabled: bool,
    amp_dtype: torch.dtype = torch.float16,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    grad_clip_norm: float = 1.0,
    log_every: int = 50,
    on_step_end: Callable[[int, float, float | None], None] | None = None,
) -> float:
    model.train()
    total_loss = 0.0
    global_step = epoch * len(dataloader)

    for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch} train")):
        inputs = {k: v.to(device) for k, v in batch.items() if k not in ("labels", "text")}
        labels = batch["labels"].to(device)

        with autocast(device.type, dtype=amp_dtype, enabled=amp_enabled):
            outputs = model(**inputs)
            loss = criterion(outputs.logits, labels)

        if grad_scaler.is_enabled():
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        optimizer.zero_grad()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()

        if on_step_end is not None and step % log_every == 0:
            lr = float(scheduler.get_last_lr()[0]) if scheduler is not None else None
            on_step_end(global_step + step, loss.item(), lr)

    return total_loss / len(dataloader)


def eval_model(
    model: PreTrainedModel,
    dataloader: DataLoader,
    device: torch.device,
    num_labels: int,
    amp_enabled: bool,
    amp_dtype: torch.dtype = torch.float16,
    desc: str = "Evaluating",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    model.eval()
    all_labels, all_probs, all_texts = [], [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc):
            labels = batch["labels"].cpu().numpy()
            inputs = {k: v.to(device) for k, v in batch.items() if k not in ("labels", "text")}

            with autocast(device.type, dtype=amp_dtype, enabled=amp_enabled):
                outputs = model(**inputs)
            probs = torch.sigmoid(outputs.logits).float().cpu().numpy()

            all_labels.append(labels)
            all_probs.append(probs)
            all_texts.extend(batch["text"])

    all_labels_arr = np.concatenate(all_labels, axis=0)
    all_probs_arr = np.concatenate(all_probs, axis=0)

    if all_labels_arr.ndim == 1:
        all_labels_arr = all_labels_arr.reshape(-1, num_labels)
    if all_probs_arr.ndim == 1:
        all_probs_arr = all_probs_arr.reshape(-1, num_labels)

    return all_labels_arr, all_probs_arr, all_texts


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------


def setup_data(
    cfg,
    tokenizer,
    device: torch.device,
) -> tuple[DataLoader, DataLoader, torch.Tensor]:
    data_dir = Path(cfg.data.preprocessed_dir)
    label_cols = cfg.model.label_cols
    batch_size = cfg.training.batch_size

    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "val.csv")

    pos_weights = calculate_pos_weights(train_df, label_cols).to(device)

    train_ds = TextClassificationDataset(train_df, tokenizer, label_cols=label_cols, max_length=cfg.model.max_length)
    val_ds = TextClassificationDataset(val_df, tokenizer, label_cols=label_cols, max_length=cfg.model.max_length)

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=2, pin_memory=pin_memory)

    return train_loader, val_loader, pos_weights


def setup_model(cfg, device: torch.device, n_train_steps: int) -> tuple:
    model_name = cfg.model.name
    mixed_precision = cfg.training.mixed_precision

    hf_config = AutoConfig.from_pretrained(
        model_name,
        num_labels=cfg.model.num_labels,
        problem_type="multi_label_classification",
    )
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=hf_config).to(device)
    optimizer = AdamW(model.parameters(), lr=cfg.training.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, n_train_steps // 10),
        num_training_steps=n_train_steps,
    )

    amp_enabled = mixed_precision and device.type in {"cuda", "mps"}
    amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    grad_scaler = GradScaler(device=device.type, enabled=amp_enabled and device.type == "cuda")

    return model, optimizer, scheduler, grad_scaler, amp_enabled, amp_dtype


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def run_training(
    model: PreTrainedModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    grad_scaler: GradScaler,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    device: torch.device,
    cfg,
    lexical_groups: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str], float]:
    epochs = cfg.training.epochs
    label_cols = cfg.model.label_cols
    num_labels = cfg.model.num_labels

    thresholds = {lab: 0.5 for lab in label_cols}
    best_mean_f1 = -1.0
    best_weights: dict | None = None
    val_labels = val_probs = val_texts = None

    def _log_step(global_step: int, loss: float, lr: float | None) -> None:
        mlflow.log_metric("train_batch_loss", loss, step=global_step)
        if lr is not None:
            mlflow.log_metric("lr", lr, step=global_step)

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            grad_scaler,
            epoch,
            amp_enabled,
            amp_dtype=amp_dtype,
            scheduler=scheduler,
            on_step_end=_log_step,
        )

        val_labels, val_probs, val_texts = eval_model(
            model,
            val_loader,
            device,
            num_labels,
            amp_enabled,
            amp_dtype=amp_dtype,
            desc=f"Epoch {epoch} val",
        )
        val_metrics = compute_metrics(val_labels, val_probs, val_texts, thresholds, label_cols, lexical_groups)

        logger.info("Epoch %d: loss=%.4f, val_metrics=%s", epoch, train_loss, val_metrics)
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        for k, v in val_metrics.items():
            mlflow.log_metric(k, v, step=epoch)

        mean_f1 = float(np.mean([val_metrics.get(f"val_{lab}_f1", 0.0) for lab in label_cols]))
        if mean_f1 > best_mean_f1:
            best_mean_f1 = mean_f1
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            logger.info("  New best model (epoch %d, mean F1=%.4f)", epoch, best_mean_f1)

    if best_weights is not None:
        model.load_state_dict(best_weights)
        logger.info("Restored best weights (mean F1=%.4f)", best_mean_f1)

    assert val_labels is not None and val_probs is not None and val_texts is not None
    return val_labels, val_probs, val_texts, best_mean_f1


# ---------------------------------------------------------------------------
# Artifact saving
# ---------------------------------------------------------------------------


def save_artifacts(
    model: PreTrainedModel,
    tokenizer,
    calibrated_thresholds: dict,
    model_dir: Path,
    registered_model_name: str,
) -> None:
    thresholds_path = model_dir / "thresholds.json"
    with thresholds_path.open("w") as f:
        json.dump(calibrated_thresholds, f)

    save_path = model_dir / "model"
    save_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    logger.info("Model and tokenizer saved to %s", save_path)

    mlflow.transformers.log_model(
        transformers_model={"model": model, "tokenizer": tokenizer},
        artifact_path="model",
        registered_model_name=registered_model_name,
        metadata={"thresholds": calibrated_thresholds},
        pip_requirements=[
            f"torch=={torch.__version__}",
            f"transformers=={transformers.__version__}",
        ],
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    cfg = get_settings()

    model_dir = Path(cfg.paths.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    seed = cfg.training.seed
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

    lexical_groups = load_group_terms(cfg.paths.sensitive_words_config)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)

    train_loader, val_loader, pos_weights = setup_data(cfg, tokenizer, device)
    if cfg.training.loss_fn == "focal":
        criterion = BinaryFocalLossWithLogits(gamma=cfg.training.focal_gamma, pos_weight=pos_weights)
    else:
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    n_train_steps = cfg.training.epochs * len(train_loader)
    model, optimizer, scheduler, grad_scaler, amp_enabled, amp_dtype = setup_model(cfg, device, n_train_steps)

    git_sha = (
        subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
        or "unknown"
    )
    run_name = f"{cfg.model.name.split('/')[-1]}_e{cfg.training.epochs}_lr{cfg.training.learning_rate}"

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags(
            {
                "git_sha": git_sha,
                "device": str(device),
            }
        )
        loss_params: dict = {"loss_fn": cfg.training.loss_fn}
        if cfg.training.loss_fn == "focal":
            loss_params["focal_gamma"] = cfg.training.focal_gamma
        mlflow.log_params(
            {
                "model_name": cfg.model.name,
                "epochs": cfg.training.epochs,
                "batch_size": cfg.training.batch_size,
                "lr": cfg.training.learning_rate,
                "max_length": cfg.model.max_length,
                "label_cols": ",".join(cfg.model.label_cols),
                "problem_type": "multi_label_classification",
                **loss_params,
            }
        )
        if pos_weights is not None:
            mlflow.log_params(
                {f"pos_weight_{lab}": round(float(pos_weights[i]), 4) for i, lab in enumerate(cfg.model.label_cols)}
            )

        val_labels, val_probs, _, best_mean_f1 = run_training(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            scheduler,
            grad_scaler,
            amp_enabled,
            amp_dtype,
            device,
            cfg,
            lexical_groups,
        )
        mlflow.log_metric("best_mean_f1", best_mean_f1)

        calibrated_thresholds = find_optimal_thresholds(val_labels, val_probs, cfg.model.label_cols)
        for k, v in calibrated_thresholds.items():
            mlflow.log_metric(f"threshold_{k}", v)

        save_artifacts(
            model,
            tokenizer,
            calibrated_thresholds,
            model_dir,
            registered_model_name=cfg.optimization.registered_model_name,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
