import os
from pathlib import Path
import mlflow
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from torch.optim import AdamW
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_recall_curve, auc, precision_score, recall_score, confusion_matrix
from tqdm import tqdm
from dotenv import load_dotenv
import json

load_dotenv()

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI")  # optional
SENSITIVE_CFG_PATH = Path("config/local_sensitive_words.json")

DATA_DIR = Path("data/preprocessed/text")
MODEL_DIR = Path("models/text_toxicity/artifacts")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
THRESHOLDS_PATH = MODEL_DIR / "thresholds.json" # path for the calibrated thresholds for each label

SEED = 42   # to ensure reproducibility
MODEL_NAME = "distilbert-base-uncased"
LABEL_COLS = ["toxicity", "hate"]   # safe can be derived from these at inference
NUM_LABELS = len(LABEL_COLS)
EPOCHS = 1
BATCH_SIZE = 16
LR = 2e-5
MAX_LENGTH = 256


def load_lexical_groups(path: Path) -> list[str]:
    if not path.exists():
        print(f"Warning! Lexical Bias Groups not found at {path}")
        return []
    with open(SENSITIVE_CFG_PATH) as f:
        sensitive_cfg = json.load(f)
    groups = sensitive_cfg.get("groups", [])
    return groups


LEXICAL_GROUPS = load_lexical_groups(SENSITIVE_CFG_PATH)


class JigsawDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=MAX_LENGTH):
        self.texts = df["text"].astype(str).tolist()
        self.labels = df[LABEL_COLS].values.astype("float32")
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        labels = self.labels[idx]

        enc = self.tokenizer(
            text,
            truncation = True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        # BCEWithLogitsLoss expects float targets for multi-label
        item["labels"] = torch.tensor(labels, dtype=torch.float32)
        item["text"] = text # raw text for lexical bias eval
        
        return item


def build_group_masks(texts: list[str], keywords: list[str]) -> np.ndarray:
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
        weight = neg / (pos + 1e-6)    # add the epsilon (1e-6) to avoid division by zero
        weights.append(weight)
    weights = torch.tensor(weights, dtype=torch.float32)
    return weights


def find_optimal_thresholds(
    all_labels: np.ndarray,
    all_probs: np.ndarray,
    search_space: np.ndarray | None = None
) -> dict:
    """ 
        Calibrates the thresholds per label to maximize F1-score
    """
    if search_space is None:
        search_space = np.linspace(0.01, 0.99, 99)

    thresholds = {}

    for i, label_name in enumerate(LABEL_COLS):
        y_true = all_labels[:, i]
        y_score = all_probs[:, i]

        best_f1 = -1
        best_t = 0.5

        for t in search_space:
            y_pred = (y_score >= t).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division='warn')

            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        
        thresholds[label_name] = best_t
    
    return thresholds


def train_one_epoch(
    model: AutoModelForSequenceClassification,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    log_every: int = 50
):
    model.train()
    total_loss = 0.0
    global_step = epoch*len(dataloader)

    for step, batch in enumerate(tqdm(dataloader, desc=f"Training epoch {epoch}")):
        inputs = {
                k: v.to(device) for k, v in batch.items() if k not in ["labels", "text"]
        }
        outputs = model(**inputs)
        loss = criterion(outputs.logits, batch['labels'])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # log batch training loss every 'log_every' steps
        if step % log_every == 0:
            mlflow.log_metric("train_batch_loss", loss.item(), step=global_step+step)
    
    return total_loss / len(dataloader)


def eval_model(
    model: AutoModelForSequenceClassification,
    dataloader: DataLoader,
    device: torch.device,
    epoch: int
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    model.eval()
    all_labels=[]
    all_probs=[]
    all_texts=[]

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Val Epoch {epoch}"):
            labels = batch["labels"].cpu().numpy()
            texts = batch["text"]
            # Remove labels and text for the inputs
            inputs = {
                k: v.to(device) for k, v in batch.items() if k not in ["labels", "text"]
            }
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.sigmoid(logits).cpu().numpy()
            all_labels.append(labels)
            all_probs.append(probs)
            all_texts.extend(texts)
    
    all_labels_arr = np.concatenate(all_labels, axis=0)
    all_probs_arr = np.concatenate(all_probs, axis=0)

    # If 1D, reshape using NUM_LABELS
    if all_labels_arr.ndim == 1:
        all_labels_arr = all_labels_arr.reshape(-1, NUM_LABELS)
    if all_probs_arr.ndim == 1:
        all_probs_arr = all_probs_arr.reshape(-1, NUM_LABELS)

    return all_labels_arr, all_probs_arr, all_texts


def _compute_labelwise_metrics_slice(
    labels: np.ndarray,
    probs: np.ndarray,
    thresholds: dict,
    prefix: str
) -> dict:
    """
        Compute all per-label metrics.
    """
    metrics = {}

    num_samples = int(labels.shape[0])
    metrics[f"{prefix}num_samples"] = num_samples

    if num_samples == 0:
        return metrics

    # Per-label metrics
    for i, label_name in enumerate(LABEL_COLS):
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
        
        prec_val = precision_score(y_true, y_pred, zero_division='warn')
        rec_val = recall_score(y_true, y_pred, zero_division='warn')
        f1_val = f1_score(y_true, y_pred, zero_division='warn')
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
    all_texts: np.ndarray,
    thresholds: dict,
) -> dict:
    metrics = {}

    # Overall metrics
    overall_metrics = _compute_labelwise_metrics_slice(
        labels=all_labels,
        probs=all_probs,
        thresholds=thresholds,
        prefix="val_"
    )
    metrics.update(overall_metrics)

    # Lexical bias analysis metrics
    if not LEXICAL_GROUPS:
        return metrics
    
    # Create mask for the texts that contain any of the words of the group
    mask = build_group_masks(all_texts, LEXICAL_GROUPS)

    # Group slice
    group_prefix = f"val_lex_group_"
    group_metrics = _compute_labelwise_metrics_slice(
        labels=all_labels[mask],
        probs=all_probs[mask],
        thresholds=thresholds,
        prefix=group_prefix
    )
    metrics.update(group_metrics)

    # Non-group slice
    non_group_prefix = f"val_non_lex_group_"
    non_group_metrics = _compute_labelwise_metrics_slice(
        labels=all_labels[~mask],
        probs=all_probs[~mask],
        thresholds=thresholds,
        prefix=non_group_prefix
    )
    metrics.update(non_group_metrics)

    # Deltas
    for label_name in LABEL_COLS:
        g_fpr = group_metrics.get(f"{group_prefix}{label_name}_FPR")
        ng_fpr = non_group_metrics.get(f"{non_group_prefix}{label_name}_FPR")
        g_tpr = group_metrics.get(f"{group_prefix}{label_name}_recall")   # recall == TPR
        ng_tpr = non_group_metrics.get(f"{non_group_prefix}{label_name}_recall")

        if g_fpr is not None and ng_fpr is not None:
            metrics[f"{group_prefix}{label_name}_FPR_delta"] = g_fpr - ng_fpr
        if g_tpr is not None and ng_tpr is not None:
            metrics[f"{group_prefix}{label_name}_TPR_delta"] = g_tpr - ng_tpr

    return metrics


def main():
    # Set the seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    
    # MLflow setup
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("text_toxicity_moderation")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    val_df = pd.read_csv(DATA_DIR / "val.csv")
    
    # Calculate the weights to handle imbalance and use them in the loss
    pos_weights = calculate_pos_weights(train_df, LABEL_COLS).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_ds = JigsawDataset(train_df, tokenizer)
    val_ds = JigsawDataset(val_df, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    config = AutoConfig.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        problem_type="multi_label_classification",
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        config=config,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=LR)

    with mlflow.start_run():
        mlflow.log_params(
            {
                "model_name": MODEL_NAME,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "lr": LR,
                "max_length": MAX_LENGTH,
                "label_cols": ",".join(LABEL_COLS),
                "problem_type": "multi_label_classification",
            }
        )

        # Start with 0.5 thresholds
        thresholds = {lab: 0.5 for lab in LABEL_COLS}

        for epoch in range(EPOCHS):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
            val_labels, val_probs, val_texts = eval_model(model, val_loader, device, epoch)
            val_metrics = compute_metrics(val_labels, val_probs, val_texts, thresholds)

            print(f"Epoch {epoch}: loss={train_loss:.4f}, val_metrics={val_metrics}")

            # log training loss and per-label metrics
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            for k, v in val_metrics.items():
                mlflow.log_metric(k, v, step=epoch)
        
        # Final calibration on validation set
        val_labels, val_probs, _ = eval_model(model, val_loader, device, epoch)
        calibrated_thresholds = find_optimal_thresholds(val_labels, val_probs)

        # Save thresholds JSON alongside the model
        THRESHOLDS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with THRESHOLDS_PATH.open("w") as f:
            json.dump(calibrated_thresholds, f)

        # Log calibrated thresholds to MLflow
        mlflow.log_params(
            {f"threshold_{k}": v for k, v in calibrated_thresholds.items()}
        )

        # Save model locally and to MLflow
        save_path = MODEL_DIR / "model"
        save_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"Model and tokenizer saved to {save_path}")

        # Log the entire folder (weights + tokenizer + config) to MLflow
        mlflow.log_artifacts(str(save_path), artifact_path="full_model")
        # Log the thresholds to MLflow
        mlflow.log_artifact(str(THRESHOLDS_PATH), artifact_path="full_model")


if __name__ == "__main__":
    main()


