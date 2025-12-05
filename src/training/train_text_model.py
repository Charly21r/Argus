import os
from pathlib import Path
import mlflow
import mlflow.pytorch
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from torch.optim import AdamW
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_recall_curve, auc, precision_score, recall_score, confusion_matrix
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI")  # optional

DATA_DIR = Path("data/preprocessed/text")
MODEL_DIR = Path("models/text_toxicity/artifacts")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "distilbert-base-uncased"
LABEL_COLS = ["toxicity", "hate"]   # safe can be derived from these at inference
NUM_LABELS = len(LABEL_COLS)
EPOCHS = 1
BATCH_SIZE = 16
LR = 2e-5
MAX_LENGTH = 256

SEED = 42   # to ensure reproducibility

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
        return item


def calculate_pos_weights(df: pd.DataFrame, labels) -> torch.Tensor:
    weights = []
    for lab in labels:
        pos = df[lab].sum()
        neg = len(df) - pos
        weight = neg / (pos + 1e-6)    # add the epsilon (1e-6) to avoid division by zero
        weights.append(weight)
    weights = torch.tensor(weights, dtype=torch.float32)
    return weights


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch: int, log_every: int = 50):
    model.train()
    total_loss = 0.0
    global_step = epoch*len(dataloader)

    for step, batch in enumerate(tqdm(dataloader, desc=f"Training epoch {epoch}")):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = criterion(outputs.logits, batch['labels'])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # log batch training loss every 'log_every' steps
        if step % log_every == 0:
            mlflow.log_metric("train_batch_loss", loss.item(), step=global_step+step)
    
    return total_loss / len(dataloader)


def eval_model(model, dataloader, device, epoch):
    model.eval()
    all_labels=[]
    all_probs=[]
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Val Epoch {epoch}"):
            labels = batch["labels"].cpu().numpy()
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            probs = torch.sigmoid(logits).cpu().numpy()
            all_labels.append(labels)
            all_probs.append(probs)
    all_labels = np.concatenate(all_labels, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)

    metrics = {}
    for i, label_name in enumerate(LABEL_COLS):
        y_true = all_labels[:, i]
        y_score = all_probs[:, i]
        y_pred = (y_score >= 0.5).astype(int)

        # Guard in case there are no positives in val for a label
        if len(np.unique(y_true)) == 1:
            roc_auc = float("nan")
            pr_auc = float("nan")
        else:
            roc_auc = roc_auc_score(y_true, y_score)
            precision, recall, _ = precision_recall_curve(y_true, y_pred)
            pr_auc = auc(recall, precision)
        
        prec_val = precision_score(y_true, y_pred, zero_division='warn')
        rec_val = recall_score(y_true, y_pred, zero_division='warn')
        f1_val = f1_score(y_true, y_pred, zero_division='warn')

        # Create a binary label confusion matrix for each label
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        acc = accuracy_score(y_true, y_pred)
        metrics[f"{label_name}_roc_auc"] = roc_auc
        metrics[f"{label_name}_pr_auc"] = pr_auc
        metrics[f"{label_name}_precision"] = prec_val
        metrics[f"{label_name}_recall"] = rec_val
        metrics[f"{label_name}_f1"] = f1_val
        metrics[f"{label_name}_TP"] = tp
        metrics[f"{label_name}_FP"] = fp
        metrics[f"{label_name}_FN"] = fn
        metrics[f"{label_name}_TN"] = tn
        metrics[f"{label_name}_accuracy"] = acc
    
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

        for epoch in range(EPOCHS):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
            metrics = eval_model(model, val_loader, device, epoch)
            print(f"Epoch {epoch}: loss={train_loss:.4f}, metrics={metrics}")

            # log training loss and per-label metrics
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            for k, v in metrics.items():
                mlflow.log_metric(k, v, step=epoch)
        
        # Save model locally and to MLflow
        save_path = MODEL_DIR / "model"
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"Model and tokenizer saved to {save_path}")

        # Log the entire folder (weights + tokenizer + config) to MLflow
        mlflow.log_artifacts(str(save_path), artifact_path="full_model")


if __name__ == "__main__":
    main()


