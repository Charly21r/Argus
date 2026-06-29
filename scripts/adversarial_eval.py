import argparse
from pathlib import Path
from random import Random

import pandas as pd
import torch
from sklearn.metrics import recall_score
from src.config import get_settings
from src.data.adversarial import ATTACKS
from src.training.train_text_model import TextClassificationDataset, eval_model
from src.utils.loading import LABEL_COLS, NUM_LABELS, load_model_and_tokenizer, load_thresholds
from torch.utils.data import DataLoader


def apply_attack(df, attack_fn, rng, intensity=1.0, text_col="text"):
    """Return a copy of df with attack_fn applied to every text row."""
    df = df.copy()
    df[text_col] = df[text_col].astype(str).apply(lambda t: attack_fn(t, rng, intensity))
    return df


def recall_per_label(labels, probs, thresholds) -> dict[str, float]:
    """Per-label recall at the calibrated thresholds."""
    out = {}
    for i, label in enumerate(LABEL_COLS):
        y_true = labels[:, i]
        y_pred = (probs[:, i] >= thresholds[label]).astype(int)
        out[label] = float(recall_score(y_true, y_pred, zero_division=0))
    return out


def score(df, model, tokenizer, device, max_length, batch_size):
    """Tokenize df, run eval_model, return (labels, probs)."""
    ds = TextClassificationDataset(df, tokenizer, LABEL_COLS, max_length)
    loader = DataLoader(ds, batch_size=batch_size)
    labels, probs, _ = eval_model(model, loader, device, NUM_LABELS, amp_enabled=False)
    return labels, probs


def run_adversarial_eval(test_path, model_path, thresholds_path, seed=42, sample=None, max_length=256, batch_size=32):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    df = pd.read_csv(test_path)
    if sample is not None:
        df = df.sample(n=min(sample, len(df)), random_state=seed).reset_index(drop=True)
    print(f"Evaluating on {len(df)} rows")

    model, tokenizer = load_model_and_tokenizer(Path(model_path), device)
    thresholds = load_thresholds(Path(thresholds_path))

    results = {}
    clean_labels, clean_probs = score(df, model, tokenizer, device, max_length, batch_size)
    results["clean"] = recall_per_label(clean_labels, clean_probs, thresholds)

    for name, fn in ATTACKS.items():
        adv_df = apply_attack(df, fn, Random(seed))  # fresh rng per attack
        labels, probs = score(adv_df, model, tokenizer, device, max_length, batch_size)
        results[name] = recall_per_label(labels, probs, thresholds)

    table = pd.DataFrame(results).T  # rows = attacks, cols = labels
    for label in LABEL_COLS:
        table[f"{label}_delta"] = table[label] - table.loc["clean", label]
    return table


def main():
    cfg = get_settings()
    parser = argparse.ArgumentParser(description="Adversarial robustness evaluation")
    parser.add_argument("--test-path", default="data/preprocessed/text/test.csv")
    parser.add_argument("--model-path", default="models/text_toxicity/hf_export")
    parser.add_argument("--thresholds-path", default="models/text_toxicity/onnx/thresholds.json")
    parser.add_argument("--sample", type=int, default=None, help="Subsample N rows for fast iteration")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    table = run_adversarial_eval(
        test_path=args.test_path,
        model_path=args.model_path,
        thresholds_path=args.thresholds_path,
        seed=args.seed,
        sample=args.sample,
        max_length=cfg.model.max_length,
        batch_size=args.batch_size,
    )

    print("\n===== Adversarial Evaluation =====")
    with pd.option_context("display.float_format", "{:.3f}".format):
        print(table)


if __name__ == "__main__":
    main()
