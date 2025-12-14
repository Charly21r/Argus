from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from data.counterfactual_augmentation import augment_with_counterfactuals

RAW_PATH = Path('data/raw/jigsaw/train.csv')
OUT_DIR = Path('data/preprocessed/text')
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.15


def load_jigsaw() -> pd.DataFrame:
    df = pd.read_csv(RAW_PATH)
    return df

def map_to_policy_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    toxicity_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult"]
    hate_cols = ["identity_hate"]

    df["toxicity"] = (df[toxicity_cols].sum(axis=1) > 0).astype(int)
    df["hate"] = (df[hate_cols].sum(axis=1) > 0).astype(int)
    
    df["safe"] = ( (df["toxicity"] == 0) & (df["hate"] == 0) ).astype(int)

    return df[["comment_text", "toxicity", "hate", "safe"]].rename(
        columns={"comment_text": "text"}
    )

def stratified_split(df: pd.DataFrame):
    # Create temporal new label for stratification
    df["strat_tmp_column"] = df["toxicity"].astype(str) + df["hate"].astype(str)

    # Split stratifying by the toxicity label
    train_val_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        stratify=df["strat_tmp_column"],
        random_state=RANDOM_STATE
    )
    
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=VAL_SIZE / (1 - TEST_SIZE),
        stratify=train_val_df["strat_tmp_column"],
        random_state=RANDOM_STATE
    )

    # Print stats on hate label too
    print("Train hate rate:", train_df["hate"].mean())
    print("Val hate rate:", val_df["hate"].mean())
    print("Test hate rate:", test_df["hate"].mean())

    # Drop the temporal column
    train_df = train_df.drop(columns=["strat_tmp_column"])
    val_df = val_df.drop(columns=["strat_tmp_column"])
    test_df = test_df.drop(columns=["strat_tmp_column"])

    return train_df, val_df, test_df

def main():
    df = load_jigsaw()
    df_unified = map_to_policy_labels(df)
    df_train, df_val, df_test = stratified_split(df_unified)

    df_train = augment_with_counterfactuals(
        df_train,
        text_col="text",
        max_augment_per_sample=1,
        frac=0.7,
        random_state=RANDOM_STATE
    )

    df_train.to_csv(OUT_DIR / "train.csv", index=False)
    df_val.to_csv(OUT_DIR / "val.csv", index=False)
    df_test.to_csv(OUT_DIR / "test.csv", index=False)

    # Save the unified data to perform initial EDA
    df_unified.to_csv(OUT_DIR / "jigsaw_unified.csv")
    
    print(f"Saved train, test, and val data to {OUT_DIR}")


if __name__ == "__main__":
    main()