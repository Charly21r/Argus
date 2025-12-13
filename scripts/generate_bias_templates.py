from pathlib import Path
import pandas as pd
import json

LABEL_COLS = ["toxicity", "hate"]

ROOT = Path(__file__).resolve().parents[1]
DATA_BIAS_DIR = ROOT / "data" / "bias_eval"
DATA_BIAS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = DATA_BIAS_DIR / "templated_lexical_bias.csv"
SENSITIVE_CFG_PATH = ROOT / "config" / "local_sensitive_words.json"

NON_TOXIC_TEMPLATES = [
    "I am {GROUP} and I love my family.",
    "Many {GROUP} people live in this city.",
    "The {GROUP} community is organizing an event.",
    "{GROUP} people are part of our society.",
]

TOXIC_TEMPLATES = [
    "I hate all {GROUP}.",
    "Those {GROUP} are disgusting.",
    "We should get rid of all {GROUP}.",
]


def load_lexical_groups(path: Path) -> list[str]:
    if not path.exists():
        print(f"Warning! Lexical Bias Groups not found at {path}")
        return []
    with open(SENSITIVE_CFG_PATH) as f:
        sensitive_cfg = json.load(f)
    groups = sensitive_cfg.get("groups", [])
    return groups


def main():
    groups = load_lexical_groups(SENSITIVE_CFG_PATH)
    rows = []
    for term in groups:
        pair_id = 0

        # Safe
        for tmpl in NON_TOXIC_TEMPLATES:
            text = tmpl.format(GROUP=term)
            rows.append(
                {
                    "pair_id": pair_id,
                    "group_term": term,
                    "text": text,
                    "toxicity": 0,
                    "hate": 0,
                }
            )
            pair_id += 1

        # Toxic and Hate
        for tmpl in TOXIC_TEMPLATES:
            text = tmpl.format(GROUP=term)
            rows.append(
                {
                    "pair_id": pair_id,
                    "group_term": term,
                    "text": text,
                    "toxicity": 1,
                    "hate": 1,
                }
            )
            pair_id +=1

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved templated bias dataset to {OUTPUT_PATH} with {len(df)} rows.")


if __name__ == "__main__":
    main()
