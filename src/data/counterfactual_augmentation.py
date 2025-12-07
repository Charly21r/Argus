import json
from pathlib import Path
import random
import pandas as pd

SENSITIVE_CFG_PATH = Path("config/local_sensitive_words.json")

def load_lexical_groups(path: Path) -> list[str]:
    if not path.exists():
        print(f"Warning! Lexical Bias Groups not found at {path}")
        return []
    with open(SENSITIVE_CFG_PATH) as f:
        sensitive_cfg = json.load(f)
    groups = sensitive_cfg.get("counterfactual_swapping", [])
    return groups

LEXICAL_GROUPS = load_lexical_groups(SENSITIVE_CFG_PATH)



def _swap_words(s, x, y):
    return y.join(part.replace(y, x) for part in s.split(x))

def _swap_groups_in_text(
    text: str,
    rng: random.Random
) -> str | None:
    """
        Given a text, find all the keywords that appear and swap each one
        with a randomly chosen alternative from LEXICAL_GROUPS.
    """
    lowered = text.lower()
    found_tokens = [tok for tok in LEXICAL_GROUPS.keys() if tok in lowered]

    if not found_tokens:
        return None

    new_text = text
    for tok in found_tokens:
        candidates = LEXICAL_GROUPS[tok]
        replacement = rng.choice(candidates)
        new_text = _swap_words(new_text, tok, replacement)

    return new_text

def augment_with_counterfactuals(
    df: pd.DataFrame,
    text_col: str = "text",
    max_augment_per_sample: int = 1,
    frac: float = 1.0,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Create group-swapped counterfactual examples from a subset of the rows.
    
    :param df: Original training dataframe
    :type df: pd.Dataframe
    :param text_col: Name of the text column
    :type text_col: str
    :param max_augment_per_sample: Max. number of counterfactuals per row
    :type max_augment_per_sample: int
    :param frac: Fraction of elegible rows to augment
    :type frac: float
    :param random_state: Random state
    :type random_state: int
    :return: New dataframe that contains the original df + counterfactual rows
    :rtype: DataFrame
    """
    rng = random.Random(random_state)

    df = df.copy()
    df[text_col] = df[text_col].astype(str)

    mask_group = df[text_col].str.lower().apply(
        lambda t: any(tok in t for tok in LEXICAL_GROUPS.keys())
    )

    elegible_df = df[mask_group]

    if frac < 1.0:
        elegible_df = elegible_df.sample(frac=frac, random_state=random_state)
    
    cf_rows = []

    for _, row in elegible_df.iterrows():
        original_text = row[text_col]

        for _ in range(max_augment_per_sample):
            swapped = _swap_groups_in_text(original_text, rng)
            if swapped is None or swapped == original_text:
                continue
            
            new_row = row.copy()
            new_row[text_col] = swapped
            cf_rows.append(new_row)
    
    if not cf_rows:
        return df
    
    cf_df = pd.DataFrame(cf_rows)
    augmented_df = pd.concat([df, cf_df], axis=0).reset_index(drop=True)
    return augmented_df
    