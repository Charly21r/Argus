"""Unit tests for counterfactual augmentation functions."""

import random
from unittest.mock import patch

import pandas as pd
import pytest

from data.counterfactual_augmentation import _swap_groups_in_text, augment_with_counterfactuals


MOCK_LEXICAL_GROUPS = {
    "christian": ["muslim", "jewish", "hindu"],
    "muslim": ["christian", "jewish", "hindu"],
}


@patch("data.counterfactual_augmentation.LEXICAL_GROUPS", MOCK_LEXICAL_GROUPS)
class TestSwapGroupsInText:
    def test_swap_found_token(self):
        rng = random.Random(42)
        result = _swap_groups_in_text("I am christian and I am proud.", rng)
        assert result is not None
        assert "christian" not in result.lower()

    def test_no_match_returns_none(self):
        rng = random.Random(42)
        result = _swap_groups_in_text("This text has no identity terms.", rng)
        assert result is None

    def test_deterministic_with_seed(self):
        rng1 = random.Random(123)
        rng2 = random.Random(123)
        text = "The christian community is large."
        assert _swap_groups_in_text(text, rng1) == _swap_groups_in_text(text, rng2)


@patch("data.counterfactual_augmentation.LEXICAL_GROUPS", MOCK_LEXICAL_GROUPS)
class TestAugmentWithCounterfactuals:
    def test_augmented_has_more_rows(self):
        df = pd.DataFrame({
            "text": ["I am christian.", "I am muslim.", "Hello world."],
            "toxicity": [0, 0, 0],
            "hate": [0, 0, 0],
        })
        result = augment_with_counterfactuals(df, text_col="text", frac=1.0)
        assert len(result) >= len(df)

    def test_labels_preserved(self):
        df = pd.DataFrame({
            "text": ["I am christian."],
            "toxicity": [1],
            "hate": [0],
        })
        result = augment_with_counterfactuals(df, text_col="text", frac=1.0)
        # All rows (original + augmented) should keep the same labels
        assert (result["toxicity"] == 1).all()
        assert (result["hate"] == 0).all()

    def test_no_match_returns_original(self):
        df = pd.DataFrame({
            "text": ["No identity terms here.", "Just a sentence."],
            "toxicity": [0, 0],
            "hate": [0, 0],
        })
        result = augment_with_counterfactuals(df, text_col="text")
        assert len(result) == len(df)

    def test_frac_limits_augmentation(self):
        df = pd.DataFrame({
            "text": [f"I am christian {i}." for i in range(100)],
            "toxicity": [0]*100,
            "hate": [0]*100,
        })
        result_full = augment_with_counterfactuals(df, frac=1.0, random_state=42)
        result_half = augment_with_counterfactuals(df, frac=0.5, random_state=42)
        assert len(result_half) < len(result_full)
