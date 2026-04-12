"""Unit tests for training utility functions."""

import numpy as np
import pandas as pd
import torch
import pytest

from training.train_text_model import (
    calculate_pos_weights,
    find_optimal_thresholds,
    build_group_masks,
)


class TestCalculatePosWeights:
    def test_balanced_dataset(self):
        df = pd.DataFrame({"toxicity": [0, 0, 1, 1], "hate": [0, 1, 0, 1]})
        weights = calculate_pos_weights(df, ["toxicity", "hate"])
        assert isinstance(weights, torch.Tensor)
        assert weights.shape == (2,)
        # Balanced → weight ≈ 1.0
        assert abs(weights[0].item() - 1.0) < 0.01
        assert abs(weights[1].item() - 1.0) < 0.01

    def test_imbalanced_dataset(self):
        df = pd.DataFrame({"toxicity": [0]*9 + [1], "hate": [0]*8 + [1]*2})
        weights = calculate_pos_weights(df, ["toxicity", "hate"])
        # toxicity: 9 neg / 1 pos = 9.0
        assert abs(weights[0].item() - 9.0) < 0.01
        # hate: 8 neg / 2 pos = 4.0
        assert abs(weights[1].item() - 4.0) < 0.01

    def test_all_positive(self):
        df = pd.DataFrame({"toxicity": [1, 1, 1]})
        weights = calculate_pos_weights(df, ["toxicity"])
        # 0 neg / 3 pos ≈ 0
        assert weights[0].item() < 0.01


class TestFindOptimalThresholds:
    def test_returns_dict_with_label_names(self, sample_labels, sample_probs):
        thresholds = find_optimal_thresholds(sample_labels, sample_probs)
        assert "toxicity" in thresholds
        assert "hate" in thresholds

    def test_thresholds_in_range(self, sample_labels, sample_probs):
        thresholds = find_optimal_thresholds(sample_labels, sample_probs)
        for v in thresholds.values():
            assert 0.0 < v < 1.0

    def test_custom_search_space(self, sample_labels, sample_probs):
        space = np.array([0.3, 0.5, 0.7])
        thresholds = find_optimal_thresholds(sample_labels, sample_probs, search_space=space)
        for v in thresholds.values():
            assert v in space


class TestBuildGroupMasks:
    def test_basic_matching(self):
        texts = ["I love cats", "dogs are great", "nothing here"]
        mask = build_group_masks(texts, ["cats", "dogs"])
        assert mask.tolist() == [True, True, False]

    def test_case_insensitive(self):
        texts = ["CATS are fun", "Dogs Rule"]
        mask = build_group_masks(texts, ["cats", "dogs"])
        assert mask.tolist() == [True, True]

    def test_empty_keywords(self):
        texts = ["hello world"]
        mask = build_group_masks(texts, [])
        assert mask.tolist() == [False]

    def test_empty_texts(self):
        mask = build_group_masks([], ["cats"])
        assert len(mask) == 0

    def test_substring_matching(self):
        texts = ["concatenate these strings"]
        mask = build_group_masks(texts, ["cat"])
        # "cat" appears as substring of "concatenate"
        assert mask.tolist() == [True]
