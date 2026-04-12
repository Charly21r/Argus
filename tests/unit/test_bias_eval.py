"""Unit tests for bias evaluation functions."""

import numpy as np
import pandas as pd
import pytest

from scripts.run_bias_eval import compute_identity_sensitivity


class TestComputeIdentitySensitivity:
    def test_basic_output_keys(self):
        df = pd.DataFrame({
            "text": ["a", "b", "c", "d"],
            "pair_id": [0, 0, 1, 1],
        })
        probs = np.array([
            [0.3, 0.2],
            [0.3, 0.2],
            [0.8, 0.9],
            [0.8, 0.9],
        ], dtype=np.float32)
        thresholds = {"toxicity": 0.5, "hate": 0.5}

        result = compute_identity_sensitivity(df, probs, thresholds)
        assert "num_samples" in result
        assert "toxicity_flip_rate" in result
        assert "hate_flip_rate" in result
        assert "toxicity_mean_abs_prob_delta" in result

    def test_identical_pairs_have_zero_flip_rate(self):
        df = pd.DataFrame({
            "text": ["a", "b"],
            "pair_id": [0, 0],
        })
        probs = np.array([[0.9, 0.8], [0.9, 0.8]], dtype=np.float32)
        thresholds = {"toxicity": 0.5, "hate": 0.5}

        result = compute_identity_sensitivity(df, probs, thresholds)
        assert result["toxicity_flip_rate"] == 0.0
        assert result["hate_flip_rate"] == 0.0

    def test_divergent_pairs_have_nonzero_flip_rate(self):
        df = pd.DataFrame({
            "text": ["a", "b"],
            "pair_id": [0, 0],
        })
        # One above threshold, one below
        probs = np.array([[0.9, 0.8], [0.1, 0.1]], dtype=np.float32)
        thresholds = {"toxicity": 0.5, "hate": 0.5}

        result = compute_identity_sensitivity(df, probs, thresholds)
        assert result["toxicity_flip_rate"] == 1.0
        assert result["hate_flip_rate"] == 1.0

    def test_missing_pair_id_raises(self):
        df = pd.DataFrame({"text": ["a"]})
        probs = np.array([[0.5, 0.5]], dtype=np.float32)
        with pytest.raises(ValueError, match="pair_id"):
            compute_identity_sensitivity(df, probs, {"toxicity": 0.5, "hate": 0.5})
