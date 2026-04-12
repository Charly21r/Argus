"""Unit tests for data preprocessing functions."""

import pandas as pd
import pytest

from data.jigsaw_preprocessing import map_to_policy_labels, stratified_split


class TestMapToPolicyLabels:
    def test_output_columns(self, jigsaw_raw_df):
        result = map_to_policy_labels(jigsaw_raw_df)
        assert list(result.columns) == ["text", "toxicity", "hate", "safe"]

    def test_toxicity_label_aggregation(self, jigsaw_raw_df):
        result = map_to_policy_labels(jigsaw_raw_df)
        # Row 0: toxic=1, obscene=1, insult=1 → toxicity=1
        assert result.iloc[0]["toxicity"] == 1
        # Row 1: all zero → toxicity=0
        assert result.iloc[1]["toxicity"] == 0

    def test_hate_label(self, jigsaw_raw_df):
        result = map_to_policy_labels(jigsaw_raw_df)
        # Row 3: identity_hate=1 → hate=1
        assert result.iloc[3]["hate"] == 1
        # Row 0: identity_hate=0 → hate=0
        assert result.iloc[0]["hate"] == 0

    def test_safe_is_inverse(self, jigsaw_raw_df):
        result = map_to_policy_labels(jigsaw_raw_df)
        for _, row in result.iterrows():
            expected_safe = int(row["toxicity"] == 0 and row["hate"] == 0)
            assert row["safe"] == expected_safe

    def test_does_not_mutate_input(self, jigsaw_raw_df):
        original_cols = list(jigsaw_raw_df.columns)
        map_to_policy_labels(jigsaw_raw_df)
        assert list(jigsaw_raw_df.columns) == original_cols

    def test_text_column_renamed(self, jigsaw_raw_df):
        result = map_to_policy_labels(jigsaw_raw_df)
        assert "text" in result.columns
        assert "comment_text" not in result.columns


class TestStratifiedSplit:
    def test_split_sizes(self, sample_df):
        train, val, test = stratified_split(sample_df)
        assert len(train) + len(val) + len(test) == len(sample_df)

    def test_no_strat_column_leak(self, sample_df):
        train, val, test = stratified_split(sample_df)
        for df in [train, val, test]:
            assert "strat_tmp_column" not in df.columns

    def test_no_row_overlap(self, sample_df):
        train, val, test = stratified_split(sample_df)
        all_indices = set(train.index) | set(val.index) | set(test.index)
        assert len(all_indices) == len(train) + len(val) + len(test)
