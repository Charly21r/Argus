"""Data validation tests — schema checks on preprocessed CSVs.

These tests verify the structure and invariants of the data pipeline output.
They run against fixtures (not real data files) to keep tests fast and self-contained.
"""

import pandas as pd
import pytest


REQUIRED_COLS = ["text", "toxicity", "hate", "safe"]
LABEL_COLS = ["toxicity", "hate", "safe"]


class TestPreprocessedDataSchema:
    def test_required_columns_present(self, sample_df):
        for col in REQUIRED_COLS:
            assert col in sample_df.columns, f"Missing column: {col}"

    def test_label_columns_are_binary(self, sample_df):
        for col in LABEL_COLS:
            unique_vals = set(sample_df[col].unique())
            assert unique_vals <= {0, 1}, f"{col} has non-binary values: {unique_vals}"

    def test_safe_is_consistent(self, sample_df):
        """safe should be 1 iff toxicity=0 AND hate=0."""
        for _, row in sample_df.iterrows():
            expected = int(row["toxicity"] == 0 and row["hate"] == 0)
            assert row["safe"] == expected, (
                f"safe={row['safe']} but toxicity={row['toxicity']}, hate={row['hate']}"
            )

    def test_no_null_text(self, sample_df):
        assert sample_df["text"].notna().all(), "Found null values in text column"

    def test_no_empty_text(self, sample_df):
        assert (sample_df["text"].str.strip() != "").all(), "Found empty text values"

    def test_no_duplicate_rows(self, sample_df):
        dupes = sample_df.duplicated(subset=["text"])
        assert not dupes.any(), f"Found {dupes.sum()} duplicate rows"

    def test_text_column_is_string(self, sample_df):
        assert sample_df["text"].dtype == object
