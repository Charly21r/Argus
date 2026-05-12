"""Shared fixtures for the test suite."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure src/ is on the path so imports work like they do in the project
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture
def sample_df():
    """Minimal DataFrame mimicking Jigsaw preprocessed data.

    Needs >=20 rows so stratified_split (test_size=0.2, val_size=0.15) produces
    test>=3 and val>=3, satisfying sklearn's requirement of >=n_classes (3) per split.
    """
    return pd.DataFrame(
        {
            "text": [
                "This is a normal comment.",
                "You are terrible and disgusting.",
                "I hate all of them.",
                "Great weather today!",
                "This is an offensive slur.",
                "Have a nice day everyone.",
                "Shut up you idiot.",
                "What a lovely morning.",
                "You should die.",
                "Nice work on the project.",
                "Good morning everyone.",
                "You are disgusting trash.",
                "I despise every one of you.",
                "Lovely day for a walk.",
                "Drop dead already.",
                "This is so wonderful.",
                "Absolute garbage human being.",
                "What a great team effort.",
                "Go kill yourself.",
                "Beautiful sunset tonight.",
            ],
            "toxicity": [0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0],
            "hate": [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
            "safe": [1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    )


@pytest.fixture
def sample_labels():
    """(N, 2) array of ground truth labels."""
    return np.array(
        [
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 0],
            [1, 0],
        ],
        dtype=np.float32,
    )


@pytest.fixture
def sample_probs():
    """(N, 2) array of predicted probabilities."""
    return np.array(
        [
            [0.1, 0.05],
            [0.9, 0.3],
            [0.85, 0.92],
            [0.2, 0.1],
            [0.7, 0.15],
        ],
        dtype=np.float32,
    )


@pytest.fixture
def jigsaw_raw_df():
    """Minimal DataFrame mimicking raw Jigsaw CSV format."""
    return pd.DataFrame(
        {
            "id": range(20),
            "comment_text": [f"comment {i}" for i in range(20)],
            "toxic": [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            "severe_toxic": [0] * 20,
            "obscene": [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "threat": [0] * 20,
            "insult": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "identity_hate": [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        }
    )
