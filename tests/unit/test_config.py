"""Unit tests for centralized configuration."""

from pathlib import Path

import pytest

from config import Settings, get_settings


class TestSettings:
    def test_defaults_load(self):
        settings = Settings()
        assert settings.model.name == "distilbert-base-uncased"
        assert settings.model.num_labels == 2
        assert settings.training.epochs == 1
        assert settings.training.batch_size == 16
        assert settings.training.learning_rate == 2e-5
        assert settings.data.test_size == 0.2

    def test_label_cols(self):
        settings = Settings()
        assert settings.model.label_cols == ["toxicity", "hate"]

    def test_paths_are_path_objects(self):
        settings = Settings()
        assert isinstance(settings.paths.model_dir, Path)
        assert isinstance(settings.data.preprocessed_dir, Path)

    def test_get_settings_returns_same_instance(self):
        # lru_cache should return the same object
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2
