"""Centralized configuration for the content moderation system.

Loads from config/training.yaml with environment variable overrides.
Env vars use the prefix CMS_ (Content Moderation System), e.g.:
  CMS_TRAINING__EPOCHS=3  (double underscore for nesting)
  CMS_TRAINING__BATCH_SIZE=32

In Colab or other environments, set env vars before the first call to
get_settings(), or call reload_settings() to pick up changes made after
the initial load.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "config" / "training.yaml"


def _load_yaml_config(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


class ModelConfig(BaseModel):
    name: str = "distilbert-base-uncased"
    num_labels: int = 2
    label_cols: list[str] = ["toxicity", "hate"]
    max_length: int = 256


class TrainingConfig(BaseModel):
    epochs: int = 1
    batch_size: int = 16
    learning_rate: float = 2e-5
    seed: int = 42
    mixed_precision: bool = True
    loss_fn: str = "bce"
    focal_gamma: float = 2.0


class DataConfig(BaseModel):
    raw_path: Path = Path("data/raw/jigsaw/train.csv")
    preprocessed_dir: Path = Path("data/preprocessed/text")
    test_size: float = 0.2
    val_size: float = 0.15


class PathsConfig(BaseModel):
    model_dir: Path = Path("models/text_toxicity/artifacts")
    sensitive_words_config: Path = Path("config/local_sensitive_words.json")


class BiasEvalConfig(BaseModel):
    templated_data_path: Path = Path("data/bias_eval/templated_lexical_bias.csv")
    batch_size: int = 32
    max_fpr_delta: float = 0.05


class OptimizationConfig(BaseModel):
    registered_model_name: str = "workspace.default.content-moderation-text"
    model_alias: str = "Production"
    onnx_dir: Path = Path("models/text_toxicity/onnx")
    quantized_dir: Path = Path("models/text_toxicity/quantized")
    validation_tolerance: float = 1e-4


class _YamlSource(PydanticBaseSettingsSource):
    def __init__(self, settings_cls: type[BaseSettings], path: Path) -> None:
        super().__init__(settings_cls)
        self._data = _load_yaml_config(path)

    def get_field_value(self, _field: Any, field_name: str) -> Any:
        return self._data.get(field_name), field_name, False

    def __call__(self) -> dict[str, Any]:
        return self._data


class Settings(BaseSettings):
    model_config = {"env_prefix": "CMS_", "env_nested_delimiter": "__"}

    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    bias_eval: BiasEvalConfig = Field(default_factory=BiasEvalConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)

    mlflow_tracking_uri: str = "file:./mlruns"

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        yaml_source = _YamlSource(settings_cls, _DEFAULT_CONFIG_PATH)
        # Priority: env vars > yaml file > defaults
        return env_settings, yaml_source


@lru_cache(maxsize=1)
def get_settings(config_path: Path = _DEFAULT_CONFIG_PATH) -> Settings:
    """Load settings from YAML config with env var overrides."""
    return Settings()


def reload_settings(config_path: Path = _DEFAULT_CONFIG_PATH) -> Settings:
    """Clear the cache and reload settings. Use in Colab after setting env vars."""
    get_settings.cache_clear()
    return get_settings(config_path)
