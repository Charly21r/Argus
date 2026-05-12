"""Shared lexicon loading for sensitive word groups.

The config file (config/local_sensitive_words.json) has two sections:
  - "groups": flat list of identity terms used for bias evaluation
  - "counterfactual_swapping": dict mapping term -> list of alternatives for CDA
"""

import json
import logging
from pathlib import Path
from typing import Any, cast

logger = logging.getLogger(__name__)

DEFAULT_SENSITIVE_CFG_PATH = Path("config/local_sensitive_words.json")


def load_sensitive_config(path: Path = DEFAULT_SENSITIVE_CFG_PATH) -> dict[str, Any]:
    """Load the full sensitive words config file."""
    if not path.exists():
        logger.warning("Lexical bias config not found at %s", path)
        return {}
    with open(path) as f:
        data = json.load(f)

    return cast(dict[str, Any], data)


def load_group_terms(path: Path = DEFAULT_SENSITIVE_CFG_PATH) -> list[str]:
    """Load flat list of identity group terms for bias evaluation."""
    cfg = load_sensitive_config(path)

    groups = cfg.get("groups", [])
    if not isinstance(groups, list):
        return []

    return groups


def load_counterfactual_swapping(path: Path = DEFAULT_SENSITIVE_CFG_PATH) -> dict[str, list[str]]:
    """Load counterfactual swapping dict (term -> alternatives) for CDA."""
    cfg = load_sensitive_config(path)

    raw = cfg.get("counterfactual_swapping", {})

    if not isinstance(raw, dict):
        return {}

    result: dict[str, list[str]] = {}
    for k, v in raw.items():
        if isinstance(v, list):
            result[k] = [str(x) for x in v]
    return result
