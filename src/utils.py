"""
Utility helpers: reproducibility, prediction normalisation, and metrics logging.
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path

import numpy as np
import torch

from .constants import (
    NEGATIVE_CLASS,
    POSITIVE_CLASS,
    _CONTROL_ALIASES,
    _DEMENTIA_ALIASES,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    """Fix all relevant random seeds for reproducibility.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info("Random seed fixed to %d.", seed)


# ---------------------------------------------------------------------------
# Prediction normalisation
# ---------------------------------------------------------------------------


def normalize_prediction(raw_pred: str) -> str:
    """Map a raw model output token to a canonical class label.

    Args:
        raw_pred: Raw string produced by the decoder.

    Returns:
        Either ``POSITIVE_CLASS`` or ``NEGATIVE_CLASS``.
    """
    token = raw_pred.strip().lower().split()[0] if raw_pred.strip() else ""
    if token in _DEMENTIA_ALIASES or POSITIVE_CLASS in token:
        return POSITIVE_CLASS
    if token in _CONTROL_ALIASES or NEGATIVE_CLASS in token:
        return NEGATIVE_CLASS
    # Ambiguous — default to negative (control) as the safer clinical fallback
    return NEGATIVE_CLASS


def normalize_label(raw_label: str) -> str:
    """Canonicalise a ground-truth label string.

    Args:
        raw_label: Raw label string from the CSV.

    Returns:
        Either ``POSITIVE_CLASS`` or ``NEGATIVE_CLASS``.
    """
    token = raw_label.strip().lower()
    if token in _DEMENTIA_ALIASES or POSITIVE_CLASS in token:
        return POSITIVE_CLASS
    return NEGATIVE_CLASS


# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logger with a sensible format.

    Args:
        level: Python logging level (default: ``logging.INFO``).
    """
    logging.basicConfig(
        format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=level,
    )


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------


def ensure_dir(path: str | Path) -> Path:
    """Create *path* (and any missing parents) if it does not exist.

    Args:
        path: Directory path to create.

    Returns:
        The resolved ``Path`` object.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
