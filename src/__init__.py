"""
phi4-ad-detection — Phi-4 Multimodal fine-tuning for Alzheimer's Disease detection.
"""

from .collator import custom_collate_fn
from .constants import (
    AD_CLASSIFICATION_PROMPT_TEMPLATE,
    ANSWER_SUFFIX,
    CLASS_LABELS,
    NEGATIVE_CLASS,
    POSITIVE_CLASS,
    _IGNORE_INDEX,
)
from .dataset import ADClassificationDataset
from .evaluate import MultipleTokenBatchStoppingCriteria, evaluate
from .model import create_model
from .utils import configure_logging, normalize_label, normalize_prediction, set_seed

__all__ = [
    "AD_CLASSIFICATION_PROMPT_TEMPLATE",
    "ANSWER_SUFFIX",
    "CLASS_LABELS",
    "NEGATIVE_CLASS",
    "POSITIVE_CLASS",
    "_IGNORE_INDEX",
    "ADClassificationDataset",
    "MultipleTokenBatchStoppingCriteria",
    "evaluate",
    "create_model",
    "custom_collate_fn",
    "configure_logging",
    "normalize_label",
    "normalize_prediction",
    "set_seed",
]
