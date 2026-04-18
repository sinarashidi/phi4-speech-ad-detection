"""
Task-specific constants for Alzheimer's Disease (AD) detection with Phi-4 Multimodal.
"""

# ---------------------------------------------------------------------------
# Classification prompt
# ---------------------------------------------------------------------------

AD_CLASSIFICATION_PROMPT_TEMPLATE: str = (
    'Transcription: "{transcription}"\n\n'
    "Based on the speech audio and its transcription, "
    "classify the speaker as dementia (Alzheimer's Disease or Related Dementia) or control "
    "(Cognitively Normal) with a single word: 'dementia' or 'control'."
)

# ---------------------------------------------------------------------------
# Class labels
# ---------------------------------------------------------------------------

POSITIVE_CLASS: str = "dementia"
NEGATIVE_CLASS: str = "control"
CLASS_LABELS: list[str] = [POSITIVE_CLASS, NEGATIVE_CLASS]

# Aliases accepted during prediction normalisation
_DEMENTIA_ALIASES: frozenset[str] = frozenset({"ad", "dem", "demen", "dementia"})
_CONTROL_ALIASES: frozenset[str] = frozenset({"cn", "ctrl", "normal", "cogn", "control"})

# ---------------------------------------------------------------------------
# Phi-4 specific tokens
# ---------------------------------------------------------------------------

ANSWER_SUFFIX: str = "<|end|><|endoftext|>"
_IGNORE_INDEX: int = -100
