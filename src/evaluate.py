"""
Evaluation loop and stopping criteria for Phi-4 AD classification.

The ``evaluate`` function is shared by both the training pipeline (called
inside ``MemoryEfficientTrainer``) and the standalone ``test.py`` script.
"""

from __future__ import annotations

import json
import logging
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.utils.data
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from tqdm import tqdm
from transformers import BatchFeature, StoppingCriteria, StoppingCriteriaList

from .collator import custom_collate_fn
from .constants import ANSWER_SUFFIX, NEGATIVE_CLASS, POSITIVE_CLASS, _IGNORE_INDEX
from .utils import normalize_label, normalize_prediction

logger = logging.getLogger(__name__)

# Maximum number of samples evaluated in a single call (guards against
# accidentally running inference over huge splits during mid-training checks).
_DEFAULT_MAX_EVAL_SAMPLES: int = 100


# ---------------------------------------------------------------------------
# Stopping criteria
# ---------------------------------------------------------------------------


class MultipleTokenBatchStoppingCriteria(StoppingCriteria):
    """Stop generation as soon as every sequence in the batch has produced one
    of the provided stop tokens.

    Args:
        stop_tokens: Tensor of shape ``(num_stop_tokens, token_length)``
            containing token sequences that signal end-of-generation.
        batch_size: Number of sequences in the batch.
    """

    def __init__(self, stop_tokens: torch.LongTensor, batch_size: int = 1) -> None:
        self.stop_tokens = stop_tokens
        self.max_stop_tokens = stop_tokens.shape[-1]
        self.stop_tokens_idx = torch.zeros(
            batch_size, dtype=torch.long, device=stop_tokens.device
        )

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any
    ) -> bool:
        generated = torch.eq(
            input_ids[:, -self.max_stop_tokens :].unsqueeze(1), self.stop_tokens
        )
        equal = torch.all(generated, dim=2)
        finished = torch.any(equal, dim=1)
        mask = self.stop_tokens_idx == 0
        self.stop_tokens_idx[finished & mask] = input_ids.shape[-1]
        return bool(torch.all(self.stop_tokens_idx > 0))


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------


def _compute_bce_loss(
    predictions: list[str],
    true_labels: list[str],
) -> float:
    """Compute a heuristic binary cross-entropy loss from hard predictions.

    Because the model produces text (not calibrated probabilities) we assign
    confidence 0.9 to correct predictions and 0.1 to incorrect ones.

    Args:
        predictions: List of normalised predicted labels.
        true_labels: List of normalised ground-truth labels.

    Returns:
        Scalar BCE loss value, or ``float('inf')`` if the input is empty.
    """
    if not predictions:
        return float("inf")

    probs, targets = [], []
    for pred, true in zip(predictions, true_labels):
        is_positive = true == POSITIVE_CLASS
        targets.append(1.0 if is_positive else 0.0)
        if pred == POSITIVE_CLASS:
            probs.append(0.9 if is_positive else 0.1)
        else:
            probs.append(0.1 if is_positive else 0.9)

    p = np.clip(np.array(probs), 1e-7, 1 - 1e-7)
    y = np.array(targets)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _build_metrics(
    raw_preds: list[str],
    raw_labels: list[str],
) -> dict[str, Any]:
    """Align, normalise, and compute all evaluation metrics.

    Args:
        raw_preds: Raw model-generated strings (one per sample).
        raw_labels: Raw decoded ground-truth strings (one per sample).

    Returns:
        Dict containing ``accuracy``, ``f1_weighted``,
        ``f1_dementia``, ``precision_weighted``, ``recall_weighted``,
        ``bce_loss``, ``num_samples``, ``raw_predictions``, and
        ``true_labels``.
    """
    preds = [normalize_prediction(p) for p in raw_preds]
    labels = [normalize_label(l) for l in raw_labels]

    if not labels:
        logger.warning("No valid labels found — returning zero metrics.")
        return {
            "accuracy": 0.0,
            "f1_weighted": 0.0,
            f"f1_{POSITIVE_CLASS}": 0.0,
            "precision_weighted": 0.0,
            "recall_weighted": 0.0,
            "bce_loss": float("inf"),
            "num_samples": 0,
            "raw_predictions": raw_preds,
            "true_labels": raw_labels,
        }

    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    f1_ad = f1_score(labels, preds, pos_label=POSITIVE_CLASS, average="binary", zero_division=0)
    bce = _compute_bce_loss(preds, labels)

    return {
        "accuracy": float(accuracy),
        "f1_weighted": float(f1),
        f"f1_{POSITIVE_CLASS}": float(f1_ad),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "bce_loss": bce,
        "num_samples": len(labels),
        "raw_predictions": raw_preds,
        "true_labels": raw_labels,
    }


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate(
    model: Any,
    processor: Any,
    eval_dataset: torch.utils.data.Dataset,
    save_path: Path | None = None,
    disable_tqdm: bool = False,
    eval_batch_size: int = 1,
    max_eval_samples: int | None = None,
    accelerator: Any | None = None,
    device: torch.device | None = None,
) -> dict[str, Any] | None:
    """Run generation-based evaluation on an AD classification dataset.

    Args:
        model: The (fine-tuned) Phi-4 model.
        processor: Corresponding multimodal processor.
        eval_dataset: Dataset whose ``__getitem__`` matches
            ``ADClassificationDataset`` output format.
        save_path: Optional JSON path to persist full results.
        disable_tqdm: Suppress the progress bar.
        eval_batch_size: Batch size for the evaluation DataLoader.
        max_eval_samples: Hard cap on evaluated samples.  Defaults to
            ``_DEFAULT_MAX_EVAL_SAMPLES``.
        accelerator: ``Accelerator`` instance (pass ``None`` for single-GPU /
            CPU inference without Accelerate).
        device: Target device. Inferred from ``accelerator`` or model
            parameters when ``None``.

    Returns:
        Metrics dict on the main process; ``None`` on non-main processes.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if device is None:
        device = accelerator.device if accelerator else next(model.parameters()).device

    model.eval()

    # ------------------------------------------------------------------ #
    # Sub-sample if requested
    # ------------------------------------------------------------------ #
    cap = max_eval_samples if max_eval_samples is not None else _DEFAULT_MAX_EVAL_SAMPLES
    indices = list(range(len(eval_dataset)))
    if len(indices) > cap:
        import random
        random.shuffle(indices)
        indices = indices[:cap]
    subset = torch.utils.data.Subset(eval_dataset, indices)
    logger.info("Evaluating %d / %d samples.", len(subset), len(eval_dataset))

    # ------------------------------------------------------------------ #
    # DataLoader
    # ------------------------------------------------------------------ #
    collate_fn = partial(custom_collate_fn, processor=processor)
    dataloader = torch.utils.data.DataLoader(
        subset,
        batch_size=eval_batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=False,
    )
    if accelerator:
        dataloader = accelerator.prepare(dataloader)

    # ------------------------------------------------------------------ #
    # Stop tokens
    # ------------------------------------------------------------------ #
    stop_strs = sorted({s for s in ["<|end|>", processor.tokenizer.eos_token] if s})
    stop_ids = processor.tokenizer(
        stop_strs, add_special_tokens=False, return_tensors="pt", padding=True
    ).input_ids.to(device)

    # ------------------------------------------------------------------ #
    # Inference loop
    # ------------------------------------------------------------------ #
    all_preds: list[str] = []
    all_labels: list[str] = []

    for batch_idx, batch in enumerate(
        tqdm(dataloader, disable=disable_tqdm, desc="Evaluating")
    ):
        if torch.cuda.is_available() and batch_idx > 0 and batch_idx % 10 == 0:
            torch.cuda.empty_cache()

        true_labels_tokenized: torch.Tensor = batch.pop("labels")

        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)

        # Decode ground-truth labels
        decoded_labels = _decode_labels(true_labels_tokenized, processor)
        all_labels.extend(decoded_labels)

        # Generate predictions
        stopping_criteria = StoppingCriteriaList(
            [MultipleTokenBatchStoppingCriteria(stop_ids, batch_size=batch.input_ids.size(0))]
        )
        try:
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                generated_ids = model.generate(
                    **batch,
                    eos_token_id=processor.tokenizer.eos_token_id,
                    max_new_tokens=3,
                    stopping_criteria=stopping_criteria,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    do_sample=False,
                    num_beams=1,
                )
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                logger.warning("OOM on batch %d — defaulting to '%s'.", batch_idx, NEGATIVE_CLASS)
                all_preds.extend([NEGATIVE_CLASS] * len(decoded_labels))
                torch.cuda.empty_cache()
                continue
            raise

        new_tokens = generated_ids[:, batch["input_ids"].shape[1] :]
        texts = processor.batch_decode(
            new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        all_preds.extend(t.strip() for t in texts)

        del batch, generated_ids, new_tokens
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    # Gather across DDP processes
    # ------------------------------------------------------------------ #
    if accelerator:
        all_preds = accelerator.gather_for_metrics(all_preds)
        all_labels = accelerator.gather_for_metrics(all_labels)

    # ------------------------------------------------------------------ #
    # Compute metrics (main process only)
    # ------------------------------------------------------------------ #
    if accelerator is None or accelerator.is_main_process:
        metrics = _build_metrics(all_preds, all_labels)
        logger.info("Evaluation metrics: %s", {k: v for k, v in metrics.items() if k not in {"raw_predictions", "true_labels"}})

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w") as fh:
                json.dump({"metrics": metrics}, fh, indent=4)
            logger.info("Results saved to '%s'.", save_path)

        return metrics

    return None


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _decode_labels(label_tensor: torch.Tensor, processor: Any) -> list[str]:
    """Decode a batch of tokenised label sequences back to strings.

    Args:
        label_tensor: Tensor of shape ``(batch, seq_len)`` containing token ids
            (with ``_IGNORE_INDEX`` padding).
        processor: Phi-4 processor.

    Returns:
        List of decoded label strings with special tokens stripped.
    """
    decoded = []
    for tokens in label_tensor:
        valid = tokens[tokens != _IGNORE_INDEX]
        valid = valid[valid != processor.tokenizer.pad_token_id]
        text = processor.decode(valid, skip_special_tokens=False)
        for suffix in (ANSWER_SUFFIX, processor.tokenizer.eos_token, "<|end|>"):
            if suffix and text.endswith(suffix):
                text = text[: -len(suffix)]
        decoded.append(text.strip())
    return decoded
