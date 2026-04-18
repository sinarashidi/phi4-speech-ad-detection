"""
Batch collation utilities for Phi-4 multimodal inputs.

``custom_collate_fn`` is designed to be used with ``functools.partial`` to
bind the ``processor`` argument before passing it to a ``DataLoader``.

    collate_fn = partial(custom_collate_fn, processor=processor)
    loader = DataLoader(dataset, collate_fn=collate_fn, ...)
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from transformers import BatchFeature

from .constants import _IGNORE_INDEX

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Low-level tensor helpers
# ---------------------------------------------------------------------------


def pad_sequence(
    sequences: list[torch.Tensor],
    padding_side: str = "right",
    padding_value: int = 0,
) -> torch.Tensor:
    """Pad a list of 1-D tensors to the same length.

    Args:
        sequences: List of 1-D tensors.
        padding_side: ``"right"`` or ``"left"`` padding.
        padding_value: Scalar fill value.

    Returns:
        2-D tensor of shape ``(len(sequences), max_length)``.
    """
    if padding_side not in {"right", "left"}:
        raise ValueError(f"padding_side must be 'right' or 'left', got '{padding_side}'.")
    max_len = max(s.size(0) for s in sequences)
    out = sequences[0].new_full((len(sequences), max_len), padding_value)
    for i, seq in enumerate(sequences):
        length = seq.size(0)
        if padding_side == "right":
            out[i, :length] = seq
        else:
            out[i, -length:] = seq
    return out


def cat_with_pad(
    tensors: list[torch.Tensor],
    dim: int,
    padding_value: float = 0.0,
) -> torch.Tensor:
    """Concatenate tensors along ``dim``, padding all other dimensions to the
    maximum size seen across the list.

    Args:
        tensors: List of tensors with the same number of dimensions.
        dim: Dimension along which to concatenate.
        padding_value: Fill value for padding.

    Returns:
        Concatenated and padded tensor.
    """
    ndim = tensors[0].dim()
    if not all(t.dim() == ndim for t in tensors):
        raise ValueError("All tensors must have the same number of dimensions.")

    out_size = [max(t.shape[i] for t in tensors) for i in range(ndim)]
    out_size[dim] = sum(t.shape[dim] for t in tensors)
    output = tensors[0].new_full(out_size, padding_value)

    offset = 0
    for t in tensors:
        slices = [slice(0, t.shape[d]) for d in range(ndim)]
        slices[dim] = slice(offset, offset + t.shape[dim])
        output[slices] = t
        offset += t.shape[dim]
    return output


# ---------------------------------------------------------------------------
# Main collate function
# ---------------------------------------------------------------------------


def custom_collate_fn(batch: list[dict[str, Any]], processor: Any) -> BatchFeature:
    """Collate a list of dataset items into a padded ``BatchFeature``.

    Invalid items (containing ``None`` tensors) are silently dropped. If every
    item in the batch is invalid an empty ``BatchFeature`` is returned and a
    warning is logged.

    Args:
        batch: List of dicts as returned by ``ADClassificationDataset.__getitem__``.
        processor: Phi-4 multimodal processor used to retrieve the pad token id.

    Returns:
        A ``BatchFeature`` ready for the model forward pass.
    """
    # ------------------------------------------------------------------ #
    # Filter invalid items
    # ------------------------------------------------------------------ #
    valid: list[dict[str, Any]] = []
    for item in batch:
        if item.get("input_ids") is None or item.get("labels") is None:
            continue
        if item.get("input_audio_embeds") is None:
            # Replace with a minimal dummy so the batch can still be formed
            item["input_audio_embeds"] = torch.zeros((1, 1), dtype=torch.float32)
            item["audio_embed_sizes"] = torch.tensor([1], dtype=torch.long)
        valid.append(item)

    if not valid:
        logger.warning("Entire batch was invalid — returning empty BatchFeature.")
        return BatchFeature({})

    pad_token_id: int = (
        processor.tokenizer.pad_token_id
        if processor.tokenizer.pad_token_id is not None
        else 0
    )

    # ------------------------------------------------------------------ #
    # Separate columns
    # ------------------------------------------------------------------ #
    input_ids_list = [item["input_ids"] for item in valid]
    labels_list = [item["labels"] for item in valid]
    audio_embeds_list = [
        item["input_audio_embeds"].unsqueeze(0)
        if item["input_audio_embeds"].dim() == 2
        else item["input_audio_embeds"]
        for item in valid
    ]
    embed_sizes_list = [
        item["audio_embed_sizes"].unsqueeze(0)
        if item["audio_embed_sizes"].dim() == 0
        else item["audio_embed_sizes"]
        for item in valid
    ]

    # ------------------------------------------------------------------ #
    # Detect training vs. evaluation mode from label shapes
    # ------------------------------------------------------------------ #
    # During evaluation, labels are short target sequences (< 10 tokens).
    # During training, labels are padded to the same length as input_ids.
    is_eval = all(lbl.shape[0] < 10 for lbl in labels_list)

    # ------------------------------------------------------------------ #
    # Pad sequences
    # ------------------------------------------------------------------ #
    input_ids = pad_sequence(input_ids_list, padding_side="left", padding_value=pad_token_id)
    attention_mask = (input_ids != pad_token_id).long()

    if is_eval:
        labels = (
            torch.stack(labels_list)
            if all(lbl.shape == labels_list[0].shape for lbl in labels_list)
            else pad_sequence(labels_list, padding_side="left", padding_value=_IGNORE_INDEX)
        )
    else:
        labels = pad_sequence(labels_list, padding_side="left", padding_value=_IGNORE_INDEX)

    # ------------------------------------------------------------------ #
    # Pad audio embeddings
    # ------------------------------------------------------------------ #
    input_audio_embeds = cat_with_pad(audio_embeds_list, dim=0)
    audio_embed_sizes = torch.cat(embed_sizes_list)
    max_patches = input_audio_embeds.size(1)

    audio_attention_mask = torch.stack([
        torch.cat([
            torch.ones(size_tensor.item(), dtype=torch.bool),
            torch.zeros(max_patches - size_tensor.item(), dtype=torch.bool),
        ])
        for size_tensor in embed_sizes_list
    ])

    # ------------------------------------------------------------------ #
    # Clean up intermediate lists
    # ------------------------------------------------------------------ #
    del input_ids_list, labels_list, audio_embeds_list, embed_sizes_list

    return BatchFeature(
        {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "input_audio_embeds": input_audio_embeds,
            "audio_embed_sizes": audio_embed_sizes,
            "audio_attention_mask": audio_attention_mask,
            "input_mode": 2,
        }
    )
