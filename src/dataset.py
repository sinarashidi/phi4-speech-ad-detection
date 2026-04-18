"""
PyTorch Dataset for Alzheimer's Disease classification from speech + transcription.

Expected CSV schema
-------------------
uid           : filename of the audio file (relative to ``audio_dir``).
transcription : manual or ASR-generated transcript of the utterance.
label         : ground-truth class — ``"dementia"`` or ``"control"``.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from .constants import (
    ANSWER_SUFFIX,
    NEGATIVE_CLASS,
    POSITIVE_CLASS,
    _IGNORE_INDEX,
)

logger = logging.getLogger(__name__)

_MAX_TRANSCRIPTION_CHARS: int = 750
_SILENT_AUDIO_LENGTH: int = 16_000  # 1 s of silence at 16 kHz


class ADClassificationDataset(Dataset):
    """Speech + transcription dataset for binary AD / control classification.

    The class handles audio loading with graceful fallback for corrupt files,
    optional dataset sub-sampling, and simple DDP sharding via ``rank`` /
    ``world_size``.

    Args:
        processor: Phi-4 multimodal processor (``AutoProcessor``).
        csv_path: Path to the metadata CSV file.
        audio_dir: Root directory that contains the audio files referenced by
            the ``uid`` column of the CSV.
        split_name: Split identifier (e.g. ``"train"``, ``"eval"``, ``"test"``).
            If ``"train"`` appears in the name the dataset returns concatenated
            input + label ids suitable for causal LM training.
        task_prompt_template: F-string template with a ``{transcription}``
            placeholder.
        positive_class: Positive class string (default ``"dementia"``).
        negative_class: Negative class string (default ``"control"``).
        max_samples: Optional cap on the number of samples (applied after
            shuffling).
        rank: Local process index for DDP sharding (default ``0``).
        world_size: Total number of processes for DDP sharding (default ``1``).
        max_audio_seconds: Maximum audio duration kept per sample (longer
            recordings are truncated at read time).
    """

    def __init__(
        self,
        processor: Any,
        csv_path: str,
        audio_dir: str,
        split_name: str,
        task_prompt_template: str,
        positive_class: str = POSITIVE_CLASS,
        negative_class: str = NEGATIVE_CLASS,
        max_samples: int | None = None,
        rank: int = 0,
        world_size: int = 1,
        max_audio_seconds: int = 30,
    ) -> None:
        import pandas as pd

        self.processor = processor
        self.task_prompt_template = task_prompt_template
        self.positive_class = positive_class
        self.negative_class = negative_class
        self.audio_dir = Path(audio_dir)
        self.max_audio_frames = max_audio_seconds * 16_000
        self.training = "train" in split_name.lower()

        df = pd.read_csv(csv_path)
        required_cols = {"uid", "transcription", "label"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(
                f"CSV '{csv_path}' is missing required column(s): {missing}"
            )

        records: list[dict[str, str]] = []
        for _, row in df.iterrows():
            uid = str(row["uid"])
            audio_path = self.audio_dir / uid
            if not audio_path.is_file():
                logger.warning(
                    "Audio file not found for uid '%s' — skipping. "
                    "(expected path: %s)",
                    uid,
                    audio_path,
                )
                continue
            transcription = str(row["transcription"])[:_MAX_TRANSCRIPTION_CHARS]
            records.append(
                {
                    "audio_path": str(audio_path),
                    "transcription_text": transcription,
                    "label": str(row["label"]),
                }
            )

        if not records:
            raise ValueError(
                f"No valid samples found. Verify CSV at '{csv_path}' and "
                f"audio directory '{audio_dir}'."
            )

        if max_samples is not None:
            random.shuffle(records)
            records = records[:max_samples]

        # Simple round-robin DDP sharding — each rank takes every world_size-th item
        if world_size > 1:
            records = records[rank::world_size]

        self.data = records
        logger.info(
            "Loaded %d samples for split '%s' (rank %d / %d).",
            len(self.data),
            split_name,
            rank,
            world_size,
        )

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | None]:
        item = self.data[idx]
        audio_data = self._load_audio(item["audio_path"])

        filled_prompt = self.task_prompt_template.format(
            transcription=item["transcription_text"]
        )
        user_message = {
            "role": "user",
            "content": f"<|audio_1|>\n{filled_prompt}",
        }

        try:
            prompt_text = self.processor.tokenizer.apply_chat_template(
                [user_message], tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(
                text=prompt_text,
                audios=[(audio_data["array"], audio_data["sampling_rate"])],
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )

            answer_str = f"{item['label']}{ANSWER_SUFFIX}"
            answer_ids = self.processor.tokenizer(
                answer_str, return_tensors="pt", add_special_tokens=False
            ).input_ids

            input_ids: torch.Tensor = inputs.input_ids[0]
            input_audio_embeds = (
                inputs.input_audio_embeds[0]
                if hasattr(inputs, "input_audio_embeds")
                else None
            )
            audio_embed_sizes = (
                inputs.audio_embed_sizes[0]
                if hasattr(inputs, "audio_embed_sizes")
                else None
            )

            if self.training:
                concatenated_ids = torch.cat([input_ids, answer_ids[0]], dim=0)
                labels = torch.full_like(concatenated_ids, _IGNORE_INDEX)
                labels[len(input_ids) :] = answer_ids[0]
            else:
                concatenated_ids = input_ids
                labels = answer_ids[0]

            del inputs, input_ids, answer_ids

            return {
                "input_ids": concatenated_ids,
                "labels": labels,
                "input_audio_embeds": input_audio_embeds,
                "audio_embed_sizes": audio_embed_sizes,
            }

        except Exception:
            logger.exception("Failed to process item %d — returning sentinel.", idx)
            return {
                "input_ids": torch.tensor([0]),
                "labels": torch.tensor([0]),
                "input_audio_embeds": None,
                "audio_embed_sizes": None,
            }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_audio(self, path: str) -> dict[str, Any]:
        """Load an audio file, truncating to ``max_audio_frames`` samples.

        Falls back to one second of silence on any read error.

        Args:
            path: Absolute path to the audio file.

        Returns:
            Dict with keys ``"array"`` (``np.ndarray``) and
            ``"sampling_rate"`` (``int``).
        """
        try:
            info = sf.info(path)
            frames_to_read = min(self.max_audio_frames, info.frames)
            array, sr = sf.read(path, frames=frames_to_read, dtype="float32")

            if array.size == 0 or not np.isfinite(array).all():
                raise ValueError("Audio array is empty or contains non-finite values.")

            return {"array": array, "sampling_rate": sr}

        except Exception:
            logger.warning(
                "Could not read audio '%s' — substituting silence.", path
            )
            return {
                "array": np.zeros(_SILENT_AUDIO_LENGTH, dtype=np.float32),
                "sampling_rate": 16_000,
            }
