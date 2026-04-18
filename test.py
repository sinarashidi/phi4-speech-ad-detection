"""
Evaluate a fine-tuned Phi-4 Multimodal model on a held-out test set.

Usage
-----
    python test.py \\
        --model_dir runs/phi4-ad-v1 \\
        --test_csv_path data/test.csv \\
        --test_audio_dir data/audio/test/

CSV format
----------
- ``uid``            — audio filename relative to ``test_audio_dir``.
- ``transcription``  — utterance transcript.
- ``label``          — ground-truth label (``"dementia"`` or ``"control"``).
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

from src import (
    AD_CLASSIFICATION_PROMPT_TEMPLATE,
    NEGATIVE_CLASS,
    POSITIVE_CLASS,
    ADClassificationDataset,
    configure_logging,
    evaluate,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core test function
# ---------------------------------------------------------------------------


def test_model(
    model_dir: str,
    test_csv_path: str,
    test_audio_dir: str,
    output_dir: str | None = None,
    max_test_samples: int | None = None,
    batch_size: int = 1,
    max_audio_seconds: int = 30,
    use_flash_attention: bool = False,
    mixed_precision: str = "bf16",
) -> dict | None:
    """Load a fine-tuned model and evaluate it on a test CSV.

    Args:
        model_dir: Directory containing the saved model and processor.
        test_csv_path: Path to the test metadata CSV.
        test_audio_dir: Root directory containing test audio files.
        output_dir: Directory where results are saved.  Defaults to
            ``{model_dir}/test_results``.
        max_test_samples: Cap on evaluated samples (``None`` = all).
        batch_size: DataLoader batch size.
        max_audio_seconds: Maximum audio duration per sample.
        use_flash_attention: Use Flash Attention 2.
        mixed_precision: ``"bf16"``, ``"fp16"``, or ``"no"``.

    Returns:
        Metrics dict, or ``None`` on failure.
    """
    if output_dir is None:
        output_dir = os.path.join(model_dir, "test_results")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Model directory : %s", model_dir)
    logger.info("Test CSV        : %s", test_csv_path)
    logger.info("Audio directory : %s", test_audio_dir)
    logger.info("Output directory: %s", output_dir)

    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    # ------------------------------------------------------------------ #
    # Processor
    # ------------------------------------------------------------------ #
    try:
        processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
        logger.info("Processor loaded from '%s'.", model_dir)
    except Exception:
        logger.warning(
            "Could not load processor from '%s'. Falling back to base model.", model_dir
        )
        processor = AutoProcessor.from_pretrained(
            "microsoft/Phi-4-multimodal-instruct", trust_remote_code=True
        )

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # ------------------------------------------------------------------ #
    # Model
    # ------------------------------------------------------------------ #
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "no": torch.float32}
    dtype = dtype_map.get(mixed_precision, torch.bfloat16)
    attn_impl = "flash_attention_2" if use_flash_attention else "sdpa"

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=dtype,
        _attn_implementation=attn_impl,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    logger.info("Model loaded and moved to %s.", device)

    if processor.tokenizer.pad_token_id is not None:
        model.config.pad_token_id = processor.tokenizer.pad_token_id

    try:
        model.set_lora_adapter("speech")
        logger.info("Using 'speech' LoRA adapter.")
    except Exception:
        logger.warning("Could not set LoRA adapter 'speech'. Proceeding without it.")

    # ------------------------------------------------------------------ #
    # Dataset
    # ------------------------------------------------------------------ #
    test_dataset = ADClassificationDataset(
        processor=processor,
        csv_path=test_csv_path,
        audio_dir=test_audio_dir,
        split_name="test",
        task_prompt_template=AD_CLASSIFICATION_PROMPT_TEMPLATE,
        positive_class=POSITIVE_CLASS,
        negative_class=NEGATIVE_CLASS,
        max_samples=max_test_samples,
        rank=0,
        world_size=1,
        max_audio_seconds=max_audio_seconds,
    )

    if len(test_dataset) == 0:
        raise ValueError("Test dataset is empty. Verify the CSV and audio directory.")

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_file = Path(output_dir) / f"test_results_{timestamp}.json"

    metrics = evaluate(
        model=model,
        processor=processor,
        eval_dataset=test_dataset,
        save_path=results_file,
        disable_tqdm=False,
        eval_batch_size=batch_size,
        max_eval_samples=max_test_samples,
        device=device,
    )

    if metrics is None:
        logger.warning("No metrics returned.")
        return None

    # ------------------------------------------------------------------ #
    # Print summary
    # ------------------------------------------------------------------ #
    _print_summary(metrics)
    _save_summary_txt(metrics, output_dir, model_dir, test_csv_path, test_audio_dir, timestamp)
    _save_summary_excel(metrics, output_dir, model_dir, test_csv_path, test_audio_dir, timestamp)

    logger.info("Results saved to '%s'.", output_dir)
    return metrics


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

_MAIN_METRICS = [
    "accuracy",
    "f1_weighted",
    f"f1_{POSITIVE_CLASS}",
    "precision_weighted",
    "recall_weighted",
    "bce_loss",
    "num_samples",
]


def _fmt(key: str, value) -> str:
    if key == "num_samples":
        return str(int(value))
    if isinstance(value, float) and value == float("inf"):
        return "inf"
    return f"{value:.4f}"


def _print_summary(metrics: dict) -> None:
    print("\n" + "=" * 55)
    print("  TEST RESULTS")
    print("=" * 55)
    for key in _MAIN_METRICS:
        if key in metrics:
            print(f"  {key:<28} {_fmt(key, metrics[key])}")
    print("=" * 55)


def _save_summary_txt(
    metrics: dict,
    output_dir: str,
    model_dir: str,
    csv_path: str,
    audio_dir: str,
    timestamp: str,
) -> None:
    summary_file = Path(output_dir) / f"test_summary_{timestamp}.txt"
    with open(summary_file, "w") as fh:
        fh.write("TEST RESULTS SUMMARY\n")
        fh.write("=" * 55 + "\n")
        fh.write(f"Model directory : {model_dir}\n")
        fh.write(f"Test CSV        : {csv_path}\n")
        fh.write(f"Audio directory : {audio_dir}\n")
        fh.write(f"Timestamp       : {timestamp}\n\n")
        fh.write("METRICS:\n")
        for key in _MAIN_METRICS:
            if key in metrics:
                fh.write(f"  {key:<28} {_fmt(key, metrics[key])}\n")
    logger.info("Summary written to '%s'.", summary_file)


def _save_summary_excel(
    metrics: dict,
    output_dir: str,
    model_dir: str,
    csv_path: str,
    audio_dir: str,
    timestamp: str,
) -> None:
    try:
        excel_file = Path(output_dir) / f"test_results_{timestamp}.xlsx"
        summary_row = {
            "model_directory": model_dir,
            "test_csv_path": csv_path,
            "test_audio_directory": audio_dir,
            "timestamp": timestamp,
        }
        for key in _MAIN_METRICS:
            if key in metrics and key != "num_samples":
                summary_row[key] = (
                    "inf" if metrics[key] == float("inf") else metrics[key]
                )
            elif key == "num_samples" and key in metrics:
                summary_row[key] = metrics[key]

        prediction_rows = []
        if "raw_predictions" in metrics and "true_labels" in metrics:
            for i, (pred, true) in enumerate(
                zip(metrics["raw_predictions"][:100], metrics["true_labels"][:100])
            ):
                prediction_rows.append(
                    {
                        "sample_idx": i,
                        "prediction": pred,
                        "true_label": true,
                        "correct": pred.strip().lower() == true.strip().lower(),
                    }
                )

        with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
            pd.DataFrame([summary_row]).to_excel(writer, sheet_name="Test Results", index=False)
            if prediction_rows:
                pd.DataFrame(prediction_rows).to_excel(
                    writer, sheet_name="Example Predictions", index=False
                )
        logger.info("Excel results saved to '%s'.", excel_file)
    except Exception:
        logger.warning("Could not save Excel file.", exc_info=True)


# ---------------------------------------------------------------------------
# Argument parsing & entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a fine-tuned Phi-4 Multimodal model on a test set."
    )
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--test_csv_path", type=str, required=True)
    parser.add_argument("--test_audio_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_test_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_audio_seconds", type=int, default=30)
    parser.add_argument("--use_flash_attention", action="store_true")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
    )
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    metrics = test_model(
        model_dir=args.model_dir,
        test_csv_path=args.test_csv_path,
        test_audio_dir=args.test_audio_dir,
        output_dir=args.output_dir,
        max_test_samples=args.max_test_samples,
        batch_size=args.batch_size,
        max_audio_seconds=args.max_audio_seconds,
        use_flash_attention=args.use_flash_attention,
        mixed_precision=args.mixed_precision,
    )
    if metrics:
        logger.info("Testing complete.")
    else:
        logger.error("Testing failed — no metrics returned.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
