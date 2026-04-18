"""
Fine-tune Phi-4 Multimodal for Alzheimer's Disease detection from speech.

Usage
-----
Launch via ``accelerate`` (see ``scripts/train.sh`` for a ready-made command):

    accelerate launch --mixed_precision bf16 train.py \\
        --model_name_or_path microsoft/Phi-4-multimodal-instruct \\
        --train_csv_path data/train.csv \\
        --train_audio_dir data/audio/train/ \\
        --eval_csv_path  data/val.csv \\
        --eval_audio_dir data/audio/val/ \\
        --output_dir runs/phi4-ad-v1

CSV format
----------
The CSV files must contain three columns:

- ``uid``            — filename of the audio file relative to the audio directory.
- ``transcription``  — utterance transcript (manual or ASR-generated).
- ``label``          — class label: ``"dementia"`` or ``"control"``.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
from functools import partial
from pathlib import Path

import pandas as pd
import torch
from accelerate import Accelerator
from transformers import AutoProcessor, Trainer, TrainingArguments

from src import (
    AD_CLASSIFICATION_PROMPT_TEMPLATE,
    NEGATIVE_CLASS,
    POSITIVE_CLASS,
    ADClassificationDataset,
    configure_logging,
    create_model,
    custom_collate_fn,
    evaluate,
    set_seed,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom trainer
# ---------------------------------------------------------------------------


class MemoryEfficientTrainer(Trainer):
    """Trainer subclass that routes mid-training evaluation to our custom
    generation-based ``evaluate()`` function and clears the CUDA cache before
    train / eval / save operations.

    The standard ``Trainer.evaluate()`` runs teacher-forced loss which is
    not meaningful for this generation task — we override it with autoregressive
    greedy decoding instead.
    """

    def __init__(self, *args, external_eval_fn=None, eval_processor=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._external_eval_fn = external_eval_fn
        self._eval_processor = eval_processor

    # ------------------------------------------------------------------
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Fall back to default behaviour for explicit callers that pass
        # ``eval_dataset`` or a custom prefix.
        if eval_dataset is not None or metric_key_prefix != "eval":
            return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

        raw = self._external_eval_fn(
            model=self.model,
            processor=self._eval_processor,
            eval_dataset=self.eval_dataset,
            disable_tqdm=True,
            eval_batch_size=self.args.per_device_eval_batch_size,
            accelerator=self.accelerator,
        )

        if not raw:
            return {}

        clean: dict[str, float] = {}
        for key, value in raw.items():
            if key.startswith("_") or isinstance(value, (list, dict, tuple)):
                continue
            prefixed = key if key in {"eval_loss", "epoch", "num_samples"} else f"{metric_key_prefix}_{key}"
            clean[prefixed] = value

        self.log(clean)
        return clean

    # ------------------------------------------------------------------
    def save_model(self, *args, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        super().save_model(*args, **kwargs)

    # ------------------------------------------------------------------
    def train(self, *args, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return super().train(*args, **kwargs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _save_results_excel(
    output_dir: str,
    args: argparse.Namespace,
    initial_metrics: dict | None,
    final_metrics: dict,
    train_size: int,
    eval_size: int,
) -> None:
    """Persist hyperparameters, per-split metrics, and sample predictions to
    an Excel workbook inside ``output_dir``.

    Args:
        output_dir: Directory where the workbook is written.
        args: Parsed command-line arguments (used for hyperparameter logging).
        initial_metrics: Metrics from the pre-training evaluation (may be ``None``).
        final_metrics: Metrics from the post-training evaluation.
        train_size: Number of training samples.
        eval_size: Number of evaluation samples.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    excel_path = os.path.join(output_dir, f"training_results_{timestamp}.xlsx")

    hyperparams = {
        "model": args.model_name_or_path,
        "epochs": args.num_train_epochs,
        "batch_size": args.batch_size_per_gpu,
        "grad_accum_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.wd,
        "max_audio_seconds": args.max_audio_seconds,
        "mixed_precision": args.mixed_precision,
        "flash_attention": args.use_flash_attention,
        "seed": args.seed,
        "train_samples": train_size,
        "eval_samples": eval_size,
    }

    scalar_metrics = {
        k: v
        for k, v in final_metrics.items()
        if not k.startswith("_") and not isinstance(v, (list, dict, tuple))
    }

    comparison: dict[str, float] = {}
    if initial_metrics:
        for key in ["accuracy", "f1_weighted", f"f1_{POSITIVE_CLASS}", "bce_loss", "num_samples"]:
            if key in initial_metrics and key in final_metrics:
                comparison[f"initial_{key}"] = initial_metrics[key]
                comparison[f"final_{key}"] = final_metrics[key]
                if key not in {"num_samples"}:
                    delta = (
                        initial_metrics[key] - final_metrics[key]
                        if key == "bce_loss"
                        else final_metrics[key] - initial_metrics[key]
                    )
                    comparison[f"improvement_{key}"] = delta

    prediction_rows = []
    if "raw_predictions" in final_metrics and "true_labels" in final_metrics:
        for i, (pred, true) in enumerate(
            zip(final_metrics["raw_predictions"][:100], final_metrics["true_labels"][:100])
        ):
            prediction_rows.append(
                {
                    "sample_idx": i,
                    "prediction": pred,
                    "true_label": true,
                    "correct": pred.strip().lower() == true.strip().lower(),
                }
            )

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        pd.DataFrame([hyperparams]).to_excel(writer, sheet_name="Hyperparameters", index=False)
        pd.DataFrame([scalar_metrics]).to_excel(writer, sheet_name="Metrics", index=False)
        if comparison:
            pd.DataFrame([comparison]).to_excel(writer, sheet_name="Metrics Comparison", index=False)
        if prediction_rows:
            pd.DataFrame(prediction_rows).to_excel(writer, sheet_name="Example Predictions", index=False)

    logger.info("Training results saved to '%s'.", excel_path)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune Phi-4 Multimodal for AD speech classification."
    )
    # Model
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="microsoft/Phi-4-multimodal-instruct",
        help="HuggingFace model id or local directory.",
    )
    # Data
    parser.add_argument("--train_csv_path", type=str, required=True)
    parser.add_argument("--train_audio_dir", type=str, required=True)
    parser.add_argument("--eval_csv_path", type=str, required=True)
    parser.add_argument("--eval_audio_dir", type=str, required=True)
    # Training
    parser.add_argument("--output_dir", type=str, default="./phi4_ad_finetuned")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--batch_size_per_gpu", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--wd", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=500)
    # Sampling limits
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    # Hardware
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
    )
    parser.add_argument("--use_flash_attention", action="store_true")
    parser.add_argument("--low_cpu_mem_usage", action="store_true")
    # Misc
    parser.add_argument("--max_audio_seconds", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_tqdm", dest="tqdm", action="store_false")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    configure_logging()
    args = parse_args()
    set_seed(args.seed)

    if args.use_flash_attention and not args.low_cpu_mem_usage:
        logger.info("Enabling low_cpu_mem_usage because Flash Attention is active.")
        args.low_cpu_mem_usage = True

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision if args.mixed_precision != "no" else None,
    )

    if accelerator.is_local_main_process:
        logger.info("Arguments: %s", vars(args))
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Processor & model
    # ------------------------------------------------------------------ #
    processor = AutoProcessor.from_pretrained(
        args.model_name_or_path, trust_remote_code=True
    )
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    model = create_model(
        args.model_name_or_path,
        use_flash_attention=args.use_flash_attention,
        low_cpu_mem_usage=args.low_cpu_mem_usage,
    )
    if processor.tokenizer.pad_token_id is not None:
        model.config.pad_token_id = processor.tokenizer.pad_token_id

    try:
        model.set_lora_adapter("speech")
        logger.info("Using 'speech' LoRA adapter.")
    except Exception:
        logger.warning("Could not set LoRA adapter 'speech'. Proceeding without it.")

    # ------------------------------------------------------------------ #
    # Datasets
    # ------------------------------------------------------------------ #
    dataset_kwargs = dict(
        processor=processor,
        task_prompt_template=AD_CLASSIFICATION_PROMPT_TEMPLATE,
        positive_class=POSITIVE_CLASS,
        negative_class=NEGATIVE_CLASS,
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
        max_audio_seconds=args.max_audio_seconds,
    )

    train_dataset = ADClassificationDataset(
        csv_path=args.train_csv_path,
        audio_dir=args.train_audio_dir,
        split_name="train",
        max_samples=args.max_train_samples,
        **dataset_kwargs,
    )
    eval_dataset = ADClassificationDataset(
        csv_path=args.eval_csv_path,
        audio_dir=args.eval_audio_dir,
        split_name="eval",
        max_samples=args.max_eval_samples,
        **dataset_kwargs,
    )

    if accelerator.is_local_main_process:
        n_train = len(train_dataset) * max(accelerator.num_processes, 1)
        n_eval = len(eval_dataset) * max(accelerator.num_processes, 1)
        logger.info("Train samples (estimated): %d | Eval samples (estimated): %d", n_train, n_eval)
        if len(train_dataset) == 0 and accelerator.num_processes == 1:
            logger.error("Training dataset is empty — exiting.")
            return

    # ------------------------------------------------------------------ #
    # Training arguments
    # ------------------------------------------------------------------ #
    fp16 = accelerator.mixed_precision == "fp16"
    bf16 = accelerator.mixed_precision == "bf16"
    if args.use_flash_attention and not (fp16 or bf16):
        bf16 = True

    warmup_steps = (
        max(
            int(0.1 * args.num_train_epochs * len(train_dataset) // args.batch_size_per_gpu),
            50,
        )
        if len(train_dataset) > 0
        else 50
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size_per_gpu,
        per_device_eval_batch_size=1,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch",
        learning_rate=args.learning_rate,
        weight_decay=args.wd,
        max_grad_norm=1.0,
        lr_scheduler_type="linear",
        warmup_steps=warmup_steps,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=1,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model=f"eval_f1_{POSITIVE_CLASS}",
        greater_is_better=True,
        fp16=fp16,
        bf16=bf16,
        remove_unused_columns=False,
        report_to="tensorboard",
        disable_tqdm=not args.tqdm,
        dataloader_num_workers=1,
        dataloader_pin_memory=False,
    )

    # ------------------------------------------------------------------ #
    # Closure that forwards CLI arguments to the shared evaluate() function
    # ------------------------------------------------------------------ #
    def _evaluate_wrapper(
        model, processor, eval_dataset, save_path=None, disable_tqdm=False,
        eval_batch_size=1, accelerator=None,
    ):
        return evaluate(
            model=model,
            processor=processor,
            eval_dataset=eval_dataset,
            save_path=save_path,
            disable_tqdm=disable_tqdm,
            eval_batch_size=eval_batch_size,
            max_eval_samples=args.max_eval_samples,
            accelerator=accelerator,
        )

    collate_fn = partial(custom_collate_fn, processor=processor)

    trainer = MemoryEfficientTrainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
        external_eval_fn=_evaluate_wrapper,
        eval_processor=processor,
        compute_metrics=lambda _: {
            "accuracy": 0.0,
            "f1_weighted": 0.0,
            f"f1_{POSITIVE_CLASS}": 0.0,
            "bce_loss": 0.0,
        },
    )

    # ------------------------------------------------------------------ #
    # Pre-training evaluation
    # ------------------------------------------------------------------ #
    initial_metrics: dict | None = None
    if accelerator.is_main_process:
        logger.info("--- Pre-training evaluation ---")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        initial_metrics = _evaluate_wrapper(
            model=trainer.model,
            processor=processor,
            eval_dataset=eval_dataset,
            save_path=Path(args.output_dir) / "eval_before_finetuning.json",
            disable_tqdm=not args.tqdm,
            eval_batch_size=args.batch_size_per_gpu,
            accelerator=accelerator,
        )
        if initial_metrics:
            scalar = {k: v for k, v in initial_metrics.items() if not isinstance(v, (list, dict, tuple))}
            logger.info("Pre-training metrics: %s", scalar)

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    if len(train_dataset) > 0:
        logger.info("--- Starting fine-tuning ---")
        trainer.train()
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            trainer.save_model(args.output_dir)
            processor.save_pretrained(args.output_dir)
            logger.info("Model and processor saved to '%s'.", args.output_dir)
    else:
        logger.warning("Training dataset is empty — skipping training.")

    # ------------------------------------------------------------------ #
    # Post-training evaluation
    # ------------------------------------------------------------------ #
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    final_metrics: dict | None = None
    if accelerator.is_main_process:
        logger.info("--- Post-training evaluation ---")
        final_metrics = _evaluate_wrapper(
            model=trainer.model,
            processor=processor,
            eval_dataset=eval_dataset,
            save_path=Path(args.output_dir) / "eval_after_finetuning.json",
            disable_tqdm=not args.tqdm,
            eval_batch_size=args.batch_size_per_gpu,
            accelerator=accelerator,
        )
        if final_metrics:
            scalar = {k: v for k, v in final_metrics.items() if not isinstance(v, (list, dict, tuple))}
            logger.info("Post-training metrics: %s", scalar)
            if len(train_dataset) > 0:
                trainer.log_metrics("eval_final", scalar)
                trainer.save_metrics("eval_final", scalar)

        # Save results to Excel
        if final_metrics:
            n_train = len(train_dataset) * max(accelerator.num_processes, 1)
            n_eval = len(eval_dataset) * max(accelerator.num_processes, 1)
            _save_results_excel(
                output_dir=args.output_dir,
                args=args,
                initial_metrics=initial_metrics,
                final_metrics=final_metrics,
                train_size=n_train,
                eval_size=n_eval,
            )


if __name__ == "__main__":
    main()
