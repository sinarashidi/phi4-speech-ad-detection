#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Fine-tune Phi-4 Multimodal for Alzheimer's Disease detection.
#
# Adjust the paths and hyper-parameters below before running.
# ---------------------------------------------------------------------------

set -euo pipefail

# ---- Paths ----------------------------------------------------------------
MODEL="microsoft/Phi-4-multimodal-instruct"
TRAIN_CSV="data/train.csv"
TRAIN_AUDIO="data/audio/train/"
EVAL_CSV="data/val.csv"
EVAL_AUDIO="data/audio/val/"
OUTPUT_DIR="runs/phi4-ad-v1"

# ---- Hyper-parameters -----------------------------------------------------
EPOCHS=3
BATCH_SIZE=1          # per GPU
GRAD_ACCUM=32         # effective batch = BATCH_SIZE × GRAD_ACCUM
LR=2e-5
MAX_AUDIO_SEC=70
SEED=42

# ---- Memory management ----------------------------------------------------
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:32"

# ---------------------------------------------------------------------------
accelerate launch \
    --mixed_precision bf16 \
    --num_processes 1 \
    --num_machines 1 \
    train.py \
        --model_name_or_path        "$MODEL" \
        --train_csv_path            "$TRAIN_CSV" \
        --train_audio_dir           "$TRAIN_AUDIO" \
        --eval_csv_path             "$EVAL_CSV" \
        --eval_audio_dir            "$EVAL_AUDIO" \
        --output_dir                "$OUTPUT_DIR" \
        --num_train_epochs          "$EPOCHS" \
        --batch_size_per_gpu        "$BATCH_SIZE" \
        --gradient_accumulation_steps "$GRAD_ACCUM" \
        --learning_rate             "$LR" \
        --logging_steps             10 \
        --save_steps                200 \
        --eval_steps                200 \
        --max_audio_seconds         "$MAX_AUDIO_SEC" \
        --seed                      "$SEED" \
        --use_flash_attention \
        --low_cpu_mem_usage
