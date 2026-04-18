#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Evaluate a fine-tuned Phi-4 AD model on a held-out test set.
#
# Adjust the paths below before running.
# ---------------------------------------------------------------------------

set -euo pipefail

# ---- Paths ----------------------------------------------------------------
MODEL_DIR="runs/phi4-ad-v1"
TEST_CSV="data/test.csv"
TEST_AUDIO="data/audio/test/"
OUTPUT_DIR="${MODEL_DIR}/test_results"

# ---- Settings -------------------------------------------------------------
BATCH_SIZE=1
MAX_AUDIO_SEC=70

# ---------------------------------------------------------------------------
python test.py \
    --model_dir             "$MODEL_DIR" \
    --test_csv_path         "$TEST_CSV" \
    --test_audio_dir        "$TEST_AUDIO" \
    --output_dir            "$OUTPUT_DIR" \
    --batch_size            "$BATCH_SIZE" \
    --max_audio_seconds     "$MAX_AUDIO_SEC" \
    --use_flash_attention \
    --mixed_precision       bf16
