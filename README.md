# **Multimodal Alzheimer's Disease detection from spontaneous speech using fine-tuned Phi-4.**

This repository fine-tunes **[Microsoft Phi-4 Multimodal Instruct](https://huggingface.co/microsoft/Phi-4-multimodal-instruct)** to classify speakers as `dementia` (Alzheimer's Disease or Related Dementia) or `control` (Cognitively Normal) from spontaneous speech recordings and their transcriptions.

The model is evaluated on the **DementiaBank Cookie Theft** picture description task. Both the raw audio stream and the corresponding transcript are fed jointly into the model, enabling it to leverage acoustic and linguistic cues simultaneously.

---

## Overview

**Task.** Binary classification — `dementia` vs. `control` — from a Cookie Theft picture description recording.

**Model.** Phi-4 Multimodal Instruct, prompted to produce a single classification token via greedy decoding. Fine-tuning uses Phi-4's built-in **speech LoRA adapter**, keeping the trainable parameter count low and training stable on small clinical corpora.

**Input.** Each sample consists of:
- A raw speech recording (`.wav`, up to `--max_audio_seconds` seconds)
- A manual or ASR-generated transcription injected into the prompt

**Hardware.** Designed to run on a single 80 GB A100 with bfloat16 mixed precision, Flash Attention 2, and gradient checkpointing.

---

## Repository Structure

```
phi4-speech-ad-detection/
│
├── train.py                  # Fine-tuning entry point
├── test.py                   # Standalone inference / evaluation entry point
├── requirements.txt
├── pyproject.toml            # Package metadata, Ruff, and pytest config
│
├── src/                      # Core library
│   ├── constants.py          # Classification prompt, class labels, special tokens
│   ├── dataset.py            # ADClassificationDataset (PyTorch Dataset)
│   ├── collator.py           # Multimodal batch collation with audio padding
│   ├── model.py              # Model factory (attention implementation, dtype)
│   ├── evaluate.py           # Generation-based evaluation loop and metrics
│   └── utils.py              # Seed control, prediction normalisation, logging
│
├── scripts/
    ├── train.sh              # Ready-to-run accelerate launch command
    └── test.sh               # Ready-to-run evaluation command

```

---

## Setup

### 1. Clone

```bash
git clone https://github.com/<your-username>/speechdx-phi4.git
cd speechdx-phi4
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

Flash Attention must be compiled from source and installed first:

```bash
pip install flash-attn==2.7.4.post1 --no-build-isolation
pip install -r requirements.txt
```

> **No compatible GPU for Flash Attention?** Remove the `flash_attn` line from
> `requirements.txt` and drop `--use_flash_attention` from all commands below.

---

## Data Format

Both train and evaluation CSVs must have exactly three columns:

| Column | Type | Description |
|---|---|---|
| `uid` | `str` | Audio filename **relative** to the audio directory |
| `transcription` | `str` | Manual or ASR-generated transcript |
| `label` | `str` | `"dementia"` or `"control"` |

```csv
uid,transcription,label
S001_cookie.wav,"The woman is washing the dishes and the boy is stealing a cookie.",dementia
S002_cookie.wav,"A woman is doing dishes while a boy reaches for cookies on a wobbly stool.",control
```

**Notes:**
- Transcriptions over 750 characters are automatically truncated.
- Audio longer than `--max_audio_seconds` is truncated at read time — no full file is loaded into RAM.
- All audio is expected to be **16 kHz**; other sample rates are accepted but may affect quality.

---

## Training

### Quick start

Edit the paths at the top of `scripts/train.sh`, then:

```bash
bash scripts/train.sh
```

### Manual launch

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:32 \
accelerate launch \
    --mixed_precision bf16 \
    --num_processes 1 \
    train.py \
        --model_name_or_path          microsoft/Phi-4-multimodal-instruct \
        --train_csv_path              data/train.csv \
        --train_audio_dir             data/audio/train/ \
        --eval_csv_path               data/val.csv \
        --eval_audio_dir              data/audio/val/ \
        --output_dir                  runs/speechdx-v1 \
        --num_train_epochs            3 \
        --batch_size_per_gpu          1 \
        --gradient_accumulation_steps 32 \
        --learning_rate               2e-5 \
        --max_audio_seconds           70 \
        --use_flash_attention \
        --low_cpu_mem_usage
```

### Key arguments

| Argument | Default | Description |
|---|---|---|
| `--num_train_epochs` | `3` | Training epochs |
| `--batch_size_per_gpu` | `1` | Per-device batch size |
| `--gradient_accumulation_steps` | `16` | Effective batch = batch × accum |
| `--learning_rate` | `2e-5` | Peak LR (linear schedule with warmup) |
| `--max_audio_seconds` | `30` | Hard cap on audio duration per sample |
| `--use_flash_attention` | off | Flash Attention 2 (requires bfloat16) |
| `--seed` | `42` | Global random seed |
| `--max_train_samples` | `None` | Subsample training set (debug) |

### Output directory

```
runs/speechdx-v1/
├── model.safetensors
├── config.json
├── processor files …
├── eval_before_finetuning.json       # Pre-training baseline
├── eval_after_finetuning.json        # Post-training metrics
└── training_results_<timestamp>.xlsx # Hyperparams + metrics + example predictions
```

Monitor training with TensorBoard:

```bash
tensorboard --logdir runs/
```

---

## Evaluation

### Quick start

```bash
bash scripts/test.sh
```

### Manual

```bash
python test.py \
    --model_dir       runs/speechdx-v1 \
    --test_csv_path   data/test.csv \
    --test_audio_dir  data/audio/test/ \
    --output_dir      runs/speechdx-v1/test_results \
    --use_flash_attention \
    --mixed_precision bf16
```

Three output files are written to `output_dir`:

| File | Contents |
|---|---|
| `test_results_<timestamp>.json` | Full predictions, labels, and all metrics |
| `test_summary_<timestamp>.txt` | Human-readable metrics summary |
| `test_results_<timestamp>.xlsx` | Excel sheet with per-sample prediction table |

---

## Results

> Update this table with your experimental results.

| Model | Split | Accuracy | F1 (weighted) | F1 (dementia) |
|---|---|---|---|---|
| Phi-4 zero-shot | Test | — | — | — |
| Phi-4 fine-tuned | Test | — | — | — |

---

## Key Design Decisions

**Generation-based classification.** Instead of attaching a linear classification head, the model is prompted to emit a single word token (`dementia` / `control`). This keeps fine-tuning aligned with the original instruction-following objective, avoids randomly-initialised head weights, and naturally handles label synonyms via the prediction normalisation layer.

**Speech LoRA adapter.** Phi-4 Multimodal ships with a dedicated LoRA adapter for speech (`set_lora_adapter('speech')`). Activating it before fine-tuning reduces trainable parameters substantially and improves stability on small clinical corpora like DementiaBank.

**Audio truncated at read time.** `soundfile.read(frames=N)` reads only the required number of frames rather than loading the full recording and slicing afterwards, cutting peak CPU memory usage for long recordings.

**Prediction normalisation.** A canonical mapping step converts common decoder variants (`"dem"`, `"AD"`, `"Dementia"`, …) to the official label string before metric computation, making evaluation robust to minor tokenisation differences across model versions.

**Memory-safe mid-training evaluation.** The default HuggingFace `Trainer.evaluate()` computes teacher-forced cross-entropy loss, which is uninformative for a generation-based classification task. `MemoryEfficientTrainer` overrides this to run the full greedy-decoding evaluation loop, clearing the CUDA cache every 10 batches to prevent OOM during training.

---

## Citation

If you use this codebase, please cite the DementiaBank dataset and the Phi-4 model:

```bibtex
@misc{phi4multimodal2024,
  title  = {Phi-4 Technical Report},
  author = {Microsoft Research},
  year   = {2024},
  url    = {https://huggingface.co/microsoft/Phi-4-multimodal-instruct}
}
```

---

## License

Released under the [MIT License](LICENSE).
The base Phi-4 model weights are governed by [Microsoft's Phi-4 model license](https://huggingface.co/microsoft/Phi-4-multimodal-instruct).
