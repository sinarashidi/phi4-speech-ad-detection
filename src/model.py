"""
Phi-4 multimodal model factory.
"""

from __future__ import annotations

import logging

import torch
from transformers import AutoModelForCausalLM

logger = logging.getLogger(__name__)


def create_model(
    model_name_or_path: str,
    use_flash_attention: bool = False,
    low_cpu_mem_usage: bool = False,
) -> AutoModelForCausalLM:
    """Load a Phi-4 multimodal causal LM from a local path or HuggingFace hub.

    Args:
        model_name_or_path: HuggingFace model id or local directory path.
        use_flash_attention: Enable Flash Attention 2. Requires ``bfloat16``
            precision and a compatible CUDA GPU.
        low_cpu_mem_usage: Stream model weights shard-by-shard to reduce peak
            CPU RAM during loading.

    Returns:
        Loaded ``AutoModelForCausalLM`` instance.
    """
    attn_impl = "flash_attention_2" if use_flash_attention else "sdpa"
    dtype = torch.bfloat16 if use_flash_attention else torch.float16

    logger.info(
        "Loading model '%s'  |  attn=%s  |  dtype=%s  |  low_cpu_mem=%s",
        model_name_or_path,
        attn_impl,
        dtype,
        low_cpu_mem_usage,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        _attn_implementation=attn_impl,
        trust_remote_code=True,
        low_cpu_mem_usage=low_cpu_mem_usage,
    )

    if use_flash_attention and torch.cuda.is_available():
        model = model.to("cuda")
        logger.info("Model moved to CUDA for Flash Attention.")

    return model
