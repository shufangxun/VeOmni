#!/usr/bin/env python3
"""
Compose a Qwen3-MoE language checkpoint and a SigLIP vision checkpoint into one
qwen3_moe_siglip_vlm checkpoint for VeOmni training.

Example:
  PATH=/root/shufangxun/VeOmni/.venv/bin:$PATH python \
    scripts/multimodal/qwen3_moe_siglip_30a3b/compose_qwen3_moe_siglip_vlm.py \
    --qwen-path Qwen/Qwen3-30B-A3B \
    --siglip-path google/siglip-so400m-patch14-384 \
    --output-dir /root/shufangxun/checkpoints/qwen3_moe_siglip_30a3b_init
"""

import argparse
import glob
import os
import shutil
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Iterable, Tuple

import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer
from transformers.models.siglip.configuration_siglip import SiglipVisionConfig

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.compose_qwen3_siglip_vlm import (
    ASSET_ALLOWLIST,
    map_siglip_key,
    resize_siglip_position_embedding,
)
from veomni.models import save_model_weights
from veomni.models.module_utils import _load_state_dict
from veomni.models.transformers.qwen3_moe_siglip_vlm.checkpoint_tensor_converter import (
    Qwen3MoeSiglipVLMCheckpointTensorConverter,
)
from veomni.models.transformers.qwen3_moe_siglip_vlm.configuration_qwen3_moe_siglip_vlm import (
    Qwen3MoeSiglipVLMConfig,
)
from veomni.models.transformers.qwen3_siglip_vlm.modeling_qwen3_siglip_vlm import PixelShuffleMLPConnector


def iter_checkpoint_tensors(path: str) -> Iterable[Tuple[str, torch.Tensor]]:
    for iterator in _load_state_dict(path):
        for name, tensor in iterator:
            yield name, tensor


def build_config(qwen_path: str, siglip_path: str, pixel_shuffle_factor: int) -> Qwen3MoeSiglipVLMConfig:
    text_config = AutoConfig.from_pretrained(qwen_path, trust_remote_code=True).to_dict()
    vision_config = SiglipVisionConfig.from_pretrained(siglip_path).to_dict()
    text_config["architectures"] = ["Qwen3MoeForCausalLM"]
    return Qwen3MoeSiglipVLMConfig(
        architectures=["Qwen3MoeSiglipVLMForConditionalGeneration"],
        text_config=text_config,
        vision_config=vision_config,
        pixel_shuffle_factor=pixel_shuffle_factor,
    )


def add_qwen_moe_weights(state_dict: OrderedDict, qwen_path: str, config: Qwen3MoeSiglipVLMConfig) -> None:
    converter = Qwen3MoeSiglipVLMCheckpointTensorConverter(num_experts=config.text_config.num_experts)
    for name, tensor in tqdm(iter_checkpoint_tensors(qwen_path), desc="Loading Qwen3-MoE"):
        if not converter.can_handle(name):
            continue
        converted = converter.convert(name, tensor.cpu())
        if converted is not None:
            state_dict[converted.name] = converted.tensor
    for converted in converter.finalize():
        state_dict[converted.name] = converted.tensor
    if "language_model.lm_head.weight" not in state_dict and "language_model.model.embed_tokens.weight" in state_dict:
        state_dict["language_model.lm_head.weight"] = state_dict["language_model.model.embed_tokens.weight"].clone()


def add_siglip_weights(state_dict: OrderedDict, siglip_path: str, config: Qwen3MoeSiglipVLMConfig) -> None:
    target_positions = (config.vision_config.image_size // config.vision_config.patch_size) ** 2
    for name, tensor in tqdm(iter_checkpoint_tensors(siglip_path), desc="Loading SigLIP"):
        mapped_name = map_siglip_key(name)
        if mapped_name is None:
            continue
        if mapped_name == "vision_tower.embeddings.position_embedding.weight":
            tensor = resize_siglip_position_embedding(tensor, target_positions)
        state_dict[mapped_name] = tensor.cpu()


def add_connector_weights(state_dict: OrderedDict, config: Qwen3MoeSiglipVLMConfig, seed: int) -> None:
    generator_state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    connector = PixelShuffleMLPConnector(
        vision_hidden_size=config.vision_config.hidden_size,
        text_hidden_size=config.text_config.hidden_size,
        pixel_shuffle_factor=config.pixel_shuffle_factor,
    )
    for name, tensor in connector.state_dict().items():
        state_dict[f"connector.{name}"] = tensor.cpu()
    torch.random.set_rng_state(generator_state)


def copy_tokenizer_assets(qwen_path: str, output_dir: str) -> None:
    try:
        tokenizer = AutoTokenizer.from_pretrained(qwen_path, trust_remote_code=True)
        tokenizer.save_pretrained(output_dir)
    except Exception:
        if not os.path.isdir(qwen_path):
            raise
        for name in ASSET_ALLOWLIST:
            src = os.path.join(qwen_path, name)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(output_dir, name))
        for pattern in ("*.model", "*.tiktoken"):
            for src in glob.glob(os.path.join(qwen_path, pattern)):
                shutil.copy2(src, os.path.join(output_dir, os.path.basename(src)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen-path", required=True, help="Qwen3-MoE language checkpoint path or HF repo id.")
    parser.add_argument("--siglip-path", required=True, help="SigLIP vision checkpoint path or HF repo id.")
    parser.add_argument("--output-dir", required=True, help="Output composed checkpoint directory.")
    parser.add_argument("--pixel-shuffle-factor", type=int, default=2, help="Connector spatial downsample factor.")
    parser.add_argument("--connector-seed", type=int, default=42, help="Seed for connector initialization.")
    parser.add_argument("--save-dtype", default="bfloat16", help="Output dtype for saved weights.")
    parser.add_argument("--shard-size-gb", type=float, default=5.0, help="Output shard size in GB.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    config = build_config(args.qwen_path, args.siglip_path, args.pixel_shuffle_factor)
    state_dict = OrderedDict()

    add_qwen_moe_weights(state_dict, args.qwen_path, config)
    add_siglip_weights(state_dict, args.siglip_path, config)
    add_connector_weights(state_dict, config, seed=args.connector_seed)

    config.save_pretrained(args.output_dir)
    copy_tokenizer_assets(args.qwen_path, args.output_dir)
    save_model_weights(
        args.output_dir,
        state_dict,
        save_dtype=args.save_dtype,
        shard_size=int(args.shard_size_gb * 1_000_000_000),
        safe_serialization=True,
    )

    print(f"Composed qwen3_moe_siglip_vlm checkpoint saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
