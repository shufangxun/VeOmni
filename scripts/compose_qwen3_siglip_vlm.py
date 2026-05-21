#!/usr/bin/env python3
"""
Compose a Qwen3 language checkpoint and a SigLIP vision checkpoint into one
qwen3_siglip_vlm checkpoint for VeOmni training.

Example:
  PATH=/root/shufangxun/VeOmni/.venv/bin:$PATH python scripts/compose_qwen3_siglip_vlm.py \
    --qwen-path Qwen/Qwen3-1.7B-Base \
    --siglip-path google/siglip-so400m-patch14-384 \
    --config-path configs/model_configs/qwen3_siglip_vlm/qwen3_siglip_vlm_1p7b.json \
    --output-dir /root/shufangxun/checkpoints/qwen3_siglip_1p7b_init
"""

import argparse
import glob
import os
import shutil
from collections import OrderedDict
from typing import Iterable, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from veomni.models import save_model_weights
from veomni.models.module_utils import _load_state_dict
from veomni.models.transformers.qwen3_siglip_vlm.configuration_qwen3_siglip_vlm import Qwen3SiglipVLMConfig
from veomni.models.transformers.qwen3_siglip_vlm.modeling_qwen3_siglip_vlm import PixelShuffleMLPConnector


ASSET_ALLOWLIST = {
    "chat_template.jinja",
    "generation_config.json",
    "merges.txt",
    "README.md",
    "special_tokens_map.json",
    "spiece.model",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
}


def iter_checkpoint_tensors(path: str) -> Iterable[Tuple[str, torch.Tensor]]:
    for iterator in _load_state_dict(path):
        for name, tensor in iterator:
            yield name, tensor


def map_qwen_key(name: str) -> Optional[str]:
    if name.startswith(("model.", "lm_head.")):
        return f"language_model.{name}"
    return None


def map_siglip_key(name: str) -> Optional[str]:
    for prefix in ("vision_model.", "model.vision_model."):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    if name.startswith(("embeddings.", "encoder.", "post_layernorm.")):
        return f"vision_tower.{name}"
    return None


def resize_siglip_position_embedding(tensor: torch.Tensor, target_num_positions: int) -> torch.Tensor:
    if tensor.shape[0] == target_num_positions:
        return tensor
    src_grid = int(tensor.shape[0] ** 0.5)
    dst_grid = int(target_num_positions**0.5)
    if src_grid * src_grid != tensor.shape[0] or dst_grid * dst_grid != target_num_positions:
        raise ValueError(
            f"Cannot resize SigLIP position embedding from {tuple(tensor.shape)} to {target_num_positions} positions."
        )
    tensor = tensor.reshape(src_grid, src_grid, -1).permute(2, 0, 1).unsqueeze(0)
    tensor = torch.nn.functional.interpolate(tensor, size=(dst_grid, dst_grid), mode="bicubic", align_corners=False)
    return tensor.squeeze(0).permute(1, 2, 0).reshape(target_num_positions, -1)


def add_qwen_weights(state_dict: OrderedDict, qwen_path: str) -> None:
    for name, tensor in tqdm(iter_checkpoint_tensors(qwen_path), desc="Loading Qwen3"):
        mapped_name = map_qwen_key(name)
        if mapped_name is not None:
            state_dict[mapped_name] = tensor.cpu()
    if "language_model.lm_head.weight" not in state_dict and "language_model.model.embed_tokens.weight" in state_dict:
        state_dict["language_model.lm_head.weight"] = state_dict["language_model.model.embed_tokens.weight"].clone()


def add_siglip_weights(state_dict: OrderedDict, siglip_path: str, config: Qwen3SiglipVLMConfig) -> None:
    target_positions = (config.vision_config.image_size // config.vision_config.patch_size) ** 2
    for name, tensor in tqdm(iter_checkpoint_tensors(siglip_path), desc="Loading SigLIP"):
        mapped_name = map_siglip_key(name)
        if mapped_name is None:
            continue
        if mapped_name == "vision_tower.embeddings.position_embedding.weight":
            tensor = resize_siglip_position_embedding(tensor, target_positions)
        state_dict[mapped_name] = tensor.cpu()


def add_connector_weights(state_dict: OrderedDict, config: Qwen3SiglipVLMConfig, seed: int) -> None:
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
    parser.add_argument("--qwen-path", required=True, help="Qwen3 language checkpoint path or HF repo id.")
    parser.add_argument("--siglip-path", required=True, help="SigLIP vision checkpoint path or HF repo id.")
    parser.add_argument("--config-path", required=True, help="qwen3_siglip_vlm config path.")
    parser.add_argument("--output-dir", required=True, help="Output composed checkpoint directory.")
    parser.add_argument("--connector-seed", type=int, default=42, help="Seed for connector initialization.")
    parser.add_argument("--save-dtype", default="bfloat16", help="Output dtype for saved weights.")
    parser.add_argument("--shard-size-gb", type=float, default=5.0, help="Output shard size in GB.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    config = Qwen3SiglipVLMConfig.from_pretrained(args.config_path)
    state_dict = OrderedDict()

    add_qwen_weights(state_dict, args.qwen_path)
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

    print(f"Composed qwen3_siglip_vlm checkpoint saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
