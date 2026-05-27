import argparse
import gc
import json
import os
import shutil
from collections import OrderedDict
from typing import TYPE_CHECKING, Optional, Sequence, Union

import torch
from safetensors.torch import save_file
from transformers import AutoConfig, AutoProcessor
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME, WEIGHTS_INDEX_NAME, WEIGHTS_NAME

from veomni.checkpoint.dcp_checkpointer import _get_sharding_plan, _process_shard
from veomni.utils import helper


if TYPE_CHECKING:
    from transformers import GenerationConfig, PretrainedConfig, PreTrainedTokenizer, ProcessorMixin

    ModelAssets = Union[GenerationConfig, PretrainedConfig, PreTrainedTokenizer, ProcessorMixin]


logger = helper.create_logger(__name__)


def _get_qwen3_siglip_chat_template() -> str:
    return (
        "{%- if not add_generation_prompt is defined -%}"
        "{%- set add_generation_prompt = false -%}"
        "{%- endif -%}"
        "{%- for message in messages -%}"
        "{{- '<|im_start|>' + message['role'] + '\n' -}}"
        "{%- if message['content'] is string -%}"
        "{{- message['content'] | trim -}}"
        "{%- else -%}"
        "{%- for content in message['content'] -%}"
        "{%- if content['type'] == 'image' or 'image' in content or 'image_url' in content -%}"
        "{%- if content['image_token_num'] is defined -%}"
        "{%- set image_token_num = content['image_token_num'] -%}"
        "{%- elif content['token_num'] is defined -%}"
        "{%- set image_token_num = content['token_num'] -%}"
        "{%- elif content['num_tokens'] is defined -%}"
        "{%- set image_token_num = content['num_tokens'] -%}"
        "{%- else -%}"
        "{%- set image_token_num = 1 -%}"
        "{%- endif -%}"
        "{{- '<|image_pad|>' * (image_token_num | int) -}}"
        "{%- elif content['type'] == 'text' or 'text' in content -%}"
        "{{- content['text'] -}}"
        "{%- endif -%}"
        "{%- endfor -%}"
        "{%- endif -%}"
        "{{- '<|im_end|>\n' -}}"
        "{%- endfor -%}"
        "{%- if add_generation_prompt -%}"
        "{{- '<|im_start|>assistant\n' -}}"
        "{%- endif -%}"
    )


def _maybe_write_qwen3_siglip_chat_template(save_path: str) -> None:
    config_path = os.path.join(save_path, "config.json")
    if not os.path.exists(config_path):
        return
    try:
        with open(config_path, encoding="utf-8") as f:
            model_type = json.load(f).get("model_type")
    except Exception as exc:
        logger.warning(f"Failed to inspect config for chat template override: {exc}")
        return
    if model_type not in {"qwen3_siglip_vlm", "qwen3_moe_siglip_vlm"}:
        return
    chat_template_path = os.path.join(save_path, "chat_template.jinja")
    with open(chat_template_path, "w", encoding="utf-8") as f:
        f.write(_get_qwen3_siglip_chat_template())
    logger.info(f"Wrote canonical {model_type} chat_template.jinja")


@torch.no_grad()
def save_model_weights(
    output_dir: Union[str, os.PathLike],
    checkpoint_path: Union[str, os.PathLike],
    save_dtype: Optional[Union[str, torch.dtype]] = "bfloat16",
    shard_size: int = 2_000_000_000,
    safe_serialization: bool = True,
    model_assets: Optional[Sequence["ModelAssets"]] = None,
) -> None:
    """Convert DCP checkpoint to HuggingFace format with shard-by-shard processing (memory-efficient)."""
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Saving model weights to {output_dir}")
    logger.info(
        f"Format: {'safetensors' if safe_serialization else 'pytorch'}, dtype={save_dtype}, shard_size={shard_size}"
    )

    # Plan shards from metadata
    logger.info("Analyzing DCP metadata and planning shards...")
    shards, total_size, all_dcp_keys = _get_sharding_plan(checkpoint_path, shard_size, save_dtype)

    logger.info(f"Found {len(all_dcp_keys)} model tensors, total size: ~{total_size / 1e9:.2f}GB")
    logger.info(f"Split into {len(shards)} shards")

    if len(shards) == 0:
        logger.warning("No model weights found! Check if checkpoint path is correct and contains 'model.' keys.")
        return

    # Process each shard
    weight_map = OrderedDict()
    num_shards = len(shards)

    for shard_idx, shard_keys in enumerate(shards):
        weights_name = SAFE_WEIGHTS_NAME if safe_serialization else WEIGHTS_NAME
        if num_shards == 1:
            filename = weights_name
        else:
            prefix, extension = weights_name.rsplit(".", maxsplit=1)
            filename = f"{prefix}-{shard_idx + 1:05d}-of-{num_shards:05d}.{extension}"

        save_path = os.path.join(output_dir, filename)
        logger.info(f"Processing shard {shard_idx + 1}/{num_shards}: {filename} ({len(shard_keys)} tensors)")

        processed_dict = _process_shard(shard_keys, checkpoint_path, save_dtype)

        # Save shard
        if safe_serialization:
            save_file(processed_dict, save_path, metadata={"format": "pt"})
        else:
            torch.save(processed_dict, save_path)

        del processed_dict
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for hf_key in shard_keys.keys():
            weight_map[hf_key] = filename

    # Save index file for multi-shard checkpoints
    if num_shards > 1:
        index = {
            "metadata": {"total_size": total_size},
            "weight_map": weight_map,
        }
        index_file = SAFE_WEIGHTS_INDEX_NAME if safe_serialization else WEIGHTS_INDEX_NAME
        with open(os.path.join(output_dir, index_file), "w", encoding="utf-8") as f:
            content = json.dumps(index, indent=2, sort_keys=True) + "\n"
            f.write(content)
        logger.info(f"Saved index file to {index_file}")

    logger.info("Weight conversion complete.")

    # Save model assets (config, tokenizer, processor)
    if model_assets is not None:
        for model_asset in model_assets:
            if hasattr(model_asset, "save_pretrained"):
                model_asset.save_pretrained(output_dir)
                logger.info(f"Saved model asset: {type(model_asset).__name__}")
            else:
                logger.warning(f"Model asset {model_asset} does not implement `save_pretrained`")


def merge_to_hf_pt(
    load_dir: str, save_path: str, model_assets_dir: Optional[str] = None, shard_size: int = 2_000_000_000
) -> None:
    """Main conversion function: load DCP from load_dir and save HF format to save_path."""
    model_assets = None
    if model_assets_dir is not None:
        logger.info(f"Loading model assets from {model_assets_dir}")
        model_assets = []
        try:
            config = AutoConfig.from_pretrained(model_assets_dir)
            model_assets.append(config)
        except Exception as e:
            logger.warning(f"Failed to load AutoConfig: {e}")

        try:
            processor = AutoProcessor.from_pretrained(model_assets_dir, trust_remote_code=True)
            model_assets.append(processor)
        except Exception as e:
            logger.warning(f"Failed to load AutoProcessor: {e}")

        if not model_assets:
            model_assets = None

    save_model_weights(save_path, load_dir, shard_size=shard_size, model_assets=model_assets)

    if model_assets_dir is not None:
        asset_names = [
            "config.json",
            "generation_config.json",
            "chat_template.json",
            "preprocessor_config.json",
            "processor_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "chat_template.jinja",
        ]
        asset_names.extend(
            filename
            for filename in sorted(os.listdir(model_assets_dir))
            if filename.endswith(".py") and os.path.isfile(os.path.join(model_assets_dir, filename))
        )
        os.makedirs(save_path, exist_ok=True)
        copied = []
        for asset_name in asset_names:
            src = os.path.join(model_assets_dir, asset_name)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(save_path, asset_name))
                copied.append(asset_name)
        if copied:
            logger.info(f"Copied raw model asset files: {copied}")
        else:
            logger.warning(f"No raw model asset files found in {model_assets_dir}")

    _maybe_write_qwen3_siglip_chat_template(save_path)


def main():
    parser = argparse.ArgumentParser(
        description="Merge DCP checkpoint to HuggingFace format (streaming optimized)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--load-dir", type=str, required=True, help="Directory containing DCP checkpoint")
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Output directory for HuggingFace format checkpoint (default: <load-dir>/hf_ckpt)",
    )
    parser.add_argument(
        "--model-assets-dir",
        type=str,
        default=None,
        help="Directory containing model config and processor (optional)",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=2_000_000_000,
        help="Maximum shard size in bytes (default: 2GB)",
    )
    args = parser.parse_args()

    load_dir = args.load_dir
    save_dir = os.path.join(load_dir, "hf_ckpt") if args.save_dir is None else args.save_dir
    model_assets_dir = args.model_assets_dir
    shard_size = args.shard_size

    merge_to_hf_pt(load_dir, save_dir, model_assets_dir, shard_size=shard_size)


if __name__ == "__main__":
    main()
