#!/usr/bin/env python3
# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Minimal Transformers-based inference for Qwen3-SigLIP VLM HF checkpoints.

The exported checkpoint is HuggingFace-style, but qwen3_siglip_vlm is a VeOmni
custom architecture. This script registers the config/model class into
Transformers' Auto registries at runtime, then loads the HF directory with
AutoModelForCausalLM.from_pretrained().
"""

import argparse
from typing import Optional

import torch
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

from veomni.data.data_transform import _siglip_patchify_images
from veomni.data.multimodal.multimodal_chat_template import SiglipQwen3ChatTemplate
from veomni.models.transformers.qwen3_siglip_vlm.configuration_qwen3_siglip_vlm import Qwen3SiglipVLMConfig
from veomni.models.transformers.qwen3_siglip_vlm.modeling_qwen3_siglip_vlm import (
    Qwen3SiglipVLMForConditionalGeneration,
)


IMAGE_PAD = "<|image_pad|>"
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


def register_qwen3_siglip_vlm() -> None:
    AutoConfig.register(Qwen3SiglipVLMConfig.model_type, Qwen3SiglipVLMConfig, exist_ok=True)
    AutoModel.register(Qwen3SiglipVLMConfig, Qwen3SiglipVLMForConditionalGeneration, exist_ok=True)
    AutoModelForCausalLM.register(Qwen3SiglipVLMConfig, Qwen3SiglipVLMForConditionalGeneration, exist_ok=True)


def build_messages(question: str, image_token_num: int = 0) -> list[dict]:
    if image_token_num <= 0:
        content = question
    else:
        content = [
            {"type": "image", "image_token_num": image_token_num},
            {"type": "text", "text": "\n" + question},
        ]
    return [{"role": "user", "content": content}]


def build_prompt(tokenizer, question: str, image_token_num: int = 0, raw_prompt: bool = False) -> str:
    if raw_prompt:
        if image_token_num <= 0:
            return question
        return IMAGE_PAD * image_token_num + "\n" + question

    tokenizer.chat_template = SiglipQwen3ChatTemplate(tokenizer).get_jinja_template()
    return tokenizer.apply_chat_template(
        build_messages(question, image_token_num=image_token_num),
        tokenize=False,
        add_generation_prompt=True,
    )


def prepare_image_inputs(
    image_path: Optional[str],
    config: Qwen3SiglipVLMConfig,
    device: torch.device,
    dtype: torch.dtype,
    image_min_pixels: int,
    image_max_pixels: int,
    max_ratio: int,
) -> tuple[dict, int]:
    if image_path is None:
        return {}, 0

    image_inputs = _siglip_patchify_images(
        [image_path],
        patch_size=config.vision_config.patch_size,
        pixel_shuffle_factor=config.pixel_shuffle_factor,
        image_min_pixels=image_min_pixels,
        image_max_pixels=image_max_pixels,
        max_ratio=max_ratio,
    )
    image_grid_hw = image_inputs["image_grid_hw"]
    grid_hw = image_grid_hw[0]
    factor = config.pixel_shuffle_factor
    image_token_num = int(((grid_hw[0].item() + factor - 1) // factor) * ((grid_hw[1].item() + factor - 1) // factor))
    return {
        "pixel_values": image_inputs["pixel_values"].to(device=device, dtype=dtype),
        "image_grid_hw": image_grid_hw.to(device=device),
    }, image_token_num


def prepare_text_inputs(tokenizer, prompt: str, device: torch.device) -> dict:
    input_ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=False)], device=device, dtype=torch.long)
    image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_PAD)
    image_mask = input_ids == image_token_id
    input_ids = input_ids.clone()
    input_ids[image_mask] = 0
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.arange(input_ids.shape[-1], device=device, dtype=torch.long).unsqueeze(0)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "image_mask": image_mask,
    }


@torch.no_grad()
def greedy_decode(
    model,
    tokenizer,
    inputs: dict,
    image_inputs: dict,
    max_new_tokens: int,
) -> list[int]:
    generated = []
    stop_ids = {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids(IM_END)}
    stop_ids.discard(None)

    for _ in range(max_new_tokens):
        outputs = model(**inputs, **image_inputs)
        next_token = int(outputs.logits[:, -1, :].argmax(dim=-1).item())
        if next_token in stop_ids:
            break
        generated.append(next_token)

        next_tensor = torch.tensor([[next_token]], device=inputs["input_ids"].device, dtype=torch.long)
        inputs["input_ids"] = torch.cat([inputs["input_ids"], next_tensor], dim=-1)
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
        inputs["position_ids"] = torch.arange(
            inputs["input_ids"].shape[-1], device=inputs["input_ids"].device, dtype=torch.long
        ).unsqueeze(0)
        next_image_mask = torch.zeros_like(next_tensor, dtype=torch.bool)
        inputs["image_mask"] = torch.cat([inputs["image_mask"], next_image_mask], dim=-1)

    return generated


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run minimal inference with a converted qwen3_siglip_vlm HF checkpoint."
    )
    parser.add_argument("--model-dir", required=True, help="Converted HF checkpoint directory.")
    parser.add_argument("--image", default=None, help="Optional image path.")
    parser.add_argument("--question", default="Describe this image.")
    parser.add_argument(
        "--raw-prompt",
        action="store_true",
        help="Bypass the chat template and feed --question as a raw language-model prompt.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--image-min-pixels", type=int, default=3136)
    parser.add_argument("--image-max-pixels", type=int, default=200704)
    parser.add_argument("--max-ratio", type=int, default=200)
    args = parser.parse_args()

    register_qwen3_siglip_vlm()

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    image_inputs, image_token_num = prepare_image_inputs(
        args.image,
        model.config,
        device=device,
        dtype=next(model.vision_tower.parameters()).dtype,
        image_min_pixels=args.image_min_pixels,
        image_max_pixels=args.image_max_pixels,
        max_ratio=args.max_ratio,
    )
    prompt = build_prompt(tokenizer, args.question, image_token_num=image_token_num, raw_prompt=args.raw_prompt)
    inputs = prepare_text_inputs(tokenizer, prompt, device=device)
    output_ids = greedy_decode(model, tokenizer, inputs, image_inputs, args.max_new_tokens)
    print(tokenizer.decode(output_ids, skip_special_tokens=True))


if __name__ == "__main__":
    main()
