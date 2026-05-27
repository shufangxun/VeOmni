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

"""HF remote-code inference smoke for OpenPangu-VL with random full-config weights."""

import argparse
from contextlib import contextmanager
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor


DEFAULT_MODEL_DIR = "/root/shufangxun/Verl-Pangu30B/models/pangu30b_vl_clean"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run HF OpenPangu-VL full-config inference with randomly initialized weights.",
        allow_abbrev=False,
    )
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="Local HF remote-code model directory.")
    parser.add_argument("--image", default=None, help="Optional image path. Omit for pure text inference.")
    parser.add_argument("--question", default="你好，简单介绍一下你自己。")
    parser.add_argument(
        "--raw-prompt",
        action="store_true",
        help="Bypass chat template and feed --question directly. Only valid without --image.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="bfloat16")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load config/processor and prepare inputs, but skip random model initialization and generation.",
    )
    parser.add_argument(
        "--allow-remote-files",
        action="store_true",
        help="Allow HF to fetch missing remote-code files. By default, only local files are used.",
    )
    return parser.parse_args()


def resolve_dtype(dtype_name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype_name]


@contextmanager
def default_dtype(dtype: torch.dtype):
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old_dtype)


def build_messages(question: str, has_image: bool) -> list[dict]:
    if not has_image:
        return [{"role": "user", "content": question}]
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        }
    ]


def load_image(image_path: str) -> Image.Image:
    return Image.open(image_path).convert("RGB")


def prepare_inputs(processor, args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> dict:
    has_image = args.image is not None
    if args.raw_prompt and has_image:
        raise ValueError(
            "--raw-prompt cannot be used with --image because image placeholders come from chat template."
        )

    if args.raw_prompt:
        prompt = args.question
        processor_kwargs = {"text": [prompt], "return_tensors": "pt"}
    else:
        if not getattr(processor, "chat_template", None):
            raise RuntimeError("Processor did not load a chat template. Check chat_template.json in the model dir.")
        prompt = processor.apply_chat_template(
            build_messages(args.question, has_image=has_image),
            tokenize=False,
            add_generation_prompt=True,
        )
        processor_kwargs = {"text": [prompt], "return_tensors": "pt"}
        if has_image:
            processor_kwargs["images"] = [load_image(args.image)]

    inputs = processor(**processor_kwargs)
    prepared = {}
    for name, value in inputs.items():
        if not isinstance(value, torch.Tensor):
            prepared[name] = value
        elif torch.is_floating_point(value):
            prepared[name] = value.to(device=device, dtype=dtype)
        else:
            prepared[name] = value.to(device=device)

    print(f"prompt={prompt}")
    print(f"input_keys={sorted(prepared.keys())}")
    print(f"input_shape={tuple(prepared['input_ids'].shape)}")
    return prepared


def load_random_model(args: argparse.Namespace, device: torch.device, dtype: torch.dtype):
    config = AutoConfig.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        local_files_only=not args.allow_remote_files,
        attn_implementation="eager",
    )
    config._attn_implementation = "eager"
    if hasattr(config, "text_config"):
        config.text_config._attn_implementation = "eager"

    if device.type == "cpu" and dtype != torch.float32:
        raise ValueError("Use --dtype float32 for CPU random initialization.")

    with torch.device(device), default_dtype(dtype):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    return model.eval()


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")
    if args.image is not None and not Path(args.image).exists():
        raise FileNotFoundError(f"Image does not exist: {args.image}")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = resolve_dtype(args.dtype)

    processor = AutoProcessor.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        local_files_only=not args.allow_remote_files,
    )
    print(f"processor={type(processor).__module__}.{type(processor).__name__}")
    print(f"chat_template_loaded={bool(getattr(processor, 'chat_template', None))}")

    inputs = prepare_inputs(processor, args, device=device, dtype=dtype)
    if args.dry_run:
        return

    model = load_random_model(args, device=device, dtype=dtype)
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
        )

    prompt_len = inputs["input_ids"].shape[-1]
    output_ids = generated[0, prompt_len:]
    print(f"model={type(model).__module__}.{type(model).__name__}")
    print(f"generated_ids={output_ids.tolist()}")
    print(processor.tokenizer.decode(output_ids, skip_special_tokens=True))


if __name__ == "__main__":
    main()
