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

import json
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Sequence, Union

import numpy as np
import torch

from veomni.utils import logging
from veomni.utils.constants import AUDIO_INPUT_INDEX, IGNORE_INDEX, IMAGE_INPUT_INDEX, VIDEO_INPUT_INDEX
from veomni.utils.registry import Registry


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, ProcessorMixin

    from .chat_template import ChatTemplate


DATA_TRANSFORM_REGISTRY = Registry("DataTransform")
logger = logging.get_logger(__name__)


def build_data_transform(transform_name: str, **kwargs) -> Callable:
    return partial(DATA_TRANSFORM_REGISTRY[transform_name], **kwargs)


def split_into_chunks(sequence: Sequence[int], chunk_size: int) -> List[List[int]]:
    """
    Splits a long sequence into chunks.
    """
    total_len = len(sequence)
    chunks = []
    for i in range(0, total_len, chunk_size):
        chunks.append(sequence[i : i + chunk_size])

    return chunks


def _get_text_example(example: Dict[str, Any], text_keys: Union[str, List[str]]) -> Any:
    if isinstance(text_keys, str):
        return example[text_keys]
    if isinstance(text_keys, list):
        for key in text_keys:
            if key in example:
                return example[key]
        raise ValueError(f"None of the keys {text_keys} are found in the example.")
    raise ValueError(f"text_keys must be a string or a list of strings, but got {type(text_keys)}")


def _get_optional_example_value(example: Dict[str, Any], keys: Union[str, List[str], None]) -> Any:
    if keys is None:
        return None
    if isinstance(keys, str):
        return example.get(keys)
    if isinstance(keys, list):
        for key in keys:
            if key in example and example[key] is not None:
                return example[key]
        return None
    raise ValueError(f"keys must be a string, a list of strings, or None, but got {type(keys)}")


def _count_image_placeholders(conversations: Sequence[Any]) -> int:
    count = 0
    for message in conversations:
        if isinstance(message, dict) and message.get("type") == "interleaved":
            values = message.get("values", [])
        else:
            values = message[1:]
        count += sum(1 for value in values if value[0] == "image")
    return count


def _validate_image_placeholder_alignment(
    preprocess: str,
    placeholder_count: int,
    loaded_image_count: int,
) -> None:
    if placeholder_count != loaded_image_count:
        raise ValueError(
            f"Image placeholder count mismatch for preprocess={preprocess}: "
            f"placeholders={placeholder_count}, loaded_images={loaded_image_count}."
        )


def _resolve_domain_name(example: Dict[str, Any], domain: str = None) -> Any:
    if domain is not None:
        return domain
    return example.get("domain_name", example.get("domain"))


@DATA_TRANSFORM_REGISTRY.register("plaintext")
def process_plaintext_example(
    example: Dict[str, Any],
    tokenizer: "PreTrainedTokenizer",
    max_seq_len: int,
    text_keys: Union[str, List[str]] = "content_split",
    **kwargs,
) -> List[Dict[str, Any]]:
    examples = []
    domain_name = _resolve_domain_name(example, kwargs.get("domain"))
    if isinstance(text_keys, str):
        text_example = example[text_keys]
    elif isinstance(text_keys, list):
        for key in text_keys:
            if key in example:
                text_example = example[key]
                break
        else:
            raise ValueError(f"None of the keys {text_keys} are found in the example.")
    else:
        raise ValueError(f"text_keys must be a string or a list of strings, but got {type(text_keys)}")

    tokens = tokenizer.encode(text_example, add_special_tokens=False) + [tokenizer.eos_token_id]
    for input_ids in split_into_chunks(tokens, max_seq_len):
        processed_example = {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor([1] * len(input_ids)),
            "labels": torch.tensor(input_ids),
        }
        if domain_name is not None:
            processed_example["domain_name"] = str(domain_name)
        examples.append(processed_example)

    return examples


@DATA_TRANSFORM_REGISTRY.register("conversation")
def process_conversation_example(
    example: Dict[str, Any],
    chat_template: "ChatTemplate",
    max_seq_len: int,
    text_keys: Union[str, List[str]] = "messages",
    **kwargs,
) -> List[Dict[str, "torch.Tensor"]]:
    domain_name = _resolve_domain_name(example, kwargs.get("domain"))
    if isinstance(text_keys, str):
        text_example = example[text_keys]
    elif isinstance(text_keys, list):
        for key in text_keys:
            if key in example:
                text_example = example[key]
                break
        else:
            raise ValueError(f"None of the keys {text_keys} are found in the example.")
    else:
        raise ValueError(f"text_keys must be a string or a list of strings, but got {type(text_keys)}")

    tokenized_example = chat_template.encode_messages(text_example, max_seq_len=max_seq_len)
    tokenized_example = {k: torch.tensor(v) for k, v in tokenized_example.items()}
    if domain_name is not None:
        tokenized_example["domain_name"] = str(domain_name)
    return [tokenized_example]


@DATA_TRANSFORM_REGISTRY.register("mixed_text")
def process_mixed_text_example(
    example: Dict[str, Any],
    chat_template: "ChatTemplate",
    max_seq_len: int,
    text_keys: Union[str, List[str], None] = None,
    preprocess: str = None,
    **kwargs,
) -> List[Dict[str, "torch.Tensor"]]:
    if preprocess is None:
        raise ValueError("mixed_text requires per-source preprocess from multisource data yaml.")

    if preprocess == "plaintext":
        text_keys = text_keys or "content_split"
        text_example = _get_text_example(example, text_keys)
        tokenized_examples = chat_template.encode_pretrain_text(text_example, max_seq_len=max_seq_len)
    elif preprocess == "conversation":
        text_keys = text_keys or "messages"
        text_example = _get_text_example(example, text_keys)
        tokenized_examples = [chat_template.encode_messages(text_example, max_seq_len=max_seq_len)]
    else:
        raise ValueError(
            f"Unsupported mixed_text preprocess: {preprocess}. Supported values are: plaintext, conversation."
        )

    domain_name = _resolve_domain_name(example, kwargs.get("domain"))
    examples = []
    for tokenized_example in tokenized_examples:
        processed_example = {k: torch.tensor(v) for k, v in tokenized_example.items()}
        if domain_name is not None:
            processed_example["domain_name"] = str(domain_name)
        examples.append(processed_example)
    return examples


@DATA_TRANSFORM_REGISTRY.register("dpo")
def process_dpo_example(
    example: Dict[str, Any],
    chat_template: "ChatTemplate" = None,
    tokenizer: "PreTrainedTokenizer" = None,
    max_seq_len: int = 2048,
    **kwargs,
) -> List[Dict[str, "torch.Tensor"]]:
    """Process a DPO preference pair into a single flat sample.

    Chosen and rejected sequences are concatenated into one 1-D tensor with
    ``position_ids`` that reset at the boundary so that flash-attention treats
    them as two independent sequences.  This format is directly compatible with
    ``MainCollator`` (packing + SP) — no DPO-specific collator is needed.

    Supported input formats:
      1. Conversation: {"chosen": [messages...], "rejected": [messages...]}
      2. Plaintext with prompt: {"prompt": str, "chosen": str, "rejected": str}

    Returns:
        A list with one dict.  Each value is a 1-D tensor of length
        ``len_chosen + len_rejected``.  Keys: ``input_ids``, ``attention_mask``,
        ``labels``, ``position_ids``.
    """
    chosen_raw = example["chosen"]
    rejected_raw = example["rejected"]

    if isinstance(chosen_raw, list):
        assert chat_template is not None, "chat_template is required for conversation-format DPO data"
        chosen_tok = chat_template.encode_messages(chosen_raw, max_seq_len=max_seq_len)
        rejected_tok = chat_template.encode_messages(rejected_raw, max_seq_len=max_seq_len)
    else:
        assert tokenizer is not None, "tokenizer is required for plaintext-format DPO data"
        prompt = example.get("prompt", "")
        chosen_text = prompt + chosen_raw
        rejected_text = prompt + rejected_raw

        chosen_ids = tokenizer.encode(chosen_text, add_special_tokens=True)[:max_seq_len]
        rejected_ids = tokenizer.encode(rejected_text, add_special_tokens=True)[:max_seq_len]
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=True) if prompt else []
        prompt_len = len(prompt_ids)

        chosen_tok = {
            "input_ids": chosen_ids,
            "attention_mask": [1] * len(chosen_ids),
            "labels": [IGNORE_INDEX] * prompt_len + chosen_ids[prompt_len:],
        }
        rejected_tok = {
            "input_ids": rejected_ids,
            "attention_mask": [1] * len(rejected_ids),
            "labels": [IGNORE_INDEX] * prompt_len + rejected_ids[prompt_len:],
        }

    def _to_tensor(v):
        return v if isinstance(v, torch.Tensor) else torch.tensor(v)

    c_ids = _to_tensor(chosen_tok["input_ids"])
    r_ids = _to_tensor(rejected_tok["input_ids"])
    c_len = c_ids.shape[-1]
    r_len = r_ids.shape[-1]

    result = {
        "input_ids": torch.cat([c_ids, r_ids]),
        "attention_mask": torch.cat(
            [_to_tensor(chosen_tok["attention_mask"]), _to_tensor(rejected_tok["attention_mask"])]
        ),
        "labels": torch.cat([_to_tensor(chosen_tok["labels"]), _to_tensor(rejected_tok["labels"])]),
        "position_ids": torch.cat([torch.arange(c_len, dtype=torch.int64), torch.arange(r_len, dtype=torch.int64)]),
    }
    return [result]


@DATA_TRANSFORM_REGISTRY.register("classification")
def process_classification_example(
    example: dict[str, Any],
    tokenizer: "PreTrainedTokenizer",
    max_seq_len: int,
    text_keys: Union[str, list[str]] = "text",
    label_key: str = "label",
    **kwargs,
) -> list[dict[str, "torch.Tensor"]]:
    """
    Convert a single raw example into one classification training sample.

    Args:
        example:
            A single record from the dataset. Expected format (minimal):
                {
                    "<text_key>":  str,   # e.g. news article / sentence
                    "<label_key>": int,   # e.g. 0..(num_labels-1)
                    ...                   # other fields are ignored
                }
            By default:
                text_key  = "text"
                label_key = "label"

        tokenizer:
            A HuggingFace tokenizer used to tokenize the input text.

        max_seq_len:
            Maximum sequence length (in tokens). Text longer than this
            will be truncated to the first `max_seq_len` tokens.

        text_keys:
            Keys in `example` that contains the raw input text. If a list, the first key found in `example` will be used.

        label_key:
            Key in `example` that contains the class id. The value should be int-like.

    Returns:
        A list with exactly one sample dict:
            {
                "input_ids":      LongTensor[L],
                "attention_mask": LongTensor[L],
                "labels":         LongTensor[L],
                "position_ids":   LongTensor[L]
            }
    """
    # 1) text
    if isinstance(text_keys, str):
        text = example[text_keys]
    elif isinstance(text_keys, list):
        for key in text_keys:
            if key in example:
                text = example[key]
                break
        else:
            raise ValueError(f"None of the keys {text_keys} are found in the example.")
    else:
        raise ValueError(f"text_keys must be a string or a list of strings, but got {type(text_keys)}")

    # 2) label
    if label_key not in example:
        raise ValueError(f"Missing label key '{label_key}' in example.")
    try:
        label_val = int(example[label_key])
    except Exception as e:
        raise ValueError(f"Label '{example[label_key]}' is not an int-like value.") from e

    # 3) tokenize
    tokens: list[int] = tokenizer.encode(text, add_special_tokens=True)

    # 4) build samples
    examples: list[dict[str, torch.Tensor]] = []

    def build_sample(seq: list[int]) -> dict[str, "torch.Tensor"]:
        L = len(seq)
        token_labels = torch.full((L,), IGNORE_INDEX, dtype=torch.long)
        token_labels[L - 1] = label_val

        sample: dict[str, torch.Tensor] = {
            "input_ids": torch.tensor(seq, dtype=torch.long),
            "attention_mask": torch.ones(len(seq), dtype=torch.long),
            "labels": token_labels,
        }
        sample["position_ids"] = torch.arange(len(seq), dtype=torch.long)
        return sample

    if len(tokens) > max_seq_len:
        tokens = tokens[:max_seq_len]

    examples.append(build_sample(tokens))
    return examples


def _process_sample_qwen_vl_base(
    sample: Dict[str, Any],
    processor: "ProcessorMixin",
    chat_template: "ChatTemplate",
    position_id_func: "Callable",
    **kwargs,
):
    from .multimodal import conv_preprocess
    from .multimodal.image_utils import fetch_images

    source = kwargs.get("source_name") or sample.get("source") or sample.get("source_name")

    if "conversations" in sample and sample["conversations"] is not None and len(sample["conversations"]) > 0:
        conversations = sample["conversations"]
    else:
        conversations = sample
    conversations = conv_preprocess(source, conversations, **kwargs)

    token_num_inputs, image_inputs, video_inputs = {}, {}, {}
    image_grid_thw, video_grid_thw = None, None
    video_metadata = None

    if "images" in sample and sample["images"]:
        images = fetch_images(sample["images"], **kwargs)
        image_inputs = processor.image_processor(images=images, return_tensors="pt")
        image_grid_thw = image_inputs["image_grid_thw"]
        merge_length = processor.image_processor.merge_size**2
        image_token_num = image_grid_thw.prod(dim=-1) // merge_length
        token_num_inputs["image"] = image_token_num

    if "videos" in sample and sample["videos"]:
        from .multimodal.video_utils import fetch_videos_metadata

        videos, metadata, _, _ = fetch_videos_metadata(sample["videos"], **kwargs)
        video_inputs = processor.video_processor(
            videos=videos, video_metadata=metadata, return_tensors="pt", return_metadata=True
        )
        video_grid_thw = video_inputs["video_grid_thw"]
        video_metadata = video_inputs.pop("video_metadata", None)

        merge_length = processor.video_processor.merge_size**2
        video_token_num = video_grid_thw.prod(dim=-1) // merge_length
        token_num_inputs["video"] = video_token_num

    # Encoding
    encode_kwargs = {}
    if video_metadata is not None:
        encode_kwargs["video_metadata"] = video_metadata

    tokenized_example = chat_template.encode_messages(conversations, token_num_inputs, **encode_kwargs)

    tokenized_example = {
        k: (v if isinstance(v, torch.Tensor) else torch.tensor(v)) for k, v in tokenized_example.items()
    }

    input_ids = tokenized_example["input_ids"]
    attention_mask = tokenized_example["attention_mask"]

    # Masks and Token Types
    tokenized_example["image_mask"] = input_ids == IMAGE_INPUT_INDEX
    tokenized_example["video_mask"] = input_ids == VIDEO_INPUT_INDEX

    # Position IDs
    position_id_func_kwargs = {
        "input_ids": input_ids.unsqueeze(0),
        "image_grid_thw": image_grid_thw,
        "video_grid_thw": video_grid_thw,
        "attention_mask": attention_mask.unsqueeze(0),
    }

    mm_token_type_ids = torch.zeros_like(input_ids)
    mm_token_type_ids[tokenized_example["image_mask"]] = 1
    mm_token_type_ids[tokenized_example["video_mask"]] = 2
    tokenized_example["mm_token_type_ids"] = mm_token_type_ids
    position_id_func_kwargs["mm_token_type_ids"] = mm_token_type_ids.unsqueeze(0)

    position_id_returns = position_id_func(**position_id_func_kwargs)
    # Squeeze position_ids to match the per-sample (no batch dim) convention
    # used everywhere else in this dict.
    position_id_returns["position_ids"] = position_id_returns["position_ids"].squeeze().clone()
    # Only position_ids is propagated into the training feature dict. The
    # rope_deltas position_id_func also returns is generation-only (KV-cache
    # decode); the training forward always receives a precomputed
    # position_ids and never derives or reads rope_deltas.
    tokenized_example["position_ids"] = position_id_returns["position_ids"]

    # Final cleanup
    tokenized_example["input_ids"][tokenized_example["image_mask"]] = 0
    tokenized_example["input_ids"][tokenized_example["video_mask"]] = 0
    tokenized_example.update(image_inputs)
    tokenized_example.update(video_inputs)
    # image_inputs / video_inputs carry the HF processor's CPU `image_grid_thw`
    # / `video_grid_thw` tensors; the collator packs them (DataCollateInfo
    # pack_dim=0) and the model's metadata_collate_func hook derives the ViT
    # metadata from them. No per-sample `.tolist()` sidecar needed here.

    return [tokenized_example]


@DATA_TRANSFORM_REGISTRY.register("qwen2_vl")
@DATA_TRANSFORM_REGISTRY.register("qwen2_5_vl")
@DATA_TRANSFORM_REGISTRY.register("qwen3_vl")
@DATA_TRANSFORM_REGISTRY.register("qwen3_vl_moe")
@DATA_TRANSFORM_REGISTRY.register("qwen3_5")
@DATA_TRANSFORM_REGISTRY.register("qwen3_5_moe")
def process_sample_qwen_vl(
    sample: Dict[str, Any],
    processor: "ProcessorMixin",
    chat_template: "ChatTemplate",
    position_id_func: "Callable",
    **kwargs,
):
    """
    Unified processing function for Qwen-VL series models.
    Automatically determines whether to use mm_token_type_ids based on transformers version.
    """
    return _process_sample_qwen_vl_base(
        sample,
        processor,
        chat_template,
        position_id_func,
        **kwargs,
    )


def _normalize_image_item(image: Any) -> Any:
    if isinstance(image, dict):
        if image.get("bytes") is not None:
            return image["bytes"]
        if image.get("path") is not None:
            return image["path"]
    return image


def _extract_image_items(sample: Dict[str, Any], image_keys: Union[str, List[str], None] = None) -> List[Any]:
    images = _get_optional_example_value(sample, image_keys)
    if images is None:
        images = sample.get("images", sample.get("image"))
    if images is None:
        return []
    if isinstance(images, list):
        return [_normalize_image_item(image) for image in images if image is not None]
    return [_normalize_image_item(images)]


def _sharegpt_conversation_to_multimodal_conversation(conversations: Any) -> List[List[Any]]:
    if isinstance(conversations, str):
        conversations = json.loads(conversations)
    if not conversations:
        raise ValueError("qwen3_siglip_vlm conversation expects non-empty ShareGPT conversations.")

    role_mapping = {"human": "user", "gpt": "assistant"}
    if conversations[0].get("from") != "human":
        conversations = conversations[1:]
    if not conversations or conversations[0].get("from") != "human":
        raise ValueError("qwen3_siglip_vlm conversation expects the first ShareGPT turn to be from human.")

    constructed = []
    for message in conversations:
        role = role_mapping.get(message.get("from"))
        if role is None:
            raise ValueError(f"Unsupported ShareGPT role for qwen3_siglip_vlm conversation: {message.get('from')}")
        value = message.get("value", "")
        if "<image>" in value:
            value = value.replace("<image>", "")
            constructed.append([role, ("image", None), ("text", value)])
        else:
            constructed.append([role, ("text", value)])
    return constructed


def _as_aligned_list(value: Any, field_name: str) -> List[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    raise ValueError(f"qwen3_siglip_vlm interleaved expects {field_name} to be an aligned list.")


def _build_qwen3_siglip_interleaved_from_aligned_lists(
    sample: Dict[str, Any],
    text_keys: Union[str, List[str], None],
    image_keys: Union[str, List[str], None],
) -> tuple[List[Dict[str, Any]], List[Any]]:
    texts = _as_aligned_list(_get_text_example(sample, text_keys or "texts"), "texts")
    images = _get_optional_example_value(sample, image_keys)
    if images is None:
        images = sample.get("images", sample.get("image"))
    images = _as_aligned_list(images, "images")
    if len(images) != len(texts):
        raise ValueError(
            f"qwen3_siglip_vlm interleaved expects aligned images/texts lists with equal length, "
            f"but got images={len(images)}, texts={len(texts)}."
        )

    values, image_items = [], []
    for index, (image, text) in enumerate(zip(images, texts)):
        has_image = image is not None
        has_text = text is not None
        if has_image == has_text:
            raise ValueError(
                "qwen3_siglip_vlm interleaved expects exactly one of images[i] or texts[i] to be non-null "
                f"at each aligned position, but got images[{index}]={image is not None}, "
                f"texts[{index}]={text is not None}."
            )
        if has_image:
            values.append(("image", None))
            image_items.append(_normalize_image_item(image))
        else:
            if not isinstance(text, str):
                raise ValueError(f"qwen3_siglip_vlm interleaved expects text items to be strings, got {type(text)}.")
            values.append(("text", text))

    return [{"type": "interleaved", "values": values}], image_items


def _siglip_patchify_images(
    images: List[Any],
    patch_size: int,
    pixel_shuffle_factor: int,
    image_mean: Sequence[float] = (0.5, 0.5, 0.5),
    image_std: Sequence[float] = (0.5, 0.5, 0.5),
    **kwargs,
) -> Dict[str, torch.Tensor]:
    from .multimodal.image_utils import fetch_images

    if not images:
        return {}

    scale_factor = patch_size * pixel_shuffle_factor
    pil_images = fetch_images(images, scale_factor=scale_factor, **kwargs)
    pixel_values, image_grid_hw = [], []
    mean = torch.tensor(image_mean, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(image_std, dtype=torch.float32).view(3, 1, 1)
    for image in pil_images:
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        tensor = (tensor - mean) / std
        patches = tensor.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
        grid_h, grid_w = patches.shape[1], patches.shape[2]
        patches = patches.permute(1, 2, 0, 3, 4).reshape(grid_h * grid_w, -1)
        pixel_values.append(patches)
        image_grid_hw.append([grid_h, grid_w])

    return {
        "pixel_values": torch.cat(pixel_values, dim=0),
        "image_grid_hw": torch.tensor(image_grid_hw, dtype=torch.long),
    }


@DATA_TRANSFORM_REGISTRY.register("qwen3_siglip_vlm")
def process_sample_qwen3_siglip_vlm(
    sample: Dict[str, Any],
    processor: "ProcessorMixin",
    chat_template: "ChatTemplate",
    position_id_func: "Callable",
    max_seq_len: int,
    text_keys: Union[str, List[str], None] = None,
    image_keys: Union[str, List[str], None] = None,
    preprocess: str = None,
    domain: str = None,
    patch_size: int = 14,
    pixel_shuffle_factor: int = 2,
    **kwargs,
):
    del processor
    if preprocess is None:
        raise ValueError("qwen3_siglip_vlm requires per-source preprocess from multisource data yaml.")

    image_items = []
    if preprocess == "plaintext":
        text_example = _get_text_example(sample, text_keys or "text")
        tokenized_examples = chat_template.encode_pretrain_text(text_example, max_seq_len=max_seq_len)
    elif preprocess == "conversation":
        raw_conversations = _get_text_example(sample, text_keys or "conversations")
        conversations = _sharegpt_conversation_to_multimodal_conversation(raw_conversations)
        image_items = _extract_image_items(sample, image_keys)
        _validate_image_placeholder_alignment(
            preprocess=preprocess,
            placeholder_count=_count_image_placeholders(conversations),
            loaded_image_count=len(image_items),
        )
    elif preprocess == "interleaved":
        conversations, image_items = _build_qwen3_siglip_interleaved_from_aligned_lists(
            sample,
            text_keys,
            image_keys,
        )
        _validate_image_placeholder_alignment(
            preprocess=preprocess,
            placeholder_count=_count_image_placeholders(conversations),
            loaded_image_count=len(image_items),
        )
    else:
        raise ValueError(
            f"Unsupported qwen3_siglip_vlm preprocess: {preprocess}. Supported values are: "
            "plaintext, conversation, interleaved."
        )

    try:
        image_inputs = _siglip_patchify_images(
            image_items,
            patch_size=patch_size,
            pixel_shuffle_factor=pixel_shuffle_factor,
            **kwargs,
        )
    except OSError as exc:
        logger.warning(
            "Skip qwen3_siglip_vlm sample with unreadable image: "
            f"id={sample.get('id')}, preprocess={preprocess}, domain={sample.get('domain', domain)}, "
            f"image_count={len(image_items)}, error={exc}"
        )
        return None
    image_grid_hw = image_inputs.get("image_grid_hw")
    image_token_nums = []
    if image_grid_hw is not None:
        image_token_nums = [
            int(
                ((grid_hw[0].item() + pixel_shuffle_factor - 1) // pixel_shuffle_factor)
                * ((grid_hw[1].item() + pixel_shuffle_factor - 1) // pixel_shuffle_factor)
            )
            for grid_hw in image_grid_hw
        ]

    if preprocess == "conversation":
        tokenized_examples = [
            chat_template.encode_messages(conversations, {"image": image_token_nums}, max_seq_len=max_seq_len)
        ]
    elif preprocess == "interleaved":
        tokenized_examples = [
            chat_template.encode_interleaved_messages(
                conversations,
                {"image": image_token_nums},
                max_seq_len=max_seq_len,
            )
        ]

    domain_name = _resolve_domain_name(sample, domain)
    examples = []
    for tokenized_example in tokenized_examples:
        processed_example = {
            k: (v if isinstance(v, torch.Tensor) else torch.tensor(v)) for k, v in tokenized_example.items()
        }
        input_ids = processed_example["input_ids"]
        attention_mask = processed_example["attention_mask"]
        image_mask = input_ids == IMAGE_INPUT_INDEX
        processed_example["image_mask"] = image_mask
        processed_example["input_ids"] = input_ids.clone()
        processed_example["input_ids"][image_mask] = 0
        processed_example["position_ids"] = position_id_func(
            input_ids=processed_example["input_ids"].unsqueeze(0),
            attention_mask=attention_mask.unsqueeze(0),
        )["position_ids"].squeeze(0)
        if image_inputs:
            processed_example.update(image_inputs)
        if domain_name is not None:
            processed_example["domain_name"] = str(domain_name)
        examples.append(processed_example)
    return examples


def _process_openpangu_vl_images(
    image_items: List[Any],
    processor: "ProcessorMixin",
    **kwargs,
) -> tuple[Dict[str, torch.Tensor], Union[torch.Tensor, None], List[int]]:
    from .multimodal.image_utils import fetch_images

    if not image_items:
        return {}, None, []

    images = fetch_images(image_items, **kwargs)
    image_inputs = processor.image_processor(images=images, return_tensors="pt")
    image_grid_thw = image_inputs["image_grid_thw"]
    merge_length = processor.image_processor.merge_size**2
    image_token_nums = (image_grid_thw.prod(dim=-1) // merge_length).tolist()
    return image_inputs, image_grid_thw, [int(token_num) for token_num in image_token_nums]


def _squeeze_openpangu_position_ids(position_ids: torch.Tensor) -> torch.Tensor:
    if position_ids.ndim == 3 and position_ids.shape[1] == 1:
        return position_ids.squeeze(1).clone()
    if position_ids.ndim == 2 and position_ids.shape[0] == 1:
        return position_ids.squeeze(0).clone()
    return position_ids.clone()


@DATA_TRANSFORM_REGISTRY.register("openpangu_vl")
@DATA_TRANSFORM_REGISTRY.register("pangu_vl")
def process_sample_openpangu_vl(
    sample: Dict[str, Any],
    processor: "ProcessorMixin",
    chat_template: "ChatTemplate",
    position_id_func: "Callable",
    max_seq_len: int,
    text_keys: Union[str, List[str], None] = None,
    image_keys: Union[str, List[str], None] = None,
    preprocess: str = None,
    domain: str = None,
    **kwargs,
):
    if preprocess is None:
        raise ValueError("openpangu_vl requires per-source preprocess from multisource data yaml.")

    image_items = []
    if preprocess == "plaintext":
        text_example = _get_text_example(sample, text_keys or "text")
        tokenized_examples = chat_template.encode_pretrain_text(text_example, max_seq_len=max_seq_len)
    elif preprocess == "conversation":
        raw_conversations = _get_text_example(sample, text_keys or "conversations")
        conversations = _sharegpt_conversation_to_multimodal_conversation(raw_conversations)
        image_items = _extract_image_items(sample, image_keys)
        _validate_image_placeholder_alignment(
            preprocess=preprocess,
            placeholder_count=_count_image_placeholders(conversations),
            loaded_image_count=len(image_items),
        )
    elif preprocess == "interleaved":
        conversations, image_items = _build_qwen3_siglip_interleaved_from_aligned_lists(
            sample,
            text_keys,
            image_keys,
        )
        _validate_image_placeholder_alignment(
            preprocess=preprocess,
            placeholder_count=_count_image_placeholders(conversations),
            loaded_image_count=len(image_items),
        )
    else:
        raise ValueError(
            f"Unsupported openpangu_vl preprocess: {preprocess}. Supported values are: "
            "plaintext, conversation, interleaved."
        )

    try:
        image_inputs, image_grid_thw, image_token_nums = _process_openpangu_vl_images(
            image_items,
            processor,
            **kwargs,
        )
    except OSError as exc:
        logger.warning(
            "Skip openpangu_vl sample with unreadable image: "
            f"id={sample.get('id')}, preprocess={preprocess}, domain={sample.get('domain', domain)}, "
            f"image_count={len(image_items)}, error={exc}"
        )
        return None

    if preprocess == "conversation":
        tokenized_examples = [
            chat_template.encode_messages(conversations, {"image": image_token_nums}, max_seq_len=max_seq_len)
        ]
    elif preprocess == "interleaved":
        tokenized_examples = [
            chat_template.encode_interleaved_messages(
                conversations,
                {"image": image_token_nums},
                max_seq_len=max_seq_len,
            )
        ]

    domain_name = _resolve_domain_name(sample, domain)
    examples = []
    for tokenized_example in tokenized_examples:
        processed_example = {
            k: (v if isinstance(v, torch.Tensor) else torch.tensor(v)) for k, v in tokenized_example.items()
        }
        input_ids = processed_example["input_ids"]
        attention_mask = processed_example["attention_mask"]
        image_mask = input_ids == IMAGE_INPUT_INDEX
        video_mask = input_ids == VIDEO_INPUT_INDEX
        position_output = position_id_func(
            input_ids=input_ids.unsqueeze(0),
            image_grid_thw=image_grid_thw,
            video_grid_thw=None,
            attention_mask=attention_mask.unsqueeze(0),
        )

        processed_example["image_mask"] = image_mask
        processed_example["video_mask"] = video_mask
        processed_example["position_ids"] = _squeeze_openpangu_position_ids(position_output["position_ids"])
        processed_example["input_ids"] = input_ids.clone()
        processed_example["input_ids"][image_mask] = 0
        processed_example["input_ids"][video_mask] = 0
        if image_inputs:
            processed_example.update(image_inputs)
        if domain_name is not None:
            processed_example["domain_name"] = str(domain_name)
        examples.append(processed_example)
    return examples


@DATA_TRANSFORM_REGISTRY.register("qwen2_5_omni")
@DATA_TRANSFORM_REGISTRY.register("qwen3_omni_moe")
def process_sample_qwen_omni(
    sample: Dict[str, Any],
    processor: "ProcessorMixin",
    position_id_func: "Callable",
    **kwargs,
):
    from .multimodal import conv_preprocess
    from .multimodal.audio_utils import fetch_audios
    from .multimodal.image_utils import fetch_images
    from .multimodal.video_utils import fetch_videos

    QWEN_OMNI_SYSTEM_MESSAGE = (
        "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
        "capable of perceiving auditory and visual inputs, as well as generating text and speech."
    )

    def get_omni_token_ids(processor: "ProcessorMixin") -> tuple[int, int, int]:
        tokenizer = getattr(processor, "tokenizer", processor)
        vocab = tokenizer.get_vocab()
        image_token_id = vocab.get("<|image_pad|>", vocab.get("<|IMAGE|>"))
        video_token_id = vocab.get("<|video_pad|>", vocab.get("<|VIDEO|>"))
        audio_token_id = vocab.get("<|audio_pad|>", vocab.get("<|AUDIO|>"))
        if image_token_id is None:
            raise ValueError("Cannot find image token (<|image_pad|> or <|IMAGE|>) in tokenizer vocab.")
        if video_token_id is None:
            raise ValueError("Cannot find video token (<|video_pad|> or <|VIDEO|>) in tokenizer vocab.")
        if audio_token_id is None:
            raise ValueError("Cannot find audio token (<|audio_pad|> or <|AUDIO|>) in tokenizer vocab.")
        return image_token_id, video_token_id, audio_token_id

    image_token_id, video_token_id, audio_token_id = get_omni_token_ids(processor)

    source = kwargs.get("source_name") or sample.get("source") or sample.get("source_name")
    conversations = (
        sample["conversations"] if ("conversations" in sample and len(sample["conversations"]) > 0) else sample
    )
    conversations = conv_preprocess(source, conversations, **kwargs)
    input_conversations = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": QWEN_OMNI_SYSTEM_MESSAGE,
                },
            ],
        },
    ]
    for conversation in conversations:
        contents = []
        for message in conversation[1:]:
            contents.append({"type": message[0], message[0]: message[1]})
        tmp_conv = {
            "role": conversation[0],
            "content": contents,
        }
        input_conversations.append(tmp_conv)
    text = processor.apply_chat_template(input_conversations, tokenize=False)

    images = sample.get("images", [])
    if images:
        images = fetch_images(images, **kwargs)
    else:
        images = []

    videos = sample.get("videos", [])
    if videos:
        videos, video_audios = fetch_videos(videos, **kwargs)
    else:
        videos, video_audios = [], []

    audios = sample.get("audios", [])
    if audios:
        audio_audios = fetch_audios(audios, **kwargs)
    else:
        audio_audios = []

    video_audios_iter = iter(video_audios)
    audio_audios_iter = iter(audio_audios)
    audios = []
    for item in input_conversations:
        for content in item["content"]:
            if content["type"] == "video":
                audios.append(next(video_audios_iter))
            elif content["type"] == "audio":
                audios.append(next(audio_audios_iter))

    model_inputs = processor(
        text=text,
        audios=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
    )
    model_inputs = model_inputs.data
    input_features = model_inputs.pop("input_features", None)
    feature_attention_mask = model_inputs.pop("feature_attention_mask", None)

    if feature_attention_mask is not None:
        audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
        valid_mask = audio_feature_lengths != 0
        input_features = input_features[valid_mask].permute(0, 2, 1)[feature_attention_mask[valid_mask].bool()]

        model_inputs["input_features"] = input_features
        model_inputs["audio_feature_lengths"] = audio_feature_lengths
    else:
        audio_feature_lengths = None

    input_ids = model_inputs["input_ids"].squeeze(0)
    image_mask = input_ids == image_token_id
    video_mask = input_ids == video_token_id
    audio_mask = input_ids == audio_token_id
    input_ids[image_mask] = IMAGE_INPUT_INDEX
    input_ids[video_mask] = VIDEO_INPUT_INDEX
    input_ids[audio_mask] = AUDIO_INPUT_INDEX

    position_id_returns = position_id_func(
        input_ids=input_ids.unsqueeze(0),
        image_grid_thw=model_inputs.get("image_grid_thw", None),
        video_grid_thw=model_inputs.get("video_grid_thw", None),
        attention_mask=model_inputs["attention_mask"],
        audio_seqlens=audio_feature_lengths,
        second_per_grids=model_inputs.pop("video_second_per_grid", None),
    )
    position_id_returns["position_ids"] = position_id_returns["position_ids"].clone()
    # Only position_ids is propagated — rope_deltas is generation-only; see
    # _process_sample_qwen_vl_base for the rationale. grid_thw tensors flow
    # through model_inputs and are packed by the collator.
    model_inputs["position_ids"] = position_id_returns["position_ids"]

    model_inputs["image_mask"] = image_mask
    model_inputs["video_mask"] = video_mask
    model_inputs["audio_mask"] = audio_mask
    input_ids[image_mask | video_mask | audio_mask] = 0
    model_inputs["input_ids"] = input_ids
    model_inputs["attention_mask"] = model_inputs["attention_mask"].squeeze(0)

    labels = torch.full_like(input_ids, fill_value=IGNORE_INDEX)
    tokenizer = getattr(processor, "tokenizer", processor)
    vocab = tokenizer.get_vocab()
    user_token_id = vocab.get("user")
    assistant_token_id = vocab.get("assistant")
    if user_token_id is None or assistant_token_id is None:
        raise ValueError("Cannot find user/assistant tokens in tokenizer vocab.")
    user_start_index = torch.where(input_ids == user_token_id)[0].tolist()
    assistant_start_index = torch.where(input_ids == assistant_token_id)[0].tolist()
    user_start_index.append(len(input_ids) + 1)
    user_i = 0
    for assis_i in assistant_start_index:
        while user_start_index[user_i] < assis_i:
            user_i += 1
        labels[assis_i + 2 : user_start_index[user_i] - 1] = input_ids[assis_i + 2 : user_start_index[user_i] - 1]
    model_inputs["labels"] = labels
    return [model_inputs]
