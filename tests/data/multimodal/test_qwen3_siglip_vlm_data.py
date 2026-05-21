import io
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from veomni.data.data_transform import process_sample_qwen3_siglip_vlm
from veomni.data.multimodal.multimodal_chat_template import SiglipQwen3ChatTemplate
from veomni.utils.constants import IGNORE_INDEX


class DummyTokenizer:
    def __init__(self):
        self.vocab = {
            "<unk>": 0,
            "<|im_start|>": 1,
            "<|im_end|>": 2,
            "\n": 3,
            "user": 4,
            "assistant": 5,
            "<|image_pad|>": 6,
            "<eos>": 7,
        }
        self.unk_token_id = 0
        self.eos_token_id = 7
        self.eos_token = "<eos>"
        self.pad_token_id = None
        self.pad_token = None

    def add_special_tokens(self, tokens):
        for token in tokens.get("additional_special_tokens", []):
            self.vocab.setdefault(token, len(self.vocab))
        if "pad_token" in tokens:
            self.pad_token = tokens["pad_token"]
            self.pad_token_id = self.vocab.setdefault(self.pad_token, len(self.vocab))
        return 0

    def convert_tokens_to_ids(self, token):
        return self.vocab.get(token, self.unk_token_id)

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        ids = []
        idx = 0
        specials = sorted(self.vocab, key=len, reverse=True)
        while idx < len(text):
            for token in specials:
                if text.startswith(token, idx):
                    ids.append(self.vocab[token])
                    idx += len(token)
                    break
            else:
                token = text[idx]
                self.vocab.setdefault(token, len(self.vocab))
                ids.append(self.vocab[token])
                idx += 1
        return ids


def _image_bytes(size=(28, 28)):
    image = Image.new("RGB", size, color=(128, 64, 32))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def _position_id_func(input_ids, attention_mask=None, **kwargs):
    del input_ids, kwargs
    return {"position_ids": torch.arange(attention_mask.shape[-1]).unsqueeze(0)}


def _build_transform_kwargs():
    tokenizer = DummyTokenizer()
    return {
        "processor": SimpleNamespace(tokenizer=tokenizer),
        "chat_template": SiglipQwen3ChatTemplate(tokenizer),
        "position_id_func": _position_id_func,
        "max_seq_len": 128,
        "patch_size": 14,
        "pixel_shuffle_factor": 2,
        "image_max_pixels": 28 * 28,
        "image_min_pixels": 28 * 28,
    }


def test_plaintext_source_does_not_require_image_fields():
    sample = {"text": "plain language sample", "domain": "pt"}
    output = process_sample_qwen3_siglip_vlm(
        sample,
        preprocess="plaintext",
        text_keys="text",
        image_keys=None,
        **_build_transform_kwargs(),
    )[0]

    assert "pixel_values" not in output
    assert output["image_mask"].sum().item() == 0
    assert output["domain_name"] == "pt"


def test_conversation_sharegpt_format_masks_user_turn_and_keeps_assistant_loss():
    sample = {
        "conversations": [
            {"from": "human", "value": "<image>\nWhat is shown?"},
            {"from": "gpt", "value": "A small image."},
        ],
        "image": {"bytes": _image_bytes(), "path": None},
    }
    output = process_sample_qwen3_siglip_vlm(
        sample,
        preprocess="conversation",
        text_keys="conversations",
        image_keys="image",
        **_build_transform_kwargs(),
    )[0]

    assert output["image_mask"].sum().item() == 1
    assert output["labels"][output["image_mask"]].eq(IGNORE_INDEX).all()
    assert output["labels"].ne(IGNORE_INDEX).any()


def test_conversation_sharegpt_text_only_requires_no_images():
    sample = {
        "conversations": [
            {"from": "human", "value": "Who wrote The Hobbit?"},
            {"from": "gpt", "value": "J. R. R. Tolkien wrote The Hobbit."},
        ],
        "domain": "text_sft",
    }
    output = process_sample_qwen3_siglip_vlm(
        sample,
        preprocess="conversation",
        text_keys="conversations",
        image_keys=None,
        **_build_transform_kwargs(),
    )[0]

    assert "pixel_values" not in output
    assert output["image_mask"].sum().item() == 0
    assert output["labels"].ne(IGNORE_INDEX).any()
    assert output["domain_name"] == "text_sft"


def test_interleaved_aligned_lists_use_image_tokens_and_text_lm_labels():
    sample = {
        "images": [{"bytes": _image_bytes(), "path": None}, None],
        "texts": [None, "a small image"],
        "domain": "caption",
    }
    output = process_sample_qwen3_siglip_vlm(
        sample,
        preprocess="interleaved",
        text_keys="texts",
        image_keys="images",
        **_build_transform_kwargs(),
    )[0]

    assert output["pixel_values"].shape == (4, 3 * 14 * 14)
    assert output["image_grid_hw"].tolist() == [[2, 2]]
    assert output["image_mask"].sum().item() == 1
    assert output["labels"][output["image_mask"]].eq(IGNORE_INDEX).all()
    assert output["labels"].ne(IGNORE_INDEX).any()
    assert output["domain_name"] == "caption"


def test_interleaved_rejects_unaligned_list_lengths():
    sample = {
        "images": [{"bytes": _image_bytes(), "path": None}, None],
        "texts": [None],
    }

    with pytest.raises(ValueError, match="equal length"):
        process_sample_qwen3_siglip_vlm(
            sample,
            preprocess="interleaved",
            text_keys="texts",
            image_keys="images",
            **_build_transform_kwargs(),
        )


def test_interleaved_rejects_ambiguous_aligned_position():
    sample = {
        "images": [{"bytes": _image_bytes(), "path": None}],
        "texts": ["text at the same position"],
    }

    with pytest.raises(ValueError, match="exactly one"):
        process_sample_qwen3_siglip_vlm(
            sample,
            preprocess="interleaved",
            text_keys="texts",
            image_keys="images",
            **_build_transform_kwargs(),
        )


def test_conversation_rejects_image_placeholder_mismatch():
    sample = {
        "conversations": [
            {"from": "human", "value": "<image>\nWhat is shown?"},
            {"from": "gpt", "value": "A small image."},
            {"from": "human", "value": "<image>\nWhat else is shown?"},
            {"from": "gpt", "value": "Another small image."},
        ],
        "image": {"bytes": _image_bytes(), "path": None},
    }

    with pytest.raises(ValueError, match="Image placeholder count mismatch"):
        process_sample_qwen3_siglip_vlm(
            sample,
            preprocess="conversation",
            text_keys="conversations",
            image_keys="image",
            **_build_transform_kwargs(),
        )


def test_interleaved_skips_unreadable_image_bytes():
    sample = {
        "id": "bad-image",
        "images": [{"bytes": b"not-an-image", "path": None}, None],
        "texts": [None, "caption after a bad image"],
        "domain": "caption",
    }

    output = process_sample_qwen3_siglip_vlm(
        sample,
        preprocess="interleaved",
        text_keys="texts",
        image_keys="images",
        **_build_transform_kwargs(),
    )

    assert output is None
