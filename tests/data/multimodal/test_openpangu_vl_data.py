import io
from types import SimpleNamespace

import torch
from PIL import Image

from veomni.data.data_transform import DATA_TRANSFORM_REGISTRY, process_sample_openpangu_vl
from veomni.data.multimodal.multimodal_chat_template import OpenPanguVLChatTemplate
from veomni.utils.constants import IGNORE_INDEX, IMAGE_INPUT_INDEX


class DummyTokenizer:
    def __init__(self):
        self.vocab = {
            "<unk>": 0,
            "<|pangu_text_start|>": 1,
            "<|message_start|>": 2,
            "<|message_end|>": 3,
            "<|vision_start|>": 4,
            "<|vision_end|>": 5,
            "<|image_pad|>": 6,
            "<|video_pad|>": 7,
            "\uff1a": 8,
            "<|pangu_text_end|>": 9,
        }
        self.unk_token_id = 0
        self.chat_template = None
        self.eos_token_id = 9
        self.eos_token = "<|pangu_text_end|>"

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


class DummyImageProcessor:
    merge_size = 2

    def __call__(self, images, return_tensors="pt"):
        del images, return_tensors
        return {
            "pixel_values": torch.zeros(4, 12),
            "image_grid_thw": torch.tensor([[1, 2, 2]]),
        }


def _image_bytes(size=(4, 4)):
    image = Image.new("RGB", size, color=(64, 128, 32))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def _position_id_func(input_ids, attention_mask=None, **kwargs):
    image_grid_thw = kwargs.get("image_grid_thw")
    if image_grid_thw is not None:
        assert (input_ids == IMAGE_INPUT_INDEX).any()
    return {"position_ids": torch.arange(attention_mask.shape[-1]).view(1, 1, -1).expand(3, 1, -1)}


def _build_transform_kwargs():
    tokenizer = DummyTokenizer()
    return {
        "processor": SimpleNamespace(tokenizer=tokenizer, image_processor=DummyImageProcessor()),
        "chat_template": OpenPanguVLChatTemplate(tokenizer),
        "position_id_func": _position_id_func,
        "max_seq_len": 128,
        "image_max_pixels": 4 * 4,
        "image_min_pixels": 4 * 4,
    }


def test_openpangu_vl_data_type_is_registered():
    assert DATA_TRANSFORM_REGISTRY["openpangu_vl"] is process_sample_openpangu_vl
    assert DATA_TRANSFORM_REGISTRY["pangu_vl"] is process_sample_openpangu_vl


def test_openpangu_vl_template_matches_pangu_message_tokens():
    tokenizer = DummyTokenizer()
    chat_template = OpenPanguVLChatTemplate(tokenizer)
    output = chat_template.encode_messages(
        [
            ["user", ("text", "hello")],
            ["assistant", ("text", "world")],
        ],
        {"image": []},
    )

    ids = output["input_ids"].tolist()
    assert ids[:4] == [
        tokenizer.convert_tokens_to_ids("<|pangu_text_start|>"),
        tokenizer.convert_tokens_to_ids("<|message_start|>"),
        tokenizer.convert_tokens_to_ids("\u7cfb"),
        tokenizer.convert_tokens_to_ids("\u7edf"),
    ]
    assert tokenizer.chat_template == chat_template.get_jinja_template()
    assert output["labels"].eq(IGNORE_INDEX).any()
    assert output["labels"].ne(IGNORE_INDEX).any()


def test_openpangu_vl_transform_supports_conversation_image_samples():
    image_sample = {
        "conversations": [
            {"from": "human", "value": "<image>What is shown?"},
            {"from": "gpt", "value": "A small image."},
        ],
        "images": [_image_bytes()],
        "domain": "raw_qa",
    }
    image_output = process_sample_openpangu_vl(
        image_sample,
        preprocess="conversation",
        text_keys="conversations",
        image_keys="images",
        domain="qa",
        **_build_transform_kwargs(),
    )[0]

    assert image_output["image_mask"].sum().item() == 1
    assert image_output["video_mask"].sum().item() == 0
    assert image_output["input_ids"][image_output["image_mask"]].eq(0).all()
    assert image_output["labels"][image_output["image_mask"]].eq(IGNORE_INDEX).all()
    assert image_output["position_ids"].shape[-1] == image_output["attention_mask"].shape[-1]
    assert image_output["image_grid_thw"].tolist() == [[1, 2, 2]]
    assert image_output["domain_name"] == "qa"


def test_openpangu_vl_transform_supports_plaintext_samples():
    text_sample = {
        "text": "plain language sample",
        "domain": "raw_web",
    }
    text_output = process_sample_openpangu_vl(
        text_sample,
        preprocess="plaintext",
        text_keys="text",
        image_keys=None,
        domain="text_web",
        **_build_transform_kwargs(),
    )[0]

    assert text_output["image_mask"].sum().item() == 0
    assert not (text_output["input_ids"] == IMAGE_INPUT_INDEX).any()
    assert "pixel_values" not in text_output
    assert text_output["labels"].ne(IGNORE_INDEX).any()
    assert text_output["domain_name"] == "text_web"


def test_openpangu_vl_plaintext_chunks_respect_max_seq_len():
    kwargs = _build_transform_kwargs()
    kwargs["max_seq_len"] = 5
    outputs = process_sample_openpangu_vl(
        {"text": "abcdefghijklmnopqrstuvwxyz"},
        preprocess="plaintext",
        text_keys="text",
        image_keys=None,
        **kwargs,
    )

    assert outputs
    assert all(output["attention_mask"].sum().item() <= kwargs["max_seq_len"] for output in outputs)


def test_openpangu_vl_multimodal_transform_keeps_sample_for_dataloader_packing():
    kwargs = _build_transform_kwargs()
    kwargs["max_seq_len"] = 16
    output = process_sample_openpangu_vl(
        {
            "id": "too-long-text",
            "conversations": [
                {"from": "human", "value": "x" * 64},
                {"from": "gpt", "value": "y" * 64},
            ],
            "domain": "raw_sft",
        },
        preprocess="conversation",
        text_keys="conversations",
        image_keys=None,
        domain="text_sft",
        **kwargs,
    )

    assert output
    assert output[0]["attention_mask"].sum().item() > kwargs["max_seq_len"]


def test_openpangu_vl_transform_supports_interleaved_samples():
    sample = {
        "images": [{"bytes": _image_bytes(), "path": None}, None],
        "texts": [None, "a small image"],
        "domain": "raw_caption",
    }
    output = process_sample_openpangu_vl(
        sample,
        preprocess="interleaved",
        text_keys="texts",
        image_keys="images",
        domain="caption",
        **_build_transform_kwargs(),
    )[0]

    assert output["image_mask"].sum().item() == 1
    assert output["labels"][output["image_mask"]].eq(IGNORE_INDEX).all()
    assert output["labels"].ne(IGNORE_INDEX).any()
    assert output["image_grid_thw"].tolist() == [[1, 2, 2]]
    assert output["domain_name"] == "caption"
