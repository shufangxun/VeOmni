import types

import pytest
import torch
import yaml
from torch.utils.data import IterableDataset

import veomni.data.dataset as dataset_module
from veomni.data.chat_template import MixedChatmlTemplate
from veomni.data.data_transform import process_mixed_text_example
from veomni.utils.constants import IGNORE_INDEX
from veomni.utils.multisource_utils import _parse_multisource_config


class DummyTokenizer:
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(ch) % 97 + 1 for ch in text if not ch.isspace()]


def test_mixed_text_plaintext_uses_pretrain_branch_and_chunks():
    template = MixedChatmlTemplate(DummyTokenizer())

    examples = process_mixed_text_example(
        {"text": "abcdef", "domain": "raw_pt"},
        chat_template=template,
        max_seq_len=4,
        preprocess="plaintext",
        text_keys="text",
        domain="pt",
    )

    assert len(examples) == 2
    assert examples[0]["labels"].tolist() == examples[0]["input_ids"].tolist()
    assert examples[1]["labels"].tolist() == examples[1]["input_ids"].tolist()
    assert examples[0]["domain_name"] == "pt"
    assert examples[1]["domain_name"] == "pt"
    assert examples[1]["input_ids"][-1].item() == DummyTokenizer.eos_token_id


def test_mixed_text_conversation_uses_chatml_branch():
    tokenizer = DummyTokenizer()
    template = MixedChatmlTemplate(tokenizer)
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "world"},
    ]

    examples = process_mixed_text_example(
        {"messages": messages, "domain_name": "sft"},
        chat_template=template,
        max_seq_len=4096,
        preprocess="conversation",
        text_keys="messages",
    )

    user_len = len(tokenizer.encode("<|im_start|>user\nhello<|im_end|>\n", add_special_tokens=False))
    labels = examples[0]["labels"]
    input_ids = examples[0]["input_ids"]
    assert labels[:user_len].tolist() == [IGNORE_INDEX] * user_len
    assert labels[user_len:].tolist() == input_ids[user_len:].tolist()
    assert examples[0]["domain_name"] == "sft"


def test_mixed_text_text_keys_override_global_fallback():
    template = MixedChatmlTemplate(DummyTokenizer())

    examples = process_mixed_text_example(
        {"global_text": "bad", "source_text": "good"},
        chat_template=template,
        max_seq_len=16,
        preprocess="plaintext",
        text_keys="source_text",
    )

    expected = DummyTokenizer().encode("good", add_special_tokens=False) + [DummyTokenizer.eos_token_id]
    assert examples[0]["input_ids"].tolist() == expected


def test_mixed_text_requires_known_preprocess():
    template = MixedChatmlTemplate(DummyTokenizer())

    with pytest.raises(ValueError, match="Unsupported mixed_text preprocess"):
        process_mixed_text_example(
            {"text": "hello"},
            chat_template=template,
            max_seq_len=16,
            preprocess="unknown",
            text_keys="text",
        )


def test_multisource_config_validates_per_source_fields():
    config = {
        "sources": ["a", "b"],
        "names": ["a", "b"],
        "preprocess": ["plaintext"],
        "schedule": [{"schedule_type": "const", "weights": [0.5, 0.5]}],
    }

    with pytest.raises(AssertionError, match="preprocess"):
        _parse_multisource_config(config)


def test_weighted_multisource_passes_per_source_transform_kwargs(tmp_path, monkeypatch):
    monkeypatch.setattr(dataset_module, "get_parallel_state", lambda: types.SimpleNamespace(dp_rank=0, dp_size=1))

    class SingleSampleDataset(IterableDataset):
        def __init__(self, transform):
            self.transform = transform

        def __iter__(self):
            sample = {"text": "hello"}
            if self.transform is not None:
                yield self.transform(sample)
            else:
                yield sample

    def fake_build_iterable_dataset(**kwargs):
        return SingleSampleDataset(kwargs["transform"])

    seen_kwargs = {}

    def transform(example, **kwargs):
        del example
        seen_kwargs.update(kwargs)
        return [
            {
                "input_ids": torch.tensor([1]),
                "attention_mask": torch.tensor([1]),
                "labels": torch.tensor([1]),
            }
        ]

    monkeypatch.setattr(dataset_module, "build_iterable_dataset", fake_build_iterable_dataset)
    data_config = {
        "sources": ["source_a", "source_b"],
        "names": ["dataset_a", "dataset_b"],
        "preprocess": ["plaintext", "conversation"],
        "text_keys": ["text", "messages"],
        "schedule": [{"schedule_type": "const", "weights": [1.0, 0.0]}],
    }
    config_path = tmp_path / "data.yaml"
    config_path.write_text(yaml.safe_dump(data_config))

    dataset = dataset_module.build_weighted_multisource_dataset(str(config_path), transform=transform)
    sample = next(iter(dataset))[0]

    assert seen_kwargs["ds_idx"] == 0
    assert seen_kwargs["source_name"] == "dataset_a"
    assert seen_kwargs["preprocess"] == "plaintext"
    assert seen_kwargs["text_keys"] == "text"
    assert sample["ds_idx"] == 0
    assert sample["source_name"] == "dataset_a"


@pytest.mark.parametrize(
    ("preprocess", "text_keys", "source_name"),
    [
        ("plaintext", "text", "pt_only"),
        ("conversation", "messages", "sft_only"),
    ],
)
def test_weighted_multisource_passes_single_source_transform_kwargs(
    tmp_path, monkeypatch, preprocess, text_keys, source_name
):
    monkeypatch.setattr(dataset_module, "get_parallel_state", lambda: types.SimpleNamespace(dp_rank=0, dp_size=1))

    class SingleSampleDataset(IterableDataset):
        def __init__(self, transform):
            self.transform = transform

        def __iter__(self):
            sample = {"text": "hello", "messages": [{"role": "assistant", "content": "hello"}]}
            yield self.transform(sample)

    def fake_build_iterable_dataset(**kwargs):
        return SingleSampleDataset(kwargs["transform"])

    seen_kwargs = {}

    def transform(example, **kwargs):
        del example
        seen_kwargs.update(kwargs)
        return [
            {
                "input_ids": torch.tensor([1]),
                "attention_mask": torch.tensor([1]),
                "labels": torch.tensor([1]),
            }
        ]

    monkeypatch.setattr(dataset_module, "build_iterable_dataset", fake_build_iterable_dataset)
    data_config = {
        "sources": ["source_a"],
        "names": [source_name],
        "preprocess": [preprocess],
        "text_keys": [text_keys],
        "schedule": [{"schedule_type": "const", "weights": [1.0]}],
    }
    config_path = tmp_path / "data.yaml"
    config_path.write_text(yaml.safe_dump(data_config))

    dataset = dataset_module.build_weighted_multisource_dataset(str(config_path), transform=transform)
    sample = next(iter(dataset))[0]

    assert seen_kwargs["ds_idx"] == 0
    assert seen_kwargs["source_name"] == source_name
    assert seen_kwargs["preprocess"] == preprocess
    assert seen_kwargs["text_keys"] == text_keys
    assert sample["ds_idx"] == 0
    assert sample["source_name"] == source_name
