import types
from collections import defaultdict

import pytest
import torch

pytest.importorskip("transformers")

from veomni.data.data_transform import process_plaintext_example  # noqa: E402
from veomni.trainer import text_trainer as text_trainer_module  # noqa: E402
from veomni.trainer.callbacks.evaluate_callback import EvaluateCallback  # noqa: E402
from veomni.trainer.text_trainer import TextTrainer  # noqa: E402


class DummyTokenizer:
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return list(range(1, len(text.split()) + 1))


def _trainer() -> TextTrainer:
    trainer = TextTrainer.__new__(TextTrainer)
    trainer.train_source_names = ["fineweb", "finewiki"]
    return trainer


def _loss_stats():
    return defaultdict(lambda: {"loss_sum": 0.0, "tokens": 0.0})


def test_plaintext_transform_copies_domain_to_all_chunks():
    examples = process_plaintext_example(
        {"text": "one two three four five", "domain": "web"},
        tokenizer=DummyTokenizer(),
        max_seq_len=3,
        text_keys="text",
    )

    assert [example["domain_name"] for example in examples] == ["web", "web"]
    assert [example["input_ids"].tolist() for example in examples] == [[1, 2, 3], [4, 5, 0]]


def test_plaintext_transform_omits_missing_domain():
    examples = process_plaintext_example(
        {"text": "one two"},
        tokenizer=DummyTokenizer(),
        max_seq_len=8,
        text_keys="text",
    )

    assert "domain_name" not in examples[0]


def test_text_trainer_accumulates_domain_loss(monkeypatch):
    monkeypatch.setattr(
        text_trainer_module,
        "get_parallel_state",
        lambda: types.SimpleNamespace(dp_group=None, sp_enabled=False),
    )
    monkeypatch.setattr(text_trainer_module, "all_reduce", lambda value, op="sum", group=None: value)

    trainer = _trainer()
    micro_batch = {
        "position_ids": torch.tensor([[0, 1, 0, 1, 2]], dtype=torch.long),
        "source_name": ["fineweb", "finewiki"],
        "domain_name": ["web", "knowledge"],
    }
    metadata = trainer._snapshot_train_loss_metadata([micro_batch], include_source=True, include_domain=True)[0]
    stats = _loss_stats()

    trainer._accumulate_train_group_loss(
        stats,
        metadata,
        "domain_names",
        torch.tensor([-1.0, -2.0, -3.0, -4.0, 0.0]),
    )

    metrics = trainer._build_train_group_metrics(
        stats,
        aggregation_time=0.001,
        metric_prefix="training/domain",
        aggregation_metric_name="training/domain_loss/aggregation_time_ms",
    )

    assert metrics["training/domain/web/loss"] == pytest.approx(1.5)
    assert metrics["training/domain/web/tokens"] == 2
    assert metrics["training/domain/knowledge/loss"] == pytest.approx(3.5)
    assert metrics["training/domain/knowledge/tokens"] == 2


def test_text_trainer_skips_missing_domain_metadata():
    trainer = _trainer()
    micro_batch = {
        "position_ids": torch.tensor([[0, 1, 0, 1]], dtype=torch.long),
        "source_name": ["fineweb", "finewiki"],
    }
    metadata = trainer._snapshot_train_loss_metadata([micro_batch], include_source=True, include_domain=True)[0]
    stats = _loss_stats()

    trainer._accumulate_train_group_loss(
        stats,
        metadata,
        "domain_names",
        torch.tensor([-1.0, -2.0, -3.0, -4.0]),
    )

    assert dict(stats) == {}


def test_evaluate_callback_accumulates_domain_loss(monkeypatch):
    import veomni.distributed.parallel_state as parallel_state

    monkeypatch.setattr(parallel_state, "get_parallel_state", lambda: types.SimpleNamespace(sp_enabled=False))

    callback = EvaluateCallback.__new__(EvaluateCallback)
    micro_batch = {
        "position_ids": torch.tensor([[0, 1, 0, 1, 2]], dtype=torch.long),
        "domain_name": ["web", "knowledge"],
    }
    metadata = callback._snapshot_domain_metadata(micro_batch)
    stats = _loss_stats()

    callback._accumulate_eval_domain_loss(
        stats,
        metadata,
        torch.tensor([-1.0, -2.0, -3.0, -4.0, 0.0]),
    )

    assert stats["web"]["loss_sum"] == pytest.approx(3.0)
    assert stats["web"]["tokens"] == 2
    assert stats["knowledge"]["loss_sum"] == pytest.approx(7.0)
    assert stats["knowledge"]["tokens"] == 2
