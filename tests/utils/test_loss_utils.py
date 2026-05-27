from types import SimpleNamespace

import torch

from veomni.utils import loss_utils


class _FakeParallelState:
    def __init__(self, *, sp_enabled: bool, fsdp_size: int = 4, sp_size: int = 1):
        self.sp_enabled = sp_enabled
        self.fsdp_size = fsdp_size
        self.sp_size = sp_size
        self.sp_group = object()


def _patch_parallel(monkeypatch, *, sp_enabled: bool, sp_size: int = 1, global_factor: int = 4):
    parallel_state = _FakeParallelState(sp_enabled=sp_enabled, sp_size=sp_size)
    monkeypatch.setattr(loss_utils, "get_parallel_state", lambda: parallel_state)

    def fake_all_reduce(value, op="sum", group=None):
        if group is parallel_state.sp_group:
            return value * sp_size
        return value * global_factor

    monkeypatch.setattr(loss_utils, "all_reduce", fake_all_reduce)
    return parallel_state


def test_global_loss_weight_matches_mean_global_loss_without_sp(monkeypatch):
    _patch_parallel(monkeypatch, sp_enabled=False, global_factor=4)
    micro = {"foundation_tokens": torch.tensor(100)}
    global_micro_batches = {"foundation_tokens": torch.tensor(100)}

    weight = loss_utils.global_loss_weight("foundation", micro, global_micro_batches)

    assert weight == 1.0


def test_global_loss_weight_keeps_aux_loss_sp_invariant(monkeypatch):
    _patch_parallel(monkeypatch, sp_enabled=True, sp_size=2, global_factor=4)
    micro = {"foundation_tokens": torch.tensor(50)}
    global_micro_batches = {"foundation_tokens": torch.tensor(100)}

    weight = loss_utils.global_loss_weight("foundation", micro, global_micro_batches)

    assert weight == 0.5
    # Two SP=2 accumulation micro-batches with local aux loss around 8 should log
    # the same global-batch-scaled aux loss as one SP=1 micro-batch.
    assert 8.32 * weight * 2 == 8.32


def test_base_trainer_postforward_scales_router_aux_loss(monkeypatch):
    from veomni.trainer.base import BaseTrainer

    _patch_parallel(monkeypatch, sp_enabled=True, sp_size=2, global_factor=4)
    trainer = object.__new__(BaseTrainer)
    trainer.micro_batch_token_len = {"foundation_tokens": torch.tensor(50)}
    trainer.micro_batches_token_len = {"foundation_tokens": torch.tensor(100)}
    trainer.model_config = SimpleNamespace(text_config=SimpleNamespace(router_aux_loss_coef=0.001))

    aux_loss = torch.tensor(8.0)
    outputs = SimpleNamespace(loss=torch.tensor(2.008), aux_loss=aux_loss)

    loss, loss_dict = trainer.postforward(outputs, {})

    torch.testing.assert_close(loss_dict["foundation_loss"], torch.tensor(1.0))
    torch.testing.assert_close(loss_dict["load_balancing_loss"], torch.tensor(4.0))
    torch.testing.assert_close(loss_dict["load_balancing_loss_weighted"], torch.tensor(0.004))
    torch.testing.assert_close(loss, torch.tensor(1.004))


def test_base_trainer_postforward_disables_router_aux_loss_by_strategy(monkeypatch):
    from veomni.trainer.base import BaseTrainer

    _patch_parallel(monkeypatch, sp_enabled=True, sp_size=2, global_factor=4)
    trainer = object.__new__(BaseTrainer)
    trainer.micro_batch_token_len = {"foundation_tokens": torch.tensor(50)}
    trainer.micro_batches_token_len = {"foundation_tokens": torch.tensor(100)}
    trainer.model_config = SimpleNamespace(text_config=SimpleNamespace(router_aux_loss_coef=0.001))
    trainer.args = SimpleNamespace(train=SimpleNamespace(moe_load_balance_strategy="aux_free"))

    aux_loss = torch.tensor(8.0)
    outputs = SimpleNamespace(loss=torch.tensor(2.008), aux_loss=aux_loss)

    loss, loss_dict = trainer.postforward(outputs, {})

    torch.testing.assert_close(loss_dict["foundation_loss"], torch.tensor(1.0))
    torch.testing.assert_close(loss_dict["load_balancing_loss"], torch.tensor(4.0))
    assert "load_balancing_loss_weighted" not in loss_dict
    torch.testing.assert_close(loss, torch.tensor(1.0))
