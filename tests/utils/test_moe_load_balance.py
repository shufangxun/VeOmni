import sys

import torch

from veomni.arguments import TrainingArguments, VeOmniArguments, parse_args
from veomni.utils.moe_monitor import MoERouterMonitor, attach_moe_router_monitor


class _DummyMoeBlock(torch.nn.Module):
    def __init__(self, num_experts: int):
        super().__init__()
        self.register_buffer("e_score_correction_bias", torch.zeros(num_experts))


class OpenPanguV2SparseMoeBlock(_DummyMoeBlock):
    pass


def test_aux_free_bias_update_uses_load_sign():
    monitor = MoERouterMonitor(num_experts=4)
    block = _DummyMoeBlock(num_experts=4)
    monitor._register_layer(block)

    stats = monitor.update_aux_free_bias(torch.tensor([[10, 5, 5, 0]]), update_rate=0.001)

    assert torch.allclose(block.e_score_correction_bias, torch.tensor([-0.001, 0.0, 0.0, 0.001]))
    assert stats["aux_free_bias/updated_layers"] == 1.0
    assert stats["aux_free_bias/update_rate"] == 0.001


def test_openpangu_sparse_moe_block_is_registered_for_external_monitoring():
    monitor = MoERouterMonitor(num_experts=4)
    model = torch.nn.Sequential(OpenPanguV2SparseMoeBlock(num_experts=4))

    assert attach_moe_router_monitor(model, monitor) == 1


def test_legacy_moe_monitor_interval_maps_to_structured_config():
    args = TrainingArguments(
        global_batch_size=1,
        micro_batch_size=1,
        moe_load_balance_monitor_interval=7,
    )

    assert args.moe_load_balance.monitor_interval == 7
    assert args.moe_load_balance.mode == "auto"


def test_structured_moe_load_balance_cli_override():
    old_argv = sys.argv
    try:
        sys.argv = [
            "test",
            "--model.config_path",
            "dummy-model",
            "--data.train_path",
            "dummy-data",
            "--train.global_batch_size",
            "1",
            "--train.micro_batch_size",
            "1",
            "--train.moe_load_balance.mode",
            "aux_free",
            "--train.moe_load_balance.monitor_interval",
            "3",
            "--train.moe_load_balance.aux_free_bias_update_rate",
            "0.002",
        ]
        args = parse_args(VeOmniArguments)
    finally:
        sys.argv = old_argv

    assert args.train.moe_load_balance.mode == "aux_free"
    assert args.train.moe_load_balance.monitor_interval == 3
    assert args.train.moe_load_balance.aux_free_bias_update_rate == 0.002
