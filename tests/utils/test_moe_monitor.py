from types import SimpleNamespace

import torch

from veomni.trainer.callbacks.trace_callback import MoERouterMonitorCallback
from veomni.utils.moe_monitor import MoERouterMonitor, ensure_router_monitor_hooks


def test_compute_vio_matches_maxvio_definition():
    load_matrix = torch.tensor([[0.5, 0.25, 0.25, 0.0]])

    vio = MoERouterMonitor.compute_vio(load_matrix)

    assert torch.allclose(vio["max_vio"], torch.tensor([1.0]))
    assert torch.allclose(vio["min_vio"], torch.tensor([-1.0]))
    assert torch.allclose(vio["avg_vio"], torch.tensor([0.5]))


def test_compute_maxvio_batch_averages_layer_maxvio():
    load_matrix = torch.tensor(
        [
            [0.5, 0.25, 0.25, 0.0],
            [0.25, 0.25, 0.25, 0.25],
        ]
    )

    assert torch.allclose(MoERouterMonitor.compute_maxvio_batch(load_matrix), torch.tensor(0.5))


def test_get_load_matrix_normalizes_and_resets_counts_without_dist():
    monitor = MoERouterMonitor(num_experts=4)
    module = torch.nn.Linear(1, 1)
    module.train()
    monitor.record(module, torch.tensor([[0, 1], [0, 2], [0, 3], [0, 0]]))

    load_matrix = monitor.get_load_matrix(global_reduce=False)

    assert torch.allclose(load_matrix, torch.tensor([[0.625, 0.125, 0.125, 0.125]]))
    assert torch.equal(monitor.get_load_matrix(global_reduce=False), torch.zeros(1, 4))


def test_moe_router_monitor_reads_nested_text_config_num_experts():
    config = SimpleNamespace(text_config=SimpleNamespace(num_experts=128))

    assert MoERouterMonitorCallback._get_num_experts(config) == 128


def test_moe_router_monitor_reads_pangu_routed_experts():
    config = SimpleNamespace(text_config=SimpleNamespace(n_routed_experts=256))

    assert MoERouterMonitorCallback._get_num_experts(config) == 256


def test_moe_router_monitor_wandb_metrics_only_include_batch_maxvio():
    metrics = MoERouterMonitorCallback._build_wandb_metrics(maxvio_batch_sum=1.5, maxvio_batch_count=3)

    assert metrics == {"moe/maxvio_batch": 0.5}


def test_record_ignores_eval_modules():
    monitor = MoERouterMonitor(num_experts=4)
    module = torch.nn.Linear(1, 1)
    module.eval()

    monitor.record(module, torch.tensor([[0, 1]]))

    assert torch.equal(monitor.get_load_matrix(global_reduce=False), torch.zeros(0, 4))


def test_aux_free_bias_update_uses_count_sign_and_resets():
    monitor = MoERouterMonitor(num_experts=4)
    module = torch.nn.Module()
    module.e_score_correction_bias = torch.zeros(4)
    module.train()
    monitor.record(module, torch.tensor([[0, 0], [0, 1], [2, 3]]))

    counts, modules = monitor.drain_counts(global_reduce=False)
    metrics = MoERouterMonitor.update_aux_free_bias(counts, modules, update_rate=0.1, return_metrics=True)

    assert torch.allclose(module.e_score_correction_bias, torch.tensor([-0.1, 0.1, 0.1, 0.1]))
    assert metrics["moe/aux_free_bias_updated_layers"] == 1.0
    assert abs(metrics["moe/aux_free_bias_max_abs"] - 0.1) < 1e-6
    assert torch.equal(monitor.get_load_matrix(global_reduce=False), torch.zeros(1, 4))


class ToyRouter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.top_k = 2
        self.num_experts = 4

    def forward(self, x):
        return x, x, torch.tensor([[0, 1]])


def test_ensure_router_monitor_hooks_is_idempotent():
    model = torch.nn.Sequential(ToyRouter())

    assert ensure_router_monitor_hooks(model) == 1
    assert ensure_router_monitor_hooks(model) == 0
