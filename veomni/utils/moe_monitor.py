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

"""MoE Router Load Balance Monitor.

Provides tools to monitor expert load distribution across MoE layers during training.
Logs the batch-level MaxVio metric averaged across MoE layers to wandb.

Architecture:
    1. Router modules (e.g. PatchQwen3MoeTopKRouter) register ``router_forward_hook``
       in their ``__init__``. This happens at model construction time.
    2. The hook is a no-op (single ``if`` check) until a ``MoERouterMonitor`` is
       created and activated via ``set_active_monitor()``. This is done by the
       trainer when ``moe_load_balance_monitor_interval > 0``.
    3. Once active, each router forward accumulates token-to-expert counts on device.
       At the end of each training step, ``get_load_matrix()`` moves counts to CPU
       and produces a normalized frequency matrix for that training batch.
    4. ``compute_maxvio_batch()`` derives the paper-style MaxVio averaged across
       MoE layers. The callback only logs that scalar at the configured interval.

This design avoids FSDP compatibility issues — hooks are on the original router modules,
not discovered through the FSDP wrapper at runtime.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
from torch import distributed as dist


# ---------------------------------------------------------------------------
# Global active monitor singleton.
# Router forward hooks check this; when None, the hook is a no-op.
# Activated by the trainer/callback via set_active_monitor().
# ---------------------------------------------------------------------------
_active_monitor: Optional["MoERouterMonitor"] = None


def get_active_monitor() -> Optional["MoERouterMonitor"]:
    """Return the currently active MoE router monitor, or None if disabled."""
    return _active_monitor


def set_active_monitor(monitor: Optional["MoERouterMonitor"]) -> None:
    """Activate or deactivate the global MoE router monitor.

    Args:
        monitor: A ``MoERouterMonitor`` instance to activate, or ``None`` to deactivate.
    """
    global _active_monitor
    _active_monitor = monitor


def router_forward_hook(module: nn.Module, input, output):
    """PyTorch forward hook registered on MoE router modules at construction time.

    When no monitor is active (``_active_monitor is None``), this is effectively
    a no-op — just a single ``if`` check per router forward, with negligible overhead.

    Expected ``output`` format: ``(router_logits, router_scores, router_indices)``
    where ``router_indices`` has shape ``[num_tokens, top_k]``.
    """
    if _active_monitor is None:
        return
    # router_indices: [num_tokens, top_k] — the selected expert indices per token
    _active_monitor.record(module, output[2])


def _is_moe_router_module(module: nn.Module) -> bool:
    class_name = module.__class__.__name__.lower()
    return "router" in class_name and hasattr(module, "top_k") and hasattr(module, "num_experts")


def ensure_router_monitor_hooks(model: nn.Module) -> int:
    """Register router monitor hooks on MoE router modules if missing.

    Returns the number of newly registered hooks. Hooks are idempotent per module
    instance via the _veomni_moe_monitor_hook_registered marker.
    """
    num_registered = 0
    for module in model.modules():
        if not _is_moe_router_module(module):
            continue
        if getattr(module, "_veomni_moe_monitor_hook_registered", False):
            continue
        module.register_forward_hook(router_forward_hook)
        module._veomni_moe_monitor_hook_registered = True
        num_registered += 1
    return num_registered


class MoERouterMonitor:
    """Monitors MoE expert load distribution via router forward hooks.

    Router modules register ``router_forward_hook`` at construction time (in the
    model patch, e.g. ``PatchQwen3MoeTopKRouter.__init__``). The hook is a no-op
    until a monitor is activated via ``set_active_monitor()``. This avoids FSDP
    compatibility issues and module-walking entirely.

    Typical usage (via ``MoERouterMonitorCallback``)::

        # At train begin:
        monitor = MoERouterMonitor(num_experts=128)
        set_active_monitor(monitor)

        # ... training runs, hooks auto-accumulate counts ...

        # At the end of each training step:
        load_matrix = monitor.get_load_matrix(current_step=step)
        maxvio_batch = MoERouterMonitor.compute_maxvio_batch(load_matrix)

        # At train end:
        set_active_monitor(None)

    Attributes:
        num_experts: Total number of experts in the MoE model.
    """

    def __init__(self, num_experts: int):
        """Initialize the monitor.

        Args:
            num_experts: Number of experts per MoE layer (e.g. 128 for Qwen3-30B-A3B).
        """
        self.num_experts = num_experts
        # Maps module id -> accumulated token counts tensor (on device, shape [num_experts]).
        # Using id(module) as key so we can track each router instance independently.
        self._counts: Dict[int, torch.Tensor] = {}
        # Ordered list of module ids, preserving layer discovery order (layer 0 first).
        self._layer_order: list = []
        # Step range tracking for heatmap captions.
        self._accumulate_start_step: int = 0
        self._accumulate_end_step: int = 0
        self._last_step_range: tuple = (0, 0)

    def record(self, module: nn.Module, router_indices: torch.Tensor):
        """Record expert selections from a single router forward pass.

        Called by ``router_forward_hook``. Accumulates on device (no CPU sync).

        Args:
            module: The router module instance (used as key via ``id(module)``).
            router_indices: Expert indices of shape ``[num_tokens, top_k]``.
        """
        mid = id(module)
        # Lazily initialize the counts tensor for new router modules.
        # The first forward pass through each router auto-registers it.
        if mid not in self._counts:
            self._layer_order.append(mid)
            device = router_indices.device
            self._counts[mid] = torch.zeros(self.num_experts, dtype=torch.long, device=device)
        # Count how many tokens were routed to each expert.
        counts = torch.bincount(
            router_indices.reshape(-1).to(torch.long),
            minlength=self.num_experts,
        )
        # Accumulate on device — detach to avoid graph retention.
        self._counts[mid] += counts.detach()

    def get_load_matrix(self, current_step: int = 0, global_reduce: bool = True) -> torch.Tensor:
        """Return the normalized load matrix and reset accumulated counts.

        This method moves data from device to CPU, causing a CUDA sync. The training
        callback calls it once per step when MoE monitoring is enabled so the recorded
        value matches the paper's per-training-batch MaxVio definition. When distributed
        training is initialized, all ranks must call this method together because
        global_reduce=True all-reduces the accumulated counts.

        Args:
            current_step: The current global training step, used to record
                the accumulation range for heatmap captions.
            global_reduce: Whether to sum counts across the default process group
                before normalizing. This gives the computation-batch/global-rank
                MaxVio used for EP/DP efficiency monitoring.

        Returns:
            A float tensor of shape ``[num_moe_layers, num_experts]`` where each
            row sums to 1.0, representing the fraction of tokens routed to each
            expert in that layer.
        """
        if not self._counts:
            return torch.zeros(0, self.num_experts)
        self._accumulate_end_step = current_step
        matrix = torch.stack([self._counts[mid] for mid in self._layer_order]).float()
        if global_reduce and dist.is_available() and dist.is_initialized():
            dist.all_reduce(matrix, op=dist.ReduceOp.SUM)
        # Move to CPU after the optional global reduction (single sync point).
        matrix = matrix.cpu()
        # Normalize each row (layer) to sum to 1.0.
        row_sums = matrix.sum(dim=1, keepdim=True).clamp(min=1.0)
        matrix = matrix / row_sums
        # Save step range for caption, then reset for next interval.
        self._last_step_range = (self._accumulate_start_step, self._accumulate_end_step)
        self._reset_counts()
        self._accumulate_start_step = current_step + 1
        return matrix

    def create_wandb_image(self, load_matrix: torch.Tensor, caption: str = None):
        """Create a wandb.Image heatmap from the normalized load matrix.

        The heatmap has expert index on the X axis, MoE layer index on the Y axis,
        and color intensity representing normalized token frequency.

        Args:
            load_matrix: Normalized ``[num_moe_layers, num_experts]`` tensor from
                ``get_load_matrix()``.
            caption: Optional caption override. If ``None``, auto-generates from
                the accumulated step range (e.g. "Steps 11-20").

        Returns:
            A ``wandb.Image`` object ready to be logged via ``wandb.log()``.
        """
        import io

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import wandb
        except ModuleNotFoundError:
            return None

        if caption is None:
            start, end = self._last_step_range
            caption = f"Steps {start}-{end}"

        fig, ax = plt.subplots(figsize=(max(8, load_matrix.shape[1] * 0.1), max(4, load_matrix.shape[0] * 0.2)))
        im = ax.imshow(load_matrix.numpy(), aspect="auto", cmap="YlOrRd")
        ax.set_xlabel("Expert Index")
        ax.set_ylabel("MoE Layer Index")
        ax.set_title(f"MoE Expert Load Distribution ({caption})")
        fig.colorbar(im, ax=ax, label="Normalized Token Frequency")
        fig.tight_layout()

        buf = io.BytesIO()
        try:
            fig.savefig(buf, format="png", dpi=100)
            plt.close(fig)
            buf.seek(0)

            from PIL import Image

            image = wandb.Image(Image.open(buf), caption=caption)
            return image
        finally:
            buf.close()

    @staticmethod
    def compute_vio(load_matrix: torch.Tensor) -> dict:
        """Compute per-layer load balance violation metrics.

        Given a normalized load matrix where each row sums to 1.0, computes the
        deviation from uniform distribution for each layer:

            deviation = normalized_freq * num_experts - 1

        Under a perfectly uniform distribution, every expert gets ``1/num_experts``
        of the tokens, so ``deviation = 0`` everywhere. Metrics:

        - **max_vio**: ``deviation.max()`` per layer. Measures the most *overloaded*
          expert. Range: ``[0, num_experts - 1]``. Closer to 0 = more balanced.
        - **min_vio**: ``deviation.min()`` per layer. Measures the most *underloaded*
          expert. Range: ``[-1, 0]``. Closer to 0 = more balanced.
        - **avg_vio**: ``|deviation|.mean()`` per layer. Measures the average absolute
          deviation from uniform. Range: ``[0, ...]``. Closer to 0 = more balanced.

        Args:
            load_matrix: Normalized ``[num_moe_layers, num_experts]`` tensor from
                ``get_load_matrix()``.

        Returns:
            Dict with keys ``"max_vio"``, ``"min_vio"``, ``"avg_vio"``, each a
            tensor of shape ``[num_moe_layers]``.
        """
        num_experts = load_matrix.shape[1]
        # deviation from uniform: 0 means perfectly balanced
        deviation = load_matrix * num_experts - 1.0
        return {
            "max_vio": deviation.max(dim=1).values,
            "min_vio": deviation.min(dim=1).values,
            "avg_vio": deviation.abs().mean(dim=1),
        }

    @staticmethod
    def compute_maxvio_batch(load_matrix: torch.Tensor) -> torch.Tensor:
        """Compute paper-style batch MaxVio averaged across MoE layers.

        The paper defines per-layer MaxVio as the maximum overload ratio:

            ``max_i((Load_i - expected_load_i) / expected_load_i)``

        ``load_matrix`` is already normalized by total routed assignments in
        each layer, so the expected load is ``1 / num_experts`` and the overload
        ratio is ``load_matrix * num_experts - 1``. The reported model-level
        value is the mean across layers.
        """
        if load_matrix.numel() == 0:
            return torch.tensor(float("nan"))
        return MoERouterMonitor.compute_vio(load_matrix)["max_vio"].mean()

    def _reset_counts(self):
        """Zero out all accumulated counts (on device) for the next interval."""
        for mid in self._counts:
            self._counts[mid].zero_()
