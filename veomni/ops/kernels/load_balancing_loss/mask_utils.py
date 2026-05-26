from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist


try:
    from ....distributed.parallel_state import get_parallel_state
except Exception:  # pragma: no cover - import-time fallback for standalone tests
    get_parallel_state = None


def get_sequence_parallel_group() -> Optional[dist.ProcessGroup]:
    """Return the active sequence-parallel process group for router-loss stats."""
    if get_parallel_state is None or not dist.is_available() or not dist.is_initialized():
        return None

    try:
        parallel_state = get_parallel_state()
        if not parallel_state.sp_enabled or parallel_state.sp_size <= 1:
            return None
        return parallel_state.sp_group
    except Exception:
        return None


def sp_all_reduce_sum_(tensor: torch.Tensor) -> torch.Tensor:
    """In-place sum over the sequence-parallel group when SP is active."""
    group = get_sequence_parallel_group()
    if group is not None:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    return tensor


class _SPAllReduceSumWithLocalBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor: torch.Tensor) -> torch.Tensor:
        reduced = tensor.clone()
        sp_all_reduce_sum_(reduced)
        return reduced

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output


def sp_all_reduce_sum_with_local_backward(tensor: torch.Tensor) -> torch.Tensor:
    """Sum over SP ranks in forward while keeping local gradients local."""
    if get_sequence_parallel_group() is None:
        return tensor
    return _SPAllReduceSumWithLocalBackward.apply(tensor)


def maybe_slice_attention_mask_for_gate_logits(
    attention_mask: Optional[torch.Tensor],
    tokens_per_layer: int,
) -> Optional[torch.Tensor]:
    """Match a global attention mask to local SP router logits when needed.

    MoE router logits are produced after sequence-parallel slicing, so each
    layer can expose only ``batch * ceil(seq_len / sp_size)`` rows on a rank
    while the model forward still passes the global ``[batch, seq_len]`` mask
    to the auxiliary loss. In that case, slice the mask with the same
    padding-and-contiguous-chunk rule used by SP inputs.
    """
    if attention_mask is None or attention_mask.ndim != 2:
        return attention_mask

    batch_size, sequence_length = attention_mask.shape
    if tokens_per_layer == batch_size * sequence_length:
        return attention_mask

    if get_parallel_state is None:
        return attention_mask

    try:
        parallel_state = get_parallel_state()
        sp_enabled = bool(parallel_state.sp_enabled)
        sp_size = int(parallel_state.sp_size)
        sp_rank = int(parallel_state.sp_rank)
    except Exception:
        return attention_mask

    if not sp_enabled or sp_size <= 1 or sp_rank < 0:
        return attention_mask

    local_seq_len = (sequence_length + sp_size - 1) // sp_size
    if tokens_per_layer != batch_size * local_seq_len:
        return attention_mask

    padded_seq_len = local_seq_len * sp_size
    if padded_seq_len != sequence_length:
        pad_shape = list(attention_mask.shape)
        pad_shape[1] = padded_seq_len - sequence_length
        padding = torch.zeros(pad_shape, dtype=attention_mask.dtype, device=attention_mask.device)
        attention_mask = torch.cat((attention_mask, padding), dim=1)

    start = sp_rank * local_seq_len
    return attention_mask[:, start : start + local_seq_len].contiguous()
