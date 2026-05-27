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

"""Patch configuration for the OpenPangu-V2 text tower used by OpenPangu-VL."""

import torch
import torch.nn as nn
from transformers.activations import ACT2FN

from veomni.ops import fused_moe_forward
from veomni.patchgen.patch_spec import PatchConfig


config = PatchConfig(
    source_module="veomni.models.transformers.openpangu_vl.modeling_openpangu_v2",
    target_file="patched_modeling_openpangu_v2_gpu.py",
    description="OpenPangu-V2 text tower with VeOmni fused MoE dispatch",
)

config.add_import("veomni.ops", names=["fused_moe_forward"])
config.add_import("veomni.ops.dispatch", names=["OpSlot"])
config.drop_import_names("use_experts_implementation", "use_kernel_func_from_hub")
config.add_post_import_block(
    """
    veomni_openpangu_v2_rms_norm = OpSlot("rms_norm", "standard")
    veomni_openpangu_v2_swiglu_mlp = OpSlot("swiglu_mlp", "standard")
    veomni_openpangu_v2_apply_rotary_pos_emb = OpSlot("rotary_pos_emb", "full")
    veomni_moe_experts_forward = OpSlot("moe_experts", "standard")
    """
)


@config.override_method(
    "OpenPanguV2RMSNorm.forward",
    description="OpSlot guard for fused RMSNorm while preserving OpenPangu-V2 eager semantics.",
)
def openpangu_v2_rmsnorm_forward_patched(self, hidden_states: torch.Tensor) -> torch.Tensor:
    if veomni_openpangu_v2_rms_norm.use_non_eager_impl:
        return veomni_openpangu_v2_rms_norm(hidden_states, self.weight, self.variance_epsilon)

    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    return self.weight * hidden_states.to(input_dtype)


@config.override_method(
    "OpenPanguV2MLP.forward",
    description="OpSlot guard for Liger SwiGLU MLP on SiLU OpenPangu-V2 dense/shared MLPs.",
)
def openpangu_v2_mlp_forward_patched(self, x):
    if self.config.hidden_act == "silu" and veomni_openpangu_v2_swiglu_mlp.use_non_eager_impl:
        return veomni_openpangu_v2_swiglu_mlp(self, x)

    down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    return down_proj


@config.replace_function(
    "apply_rotary_pos_emb",
    description="OpSlot guard for fused text RoPE; interleaved OpenPangu RoPE stays on the eager path.",
)
def openpangu_v2_apply_rotary_pos_emb_patched(q, k, cos, sin, unsqueeze_dim=1):
    if veomni_openpangu_v2_apply_rotary_pos_emb.use_non_eager_impl:
        return veomni_openpangu_v2_apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)

    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


@config.replace_class("OpenPanguV2Experts", description="Use explicit VeOmni fused MoE dispatch path")
class PatchedOpenPanguV2Experts(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.n_routed_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        if veomni_moe_experts_forward.use_non_eager_impl:
            return fused_moe_forward(
                num_experts=self.num_experts,
                routing_weights=top_k_weights.to(final_hidden_states.dtype),
                selected_experts=top_k_index,
                hidden_states=hidden_states,
                fc1_1_weight=None,
                fc1_2_weight=None,
                fc2_weight=self.down_proj,
                fc1_1_2_weight=self.gate_up_proj,
            )

        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate_up = torch.nn.functional.linear(current_state, self.gate_up_proj[expert_idx])
            gate, up = gate_up.chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = torch.nn.functional.linear(current_hidden_states, self.down_proj[expert_idx])
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states
