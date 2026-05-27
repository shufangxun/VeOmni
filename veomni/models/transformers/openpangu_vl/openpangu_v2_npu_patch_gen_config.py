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

from veomni.models.transformers.openpangu_vl.openpangu_v2_gpu_patch_gen_config import (
    PatchedOpenPanguV2Experts,
    openpangu_v2_apply_rotary_pos_emb_patched,
    openpangu_v2_mlp_forward_patched,
    openpangu_v2_rmsnorm_forward_patched,
)
from veomni.patchgen.patch_spec import PatchConfig


config = PatchConfig(
    source_module="veomni.models.transformers.openpangu_vl.modeling_openpangu_v2",
    target_file="patched_modeling_openpangu_v2_npu.py",
    description="OpenPangu-V2 text tower with VeOmni fused MoE dispatch for NPU import compatibility",
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
config.override_method(
    "OpenPanguV2RMSNorm.forward",
    replacement=openpangu_v2_rmsnorm_forward_patched,
    description="OpSlot guard for fused RMSNorm while preserving OpenPangu-V2 eager semantics.",
)
config.override_method(
    "OpenPanguV2MLP.forward",
    replacement=openpangu_v2_mlp_forward_patched,
    description="OpSlot guard for Liger SwiGLU MLP on SiLU OpenPangu-V2 dense/shared MLPs.",
)
config.replace_function(
    "apply_rotary_pos_emb",
    replacement=openpangu_v2_apply_rotary_pos_emb_patched,
    description="OpSlot guard for fused text RoPE; interleaved OpenPangu RoPE stays on the eager path.",
)
config.replace_class(
    "OpenPanguV2Experts",
    replacement=PatchedOpenPanguV2Experts,
    description="Use explicit VeOmni fused MoE dispatch path",
)
