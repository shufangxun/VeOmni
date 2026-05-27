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

from veomni.models.transformers.openpangu_vl.openpangu_vl_gpu_patch_gen_config import (
    OpenPanguVLPositionModel,
    get_openpangu_vl_position_id,
    openpangu_vl_forward_patched,
    openpangu_vl_get_position_id_func_patched,
    openpangu_vl_mlp_forward_patched,
    openpangu_vl_model_forward_patched,
    openpangu_vl_model_get_image_features_patched,
    openpangu_vl_model_get_position_id_func_patched,
    openpangu_vl_model_get_video_features_patched,
    openpangu_vl_parse_preprocess_params_patched,
    openpangu_vl_rmsnorm_forward_patched,
    openpangu_vl_vision_attention_forward_patched,
    openpangu_vl_vision_block_forward_patched,
    openpangu_vl_vision_dummy_forward_patched,
    openpangu_vl_vision_forward_patched,
)
from veomni.patchgen.patch_spec import PatchConfig


config = PatchConfig(
    source_module="veomni.models.transformers.openpangu_vl.modeling_openpangu_vl",
    target_file="patched_modeling_openpangu_vl_npu.py",
    description="OpenPangu-VL NPU import-compatible modeling",
)

config.drop_import_names(
    "ALL_ATTENTION_FUNCTIONS",
    "AutoProcessor",
    "OpenPanguV2Model",
    "PreTrainedModel",
    "torch_npu",
    "transfer_to_npu",
)
config.add_import(
    "veomni.models.transformers.openpangu_vl.generated.patched_modeling_openpangu_v2_npu",
    names=[
        "OpenPanguV2Model",
        "veomni_moe_experts_forward",
        "veomni_openpangu_v2_apply_rotary_pos_emb",
        "veomni_openpangu_v2_rms_norm",
        "veomni_openpangu_v2_swiglu_mlp",
    ],
)
config.add_import("torch.distributed", alias="dist", is_from_import=False)
config.add_import(
    "transformers.modeling_utils",
    names=["ALL_ATTENTION_FUNCTIONS", "PreTrainedModel", "is_flash_attention_requested"],
)
config.add_import("veomni.distributed.parallel_state", names=["get_parallel_state"])
config.add_import(
    "veomni.distributed.sequence_parallel",
    names=["gather_outputs", "pad_tensor", "slice_input_tensor", "sp_pad_and_slice", "unpad_tensor"],
)
config.add_import("veomni.ops.dispatch", names=["OpSlot"])
config.add_post_import_block(
    """
    try:
        if is_torch_npu_available() and hasattr(torch, "npu") and "910" in torch.npu.get_device_name():
            import torch_npu

            NPU_ATTN_INFR = True
        else:
            NPU_ATTN_INFR = False
    except Exception:
        NPU_ATTN_INFR = False
    """
)
config.add_post_import_block(
    """
    veomni_openpangu_vl_rms_norm = OpSlot("rms_norm", "standard")
    veomni_openpangu_vl_swiglu_mlp = OpSlot("swiglu_mlp", "standard")
    veomni_causal_lm_loss = OpSlot("cross_entropy_loss", "causal")
    _openpangu_v2_op_slots_for_binding = (
        veomni_moe_experts_forward,
        veomni_openpangu_v2_apply_rotary_pos_emb,
        veomni_openpangu_v2_rms_norm,
        veomni_openpangu_v2_swiglu_mlp,
    )
    """
)
config.add_helper(OpenPanguVLPositionModel)
config.add_helper(get_openpangu_vl_position_id)
config.override_method(
    "PanguEmbeddedRMSNorm.forward",
    replacement=openpangu_vl_rmsnorm_forward_patched,
    description="OpSlot guard for fused OpenPangu-VL RMSNorm while preserving eager semantics.",
)
config.override_method(
    "OpenPanguVLMLP.forward",
    replacement=openpangu_vl_mlp_forward_patched,
    description="OpSlot guard for SiLU OpenPangu-VL MLP; non-SwiGLU activations stay eager.",
)
config.override_method(
    "OpenPanguVLModel._parse_preprocess_params",
    replacement=openpangu_vl_parse_preprocess_params_patched,
    description="Avoid processor loading during model construction; preprocessing stays in the processor.",
)
config.override_method(
    "OpenPanguVLVisionAttention.forward",
    replacement=openpangu_vl_vision_attention_forward_patched,
    description="Use varlen FlashAttention with visual cu_seqlens and precomputed max_seqlen.",
)
config.override_method(
    "OpenPanguVLVisionBlock.forward",
    replacement=openpangu_vl_vision_block_forward_patched,
    description="Propagate precomputed max_seqlen to visual attention.",
)
config.override_method(
    "OpenPanguVisionTransformerPretrainedModel.forward",
    replacement=openpangu_vl_vision_forward_patched,
    description="VeOmni SP-aware visual forward with varlen FlashAttention and window attention.",
)
config.override_method(
    "OpenPanguVisionTransformerPretrainedModel.dummy_forward",
    replacement=openpangu_vl_vision_dummy_forward_patched,
    description="Run an SP-aware tiny vision path so FSDP ranks with text-only batches still touch visual parameters.",
)
config.override_method(
    "OpenPanguVLModel.get_image_features",
    replacement=openpangu_vl_model_get_image_features_patched,
    description="Return flat projected image embeddings; outer forward handles SP gather and masked scatter.",
)
config.override_method(
    "OpenPanguVLModel.get_video_features",
    replacement=openpangu_vl_model_get_video_features_patched,
    description="Return flat projected video embeddings; outer forward handles SP gather and masked scatter.",
)
config.override_method(
    "OpenPanguVLModel.get_position_id_func",
    replacement=openpangu_vl_model_get_position_id_func_patched,
    description="Expose a picklable multimodal position-id function using VeOmni placeholder ids.",
)
config.override_method(
    "OpenPanguVL.get_position_id_func",
    replacement=openpangu_vl_get_position_id_func_patched,
    description="Delegate top-level position-id preprocessing to the OpenPangu-VL base model helper.",
)
config.override_method(
    "OpenPanguVLModel.forward",
    replacement=openpangu_vl_model_forward_patched,
    description="Use VeOmni multimodal masks with zeroed placeholder ids and precomputed position ids.",
)
config.override_method(
    "OpenPanguVL.forward",
    replacement=openpangu_vl_forward_patched,
    description="Use VeOmni fused causal loss where available while preserving upstream OpenPangu-VL behavior.",
)
