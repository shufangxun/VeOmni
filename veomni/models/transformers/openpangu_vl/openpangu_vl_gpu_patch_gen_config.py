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

"""Patch configuration for OpenPangu-VL GPU modeling."""

from typing import Callable, Optional, Union

import torch
import torch.distributed as dist
from transformers.modeling_utils import is_flash_attention_requested
from transformers.processing_utils import Unpack

from veomni.distributed.parallel_state import get_parallel_state
from veomni.distributed.sequence_parallel import (
    gather_outputs,
    pad_tensor,
    slice_input_tensor,
    sp_pad_and_slice,
    unpad_tensor,
)
from veomni.models.transformers.openpangu_vl.modeling_openpangu_vl import (
    KwargsForCausalLM,
    OpenPanguVLCausalLMOutputWithPast,
    OpenPanguVLModelOutputWithPast,
)
from veomni.patchgen.patch_spec import PatchConfig


config = PatchConfig(
    source_module="veomni.models.transformers.openpangu_vl.modeling_openpangu_vl",
    target_file="patched_modeling_openpangu_vl_gpu.py",
    description="OpenPangu-VL with patched OpenPangu-V2 text tower and VeOmni-safe initialization",
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
    "veomni.models.transformers.openpangu_vl.generated.patched_modeling_openpangu_v2_gpu",
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


@config.override_method(
    "PanguEmbeddedRMSNorm.forward",
    description="OpSlot guard for fused OpenPangu-VL RMSNorm while preserving eager semantics.",
)
def openpangu_vl_rmsnorm_forward_patched(self, hidden_states):
    if veomni_openpangu_vl_rms_norm.use_non_eager_impl:
        return veomni_openpangu_vl_rms_norm(hidden_states, self.weight, self.variance_epsilon)

    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    return self.weight * hidden_states.to(input_dtype)


@config.override_method(
    "OpenPanguVLMLP.forward",
    description="OpSlot guard for SiLU OpenPangu-VL MLP; non-SwiGLU activations stay eager.",
)
def openpangu_vl_mlp_forward_patched(self, hidden_state):
    if self.hidden_act == "silu" and veomni_openpangu_vl_swiglu_mlp.use_non_eager_impl:
        return veomni_openpangu_vl_swiglu_mlp(self, hidden_state)

    if self.hidden_act == "silu":
        x_gate = self.gate_proj(hidden_state)
        x_gate = self.act_fn(x_gate)
        x_up = self.up_proj(hidden_state)
        intermediate_parallel = x_gate * x_up
    else:
        x_up = self.up_proj(hidden_state)
        intermediate_parallel = self.act_fn(x_up)
    x_down = self.down_proj(intermediate_parallel)
    return x_down


@config.add_helper
class OpenPanguVLPositionModel:
    def __init__(self, config):
        self.config = config

    def _get_llm_pos_ids_for_vision(
        self,
        start_idx: int,
        vision_idx: int,
        spatial_merge_size: int,
        t_index: list[int],
        grid_hs: torch.Tensor,
        grid_ws: torch.Tensor,
    ) -> torch.Tensor:
        llm_grid_h = grid_hs[vision_idx] // spatial_merge_size
        llm_grid_w = grid_ws[vision_idx] // spatial_merge_size
        h_index = (
            torch.arange(llm_grid_h)
            .to(llm_grid_h.device)
            .view(1, -1, 1)
            .expand(len(t_index), -1, llm_grid_w)
            .flatten()
        )
        w_index = (
            torch.arange(llm_grid_w)
            .to(llm_grid_h.device)
            .view(1, 1, -1)
            .expand(len(t_index), llm_grid_h, -1)
            .flatten()
        )
        t_index_tensor = t_index.to(llm_grid_h.device).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
        llm_pos_ids = torch.stack([t_index_tensor, h_index, w_index])
        return llm_pos_ids + start_idx


@config.add_helper
def get_openpangu_vl_position_id(main_func, self, **kwargs):
    position_ids, rope_deltas = main_func(self, **kwargs)
    return {"position_ids": position_ids, "rope_deltas": rope_deltas}


@config.override_method(
    "OpenPanguVLModel._parse_preprocess_params",
    description="Avoid processor loading during model construction; preprocessing stays in the processor.",
)
def openpangu_vl_parse_preprocess_params_patched(self, vision_config):
    self.channel = vision_config.in_channels
    self.patch_size = vision_config.patch_size
    attn_implementation = getattr(self.config, "_attn_implementation", None)
    if attn_implementation is not None:
        self.visual.config._attn_implementation = attn_implementation
        self.language_model.config._attn_implementation = attn_implementation


@config.override_method(
    "OpenPanguVLVisionAttention.forward",
    description="Use varlen FlashAttention with visual cu_seqlens and precomputed max_seqlen.",
)
def openpangu_vl_vision_attention_forward_patched(
    self,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    rotary_pos_emb: Optional[torch.Tensor] = None,
    position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    attention_mask: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    seq_length = hidden_states.shape[0]
    query_states, key_states, value_states = (
        self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
    )
    if position_embeddings is None:
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)  # noqa: F821

    query_states = query_states.transpose(0, 1).unsqueeze(0)
    key_states = key_states.transpose(0, 1).unsqueeze(0)
    value_states = value_states.transpose(0, 1).unsqueeze(0)

    attention_interface: Callable = eager_attention_forward  # noqa: F821
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]  # noqa: F821

    if not self.training and NPU_ATTN_INFR:  # noqa: F821
        if isinstance(cu_seqlens, torch.Tensor):
            cu_seqlens = cu_seqlens.tolist()

        q, k, v = [rearrange(x, "b n s d -> (b s) n d") for x in [query_states, key_states, value_states]]  # noqa: F821
        attn_output = torch_npu.npu_fusion_attention(  # noqa: F821
            q,
            k,
            v,
            self.num_heads,
            "TND",
            pse=None,
            padding_mask=None,
            atten_mask=None,
            scale=self.scaling,
            pre_tockens=1048576,
            next_tockens=0,
            keep_prob=1.0,
            inner_precise=0,
            sparse_mode=0,
            actual_seq_qlen=cu_seqlens,
            actual_seq_kvlen=cu_seqlens,
        )[0]
    elif is_flash_attention_requested(self.config):
        attn_output, _ = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask=None,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            cu_seq_lens_q=cu_seqlens,
            cu_seq_lens_k=cu_seqlens,
            max_length_q=max_seqlen,
            max_length_k=max_seqlen,
            is_causal=False,
            skip_ulysses=True,
            **kwargs,
        )
    else:
        lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        qkv_states = (query_states, key_states, value_states)
        splits = [torch.split(tensor, lengths.tolist(), dim=2) for tensor in qkv_states]
        attn_outputs = [
            attention_interface(
                self,
                q,
                k,
                v,
                attention_mask=attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                is_causal=False,
                **kwargs,
            )[0]
            for q, k, v in zip(*splits)
        ]
        attn_output = torch.cat(attn_outputs, dim=1)

    attn_output = attn_output.reshape(seq_length, -1).contiguous()
    attn_output = self.proj(attn_output)
    return attn_output


@config.override_method(
    "OpenPanguVLVisionBlock.forward",
    description="Propagate precomputed max_seqlen to visual attention.",
)
def openpangu_vl_vision_block_forward_patched(
    self,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    rotary_pos_emb: Optional[torch.Tensor] = None,
    position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    attention_mask: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    hidden_states = hidden_states + self.attn(
        self.norm1(hidden_states),
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        rotary_pos_emb=rotary_pos_emb,
        position_embeddings=position_embeddings,
        attention_mask=attention_mask,
        **kwargs,
    )
    hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
    return hidden_states


@config.override_method(
    "OpenPanguVisionTransformerPretrainedModel.forward",
    description="VeOmni SP-aware visual forward with varlen FlashAttention and window attention.",
)
def openpangu_vl_vision_forward_patched(
    self,
    hidden_states: torch.Tensor,
    grid_thw: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    hidden_states = self.patch_embed(hidden_states)
    rotary_pos_emb = self.rot_pos_emb(grid_thw)
    window_index, cu_window_seqlens = self.get_window_index(grid_thw)
    cu_window_seqlens = torch.tensor(
        cu_window_seqlens,
        device=hidden_states.device,
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

    cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
        dim=0,
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)  # noqa: F821

    unpadded_dim_size = cu_seqlens[-1]
    sp_padding_size = 0
    if get_parallel_state().sp_enabled:
        hidden_states = gather_outputs(hidden_states, gather_dim=0, group=get_parallel_state().sp_group)
        sp_padding_size = hidden_states.size(0) - unpadded_dim_size
        if sp_padding_size > 0:
            hidden_states = unpad_tensor(hidden_states, dim=0, padding_size=sp_padding_size)

    seq_len, _ = hidden_states.size()
    hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
    hidden_states = hidden_states[window_index, :, :]
    hidden_states = hidden_states.reshape(seq_len, -1)
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
    rotary_pos_emb = rotary_pos_emb[window_index, :, :]
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)

    if get_parallel_state().sp_enabled:
        if sp_padding_size > 0:
            hidden_states = pad_tensor(hidden_states, dim=0, padding_size=sp_padding_size)
            emb = pad_tensor(emb, dim=0, padding_size=sp_padding_size)
            new_cumsum = cu_seqlens[-1] + sp_padding_size
            cu_seqlens = torch.cat([cu_seqlens, new_cumsum.unsqueeze(0)], dim=0)
            cu_window_seqlens = torch.cat([cu_window_seqlens, new_cumsum.unsqueeze(0)], dim=0)
        hidden_states = slice_input_tensor(hidden_states, dim=0, group=get_parallel_state().sp_group)
        emb = sp_pad_and_slice(emb, dim=0)

    position_embeddings = (emb.cos(), emb.sin())
    max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().detach().cpu().item()
    win_max_seqlen = (cu_window_seqlens[1:] - cu_window_seqlens[:-1]).max().detach().cpu().item()

    intermediates = []
    for layer_num, blk in enumerate(self.blocks):
        if layer_num in self.fullatt_block_indexes:
            cu_seqlens_now = cu_seqlens
            max_seqlen_now = max_seqlen
        else:
            cu_seqlens_now = cu_window_seqlens
            max_seqlen_now = win_max_seqlen

        hidden_states = blk(
            hidden_states,
            cu_seqlens=cu_seqlens_now,
            max_seqlen=max_seqlen_now,
            position_embeddings=position_embeddings,
            attention_mask=None,
            **kwargs,
        )
        if layer_num in self.take_indices:
            intermediates.append(hidden_states)

    if self.use_gatedmerger:
        hidden_states = self.merger(hidden_states)
    else:
        image_embeddings_list = []
        for idx, sl in enumerate(self.select_layer):
            image_embeddings_list.append(self.merger[idx](intermediates[sl]))
        hidden_states = sum(image_embeddings_list)

    reverse_indices = torch.argsort(window_index)
    if get_parallel_state().sp_enabled:
        merged_sp_padding = sp_padding_size // self.spatial_merge_unit
        hidden_states = gather_outputs(hidden_states, gather_dim=0, group=get_parallel_state().sp_group)
        if merged_sp_padding > 0:
            hidden_states = unpad_tensor(hidden_states, dim=0, padding_size=merged_sp_padding)
        hidden_states = hidden_states[reverse_indices, :]
        if merged_sp_padding > 0:
            hidden_states = pad_tensor(hidden_states, dim=0, padding_size=merged_sp_padding)
        hidden_states = slice_input_tensor(hidden_states, dim=0, group=get_parallel_state().sp_group)
    else:
        hidden_states = hidden_states[reverse_indices, :]

    return hidden_states


@config.override_method(
    "OpenPanguVisionTransformerPretrainedModel.dummy_forward",
    description="Run an SP-aware tiny vision path so FSDP ranks with text-only batches still touch visual parameters.",
)
def openpangu_vl_vision_dummy_forward_patched(self):
    param = next(self.parameters())
    sp_size = get_parallel_state().sp_size if get_parallel_state().sp_enabled else 1
    grid_h = self.spatial_merge_size * sp_size
    grid_w = self.spatial_merge_size
    grid_thw = torch.tensor(
        [[1, grid_h, grid_w]],
        device=param.device,
        dtype=torch.long,
    )
    num_patches = int(grid_h * grid_w)
    hidden_states = torch.zeros(
        num_patches,
        self.patch_embed.input_size,
        device=param.device,
        dtype=param.dtype,
    )
    if get_parallel_state().sp_enabled:
        hidden_states = sp_pad_and_slice(hidden_states, dim=0, pad_value=0, pad_scale=self.spatial_merge_unit)
    return self(hidden_states, grid_thw)


@config.override_method(
    "OpenPanguVLModel.get_image_features",
    description="Return flat projected image embeddings; outer forward handles SP gather and masked scatter.",
)
def openpangu_vl_model_get_image_features_patched(
    self,
    pixel_values: torch.FloatTensor,
    image_grid_thw: Optional[torch.LongTensor] = None,
):
    pixel_values = pixel_values.type(self.visual.dtype)
    image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
    image_embeds = self.visual.vision_projection(image_embeds)
    return image_embeds


@config.override_method(
    "OpenPanguVLModel.get_video_features",
    description="Return flat projected video embeddings; outer forward handles SP gather and masked scatter.",
)
def openpangu_vl_model_get_video_features_patched(
    self,
    pixel_values_videos: torch.FloatTensor,
    video_grid_thw: Optional[torch.LongTensor] = None,
):
    pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
    video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
    video_embeds = self.visual.vision_projection(video_embeds)
    return video_embeds


@config.override_method(
    "OpenPanguVLModel.get_position_id_func",
    description="Expose a picklable multimodal position-id function using VeOmni placeholder ids.",
)
def openpangu_vl_model_get_position_id_func_patched(self):
    import copy
    from functools import partial

    from veomni.utils.constants import IMAGE_INPUT_INDEX, VIDEO_INPUT_INDEX

    fake_config = copy.copy(self.config)
    fake_config.image_token_id = IMAGE_INPUT_INDEX
    fake_config.video_token_id = VIDEO_INPUT_INDEX
    fake_model = OpenPanguVLPositionModel(fake_config)
    return partial(get_openpangu_vl_position_id, OpenPanguVLModel.get_rope_index, fake_model)  # noqa: F821


@config.override_method(
    "OpenPanguVL.get_position_id_func",
    description="Delegate top-level position-id preprocessing to the OpenPangu-VL base model helper.",
)
def openpangu_vl_get_position_id_func_patched(self):
    return self.model.get_position_id_func()


@config.override_method(
    "OpenPanguVLModel.forward",
    description="Use VeOmni multimodal masks with zeroed placeholder ids and precomputed position ids.",
)
def openpangu_vl_model_forward_patched(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[list[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    **kwargs: Unpack[KwargsForCausalLM],
) -> Union[tuple, OpenPanguVLModelOutputWithPast]:
    """
    image_grid_thw (`torch.LongTensor`, *optional*):
        Temporal, height, and width grid for each image.
    video_grid_thw (`torch.LongTensor`, *optional*):
        Temporal, height, and width grid for each video.
    rope_deltas (`torch.LongTensor`, *optional*):
        Difference between sequence length and multimodal RoPE positions.
    second_per_grid_ts (`torch.Tensor`, *optional*):
        Time interval per video grid step.
    """

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    image_mask = kwargs.pop("image_mask", None)
    video_mask = kwargs.pop("video_mask", None)
    input_ids_for_mask = input_ids

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)
        if self.use_mhc:
            inputs_embeds = self.get_input_embeddings()(input_ids).repeat(1, 1, self.mhc_num_stream)

        if input_ids is not None and get_parallel_state().sp_enabled:
            input_ids_list = [torch.zeros_like(input_ids) for _ in range(get_parallel_state().sp_size)]
            dist.all_gather(input_ids_list, input_ids, group=get_parallel_state().sp_group)
            input_ids_for_mask = torch.cat(input_ids_list, dim=1)
            if image_mask is not None and image_mask.shape[-1] != input_ids_for_mask.shape[-1]:
                image_mask_list = [torch.zeros_like(image_mask) for _ in range(get_parallel_state().sp_size)]
                dist.all_gather(image_mask_list, image_mask, group=get_parallel_state().sp_group)
                image_mask = torch.cat(image_mask_list, dim=1)
            if video_mask is not None and video_mask.shape[-1] != input_ids_for_mask.shape[-1]:
                video_mask_list = [torch.zeros_like(video_mask) for _ in range(get_parallel_state().sp_size)]
                dist.all_gather(video_mask_list, video_mask, group=get_parallel_state().sp_group)
                video_mask = torch.cat(video_mask_list, dim=1)

        if image_mask is None and input_ids_for_mask is not None:
            image_mask = input_ids_for_mask == self.config.image_token_id
        if video_mask is None and input_ids_for_mask is not None:
            video_mask = input_ids_for_mask == self.config.video_token_id

        rope_input_ids = input_ids_for_mask
        if input_ids_for_mask is not None and (image_mask is not None or video_mask is not None):
            rope_input_ids = input_ids_for_mask.clone()
            if image_mask is not None:
                rope_input_ids = rope_input_ids.masked_fill(image_mask.to(torch.bool), self.config.image_token_id)
            if video_mask is not None:
                rope_input_ids = rope_input_ids.masked_fill(video_mask.to(torch.bool), self.config.video_token_id)

        flash_attn_kwargs = {}
        for key in ["cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k"]:
            if key in kwargs:
                flash_attn_kwargs[key] = kwargs.pop(key)

        if get_parallel_state().sp_enabled:
            inputs_embeds = gather_outputs(inputs_embeds, gather_dim=1, group=get_parallel_state().sp_group)

        if pixel_values is not None:
            if image_mask is None:
                raise ValueError("image_mask is required when pixel_values are provided.")
            image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            if isinstance(image_embeds, (tuple, list)):
                image_embeds = torch.cat(image_embeds, dim=0)
            if get_parallel_state().sp_enabled:
                image_embeds = gather_outputs(image_embeds, gather_dim=0, group=get_parallel_state().sp_group)
            if (
                not get_parallel_state().sp_enabled
                and not is_torchdynamo_compiling()
                and image_mask.sum() != image_embeds.shape[0]
            ):
                raise ValueError(
                    "Image features and image tokens do not match: "
                    f"tokens: {image_mask.sum()}, features {image_embeds.shape[0]}"
                )

            mask_expanded = image_mask.unsqueeze(-1).expand_as(inputs_embeds)
            image_mask_expanded = mask_expanded.to(inputs_embeds.device)
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask_expanded, image_embeds)

        if pixel_values_videos is not None:
            if video_mask is None:
                raise ValueError("video_mask is required when pixel_values_videos are provided.")
            video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
            if isinstance(video_embeds, (tuple, list)):
                video_embeds = torch.cat(video_embeds, dim=0)
            if get_parallel_state().sp_enabled:
                video_embeds = gather_outputs(video_embeds, gather_dim=0, group=get_parallel_state().sp_group)
            if (
                not get_parallel_state().sp_enabled
                and not is_torchdynamo_compiling()
                and video_mask.sum() != video_embeds.shape[0]
            ):
                raise ValueError(
                    "Video features and video tokens do not match: "
                    f"tokens: {video_mask.sum()}, features {video_embeds.shape[0]}"
                )

            mask_expanded = video_mask.unsqueeze(-1).expand_as(inputs_embeds)
            video_mask_expanded = mask_expanded.to(inputs_embeds.device)
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask_expanded, video_embeds)

        if pixel_values is None and pixel_values_videos is None:
            if get_parallel_state().fsdp_enabled:
                fake_embeds = self.visual.vision_projection(self.visual.dummy_forward()).mean() * 0.0
                inputs_embeds = inputs_embeds + fake_embeds.to(inputs_embeds.device, inputs_embeds.dtype)

        if get_parallel_state().sp_enabled:
            inputs_embeds = slice_input_tensor(inputs_embeds, dim=1, group=get_parallel_state().sp_group)

        kwargs.update(flash_attn_kwargs)
    else:
        rope_input_ids = input_ids

    if position_ids is None:
        if get_parallel_state().sp_enabled:
            raise RuntimeError(
                "OpenPanguVLModel.forward: position_ids is None while sequence parallel is enabled; "
                "multimodal position_ids must be precomputed by the VeOmni data pipeline."
            )
        attention_mask_tensor = (
            attention_mask if not isinstance(attention_mask, dict) else attention_mask["full_attention"]
        )
        if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
            attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
            attention_mask_tensor = attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
            attention_mask_tensor = (1.0 - attention_mask_tensor).int()

        prefill_compiled_stage = is_torchdynamo_compiling() and (
            (rope_input_ids is not None and rope_input_ids.shape[1] != 1)
            or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
        )
        prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
            (cache_position is not None and cache_position[0] == 0)
            or (past_key_values is None or past_key_values.get_seq_length() == 0)
        )
        if (prefill_compiled_stage or prefill_noncompiled_stage) or self.rope_deltas is None:
            position_ids, rope_deltas = self.get_rope_index(
                rope_input_ids,
                image_grid_thw,
                video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                attention_mask=attention_mask_tensor,
            )
            self.rope_deltas = rope_deltas
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            if cache_position is not None:
                delta = (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
            else:
                delta = 0
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)
            if cache_position is not None:
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
            position_ids = position_ids.add(delta)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

    if (
        position_ids is not None
        and position_ids.ndim == 3
        and position_ids.shape[0] != 3
        and position_ids.shape[1] == 3
    ):
        position_ids = position_ids.transpose(0, 1).contiguous()

    outputs = self.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=True,
        cache_position=cache_position,
        **kwargs,
    )

    return OpenPanguVLModelOutputWithPast(
        last_hidden_state=outputs.last_hidden_state,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.rope_deltas,
    )


@config.override_method(
    "OpenPanguVL.forward",
    description="Use VeOmni fused causal loss where available while preserving upstream OpenPangu-VL behavior.",
)
def openpangu_vl_forward_patched(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[list[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    **kwargs: Unpack[KwargsForCausalLM],
) -> Union[tuple, OpenPanguVLCausalLMOutputWithPast]:
    """
    labels (`torch.LongTensor`, *optional*):
        Labels for language-model loss computation.
    image_grid_thw (`torch.LongTensor`, *optional*):
        Temporal, height, and width grid for each image.
    video_grid_thw (`torch.LongTensor`, *optional*):
        Temporal, height, and width grid for each video.
    rope_deltas (`torch.LongTensor`, *optional*):
        Difference between sequence length and multimodal RoPE positions.
    second_per_grid_ts (`torch.Tensor`, *optional*):
        Time interval per video grid step.
    """
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )

    outputs = self.model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        second_per_grid_ts=second_per_grid_ts,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=True,
        cache_position=cache_position,
        **kwargs,
    )

    hidden_states = outputs[0]

    loss = None
    logits = None
    if labels is not None:
        if veomni_causal_lm_loss.use_non_eager_impl:
            loss_outputs = veomni_causal_lm_loss(
                logits=None,
                labels=labels,
                vocab_size=self.config.text_config.vocab_size,
                hidden_states=hidden_states,
                weights=self.lm_head.weight,
                **kwargs,
            )
            loss = loss_outputs[0] if isinstance(loss_outputs, tuple) else loss_outputs
        else:
            logits = self.lm_head(hidden_states)
            loss_outputs = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.text_config.vocab_size,
                hidden_states=hidden_states,
                weights=self.lm_head.weight,
                **kwargs,
            )
            if isinstance(loss_outputs, tuple) and len(loss_outputs) > 2 and loss_outputs[2] is not None:
                logits = None
    else:
        logits = self.lm_head(hidden_states)

    if labels is not None and loss is None:
        loss = loss_outputs[0] if isinstance(loss_outputs, tuple) else loss_outputs

    return OpenPanguVLCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=outputs.rope_deltas,
    )
