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

from typing import Optional

import torch
import torch.distributed as dist
from transformers.modeling_utils import PreTrainedModel

from veomni.distributed.parallel_state import get_parallel_state
from veomni.distributed.sequence_parallel import gather_outputs, slice_input_tensor
from veomni.utils.constants import IMAGE_INPUT_INDEX
from veomni.utils.model_outputs import MoeCausalLMOutputWithLogProbs

from ..qwen3_moe.generated import patched_modeling_qwen3_moe_gpu as qwen3_moe_modeling
from ..qwen3_moe.generated.patched_modeling_qwen3_moe_gpu import Qwen3MoeForCausalLM
from ..qwen3_siglip_vlm.modeling_qwen3_siglip_vlm import PixelShuffleMLPConnector, SiglipNativeVisionTower
from .configuration_qwen3_moe_siglip_vlm import Qwen3MoeSiglipVLMConfig


# Re-export Qwen3-MoE OpSlots so build_foundation_model() binds the nested
# language_model kernels when this wrapper is the top-level model class.
veomni_rms_norm = qwen3_moe_modeling.veomni_rms_norm
veomni_apply_rotary_pos_emb = qwen3_moe_modeling.veomni_apply_rotary_pos_emb
veomni_swiglu_mlp = qwen3_moe_modeling.veomni_swiglu_mlp
veomni_moe_experts_forward = qwen3_moe_modeling.veomni_moe_experts_forward
veomni_causal_lm_loss = qwen3_moe_modeling.veomni_causal_lm_loss
veomni_load_balancing_loss = qwen3_moe_modeling.veomni_load_balancing_loss


class Qwen3MoeSiglipVLMForConditionalGeneration(PreTrainedModel):
    config_class = Qwen3MoeSiglipVLMConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _supports_flash_attn = True
    _no_split_modules = ["Qwen3MoeDecoderLayer", "SiglipNativeEncoderLayer"]

    def __init__(self, config: Qwen3MoeSiglipVLMConfig):
        super().__init__(config)
        self.language_model = Qwen3MoeForCausalLM(config.text_config)
        self.vision_tower = SiglipNativeVisionTower(config)
        self.connector = PixelShuffleMLPConnector(
            vision_hidden_size=config.vision_config.hidden_size,
            text_hidden_size=config.text_config.hidden_size,
            pixel_shuffle_factor=config.pixel_shuffle_factor,
        )
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self.language_model.gradient_checkpointing_disable()

    def get_position_id_func(self):
        def get_position_ids(input_ids, attention_mask=None, **kwargs):
            del input_ids, kwargs
            if attention_mask is None:
                raise ValueError("attention_mask is required to build qwen3_moe_siglip_vlm position_ids.")
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)
            return {"position_ids": position_ids}

        return get_position_ids

    def _get_image_mask(self, input_ids: torch.Tensor, image_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if image_mask is not None:
            return image_mask
        if get_parallel_state().sp_enabled:
            input_ids_list = [torch.zeros_like(input_ids) for _ in range(get_parallel_state().sp_size)]
            dist.all_gather(input_ids_list, input_ids, group=get_parallel_state().sp_group)
            input_ids = torch.cat(input_ids_list, dim=-1)
        return input_ids == IMAGE_INPUT_INDEX

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_hw: Optional[torch.Tensor] = None,
        image_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> MoeCausalLMOutputWithLogProbs:
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        fake_vision_loss = None
        parallel_state = get_parallel_state()

        if parallel_state.sp_enabled:
            inputs_embeds = gather_outputs(inputs_embeds, gather_dim=1, group=parallel_state.sp_group)

        if pixel_values is not None and image_grid_hw is not None and pixel_values.numel() > 0:
            image_mask = self._get_image_mask(input_ids, image_mask)
            vision_features = self.vision_tower(pixel_values, image_grid_hw)
            if parallel_state.sp_enabled:
                unpadded_vision_tokens = int(image_grid_hw.prod(dim=-1).sum().item())
                vision_features = gather_outputs(
                    vision_features,
                    gather_dim=0,
                    padding_dim=0,
                    unpad_dim_size=unpadded_vision_tokens,
                    group=parallel_state.sp_group,
                )
            image_features = self.connector(vision_features, image_grid_hw).to(
                device=inputs_embeds.device, dtype=inputs_embeds.dtype
            )
            embeds_image_mask = (
                image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device, non_blocking=True)
            )
            inputs_embeds = inputs_embeds.masked_scatter(embeds_image_mask, image_features)
        elif parallel_state.fsdp_enabled:
            dummy_grid_hw = torch.tensor(
                [[parallel_state.sp_size if parallel_state.sp_enabled else 1, 1]],
                dtype=torch.long,
                device=inputs_embeds.device,
            )
            dummy_vision_features = self.vision_tower.dummy_forward(dummy_grid_hw)
            if parallel_state.sp_enabled:
                dummy_vision_features = gather_outputs(
                    dummy_vision_features,
                    gather_dim=0,
                    padding_dim=0,
                    unpad_dim_size=int(dummy_grid_hw.prod(dim=-1).sum().item()),
                    group=parallel_state.sp_group,
                )
            fake_vision_loss = self.connector(dummy_vision_features, dummy_grid_hw).mean() * 0.0

        if parallel_state.sp_enabled:
            inputs_embeds = slice_input_tensor(inputs_embeds, dim=1, group=parallel_state.sp_group)

        outputs = self.language_model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
            **kwargs,
        )

        if fake_vision_loss is None or outputs.loss is None:
            return outputs

        return MoeCausalLMOutputWithLogProbs(
            loss=outputs.loss + fake_vision_loss,
            aux_loss=outputs.aux_loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
            log_probs=outputs.log_probs,
            entropy=outputs.entropy,
        )

    def get_parallel_plan(self):
        from .parallel_plan import get_parallel_plan

        return get_parallel_plan()
