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

import math
from typing import Callable, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from transformers.modeling_utils import PreTrainedModel
from transformers.models.siglip.modeling_siglip import (
    ALL_ATTENTION_FUNCTIONS,
    SiglipMLP,
    SiglipVisionEmbeddings,
    eager_attention_forward,
)

from veomni.distributed.parallel_state import get_parallel_state
from veomni.distributed.sequence_parallel import gather_outputs, pad_tensor, slice_input_tensor, sp_pad_and_slice
from veomni.ops.dispatch import OpSlot
from veomni.utils.constants import IMAGE_INPUT_INDEX
from veomni.utils.model_outputs import CausalLMOutputWithLogProbs

from ..attention_utils import VARLEN_ATTENTION_TYPES
from .configuration_qwen3_siglip_vlm import Qwen3SiglipVLMConfig


veomni_causal_lm_loss = OpSlot("cross_entropy_loss", "causal")


try:
    from ..qwen3.generated.patched_modeling_qwen3_gpu import Qwen3ForCausalLM
except ImportError:
    from transformers import Qwen3ForCausalLM


class SiglipNativeAttention(nn.Module):
    """SigLIP attention over a flat sequence of packed visual tokens."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and "
                f"`num_heads`: {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        **kwargs,
    ) -> torch.Tensor:
        seq_length, embed_dim = hidden_states.shape
        queries = self.q_proj(hidden_states)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)

        queries = queries.view(seq_length, self.num_heads, self.head_dim).transpose(0, 1).unsqueeze(0)
        keys = keys.view(seq_length, self.num_heads, self.head_dim).transpose(0, 1).unsqueeze(0)
        values = values.view(seq_length, self.num_heads, self.head_dim).transpose(0, 1).unsqueeze(0)

        if get_parallel_state().sp_enabled and self.config._attn_implementation not in VARLEN_ATTENTION_TYPES:
            raise ValueError("SigLIP vision SP requires a varlen attention implementation.")

        if self.config._attn_implementation in VARLEN_ATTENTION_TYPES:
            attention_interface: Callable = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
            attn_output, _ = attention_interface(
                self,
                queries,
                keys,
                values,
                attention_mask=None,
                scaling=self.scale,
                dropout=0.0 if not self.training else self.dropout,
                cu_seq_lens_q=cu_seqlens,
                cu_seq_lens_k=cu_seqlens,
                max_length_q=max_seqlen,
                max_length_k=max_seqlen,
                is_causal=False,
                **kwargs,
            )
        else:
            lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            splits = [torch.split(tensor, lengths.tolist(), dim=2) for tensor in (queries, keys, values)]
            attn_outputs = [
                eager_attention_forward(
                    self,
                    q,
                    k,
                    v,
                    attention_mask=None,
                    scaling=self.scale,
                    dropout=0.0 if not self.training else self.dropout,
                    is_causal=False,
                    **kwargs,
                )[0]
                for q, k, v in zip(*splits)
            ]
            attn_output = torch.cat(attn_outputs, dim=1)

        attn_output = attn_output.reshape(seq_length, embed_dim).contiguous()
        return self.out_proj(attn_output)


class SiglipNativeEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.self_attn = SiglipNativeAttention(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)

    def forward(self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor, max_seqlen: int, **kwargs):
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, **kwargs)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class SiglipNativeEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([SiglipNativeEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor, max_seqlen: int, **kwargs):
        for layer in self.layers:
            hidden_states = layer(hidden_states, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, **kwargs)
        return hidden_states


class SiglipNativeVisionTower(nn.Module):
    """SigLIP vision tower adapted to packed native-resolution patch sequences."""

    def __init__(self, config: Qwen3SiglipVLMConfig):
        super().__init__()
        self.config = config.vision_config
        self.embeddings = SiglipVisionEmbeddings(self.config)
        self.encoder = SiglipNativeEncoder(self.config)
        self.post_layernorm = nn.LayerNorm(self.config.hidden_size, eps=self.config.layer_norm_eps)
        self.patch_dim = self.config.num_channels * self.config.patch_size * self.config.patch_size

    def _interpolate_position_embedding(self, grid_hw: torch.Tensor) -> torch.Tensor:
        base_grid = int(math.sqrt(self.embeddings.position_embedding.num_embeddings))
        pos = self.embeddings.position_embedding.weight.reshape(base_grid, base_grid, -1)
        pos = pos.permute(2, 0, 1).unsqueeze(0)
        h, w = int(grid_hw[0].item()), int(grid_hw[1].item())
        pos = torch.nn.functional.interpolate(pos, size=(h, w), mode="bicubic", align_corners=False)
        return pos.squeeze(0).permute(1, 2, 0).reshape(h * w, -1)

    def forward(self, pixel_values: torch.Tensor, image_grid_hw: torch.Tensor) -> torch.Tensor:
        if pixel_values.shape[-1] != self.patch_dim:
            raise ValueError(f"Expected flattened SigLIP patches with dim={self.patch_dim}, got {pixel_values.shape}.")

        weight = self.embeddings.patch_embedding.weight.flatten(1)
        bias = self.embeddings.patch_embedding.bias
        patch_embeds = torch.nn.functional.linear(pixel_values, weight, bias)

        if image_grid_hw.numel() == 0:
            return patch_embeds.new_zeros((0, self.config.hidden_size))

        position_embeds = []
        cu_seqlens = []
        offset = 0
        for grid_hw in image_grid_hw:
            patch_count = int(grid_hw[0].item() * grid_hw[1].item())
            cur_pos = self._interpolate_position_embedding(grid_hw).to(
                device=patch_embeds.device, dtype=patch_embeds.dtype
            )
            position_embeds.append(cur_pos)
            offset += patch_count
            cu_seqlens.append(offset)

        parallel_state = get_parallel_state()
        if not parallel_state.sp_enabled and offset != patch_embeds.shape[0]:
            raise ValueError(
                f"Image grid patch count {offset} does not match pixel_values length {patch_embeds.shape[0]}."
            )

        position_embeds = torch.cat(position_embeds, dim=0)
        cu_seqlens = torch.tensor(cu_seqlens, device=patch_embeds.device, dtype=torch.int32)
        cu_seqlens = torch.cat([torch.zeros(1, device=patch_embeds.device, dtype=torch.int32), cu_seqlens])

        if parallel_state.sp_enabled:
            unpadded_dim_size = cu_seqlens[-1]
            sp_padding_size = patch_embeds.shape[0] * parallel_state.sp_size - unpadded_dim_size
            if sp_padding_size < 0:
                raise ValueError(
                    "SP-sliced SigLIP patch sequence is shorter than expected from image_grid_hw: "
                    f"local_patches={patch_embeds.shape[0]}, sp_size={parallel_state.sp_size}, "
                    f"unpadded_total={int(unpadded_dim_size.item())}."
                )
            if sp_padding_size > 0:
                position_embeds = pad_tensor(position_embeds, dim=0, padding_size=int(sp_padding_size.item()))
                new_cumsum = cu_seqlens[-1] + sp_padding_size
                cu_seqlens = torch.cat([cu_seqlens, new_cumsum.unsqueeze(0)], dim=0)
            position_embeds = sp_pad_and_slice(position_embeds, dim=0)

        if position_embeds.shape[0] != patch_embeds.shape[0]:
            raise ValueError(
                "SigLIP position embeddings do not match patch embeddings after SP slicing: "
                f"positions={position_embeds.shape[0]}, patches={patch_embeds.shape[0]}."
            )

        hidden_states = patch_embeds + position_embeds
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().detach().cpu().item()
        hidden_states = self.encoder(hidden_states, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        return self.post_layernorm(hidden_states)

    def dummy_forward(self, image_grid_hw: Optional[torch.Tensor] = None):
        device = self.embeddings.patch_embedding.weight.device
        dtype = self.embeddings.patch_embedding.weight.dtype
        if image_grid_hw is None:
            sp_size = get_parallel_state().sp_size if get_parallel_state().sp_enabled else 1
            image_grid_hw = torch.tensor([[sp_size, 1]], dtype=torch.long, device=device)
        local_patch_count = image_grid_hw.prod(dim=-1).sum().item()
        if get_parallel_state().sp_enabled:
            local_patch_count = math.ceil(local_patch_count / get_parallel_state().sp_size)
        pixel_values = torch.zeros((local_patch_count, self.patch_dim), dtype=dtype, device=device)
        return self(pixel_values=pixel_values, image_grid_hw=image_grid_hw)


class PixelShuffleMLPConnector(nn.Module):
    def __init__(self, vision_hidden_size: int, text_hidden_size: int, pixel_shuffle_factor: int = 2):
        super().__init__()
        self.pixel_shuffle_factor = pixel_shuffle_factor
        self.proj = nn.Sequential(
            nn.Linear(vision_hidden_size * pixel_shuffle_factor * pixel_shuffle_factor, text_hidden_size),
            nn.GELU(),
            nn.Linear(text_hidden_size, text_hidden_size),
        )

    def forward(self, vision_features: torch.Tensor, image_grid_hw: torch.Tensor) -> torch.Tensor:
        segments = []
        offset = 0
        factor = self.pixel_shuffle_factor
        for grid_hw in image_grid_hw:
            h, w = int(grid_hw[0].item()), int(grid_hw[1].item())
            length = h * w
            cur = vision_features[offset : offset + length].reshape(h, w, -1)
            pad_h = (factor - h % factor) % factor
            pad_w = (factor - w % factor) % factor
            if pad_h or pad_w:
                cur = torch.nn.functional.pad(cur, (0, 0, 0, pad_w, 0, pad_h))
            new_h, new_w = cur.shape[0] // factor, cur.shape[1] // factor
            cur = cur.reshape(new_h, factor, new_w, factor, -1).permute(0, 2, 1, 3, 4)
            cur = cur.reshape(new_h * new_w, factor * factor * cur.shape[-1])
            segments.append(self.proj(cur))
            offset += length
        return torch.cat(segments, dim=0) if segments else vision_features.new_zeros((0, self.proj[-1].out_features))


class Qwen3SiglipVLMForConditionalGeneration(PreTrainedModel):
    config_class = Qwen3SiglipVLMConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _supports_flash_attn = True
    _no_split_modules = ["Qwen3DecoderLayer", "SiglipNativeEncoderLayer"]

    def __init__(self, config: Qwen3SiglipVLMConfig):
        super().__init__(config)
        self.language_model = Qwen3ForCausalLM(config.text_config)
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
                raise ValueError("attention_mask is required to build qwen3_siglip_vlm position_ids.")
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
    ) -> CausalLMOutputWithLogProbs:
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

        outputs = self.language_model.model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state

        loss = None
        logits = None
        log_probs = None
        entropy = None
        if labels is not None:
            if veomni_causal_lm_loss.use_non_eager_impl:
                loss, logits, log_probs, entropy = veomni_causal_lm_loss(
                    logits=logits,
                    labels=labels,
                    vocab_size=self.language_model.config.vocab_size,
                    hidden_states=hidden_states,
                    weights=self.language_model.lm_head.weight,
                    **kwargs,
                )
            else:
                logits = self.language_model.lm_head(hidden_states)
                loss_outputs = self.language_model.loss_function(
                    logits=logits,
                    labels=labels,
                    vocab_size=self.language_model.config.vocab_size,
                    **kwargs,
                )
                if isinstance(loss_outputs, tuple):
                    loss, _, log_probs, entropy = loss_outputs
                else:
                    loss = loss_outputs
            if fake_vision_loss is not None:
                loss = loss + fake_vision_loss
        else:
            logits = self.language_model.lm_head(hidden_states)

        return CausalLMOutputWithLogProbs(
            loss=loss,
            logits=logits,
            log_probs=log_probs,
            entropy=entropy,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
