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

from ...loader import MODEL_CONFIG_REGISTRY, MODELING_REGISTRY
from .checkpoint_tensor_converter import create_qwen3_moe_siglip_vlm_checkpoint_tensor_converter
from .configuration_qwen3_moe_siglip_vlm import Qwen3MoeSiglipVLMConfig
from .modeling_qwen3_moe_siglip_vlm import Qwen3MoeSiglipVLMForConditionalGeneration


@MODEL_CONFIG_REGISTRY.register("qwen3_moe_siglip_vlm")
def register_qwen3_moe_siglip_vlm_config():
    return Qwen3MoeSiglipVLMConfig


@MODELING_REGISTRY.register("qwen3_moe_siglip_vlm")
def register_qwen3_moe_siglip_vlm_modeling(architecture: str):
    del architecture
    model_cls = Qwen3MoeSiglipVLMForConditionalGeneration
    model_cls._create_checkpoint_tensor_converter = staticmethod(
        create_qwen3_moe_siglip_vlm_checkpoint_tensor_converter
    )
    return model_cls
