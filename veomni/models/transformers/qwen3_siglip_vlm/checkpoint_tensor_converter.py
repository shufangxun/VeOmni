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

from typing import List, Optional

from torch import Tensor

from ...checkpoint_tensor_loading import ConvertedCheckpointTensor


class Qwen3SiglipVLMCheckpointTensorConverter:
    """Load bare Qwen3 language checkpoints into the nested language model module."""

    def can_handle(self, name: str) -> bool:
        return name.startswith(("model.", "lm_head."))

    def convert(self, name: str, tensor: Tensor) -> Optional[ConvertedCheckpointTensor]:
        return ConvertedCheckpointTensor(name=f"language_model.{name}", tensor=tensor)

    def finalize(self) -> List[ConvertedCheckpointTensor]:
        return []


def create_qwen3_siglip_vlm_checkpoint_tensor_converter(model):
    return Qwen3SiglipVLMCheckpointTensorConverter()
