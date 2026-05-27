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

"""Runtime checkpoint tensor converter for OpenPangu-VL MoE checkpoints."""

import re
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from ...checkpoint_tensor_loading import ConvertedCheckpointTensor


_SPLIT_EXPERT_PATTERN = re.compile(
    r"^(?P<prefix>(?:model\.language_model|language_model|model)?\.?layers\.\d+\.mlp)\."
    r"experts\.(?P<expert>\d+)\."
    r"(?P<proj>gate_proj|up_proj|down_proj)\.weight$"
)


class OpenPanguVLCheckpointTensorConverter:
    """Convert split per-expert checkpoint tensors into fused expert tensors."""

    def __init__(self, num_experts: int):
        self.num_experts = num_experts
        self._expert_buffer: Dict[Tuple[str, str], Dict[int, Tensor]] = {}
        self._stacked_buffer: Dict[str, Dict[str, Tensor]] = {}

    def can_handle(self, name: str) -> bool:
        return bool(_SPLIT_EXPERT_PATTERN.match(name))

    def convert(self, name: str, tensor: Tensor) -> Optional[ConvertedCheckpointTensor]:
        match = _SPLIT_EXPERT_PATTERN.match(name)
        if not match:
            return None

        prefix = match.group("prefix")
        expert_id = int(match.group("expert"))
        proj_name = match.group("proj")
        if not 0 <= expert_id < self.num_experts:
            raise RuntimeError(
                f"OpenPanguVL checkpoint converter: expert id {expert_id} out of range for {name} "
                f"(num_experts={self.num_experts})"
            )

        buf_key = (prefix, proj_name)
        self._expert_buffer.setdefault(buf_key, {})[expert_id] = tensor
        if len(self._expert_buffer[buf_key]) < self.num_experts:
            return None

        stacked = torch.stack([self._expert_buffer[buf_key][i] for i in range(self.num_experts)])
        del self._expert_buffer[buf_key]

        if proj_name == "down_proj":
            return ConvertedCheckpointTensor(f"{prefix}.experts.down_proj", stacked)

        self._stacked_buffer.setdefault(prefix, {})[proj_name] = stacked
        if "gate_proj" in self._stacked_buffer[prefix] and "up_proj" in self._stacked_buffer[prefix]:
            gate = self._stacked_buffer[prefix].pop("gate_proj")
            up = self._stacked_buffer[prefix].pop("up_proj")
            if not self._stacked_buffer[prefix]:
                del self._stacked_buffer[prefix]
            return ConvertedCheckpointTensor(f"{prefix}.experts.gate_up_proj", torch.cat([gate, up], dim=1))
        return None

    def finalize(self) -> List[ConvertedCheckpointTensor]:
        errors: List[str] = []
        if self._expert_buffer:
            unflushed = {k: len(v) for k, v in self._expert_buffer.items()}
            errors.append(
                f"unflushed per-expert buffer (incomplete experts, expected {self.num_experts}): {unflushed}"
            )
        if self._stacked_buffer:
            unflushed = {k: list(v.keys()) for k, v in self._stacked_buffer.items()}
            errors.append(f"unflushed stacked buffer (missing gate/up pair): {unflushed}")
        if errors:
            raise RuntimeError(
                "OpenPanguVL checkpoint converter: incomplete checkpoint detected. " + "; ".join(errors)
            )
        return []


def create_openpangu_vl_checkpoint_tensor_converter(model):
    config = model.config
    text_config = getattr(config, "text_config", config)
    return OpenPanguVLCheckpointTensorConverter(num_experts=text_config.n_routed_experts)
