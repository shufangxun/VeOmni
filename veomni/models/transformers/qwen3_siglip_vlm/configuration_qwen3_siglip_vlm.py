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

from typing import Any, Dict, Optional

from transformers import PretrainedConfig
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.siglip.configuration_siglip import SiglipVisionConfig


class Qwen3SiglipVLMConfig(PretrainedConfig):
    model_type = "qwen3_siglip_vlm"
    sub_configs = {"text_config": Qwen3Config, "vision_config": SiglipVisionConfig}

    def __init__(
        self,
        text_config: Optional[Dict[str, Any]] = None,
        vision_config: Optional[Dict[str, Any]] = None,
        image_token_id: int = -200,
        pixel_shuffle_factor: int = 2,
        ignore_index: int = -100,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.text_config = Qwen3Config(**(text_config or {}))
        self.vision_config = SiglipVisionConfig(**(vision_config or {}))
        self.image_token_id = image_token_id
        self.pixel_shuffle_factor = pixel_shuffle_factor
        self.ignore_index = ignore_index
        self.tie_word_embeddings = getattr(self.text_config, "tie_word_embeddings", False)

    @property
    def vocab_size(self) -> int:
        return self.text_config.vocab_size
