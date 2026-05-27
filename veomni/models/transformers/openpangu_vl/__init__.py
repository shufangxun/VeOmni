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

from ....utils.device import IS_NPU_AVAILABLE
from ...loader import MODEL_CONFIG_REGISTRY, MODELING_REGISTRY
from .configuration_openpangu_vl import OpenPanguVLConfig, OpenPanguVLTextConfig, OpenPanguVLVisionConfig


@MODEL_CONFIG_REGISTRY.register("openpangu_vl")
def register_openpangu_vl_config():
    return OpenPanguVLConfig


@MODEL_CONFIG_REGISTRY.register("openpangu_vl_text")
def register_openpangu_vl_text_config():
    return OpenPanguVLTextConfig


@MODEL_CONFIG_REGISTRY.register("openpangu_vl_vision")
def register_openpangu_vl_vision_config():
    return OpenPanguVLVisionConfig


@MODELING_REGISTRY.register("openpangu_vl")
def register_openpangu_vl_modeling(architecture: str):
    from .checkpoint_tensor_converter import create_openpangu_vl_checkpoint_tensor_converter
    from .parallel_plan import get_parallel_plan

    if IS_NPU_AVAILABLE:
        from .generated.patched_modeling_openpangu_vl_npu import OpenPanguVL, OpenPanguVLModel, OpenPanguVLTextModel
    else:
        from .generated.patched_modeling_openpangu_vl_gpu import OpenPanguVL, OpenPanguVLModel, OpenPanguVLTextModel

    def _get_top_level_parallel_plan(self=None):
        return get_parallel_plan()

    def _get_base_model_parallel_plan(self=None):
        return get_parallel_plan("language_model")

    def _get_text_model_parallel_plan(self=None):
        return get_parallel_plan("")

    for model_cls in (OpenPanguVL, OpenPanguVLModel, OpenPanguVLTextModel):
        model_cls._create_checkpoint_tensor_converter = staticmethod(create_openpangu_vl_checkpoint_tensor_converter)

    OpenPanguVL.get_parallel_plan = _get_top_level_parallel_plan
    OpenPanguVLModel.get_parallel_plan = _get_base_model_parallel_plan
    OpenPanguVLTextModel.get_parallel_plan = _get_text_model_parallel_plan
    OpenPanguVLTextModel._checkpoint_conversion_mapping = {r"^model\.": ""}

    if "TextModel" in architecture:
        return OpenPanguVLTextModel
    if "Model" in architecture:
        return OpenPanguVLModel
    return OpenPanguVL


@MODELING_REGISTRY.register("openpangu_vl_text")
def register_openpangu_vl_text_modeling(architecture: str):
    del architecture

    if IS_NPU_AVAILABLE:
        from .generated.patched_modeling_openpangu_vl_npu import OpenPanguVLTextModel
    else:
        from .generated.patched_modeling_openpangu_vl_gpu import OpenPanguVLTextModel

    from .checkpoint_tensor_converter import create_openpangu_vl_checkpoint_tensor_converter
    from .parallel_plan import get_parallel_plan

    def _get_text_model_parallel_plan(self=None):
        return get_parallel_plan("")

    OpenPanguVLTextModel._create_checkpoint_tensor_converter = staticmethod(
        create_openpangu_vl_checkpoint_tensor_converter
    )
    OpenPanguVLTextModel.get_parallel_plan = _get_text_model_parallel_plan
    OpenPanguVLTextModel._checkpoint_conversion_mapping = {r"^model\.": ""}
    return OpenPanguVLTextModel
