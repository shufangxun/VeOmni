from types import SimpleNamespace

import pytest
import torch

from tests.tools.training_utils import make_eager_ops_config
from veomni.models.checkpoint_tensor_loading import maybe_convert_checkpoint_tensor
from veomni.models.loader import MODELING_REGISTRY
from veomni.models.module_utils import _convert_weight_key
from veomni.models.transformers.openpangu_vl.checkpoint_tensor_converter import (
    OpenPanguVLCheckpointTensorConverter,
    create_openpangu_vl_checkpoint_tensor_converter,
)
from veomni.models.transformers.openpangu_vl.configuration_openpangu_vl import OpenPanguVLConfig
from veomni.models.transformers.openpangu_vl.generated import (
    patched_modeling_openpangu_v2_gpu as openpangu_v2_modeling,
)
from veomni.models.transformers.openpangu_vl.generated import (
    patched_modeling_openpangu_vl_gpu as openpangu_vl_modeling,
)
from veomni.models.transformers.openpangu_vl.generated.patched_modeling_openpangu_vl_gpu import (
    OpenPanguVL,
    OpenPanguVLModel,
    OpenPanguVLTextModel,
)
from veomni.ops import apply_ops_config
from veomni.utils.constants import IGNORE_INDEX, IMAGE_INPUT_INDEX


TOY_CONFIG = "tests/toy_config/openpangu_vl_toy"


def _toy_model(attn_implementation: str = "eager") -> OpenPanguVL:
    apply_ops_config(make_eager_ops_config(attn_implementation=attn_implementation))
    config = OpenPanguVLConfig.from_pretrained(TOY_CONFIG, attn_implementation=attn_implementation)
    return OpenPanguVL(config)


def test_openpangu_vl_registered():
    assert "openpangu_vl" in MODELING_REGISTRY.valid_keys()
    model_cls = MODELING_REGISTRY["openpangu_vl"]("OpenPanguVL")
    assert model_cls is OpenPanguVL
    assert hasattr(model_cls, "_create_checkpoint_tensor_converter")
    assert hasattr(model_cls, "get_parallel_plan")

    assert MODELING_REGISTRY["openpangu_vl"]("OpenPanguVLModel") is OpenPanguVLModel
    assert MODELING_REGISTRY["openpangu_vl"]("OpenPanguVLTextModel") is OpenPanguVLTextModel
    assert MODELING_REGISTRY["openpangu_vl_text"]("OpenPanguVLTextModel") is OpenPanguVLTextModel


def test_openpangu_vl_exposes_nested_text_moe_opslot():
    assert openpangu_vl_modeling.veomni_moe_experts_forward is openpangu_v2_modeling.veomni_moe_experts_forward


def test_openpangu_vl_does_not_force_visual_attention_to_eager():
    model = _toy_model("veomni_flash_attention_2_with_sp")

    assert model.model.visual.config._attn_implementation == "veomni_flash_attention_2_with_sp"
    assert model.model.language_model.config._attn_implementation == "veomni_flash_attention_2_with_sp"


def test_openpangu_vl_text_forward():
    model = _toy_model()
    input_ids = torch.tensor([[1, 7, 8, 2]])
    attention_mask = torch.ones_like(input_ids)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)

    assert outputs.loss is not None
    assert outputs.logits.shape == (1, 4, model.config.vocab_size)
    outputs.loss.backward()


def test_openpangu_vl_text_forward_touches_vision_projection_under_fsdp(monkeypatch):
    model = _toy_model()
    input_ids = torch.tensor([[1, 7, 8, 2]])
    attention_mask = torch.ones_like(input_ids)
    fake_parallel_state = SimpleNamespace(fsdp_enabled=True, sp_enabled=False)
    monkeypatch.setattr(openpangu_vl_modeling, "get_parallel_state", lambda: fake_parallel_state)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
    outputs.loss.backward()

    assert model.model.visual.vision_projection.fc1.weight.grad is not None


def test_openpangu_vl_image_forward():
    model = _toy_model()
    image_token_id = model.config.image_token_id
    input_ids = torch.tensor(
        [[1, model.config.vision_start_token_id, image_token_id, model.config.vision_end_token_id, 2]]
    )
    attention_mask = torch.ones_like(input_ids)
    pixel_values = torch.randn(4, 3 * 2 * 2)
    image_grid_thw = torch.tensor([[1, 2, 2]])

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        labels=input_ids,
    )

    assert outputs.loss is not None
    assert outputs.logits.shape == (1, 5, model.config.vocab_size)


def test_openpangu_vl_image_features_are_flat_projected_tensor():
    model = _toy_model()
    pixel_values = torch.randn(4, 3 * 2 * 2)
    image_grid_thw = torch.tensor([[1, 2, 2]])

    image_features = model.model.get_image_features(pixel_values, image_grid_thw)

    assert isinstance(image_features, torch.Tensor)
    assert image_features.shape == (1, model.config.text_config.hidden_size)


def test_openpangu_vl_image_forward_accepts_veomni_masked_placeholders():
    model = _toy_model()
    position_input_ids = torch.tensor(
        [[1, model.config.vision_start_token_id, IMAGE_INPUT_INDEX, model.config.vision_end_token_id, 2]]
    )
    image_mask = position_input_ids == IMAGE_INPUT_INDEX
    input_ids = position_input_ids.masked_fill(image_mask, 0)
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    labels[image_mask] = IGNORE_INDEX
    pixel_values = torch.randn(4, 3 * 2 * 2)
    image_grid_thw = torch.tensor([[1, 2, 2]])

    position_output = model.get_position_id_func()(
        input_ids=position_input_ids,
        image_grid_thw=image_grid_thw,
        attention_mask=attention_mask,
    )
    assert position_output["position_ids"].shape == (3, 1, 5)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        image_mask=image_mask,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        labels=labels,
    )

    assert outputs.loss is not None
    assert outputs.logits.shape == (1, 5, model.config.vocab_size)


def test_openpangu_vl_image_forward_accepts_collator_position_ids():
    model = _toy_model()
    position_input_ids = torch.tensor(
        [[1, model.config.vision_start_token_id, IMAGE_INPUT_INDEX, model.config.vision_end_token_id, 2]]
    )
    image_mask = position_input_ids == IMAGE_INPUT_INDEX
    input_ids = position_input_ids.masked_fill(image_mask, 0)
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    labels[image_mask] = IGNORE_INDEX
    pixel_values = torch.randn(4, 3 * 2 * 2)
    image_grid_thw = torch.tensor([[1, 2, 2]])

    position_output = model.get_position_id_func()(
        input_ids=position_input_ids,
        image_grid_thw=image_grid_thw,
        attention_mask=attention_mask,
    )
    position_ids = position_output["position_ids"].transpose(0, 1).contiguous()
    assert position_ids.shape == (1, 3, 5)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        image_mask=image_mask,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        position_ids=position_ids,
        labels=labels,
    )

    assert outputs.loss is not None
    assert outputs.logits.shape == (1, 5, model.config.vocab_size)


def test_openpangu_vl_parallel_plan_targets_language_model_experts():
    model_cls = MODELING_REGISTRY["openpangu_vl"]("OpenPanguVL")
    plan = model_cls.get_parallel_plan()
    ep_plan = plan.extra_parallel_plan["ep"]
    assert "model.language_model.layers.*.mlp.experts.gate_up_proj" in ep_plan
    assert "model.language_model.layers.*.mlp.experts.down_proj" in ep_plan

    base_plan = OpenPanguVLModel.get_parallel_plan()
    assert "language_model.layers.*.mlp.experts.gate_up_proj" in base_plan.extra_parallel_plan["ep"]

    text_plan = OpenPanguVLTextModel.get_parallel_plan()
    assert "layers.*.mlp.experts.gate_up_proj" in text_plan.extra_parallel_plan["ep"]


def test_openpangu_vl_config_keeps_nested_model_types():
    config = OpenPanguVLConfig.from_pretrained(TOY_CONFIG)
    assert config.model_type == "openpangu_vl"
    assert config.text_config.model_type == "openpangu_vl_text"

    nested_config = OpenPanguVLConfig(
        architectures=["OpenPanguVL"],
        model_type="openpangu_vl",
        text_config={
            "architectures": ["OpenPanguVLTextModel"],
            "model_type": "openpangu_vl_text",
            "vocab_size": 32,
        },
    )
    assert nested_config.model_type == "openpangu_vl"
    assert nested_config.architectures == ["OpenPanguVL"]
    assert nested_config.text_config.model_type == "openpangu_vl_text"
    assert nested_config.vocab_size == 32


def test_openpangu_vl_checkpoint_key_mapping_matches_hf_layout():
    model = _toy_model()

    assert (
        _convert_weight_key("model.layers.13.mlp.experts.0.gate_proj.weight", model)
        == "model.language_model.layers.13.mlp.experts.0.gate_proj.weight"
    )
    assert _convert_weight_key("visual.blocks.0.attn.proj.bias", model) == "model.visual.blocks.0.attn.proj.bias"
    assert _convert_weight_key("lm_head.weight", model) == "lm_head.weight"


@pytest.mark.parametrize(
    "prefix",
    [
        "model.language_model.layers.0.mlp",
        "language_model.layers.0.mlp",
        "layers.0.mlp",
    ],
)
def test_openpangu_vl_converter_merges_split_experts(prefix):
    converter = OpenPanguVLCheckpointTensorConverter(num_experts=2)
    hidden_size = 4
    intermediate_size = 3

    for expert in range(2):
        name = f"{prefix}.experts.{expert}.gate_proj.weight"
        assert converter.convert(name, torch.full((intermediate_size, hidden_size), expert)) is None

    up_result = None
    for expert in range(2):
        name = f"{prefix}.experts.{expert}.up_proj.weight"
        up_result = converter.convert(name, torch.full((intermediate_size, hidden_size), expert + 0.1))

    assert up_result is not None
    assert up_result.name == f"{prefix}.experts.gate_up_proj"
    assert up_result.tensor.shape == (2, 2 * intermediate_size, hidden_size)

    down_result = None
    for expert in range(2):
        name = f"{prefix}.experts.{expert}.down_proj.weight"
        down_result = converter.convert(name, torch.full((hidden_size, intermediate_size), expert + 0.2))

    assert down_result is not None
    assert down_result.name == f"{prefix}.experts.down_proj"
    assert down_result.tensor.shape == (2, hidden_size, intermediate_size)
    assert converter.finalize() == []


def test_openpangu_vl_converter_raises_on_incomplete_checkpoint():
    converter = OpenPanguVLCheckpointTensorConverter(num_experts=2)
    converter.convert("model.language_model.layers.0.mlp.experts.0.down_proj.weight", torch.randn(4, 3))

    with pytest.raises(RuntimeError, match="incomplete checkpoint detected"):
        converter.finalize()


def test_openpangu_vl_converter_factory_uses_text_config_expert_count():
    model = SimpleNamespace(config=SimpleNamespace(text_config=SimpleNamespace(n_routed_experts=8)))
    converter = create_openpangu_vl_checkpoint_tensor_converter(model)
    assert isinstance(converter, OpenPanguVLCheckpointTensorConverter)
    assert converter.num_experts == 8


def test_openpangu_vl_converter_passthrough_for_non_expert_key():
    converter = OpenPanguVLCheckpointTensorConverter(num_experts=2)
    tensor = torch.randn(4, 4)
    result = maybe_convert_checkpoint_tensor(
        "model.language_model.layers.0.self_attn.qkv_proj.weight", tensor, converter
    )
    assert result is not None
    assert result.name == "model.language_model.layers.0.self_attn.qkv_proj.weight"
    assert result.tensor is tensor
