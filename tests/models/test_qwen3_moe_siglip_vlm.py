from types import SimpleNamespace

import torch

from tests.tools.training_utils import make_eager_ops_config
from veomni.models.transformers.qwen3_moe_siglip_vlm import modeling_qwen3_moe_siglip_vlm
from veomni.models.transformers.qwen3_moe_siglip_vlm.checkpoint_tensor_converter import (
    Qwen3MoeSiglipVLMCheckpointTensorConverter,
)
from veomni.models.transformers.qwen3_moe_siglip_vlm.configuration_qwen3_moe_siglip_vlm import (
    Qwen3MoeSiglipVLMConfig,
)
from veomni.models.transformers.qwen3_moe_siglip_vlm.modeling_qwen3_moe_siglip_vlm import (
    Qwen3MoeSiglipVLMForConditionalGeneration,
)
from veomni.ops import apply_ops_config


def test_qwen3_moe_siglip_vlm_forward_with_one_image_token():
    apply_ops_config(make_eager_ops_config())
    config = Qwen3MoeSiglipVLMConfig.from_pretrained("tests/toy_config/qwen3_moe_siglip_vlm_toy")
    model = Qwen3MoeSiglipVLMForConditionalGeneration(config)
    input_ids = torch.tensor([[0, 8, 9, 2]])
    attention_mask = torch.ones_like(input_ids)
    image_mask = torch.tensor([[True, False, False, False]])
    pixel_values = torch.randn(4, 3 * 14 * 14)
    image_grid_hw = torch.tensor([[2, 2]])

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        image_mask=image_mask,
        pixel_values=pixel_values,
        image_grid_hw=image_grid_hw,
        labels=input_ids,
    )

    assert outputs.loss is not None
    assert outputs.aux_loss is not None
    outputs.loss.backward()


def test_qwen3_moe_siglip_vlm_uses_dummy_vision_forward_for_fsdp_text_only_batch(monkeypatch):
    apply_ops_config(make_eager_ops_config())
    config = Qwen3MoeSiglipVLMConfig.from_pretrained("tests/toy_config/qwen3_moe_siglip_vlm_toy")
    model = Qwen3MoeSiglipVLMForConditionalGeneration(config)
    input_ids = torch.tensor([[8, 9, 10, 2]])
    attention_mask = torch.ones_like(input_ids)

    monkeypatch.setattr(
        modeling_qwen3_moe_siglip_vlm,
        "get_parallel_state",
        lambda: SimpleNamespace(fsdp_enabled=True, sp_enabled=False),
    )

    calls = 0
    original_dummy_forward = model.vision_tower.dummy_forward

    def wrapped_dummy_forward(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_dummy_forward(*args, **kwargs)

    model.vision_tower.dummy_forward = wrapped_dummy_forward
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)

    assert calls == 1
    assert outputs.loss is not None


def test_qwen3_moe_siglip_vlm_reexports_qwen3_moe_op_slots():
    qwen3_moe_modeling = modeling_qwen3_moe_siglip_vlm.qwen3_moe_modeling
    assert modeling_qwen3_moe_siglip_vlm.veomni_rms_norm is qwen3_moe_modeling.veomni_rms_norm
    assert modeling_qwen3_moe_siglip_vlm.veomni_apply_rotary_pos_emb is qwen3_moe_modeling.veomni_apply_rotary_pos_emb
    assert modeling_qwen3_moe_siglip_vlm.veomni_swiglu_mlp is qwen3_moe_modeling.veomni_swiglu_mlp
    assert modeling_qwen3_moe_siglip_vlm.veomni_moe_experts_forward is qwen3_moe_modeling.veomni_moe_experts_forward
    assert modeling_qwen3_moe_siglip_vlm.veomni_causal_lm_loss is qwen3_moe_modeling.veomni_causal_lm_loss
    assert modeling_qwen3_moe_siglip_vlm.veomni_load_balancing_loss is qwen3_moe_modeling.veomni_load_balancing_loss


def test_qwen3_moe_siglip_vlm_parallel_plan_prefixes_language_model_experts():
    plan = Qwen3MoeSiglipVLMForConditionalGeneration.get_parallel_plan(None)
    ep_plan = plan.extra_parallel_plan["ep"]
    assert "language_model.model.layers.*.mlp.experts.gate_up_proj" in ep_plan
    assert "language_model.model.layers.*.mlp.experts.down_proj" in ep_plan


def test_qwen3_moe_siglip_vlm_converter_maps_bare_language_keys():
    converter = Qwen3MoeSiglipVLMCheckpointTensorConverter(num_experts=4)
    tensor = torch.randn(2, 3)
    converted = converter.convert("model.embed_tokens.weight", tensor)
    assert converted is not None
    assert converted.name == "language_model.model.embed_tokens.weight"
    assert converted.tensor is tensor


def test_qwen3_moe_siglip_vlm_converter_merges_split_experts_under_language_model_prefix():
    converter = Qwen3MoeSiglipVLMCheckpointTensorConverter(num_experts=2)
    hidden_size = 4
    intermediate_size = 3

    down_results = []
    for expert in range(2):
        name = f"model.layers.0.mlp.experts.{expert}.down_proj.weight"
        down_results.append(converter.convert(name, torch.full((hidden_size, intermediate_size), expert + 0.2)))
    assert down_results[0] is None
    assert down_results[1] is not None
    assert down_results[1].name == "language_model.model.layers.0.mlp.experts.down_proj"
    assert down_results[1].tensor.shape == (2, hidden_size, intermediate_size)

    for expert in range(2):
        name = f"language_model.model.layers.0.mlp.experts.{expert}.gate_proj.weight"
        assert converter.convert(name, torch.full((intermediate_size, hidden_size), expert)) is None

    up_results = []
    for expert in range(2):
        name = f"language_model.model.layers.0.mlp.experts.{expert}.up_proj.weight"
        up_results.append(converter.convert(name, torch.full((intermediate_size, hidden_size), expert + 0.1)))

    assert up_results[0] is None
    assert up_results[1] is not None
    assert up_results[1].name == "language_model.model.layers.0.mlp.experts.gate_up_proj"
    assert up_results[1].tensor.shape == (2, 2 * intermediate_size, hidden_size)
    converter.finalize()


def test_qwen3_moe_siglip_vlm_declares_flash_attention_support():
    assert Qwen3MoeSiglipVLMForConditionalGeneration._supports_flash_attn is True
