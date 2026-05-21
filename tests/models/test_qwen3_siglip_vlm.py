from types import SimpleNamespace

import torch

import veomni.distributed.sequence_parallel.data as sp_data
from veomni.models.transformers.qwen3_siglip_vlm import modeling_qwen3_siglip_vlm
from veomni.models.transformers.qwen3_siglip_vlm.configuration_qwen3_siglip_vlm import Qwen3SiglipVLMConfig
from veomni.models.transformers.qwen3_siglip_vlm.modeling_qwen3_siglip_vlm import (
    Qwen3SiglipVLMForConditionalGeneration,
)


def test_qwen3_siglip_vlm_forward_with_one_image_token():
    config = Qwen3SiglipVLMConfig.from_pretrained("tests/toy_config/qwen3_siglip_vlm_toy")
    model = Qwen3SiglipVLMForConditionalGeneration(config)
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
    )

    assert outputs.logits.shape[:2] == input_ids.shape
    outputs.logits.sum().backward()


def test_qwen3_siglip_vlm_forward_batches_multiple_images_into_one_encoder_call():
    config = Qwen3SiglipVLMConfig.from_pretrained("tests/toy_config/qwen3_siglip_vlm_toy")
    model = Qwen3SiglipVLMForConditionalGeneration(config)
    input_ids = torch.tensor([[0, 8, 0, 9]])
    attention_mask = torch.ones_like(input_ids)
    image_mask = torch.tensor([[True, False, True, False]])
    pixel_values = torch.randn(6, 3 * 14 * 14)
    image_grid_hw = torch.tensor([[2, 2], [1, 2]])

    calls = 0
    original_forward = model.vision_tower.encoder.forward

    def wrapped_forward(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_forward(*args, **kwargs)

    model.vision_tower.encoder.forward = wrapped_forward

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        image_mask=image_mask,
        pixel_values=pixel_values,
        image_grid_hw=image_grid_hw,
    )

    assert calls == 1
    assert outputs.logits.shape[:2] == input_ids.shape


def test_qwen3_siglip_vlm_uses_dummy_vision_forward_for_fsdp_text_only_batch(monkeypatch):
    config = Qwen3SiglipVLMConfig.from_pretrained("tests/toy_config/qwen3_siglip_vlm_toy")
    model = Qwen3SiglipVLMForConditionalGeneration(config)
    input_ids = torch.tensor([[8, 9, 10, 2]])
    attention_mask = torch.ones_like(input_ids)

    monkeypatch.setattr(
        modeling_qwen3_siglip_vlm,
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


def test_qwen3_siglip_vlm_vision_tower_slices_position_embeddings_for_sp(monkeypatch):
    config = Qwen3SiglipVLMConfig.from_pretrained("tests/toy_config/qwen3_siglip_vlm_toy")
    model = Qwen3SiglipVLMForConditionalGeneration(config)
    pixel_values = torch.randn(2, 3 * 14 * 14)
    image_grid_hw = torch.tensor([[2, 2]])

    parallel_state = SimpleNamespace(sp_enabled=True, sp_size=2, sp_rank=0)
    monkeypatch.setattr(modeling_qwen3_siglip_vlm, "get_parallel_state", lambda: parallel_state)
    monkeypatch.setattr(sp_data, "get_parallel_state", lambda: parallel_state)

    calls = 0

    def fake_encoder_forward(hidden_states, cu_seqlens, max_seqlen, **kwargs):
        nonlocal calls
        del kwargs
        calls += 1
        assert hidden_states.shape[0] == 2
        assert cu_seqlens.tolist() == [0, 4]
        assert max_seqlen == 4
        return hidden_states

    model.vision_tower.encoder.forward = fake_encoder_forward

    outputs = model.vision_tower(pixel_values, image_grid_hw)

    assert calls == 1
    assert outputs.shape[0] == 2


def test_qwen3_siglip_vlm_declares_flash_attention_support():
    assert Qwen3SiglipVLMForConditionalGeneration._supports_flash_attn is True
