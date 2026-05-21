import importlib.util
from pathlib import Path

import torch


_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "compose_qwen3_siglip_vlm.py"
_SPEC = importlib.util.spec_from_file_location("compose_qwen3_siglip_vlm", _SCRIPT_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
map_qwen_key = _MODULE.map_qwen_key
map_siglip_key = _MODULE.map_siglip_key
resize_siglip_position_embedding = _MODULE.resize_siglip_position_embedding


def test_compose_qwen3_siglip_key_mapping():
    assert map_qwen_key("model.layers.0.self_attn.q_proj.weight") == (
        "language_model.model.layers.0.self_attn.q_proj.weight"
    )
    assert map_qwen_key("lm_head.weight") == "language_model.lm_head.weight"
    assert map_qwen_key("visual.foo") is None

    assert map_siglip_key("vision_model.embeddings.patch_embedding.weight") == (
        "vision_tower.embeddings.patch_embedding.weight"
    )
    assert map_siglip_key("model.vision_model.encoder.layers.0.mlp.fc1.weight") == (
        "vision_tower.encoder.layers.0.mlp.fc1.weight"
    )
    assert map_siglip_key("text_model.encoder.layers.0.weight") is None


def test_resize_siglip_position_embedding():
    tensor = torch.arange(4 * 3, dtype=torch.float32).reshape(4, 3)
    resized = resize_siglip_position_embedding(tensor, target_num_positions=16)
    assert resized.shape == (16, 3)
