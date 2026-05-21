import importlib.util
from pathlib import Path

from transformers import AutoConfig, AutoModelForCausalLM


_REPO_ROOT = Path(__file__).resolve().parents[2]
_MERGE_SCRIPT_PATH = _REPO_ROOT / "scripts" / "merge_dcp_to_hf.py"
_INFER_SCRIPT_PATH = _REPO_ROOT / "scripts" / "multimodal" / "qwen3_siglip_1p7b" / "infer_hf.py"


def _load_script(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_merge_dcp_to_hf_copies_raw_assets_when_auto_assets_fail(tmp_path, monkeypatch):
    module = _load_script(_MERGE_SCRIPT_PATH, "merge_dcp_to_hf_for_test")
    load_dir = tmp_path / "global_step_10"
    save_dir = tmp_path / "hf_ckpt"
    assets_dir = tmp_path / "assets"
    load_dir.mkdir()
    assets_dir.mkdir()

    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        (assets_dir / name).write_text(f"asset:{name}")

    monkeypatch.setattr(
        module.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("custom config")),
    )
    monkeypatch.setattr(
        module.AutoProcessor,
        "from_pretrained",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("custom processor")),
    )

    def fake_save_model_weights(output_dir, checkpoint_path, shard_size, model_assets):
        assert checkpoint_path == str(load_dir)
        assert shard_size == 2_000_000_000
        assert model_assets is None
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / "model.safetensors").write_bytes(b"fake")

    monkeypatch.setattr(module, "save_model_weights", fake_save_model_weights)

    module.merge_to_hf_pt(str(load_dir), str(save_dir), model_assets_dir=str(assets_dir))

    assert (save_dir / "model.safetensors").read_bytes() == b"fake"
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        assert (save_dir / name).read_text() == f"asset:{name}"


def test_qwen3_siglip_infer_registers_transformers_auto_classes():
    module = _load_script(_INFER_SCRIPT_PATH, "qwen3_siglip_infer_for_test")
    module.register_qwen3_siglip_vlm()

    config = AutoConfig.for_model("qwen3_siglip_vlm")

    assert type(config).__name__ == "Qwen3SiglipVLMConfig"
    assert AutoModelForCausalLM._model_mapping[type(config)].__name__ == "Qwen3SiglipVLMForConditionalGeneration"


def test_merge_dcp_to_hf_writes_qwen3_siglip_canonical_chat_template(tmp_path, monkeypatch):
    module = _load_script(_MERGE_SCRIPT_PATH, "merge_dcp_to_hf_qwen3_siglip_for_test")
    load_dir = tmp_path / "global_step_10"
    save_dir = tmp_path / "hf_ckpt"
    assets_dir = tmp_path / "assets"
    load_dir.mkdir()
    assets_dir.mkdir()

    (assets_dir / "config.json").write_text('{"model_type": "qwen3_siglip_vlm"}')
    (assets_dir / "chat_template.jinja").write_text("stale-kimi-template")

    monkeypatch.setattr(
        module.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("custom config")),
    )
    monkeypatch.setattr(
        module.AutoProcessor,
        "from_pretrained",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("custom processor")),
    )

    def fake_save_model_weights(output_dir, checkpoint_path, shard_size, model_assets):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / "model.safetensors").write_bytes(b"fake")

    monkeypatch.setattr(module, "save_model_weights", fake_save_model_weights)

    module.merge_to_hf_pt(str(load_dir), str(save_dir), model_assets_dir=str(assets_dir))

    chat_template = (save_dir / "chat_template.jinja").read_text()
    assert "stale-kimi-template" not in chat_template
    assert "<|image_pad|>" in chat_template
    assert "image_token_num" in chat_template
    assert "<|vision_start|>" not in chat_template
    assert "<|media_start|>" not in chat_template


def test_qwen3_siglip_infer_builds_hf_chat_template_messages():
    module = _load_script(_INFER_SCRIPT_PATH, "qwen3_siglip_infer_prompt_for_test")

    messages = module.build_messages("Describe this image.", image_token_num=3)

    assert messages == [
        {
            "role": "user",
            "content": [
                {"type": "image", "image_token_num": 3},
                {"type": "text", "text": "\nDescribe this image."},
            ],
        }
    ]


def test_qwen3_siglip_infer_raw_prompt_bypasses_chat_template():
    module = _load_script(_INFER_SCRIPT_PATH, "qwen3_siglip_infer_raw_prompt_for_test")

    assert module.build_prompt(None, "Leo Messi is from", raw_prompt=True) == "Leo Messi is from"
    assert module.build_prompt(None, "Describe this image.", image_token_num=3, raw_prompt=True) == (
        "<|image_pad|><|image_pad|><|image_pad|>\nDescribe this image."
    )
