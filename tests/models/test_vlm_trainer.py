from types import SimpleNamespace

import pytest

from veomni.models import build_foundation_model
from veomni.trainer.vlm_trainer import (
    SINGLE_LR_VLM_MODEL_TYPES,
    VeOmniVLMArguments,
    VLMMDataArguments,
    VLMMModelArguments,
    VLMTrainer,
    _get_openpangu_vl_connector_module,
    _get_vlm_visual_module,
)

from ..tools.training_utils import make_eager_ops_config


_FREEZE_VIT_VLM_CASES = [
    pytest.param("./tests/toy_config/qwen2vl_toy/config.json", id="qwen2_vl"),
    pytest.param("./tests/toy_config/qwen3_5_toy/config.json", id="qwen3_5"),
    pytest.param("./tests/toy_config/qwen3_5_moe_toy/config.json", id="qwen3_5_moe"),
    pytest.param("./tests/toy_config/qwen25vl_toy/config.json", id="qwen2_5_vl"),
    pytest.param("./tests/toy_config/qwen3vl_toy/config.json", id="qwen3_vl"),
    pytest.param("./tests/toy_config/qwen3vlmoe_toy/config.json", id="qwen3_vl_moe"),
]


@pytest.mark.parametrize(
    "freeze_vit",
    [
        pytest.param(False, id="freeze_vit_disabled"),
        pytest.param(True, id="freeze_vit_enabled"),
    ],
)
@pytest.mark.parametrize("config_path", _FREEZE_VIT_VLM_CASES)
def test_freeze_vit_on_vlm_model(config_path, freeze_vit):
    # This test only constructs the model on `meta` and verifies freeze
    # behaviour — it never runs forward. Use an all-eager ops config so the
    # build works everywhere: it pins every per-op field (including the
    # Qwen3.5 GatedDeltaNet trio that has no FLA backend on NPU and the
    # GPU-only liger/triton defaults that fail NPU validation). Eager paths
    # that raise only at forward time are fine because this test never
    # forwards.
    ops_implementation = make_eager_ops_config()
    model = build_foundation_model(
        config_path=config_path,
        weights_path=None,
        torch_dtype="float32",
        init_device="meta",
        ops_implementation=ops_implementation,
    )
    visual = _get_vlm_visual_module(model)
    assert visual is not None

    args = VeOmniVLMArguments(
        model=VLMMModelArguments(
            config_path=config_path,
            ops_implementation=make_eager_ops_config(),
        ),
        data=VLMMDataArguments(train_path="dummy"),
    )
    args.train.freeze_vit = freeze_vit

    trainer = VLMTrainer.__new__(VLMTrainer)
    trainer.base = SimpleNamespace(
        args=args,
        model=model,
        model_config=model.config,
    )

    trainer._freeze_model_module()

    if freeze_vit:
        assert all(not param.requires_grad for param in visual.parameters())
    else:
        assert all(param.requires_grad for param in visual.parameters())


def _build_openpangu_vl_toy_model():
    return build_foundation_model(
        config_path="./tests/toy_config/openpangu_vl_toy/config.json",
        weights_path=None,
        torch_dtype="float32",
        init_device="meta",
        ops_implementation=make_eager_ops_config(),
    )


def _build_openpangu_vl_trainer(model):
    args = VeOmniVLMArguments(
        model=VLMMModelArguments(
            config_path="./tests/toy_config/openpangu_vl_toy/config.json",
            ops_implementation=make_eager_ops_config(),
        ),
        data=VLMMDataArguments(train_path="dummy"),
    )
    trainer = VLMTrainer.__new__(VLMTrainer)
    trainer.base = SimpleNamespace(
        args=args,
        model=model,
        model_config=model.config,
    )
    return trainer


def test_openpangu_vl_freeze_vit_keeps_connector_trainable():
    model = _build_openpangu_vl_toy_model()
    trainer = _build_openpangu_vl_trainer(model)
    trainer.base.args.train.freeze_vit = True

    trainer._freeze_model_module()

    visual = _get_vlm_visual_module(model)
    connector = _get_openpangu_vl_connector_module(model)
    connector_param_ids = {id(param) for param in connector.parameters()}

    assert all(param.requires_grad for param in connector.parameters())
    assert all(not param.requires_grad for param in visual.parameters() if id(param) not in connector_param_ids)
    assert all(param.requires_grad for param in model.language_model.parameters())


def test_openpangu_vl_freeze_connector_and_llm():
    model = _build_openpangu_vl_toy_model()
    trainer = _build_openpangu_vl_trainer(model)
    trainer.base.args.train.freeze_connector = True
    trainer.base.args.train.freeze_llm = True

    trainer._freeze_model_module()

    connector = _get_openpangu_vl_connector_module(model)
    assert all(not param.requires_grad for param in connector.parameters())
    assert all(not param.requires_grad for param in model.language_model.parameters())


def test_openpangu_vl_optimizer_uses_single_lr_group(monkeypatch):
    model = _build_openpangu_vl_toy_model()
    trainer = _build_openpangu_vl_trainer(model)
    trainer.base.args.train.optimizer.lr = 1.0e-5
    trainer.base.args.train.vit_lr = 1.0e-7
    captured = {}

    def fake_build_optimizer(*args, **kwargs):
        del args
        captured["param_groups"] = kwargs["param_groups"]
        return object()

    monkeypatch.setattr("veomni.trainer.vlm_trainer.build_optimizer", fake_build_optimizer)

    assert "openpangu_vl" in SINGLE_LR_VLM_MODEL_TYPES
    trainer._build_optimizer()

    assert len(captured["param_groups"]) == 1
    assert captured["param_groups"][0]["lr"] == trainer.base.args.train.optimizer.lr
