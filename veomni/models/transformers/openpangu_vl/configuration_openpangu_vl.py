# coding=utf-8
"""Configuration classes for the Pangu 30B vision-language model.

This file intentionally keeps the VL wrapper separate from the historical
Omni config. The text tower remains OpenPanguV2; the top-level config only
adds vision settings and multimodal token ids.
"""

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_rope_utils import rope_config_validation

from .configuration_openpangu_v2 import OpenPanguV2Config


_WRAPPER_ONLY_TEXT_CONFIG_KEYS = {
    "architectures",
    "auto_map",
    "model_architectures",
    "model_type",
    "text_config",
    "vision_config",
}


def _sanitize_text_config_kwargs(values):
    return {key: value for key, value in values.items() if key not in _WRAPPER_ONLY_TEXT_CONFIG_KEYS}


class OpenPanguVLVisionConfig(PretrainedConfig):
    model_type = "openpangu_vl_vision"
    base_config_key = "vision_config"

    def __init__(
        self,
        depth=32,
        hidden_size=3584,
        hidden_act="silu",
        intermediate_size=3420,
        num_heads=16,
        in_channels=3,
        patch_size=14,
        spatial_merge_size=2,
        temporal_patch_size=2,
        tokens_per_second=4,
        window_size=112,
        out_hidden_size=3584,
        fullatt_block_indexes=None,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.depth = depth
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.tokens_per_second = tokens_per_second
        self.window_size = window_size
        self.fullatt_block_indexes = fullatt_block_indexes if fullatt_block_indexes is not None else [7, 15, 23, 31]
        self.out_hidden_size = out_hidden_size
        self.initializer_range = initializer_range


class OpenPanguVLTextConfig(OpenPanguV2Config):
    model_type = "openpangu_vl_text"
    base_config_key = "text_config"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        num_hidden_layers=34,
        num_attention_heads=32,
        num_key_value_heads=8,
        rms_norm_eps=1e-5,
        hidden_size=4096,
        hidden_act="silu",
        intermediate_size=12800,
        initializer_range=0.02,
        tie_word_embeddings=False,
        use_sliding_window=False,
        sliding_window=None,
        max_window_layers=80,
        vocab_size=153376,
        max_position_embeddings=32768,
        use_cache=True,
        rope_theta=64000000.0,
        attention_dropout=0.0,
        rope_scaling=None,
        image_token_id=None,
        video_token_id=None,
        **kwargs,
    ):
        super().__init__(tie_word_embeddings=tie_word_embeddings, num_hidden_layers=num_hidden_layers, **kwargs)

        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.rms_norm_eps = rms_norm_eps
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.initializer_range = initializer_range
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window
        self.max_window_layers = max_window_layers
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout
        self.rope_scaling = rope_scaling

        if self.rope_scaling is not None and "type" in self.rope_scaling:
            if self.rope_scaling["type"] == "mrope":
                self.rope_scaling["type"] = "default"
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        if self.rope_parameters is None:
            self.rope_parameters = dict(self.rope_scaling or {"rope_type": "default", "type": "default"})
        self.rope_parameters.pop("rotary_mode", None)
        self.rope_parameters.setdefault("rope_type", self.rope_parameters.get("type", "default"))
        self.rope_parameters.setdefault("type", self.rope_parameters["rope_type"])
        self.rope_parameters.setdefault("rope_theta", rope_theta)
        self.rope_parameters.setdefault("partial_rotary_factor", getattr(self, "partial_rotary_factor", 1.0))
        rope_config_validation(self, ignore_keys={"mrope_section", "mrope_interleaved"})
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id


class OpenPanguVLConfig(PretrainedConfig):
    model_type = "openpangu_vl"
    sub_configs = {
        "vision_config": OpenPanguVLVisionConfig,
        "text_config": OpenPanguVLTextConfig,
    }
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        image_token_id=148909,
        video_token_id=148910,
        vision_start_token_id=148907,
        vision_end_token_id=148908,
        position_id_per_seconds=25,
        seconds_per_chunk=2,
        user_token_id=872,
        initializer_range=0.02,
        tie_word_embeddings=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()
        else:
            self.vision_config = vision_config

        text_config_values = _sanitize_text_config_kwargs(text_config) if isinstance(text_config, dict) else None
        text_config_kwargs = _sanitize_text_config_kwargs(kwargs)
        if isinstance(text_config_values, dict):
            self.text_config = self.sub_configs["text_config"](**text_config_values)
        elif text_config is None:
            self.text_config = self.sub_configs["text_config"](**text_config_kwargs)
        else:
            self.text_config = text_config

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        self.user_token_id = user_token_id
        self.position_id_per_seconds = position_id_per_seconds
        self.seconds_per_chunk = seconds_per_chunk
        self.initializer_range = initializer_range
        self.tie_word_embeddings = tie_word_embeddings

        if isinstance(text_config_values, dict):
            for key, value in text_config_values.items():
                setattr(self, key, value)


__all__ = [
    "OpenPanguVLConfig",
    "OpenPanguVLTextConfig",
    "OpenPanguVLVisionConfig",
]
