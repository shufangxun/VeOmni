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

import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import torch
from torch import distributed as dist

from ..arguments import DataArguments, ModelArguments, TrainingArguments, VeOmniArguments
from ..data import MainCollator, build_data_transform, build_multimodal_chat_template
from ..distributed.clip_grad_norm import veomni_clip_grad_norm
from ..distributed.parallel_state import get_parallel_state
from ..models import build_foundation_model, build_processor, build_tokenizer
from ..optim import build_optimizer
from ..utils import helper
from ..utils.device import synchronize
from ..utils.dist_utils import all_reduce
from ..utils.loss_utils import count_loss_token
from ..utils.model_utils import pretty_print_trainable_parameters
from ..utils.multisource_utils import parse_multisource_config
from .base import BaseTrainer


logger = helper.create_logger(__name__)
MAX_PIXELS = 768 * 28 * 28


def _get_vlm_visual_module(model):
    # Qwen-VL wrappers are not consistent across transformers versions:
    # older releases may expose `visual` directly on the conditional model
    # for backward compatibility, while newer ones only keep `model.visual`.
    visual = getattr(model, "visual", None)
    if visual is not None:
        return visual

    inner_model = getattr(model, "model", None)
    if inner_model is not None:
        return getattr(inner_model, "visual", None)

    return None


@dataclass
class VLMTrainingArguments(TrainingArguments):
    freeze_vit: bool = field(
        default=False,
        metadata={"help": "Whether or not to freeze the vit parameters."},
    )
    freeze_audio_tower: bool = field(
        default=False,
        metadata={"help": "Whether or not to freeze the audio tower parameters."},
    )
    vit_lr: float = field(
        default=1e-6,
        metadata={"help": "Maximum learning rate for vit parameters."},
    )
    freeze_connector: bool = field(
        default=False,
        metadata={"help": "Whether or not to freeze the VLM connector parameters."},
    )
    freeze_llm: bool = field(
        default=False,
        metadata={"help": "Whether or not to freeze the language model parameters."},
    )
    connector_lr: Optional[float] = field(
        default=None,
        metadata={"help": "Maximum learning rate for connector parameters."},
    )
    llm_lr: Optional[float] = field(
        default=None,
        metadata={"help": "Maximum learning rate for language model parameters."},
    )


@dataclass
class VLMMDataArguments(DataArguments):
    data_type: str = field(
        default="qwen3_vl",
        metadata={"help": "Name of the multimodal data transform registered in DATA_TRANSFORM_REGISTRY."},
    )
    mm_configs: Optional[Dict] = field(
        default_factory=dict,
        metadata={"help": "Config for multimodal input."},
    )


@dataclass
class VLMMModelArguments(ModelArguments):
    encoder_data_balance: Optional[bool] = field(
        default=False, metadata={"help": "Whether to balance encoder data for qwen3-vl model"}
    )
    encoder_data_balance_sorting_algo: Optional[str] = field(
        default="post_mbs_balancing_greedy_without_pad",
        metadata={
            "help": "The sorting algorithm of encoder data balance. All viable algorithms are defined in "
            "veomni/utils/data_balance/balance_sorting_algo.py, SORTING_ALGO_FUNC"
        },
    )


@dataclass
class VeOmniVLMArguments(VeOmniArguments):
    model: "VLMMModelArguments" = field(default_factory=VLMMModelArguments)
    data: "VLMMDataArguments" = field(default_factory=VLMMDataArguments)
    train: "VLMTrainingArguments" = field(default_factory=VLMTrainingArguments)


class VLMTrainer:
    def __init__(self, args: VeOmniVLMArguments):
        # BaseTrainer.__init__ is NOT called here; we call its private
        # helpers one-by-one so the sequence is explicit.
        self.base = BaseTrainer.__new__(BaseTrainer)
        self.base.args = args

        self.base._setup()

        # rewrite build model to support data balancing
        self._build_model()

        # rewrite freeze_model_module to support freeze multimodal encoder, etc.
        self._freeze_model_module()

        # rewrite build_model_assets to support chat_template and processor for multimodal datasets
        self._build_model_assets()

        # rewrite build_data_transform to support multimodal transform
        self._build_data_transform()

        self.base._build_dataset()

        # rewrite build_collate_fn to support multimodal collate_fn
        self._build_collate_fn()

        self.base._build_dataloader()
        self.base._build_parallelized_model()

        # rewrite build_optimizer to support different lr param groups
        self._build_optimizer()

        self.base._build_lr_scheduler()
        self.base._build_training_context()
        self.base._init_callbacks()
        self._build_train_source_names()

    def _build_train_source_names(self) -> None:
        args: VeOmniVLMArguments = self.base.args
        self.train_source_names = []
        if args.data.enable_multisource:
            self.train_source_names = list(parse_multisource_config(args.data.train_path)["names"])

    def _build_model(self):
        args: VeOmniVLMArguments = self.base.args
        logger.info_rank0("Build model")
        self.base.model = build_foundation_model(
            config_path=args.model.config_path,
            weights_path=args.model.model_path,
            torch_dtype="float32" if args.train.accelerator.fsdp_config.mixed_precision.enable else "bfloat16",
            init_device=args.train.init_device,
            encoder_data_balance=args.model.encoder_data_balance,
            encoder_data_balance_sorting_algo=args.model.encoder_data_balance_sorting_algo,
            ops_implementation=args.model.ops_implementation,
            config_kwargs=args.model.model_config,
        )
        self.base.model_config = self.base.model.config

    def _freeze_model_module(self):
        args: VeOmniVLMArguments = self.base.args
        model_config = self.base.model_config
        if model_config.model_type in ("qwen2_5_omni", "qwen3_omni_moe"):
            self.base.model.disable_talker()

        if args.train.freeze_vit:
            if model_config.model_type == "qwen3_siglip_vlm":
                self.base.model.vision_tower.requires_grad_(False)
            elif model_config.model_type in ("qwen2_5_omni", "qwen3_omni_moe"):
                self.base.model.thinker.visual.requires_grad_(False)
                self.base.model.thinker.visual.merger.requires_grad_(True)
            else:
                # Resolve both paths so freeze_vit works for the transformers v4-style Back Compatible alias
                # and the nested layout used by the v5-style Qwen3.5 models.
                visual = _get_vlm_visual_module(self.base.model)
                if visual is None:
                    raise AttributeError(f"Cannot find visual module for model_type={model_config.model_type}.")
                visual.requires_grad_(False)

        if args.train.freeze_audio_tower and model_config.model_type in ("qwen2_5_omni", "qwen3_omni_moe"):
            self.base.model.thinker.audio_tower.requires_grad_(False)
            # Qwen2.5-Omni uses audio_tower.proj; Qwen3-Omni-MoE uses audio_tower.proj1.
            audio_proj = (
                getattr(self.base.model.thinker.audio_tower, "proj1", None) or self.base.model.thinker.audio_tower.proj
            )
            audio_proj.requires_grad_(True)

        if model_config.model_type == "qwen3_siglip_vlm":
            if args.train.freeze_connector:
                self.base.model.connector.requires_grad_(False)
            if args.train.freeze_llm:
                self.base.model.language_model.requires_grad_(False)

        pretty_print_trainable_parameters(self.base.model)
        helper.print_device_mem_info("VRAM usage after building model")

    def _build_model_assets(self):
        args: VeOmniVLMArguments = self.base.args
        if self.base.model_config.model_type == "qwen3_siglip_vlm":
            tokenizer = build_tokenizer(args.model.tokenizer_path)
            self.base.processor = SimpleNamespace(tokenizer=tokenizer)
        else:
            self.base.processor = build_processor(args.model.tokenizer_path, max_pixels=MAX_PIXELS)
        if self.base.model_config.model_type not in ("qwen2_5_omni", "qwen3_omni_moe"):
            self.base.chat_template = build_multimodal_chat_template(
                args.data.chat_template, self.base.processor.tokenizer
            )
            self.base.model_assets = [self.base.processor, self.base.chat_template]
        else:
            self.base.chat_template = None
            self.base.model_assets = [self.base.processor]

    def _build_data_transform(self):
        args: VeOmniVLMArguments = self.base.args

        self.base.data_transform = build_data_transform(
            args.data.data_type,
            processor=self.base.processor,
            chat_template=self.base.chat_template,
            position_id_func=self.base.model.get_position_id_func(),
            max_seq_len=args.data.max_seq_len,
            text_keys=args.data.text_keys,
            **args.data.mm_configs,
        )

    def _build_collate_fn(self):
        if self.base.model_config.model_type in ("qwen2_5_omni", "qwen3_omni_moe"):
            data_collate_info = {
                "audio_feature_lengths": (0, False, None, None),
                "input_features": (0, True, 0, 1),
                "audio_mask": (-1, False, 0, 1),
            }
        else:
            data_collate_info = {}
        seq_classification = self.base.args.data.data_type == "classification"
        pad_to_length = self.base.args.train.pad_to_length
        self.base.collate_fn = MainCollator(
            pad_to_length=pad_to_length,
            seq_classification=seq_classification,
            data_collate_info=data_collate_info,
        )

    def _build_optimizer(self):
        args: VeOmniVLMArguments = self.base.args

        if self.base.model_config.model_type == "qwen3_siglip_vlm":
            vit_params, connector_params, llm_params, other_params = [], [], [], []
            for name, param in self.base.model.named_parameters():
                if param.requires_grad:
                    if name.startswith("vision_tower."):
                        vit_params.append(param)
                    elif name.startswith("connector."):
                        connector_params.append(param)
                    elif name.startswith("language_model."):
                        llm_params.append(param)
                    else:
                        other_params.append(param)
            param_groups = [
                {"params": vit_params, "lr": args.train.vit_lr},
                {"params": connector_params, "lr": args.train.connector_lr or args.train.optimizer.lr},
                {"params": llm_params, "lr": args.train.llm_lr or args.train.optimizer.lr},
                {"params": other_params, "lr": args.train.optimizer.lr},
            ]
            param_groups = [group for group in param_groups if group["params"]]
        else:
            vit_params, other_params = [], []
            for name, param in self.base.model.named_parameters():
                if param.requires_grad:
                    if "visual" in name:
                        vit_params.append(param)
                    else:
                        other_params.append(param)

            param_groups = [
                {"params": vit_params, "lr": args.train.vit_lr},
                {"params": other_params, "lr": args.train.optimizer.lr},
            ]

        # Build optimizer
        self.base.optimizer = build_optimizer(
            self.base.model,
            lr=args.train.optimizer.lr,
            weight_decay=args.train.optimizer.weight_decay,
            fused=True,
            optimizer_type=args.train.optimizer.type,
            param_groups=param_groups,
            no_decay_modules=args.train.optimizer.no_decay_modules,
            no_decay_params=args.train.optimizer.no_decay_params,
        )

    def on_train_begin(self):
        self.base.on_train_begin()

    def on_train_end(self):
        self.base.on_train_end()

    def on_epoch_begin(self):
        self.base.on_epoch_begin()

    def on_epoch_end(self):
        self.base.on_epoch_end()

    def on_step_begin(self, micro_batches=None):
        self.base.on_step_begin(micro_batches=micro_batches)

    def on_step_end(self, loss=None, loss_dict=None, grad_norm=None):
        self.base.on_step_end(loss=loss, loss_dict=loss_dict, grad_norm=grad_norm)

    def _should_log_train_source_loss(self) -> bool:
        args: VeOmniVLMArguments = self.base.args
        interval = args.train.log_train_source_loss_steps
        return bool(
            interval
            and args.data.enable_multisource
            and not get_parallel_state().sp_enabled
            and self.base.state.global_step % interval == 0
        )

    def _should_log_train_domain_loss(self) -> bool:
        args: VeOmniVLMArguments = self.base.args
        interval = args.train.log_train_domain_loss_steps
        return bool(interval and not get_parallel_state().sp_enabled and self.base.state.global_step % interval == 0)

    @staticmethod
    def _as_list(value: Any) -> List[Any]:
        if value is None:
            return []
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().view(-1).tolist()
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]

    def _source_name_from_idx(self, ds_idx: int) -> str:
        if 0 <= ds_idx < len(self.train_source_names):
            return self.train_source_names[ds_idx]
        return f"source_{ds_idx}"

    def _snapshot_train_loss_metadata(
        self,
        micro_batches: List[Dict[str, Any]],
        include_source: bool,
        include_domain: bool,
    ) -> List[Dict[str, Any]]:
        metadata = []
        for micro_batch in micro_batches:
            source_names = []
            domain_names = []
            if include_source:
                ds_indices = [int(idx) for idx in self._as_list(micro_batch.get("ds_idx"))]
                source_names = [str(name) for name in self._as_list(micro_batch.get("source_name"))]
                if not source_names:
                    source_names = [self._source_name_from_idx(ds_idx) for ds_idx in ds_indices]
            if include_domain:
                domain_names = [str(name) for name in self._as_list(micro_batch.get("domain_name"))]

            position_ids = micro_batch.get("position_ids")
            if position_ids is None:
                metadata.append({"source_names": [], "domain_names": [], "segment_starts": []})
                continue

            position_ids = position_ids.detach().cpu()
            if position_ids.dim() == 3:
                position_ids = position_ids[:, 0, :]
            position_ids = position_ids.reshape(-1)
            segment_starts = (position_ids == 0).nonzero(as_tuple=False).view(-1).tolist()
            if not segment_starts or segment_starts[0] != 0:
                segment_starts = [0, *segment_starts]

            if include_source and len(source_names) != len(segment_starts):
                logger.warning_once(
                    "Skip training source loss for a micro batch because source metadata does not match packed "
                    "segments: "
                    f"num_sources={len(source_names)}, num_segments={len(segment_starts)}."
                )
                source_names = []
            if include_domain and len(domain_names) != len(segment_starts):
                logger.warning_once(
                    "Skip training domain loss for a micro batch because domain metadata does not match packed "
                    f"segments: num_domains={len(domain_names)}, num_segments={len(segment_starts)}."
                )
                domain_names = []

            metadata.append(
                {"source_names": source_names, "domain_names": domain_names, "segment_starts": segment_starts}
            )
        return metadata

    def _accumulate_train_group_loss(
        self,
        group_stats: Dict[str, Dict[str, float]],
        micro_batch_metadata: Dict[str, Any],
        group_key: str,
        log_probs: torch.Tensor,
    ) -> None:
        group_names = micro_batch_metadata[group_key]
        segment_starts = micro_batch_metadata["segment_starts"]
        if not group_names or log_probs is None:
            return

        log_probs = log_probs.detach().float().reshape(-1)
        seq_len = log_probs.numel()
        segment_ends = [*segment_starts[1:], seq_len]

        for group_name, start, end in zip(group_names, segment_starts, segment_ends):
            end = min(end, log_probs.numel())
            if end <= start:
                continue

            group_log_probs = log_probs[start:end]
            valid_mask = group_log_probs != 0
            token_count = int(valid_mask.sum().item())
            if token_count == 0:
                continue

            item = group_stats[group_name]
            item["loss_sum"] += float((-group_log_probs[valid_mask]).sum().item())
            item["tokens"] += token_count

    @staticmethod
    def _metric_group_name(group_name: str) -> str:
        return group_name.replace("/", "_")

    def _ordered_train_group_names(
        self,
        group_stats: Dict[str, Dict[str, float]],
        ordered_group_names: Optional[List[str]] = None,
    ) -> List[str]:
        group_names = list(ordered_group_names or [])
        known_names = set(group_names)
        local_extra_names = sorted(set(group_stats.keys()) - known_names)

        if ordered_group_names is not None and not local_extra_names:
            return group_names

        if dist.is_available() and dist.is_initialized():
            dp_group = get_parallel_state().dp_group
            gathered_names = [None for _ in range(dist.get_world_size(group=dp_group))]
            dist.all_gather_object(gathered_names, local_extra_names, group=dp_group)
            extra_names = sorted({name for names in gathered_names for name in names})
        else:
            extra_names = local_extra_names

        group_names.extend(name for name in extra_names if name not in known_names)
        return group_names

    def _build_train_group_metrics(
        self,
        group_stats: Dict[str, Dict[str, float]],
        aggregation_time: float,
        metric_prefix: str,
        aggregation_metric_name: str,
        ordered_group_names: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        metrics = {}
        group_names = self._ordered_train_group_names(group_stats, ordered_group_names)

        for group_name in group_names:
            local_stats = group_stats.get(group_name, {})
            loss_sum = float(local_stats.get("loss_sum", 0.0))
            tokens = float(local_stats.get("tokens", 0.0))
            global_loss_sum, global_tokens = all_reduce(
                (loss_sum, tokens), op="sum", group=get_parallel_state().dp_group
            )
            if global_tokens == 0:
                continue

            loss = global_loss_sum / global_tokens
            metric_name = self._metric_group_name(group_name)
            metrics[f"{metric_prefix}/{metric_name}/loss"] = loss
            metrics[f"{metric_prefix}/{metric_name}/ppl"] = math.exp(loss) if loss < 100 else float("inf")
            metrics[f"{metric_prefix}/{metric_name}/tokens"] = global_tokens

        max_aggregation_time = all_reduce(aggregation_time, op="max", group=get_parallel_state().dp_group)
        metrics[aggregation_metric_name] = max_aggregation_time * 1000
        return metrics

    def train_step(
        self,
        data_iterator: Any,
    ) -> Dict[str, float]:
        args: VeOmniVLMArguments = self.base.args
        self.base.state.global_step += 1

        micro_batches: List[Dict[str, Any]] = next(data_iterator)
        log_train_source_loss = self._should_log_train_source_loss()
        log_train_domain_loss = self._should_log_train_domain_loss()
        log_train_group_loss = log_train_source_loss or log_train_domain_loss
        train_loss_metadata = (
            self._snapshot_train_loss_metadata(
                micro_batches,
                include_source=log_train_source_loss,
                include_domain=log_train_domain_loss,
            )
            if log_train_group_loss
            else None
        )
        train_source_stats: Dict[str, Dict[str, float]] = defaultdict(lambda: {"loss_sum": 0.0, "tokens": 0.0})
        train_domain_stats: Dict[str, Dict[str, float]] = defaultdict(lambda: {"loss_sum": 0.0, "tokens": 0.0})
        train_source_aggregation_time = 0.0
        train_domain_aggregation_time = 0.0

        self.on_step_begin(micro_batches=micro_batches)

        # Forward and backward for each micro batch
        synchronize()

        total_loss = 0.0
        total_loss_dict = defaultdict(int)

        # token num for fixed_ce_loss in postforward
        self.base.micro_batches_token_len = count_loss_token(micro_batches)
        num_micro_steps = len(micro_batches)
        # forward and backward pass with gradient_accumulationsteps
        for micro_step, micro_batch in enumerate(micro_batches):
            self.base.model_reshard(micro_step, num_micro_steps)
            loss: torch.Tensor
            loss_dict: Dict[str, torch.Tensor]
            # token num for fixed_ce_loss in postforward
            self.base.micro_batch_token_len = count_loss_token(micro_batch)
            self.base.return_train_source_log_probs = log_train_group_loss
            loss, loss_dict = self.base.forward_backward_step(micro_batch)
            self.base.return_train_source_log_probs = False

            if log_train_source_loss:
                start_time = time.perf_counter()
                self._accumulate_train_group_loss(
                    train_source_stats,
                    train_loss_metadata[micro_step],
                    "source_names",
                    self.base.last_train_log_probs,
                )
                train_source_aggregation_time += time.perf_counter() - start_time
            if log_train_domain_loss:
                start_time = time.perf_counter()
                self._accumulate_train_group_loss(
                    train_domain_stats,
                    train_loss_metadata[micro_step],
                    "domain_names",
                    self.base.last_train_log_probs,
                )
                train_domain_aggregation_time += time.perf_counter() - start_time

            total_loss += loss.item()
            for k, v in loss_dict.items():
                total_loss_dict[k] += v.item()

        # Gradient clipping
        grad_norm = veomni_clip_grad_norm(self.base.model, args.train.optimizer.max_grad_norm)

        # Optimizer and scheduler step
        self.base.optimizer.step()
        self.base.lr_scheduler.step()
        self.base.optimizer.zero_grad()

        self.base.step_train_source_metrics = (
            self._build_train_group_metrics(
                train_source_stats,
                train_source_aggregation_time,
                "source_loss",
                "source_loss/aggregation_time_ms",
                ordered_group_names=list(self.train_source_names),
            )
            if log_train_source_loss
            else {}
        )
        if log_train_domain_loss:
            self.base.step_train_source_metrics.update(
                self._build_train_group_metrics(
                    train_domain_stats,
                    train_domain_aggregation_time,
                    "domain_loss",
                    "domain_loss/aggregation_time_ms",
                )
            )
        self.on_step_end(loss=total_loss, loss_dict=total_loss_dict, grad_norm=grad_norm)

    def train(self):
        args: VeOmniVLMArguments = self.base.args
        self.on_train_begin()
        logger.info(
            f"Rank{args.train.local_rank} Start training. "
            f"Start step: {self.base.start_step}. "
            f"Train steps: {args.train_steps}. "
            f"Start epoch: {self.base.start_epoch}. "
            f"Train epochs: {args.train.num_train_epochs}."
        )

        for epoch in range(self.base.start_epoch, args.train.num_train_epochs):
            if hasattr(self.base.train_dataloader, "set_epoch"):
                self.base.train_dataloader.set_epoch(epoch)
            self.base.state.epoch = epoch

            self.on_epoch_begin()

            # Create a batch generator
            data_iterator = iter(self.base.train_dataloader)

            for _ in range(self.base.start_step, args.train_steps):
                try:
                    self.train_step(data_iterator)
                except StopIteration:
                    logger.info(f"epoch:{epoch} Dataloader finished with drop_last {args.data.dataloader.drop_last}")
                    break

            self.on_epoch_end()

            self.base.start_step = 0
            helper.print_device_mem_info(f"VRAM usage after epoch {epoch + 1}")

        self.on_train_end()

        synchronize()

        self.base.destroy_distributed()
