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
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import asdict
from functools import partial
from typing import TYPE_CHECKING, Any, Dict

import torch

from veomni.trainer.callbacks.base import TrainerState
from veomni.utils import logging
from veomni.utils.dist_utils import all_reduce
from veomni.utils.multisource_utils import parse_multisource_config

from .base import Callback


if TYPE_CHECKING:
    from ..base import BaseTrainer


logger = logging.get_logger(__name__)


class EvaluateCallback(Callback):
    def __init__(self, trainer: "BaseTrainer") -> None:
        super().__init__(trainer)
        self.eval_dataloader = None
        self.eval_dataloaders = []

    def on_train_begin(self, state: TrainerState, **kwargs):
        args = self.trainer.args
        if not self._enabled():
            return

        from veomni.data import build_dataloader, build_dataset

        eval_path = args.data.eval_path
        data_kwargs = asdict(args.data)
        data_kwargs.pop("train_path", None)
        data_kwargs.pop("eval_path", None)
        data_kwargs.pop("source_name", None)

        logger.info_rank0(f"Building validation dataset from {eval_path}.")
        if eval_path.endswith(".yaml"):
            eval_config = parse_multisource_config(eval_path)
            eval_datasets = []
            eval_preprocesses = eval_config.get("preprocess")
            eval_text_keys = eval_config.get("text_keys")
            for idx, (source, source_name) in enumerate(zip(eval_config["sources"], eval_config["names"])):
                source_transform = self.trainer.data_transform
                source_dataset_kwargs = {}
                if eval_preprocesses is not None or eval_text_keys is not None:
                    source_transform_kwargs = {"ds_idx": idx, "source_name": source_name}
                    if eval_preprocesses is not None:
                        source_transform_kwargs["preprocess"] = eval_preprocesses[idx]
                    if eval_text_keys is not None:
                        source_transform_kwargs["text_keys"] = eval_text_keys[idx]
                    source_transform = partial(self.trainer.data_transform, **source_transform_kwargs)
                else:
                    source_dataset_kwargs["source_name"] = source_name

                eval_datasets.append(
                    build_dataset(
                        dataset_name=args.data.datasets_type,
                        train_path=source,
                        transform=source_transform,
                        seed=args.train.seed,
                        shuffle=False,
                        **source_dataset_kwargs,
                        **data_kwargs,
                    )
                )
            eval_source_names = list(eval_config["names"])
        else:
            eval_datasets = [
                build_dataset(
                    dataset_name=args.data.datasets_type,
                    train_path=eval_path,
                    transform=self.trainer.data_transform,
                    seed=args.train.seed,
                    shuffle=False,
                    **data_kwargs,
                )
            ]
            eval_source_names = ["eval"]

        dataloader_kwargs = asdict(args.data.dataloader)
        dataloader_type = dataloader_kwargs.pop("type")
        dataloader_kwargs["drop_last"] = False
        if args.train.eval_dataloader_num_workers is not None:
            dataloader_kwargs["num_workers"] = args.train.eval_dataloader_num_workers
            if args.train.eval_dataloader_num_workers == 0:
                dataloader_kwargs["prefetch_factor"] = None
        self.eval_dataloaders = [
            (
                source_name,
                build_dataloader(
                    dataloader_type=dataloader_type,
                    dataset=eval_dataset,
                    micro_batch_size=args.train.eval_micro_batch_size,
                    global_batch_size=args.train.eval_global_batch_size,
                    dataloader_batch_size=args.train.eval_dataloader_batch_size,
                    max_seq_len=args.data.max_seq_len,
                    train_steps=-1,
                    bsz_warmup_ratio=0,
                    bsz_warmup_init_mbtoken=args.train.bsz_warmup_init_mbtoken,
                    dyn_bsz=args.train.dyn_bsz,
                    dyn_bsz_runtime=args.train.dyn_bsz_runtime,
                    dyn_bsz_buffer_size=args.data.dyn_bsz_buffer_size,
                    seed=args.train.seed,
                    collate_fn=self.trainer.collate_fn,
                    shuffle=False,
                    # Validation must stop when the fixed validation set is exhausted.
                    # Training keeps the default repeat_on_exhaustion=True.
                    repeat_on_exhaustion=False,
                    **dataloader_kwargs,
                ),
            )
            for source_name, eval_dataset in zip(eval_source_names, eval_datasets)
        ]
        self.eval_dataloader = self.eval_dataloaders[0][1] if self.eval_dataloaders else None

    def on_epoch_end(self, state: TrainerState, **kwargs):
        args = self.trainer.args
        if args.train.eval_epochs and (state.epoch + 1) % args.train.eval_epochs == 0:
            self._evaluate(state)

    def on_step_end(self, state: TrainerState, **kwargs) -> None:
        args = self.trainer.args
        if args.train.eval_steps and state.global_step % args.train.eval_steps == 0:
            self._evaluate(state)

    def _enabled(self) -> bool:
        args = self.trainer.args
        return bool(args.data.eval_path)

    def _prepare_micro_batch(self, micro_batch: Dict[str, Any]) -> Dict[str, Any]:
        # Multi-source metadata is for logging only and must not be passed to the model.
        micro_batch.pop("padding_flag", None)
        micro_batch.pop("ds_idx", None)
        micro_batch.pop("source_name", None)
        micro_batch.pop("domain_name", None)
        micro_batch.pop("domain", None)
        micro_batch.pop("cur_token_num", None)
        return self.trainer.preforward(micro_batch)

    @staticmethod
    def _as_list(value: Any) -> list[Any]:
        if value is None:
            return []
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().view(-1).tolist()
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]

    def _snapshot_domain_metadata(self, micro_batch: Dict[str, Any]) -> Dict[str, Any]:
        from veomni.distributed.parallel_state import get_parallel_state

        if get_parallel_state().sp_enabled:
            return {"domain_names": [], "segment_starts": []}

        domain_names = [str(name) for name in self._as_list(micro_batch.get("domain_name"))]
        position_ids = micro_batch.get("position_ids")
        if not domain_names or position_ids is None:
            return {"domain_names": [], "segment_starts": []}

        position_ids = position_ids.detach().cpu()
        if position_ids.dim() == 3:
            position_ids = position_ids[:, 0, :]
        position_ids = position_ids.reshape(-1)
        segment_starts = (position_ids == 0).nonzero(as_tuple=False).view(-1).tolist()
        if not segment_starts or segment_starts[0] != 0:
            segment_starts = [0, *segment_starts]

        if len(domain_names) != len(segment_starts):
            logger.warning_once(
                "Skip eval domain loss for a micro batch because domain metadata does not match packed "
                f"segments: num_domains={len(domain_names)}, num_segments={len(segment_starts)}."
            )
            return {"domain_names": [], "segment_starts": []}

        return {"domain_names": domain_names, "segment_starts": segment_starts}

    def _accumulate_eval_domain_loss(
        self,
        domain_stats: Dict[str, Dict[str, float]],
        domain_metadata: Dict[str, Any],
        log_probs: torch.Tensor,
    ) -> None:
        domain_names = domain_metadata["domain_names"]
        segment_starts = domain_metadata["segment_starts"]
        if not domain_names or log_probs is None:
            return

        log_probs = log_probs.detach().float().reshape(-1)
        segment_ends = [*segment_starts[1:], log_probs.numel()]
        for domain_name, start, end in zip(domain_names, segment_starts, segment_ends):
            end = min(end, log_probs.numel())
            if end <= start:
                continue

            domain_log_probs = log_probs[start:end]
            valid_mask = domain_log_probs != 0
            token_count = int(valid_mask.sum().item())
            if token_count == 0:
                continue

            item = domain_stats[domain_name]
            item["loss_sum"] += float((-domain_log_probs[valid_mask]).sum().item())
            item["tokens"] += token_count

    def _forward_eval_micro_batch(
        self, micro_batch: Dict[str, Any]
    ) -> tuple[float, Dict[str, float], Dict[str, Dict[str, float]]]:
        from veomni.ops.batch_invariant_ops import set_batch_invariant_mode
        from veomni.utils.loss_utils import count_loss_token

        self.trainer.micro_batch_token_len = count_loss_token(micro_batch)
        domain_metadata = self._snapshot_domain_metadata(micro_batch)
        micro_batch = self._prepare_micro_batch(micro_batch)
        model_kwargs = {}
        if domain_metadata["domain_names"]:
            model_kwargs["return_log_probs_with_loss"] = True

        with (
            self.trainer.model_fwd_context,
            set_batch_invariant_mode(self.trainer.args.train.enable_batch_invariant_mode),
        ):
            outputs = self.trainer.model(**micro_batch, use_cache=False, **model_kwargs)

        loss, loss_dict = self.trainer.postforward(outputs, micro_batch)
        domain_stats: Dict[str, Dict[str, float]] = defaultdict(lambda: {"loss_sum": 0.0, "tokens": 0.0})
        self._accumulate_eval_domain_loss(domain_stats, domain_metadata, getattr(outputs, "log_probs", None))
        return loss.item(), {k: v.item() for k, v in loss_dict.items()}, domain_stats

    def _evaluate_dataloader(self, dataloader) -> Dict[str, Any]:
        from veomni.distributed.parallel_state import get_parallel_state
        from veomni.utils.loss_utils import count_loss_token

        dp_size = get_parallel_state().dp_size
        loss_numer = 0.0
        loss_dict_numer: Dict[str, float] = defaultdict(float)
        total_tokens = 0.0
        dropped_tokens = 0.0
        dropped_rank_batches = 0.0
        eval_steps = 0
        domain_stats: Dict[str, Dict[str, float]] = defaultdict(lambda: {"loss_sum": 0.0, "tokens": 0.0})

        data_iterator = iter(dataloader)

        while True:
            if self.trainer.args.train.eval_max_steps > 0 and eval_steps >= self.trainer.args.train.eval_max_steps:
                break

            try:
                micro_batches = self._normalize_micro_batches(next(data_iterator))
                has_batch = int(bool(micro_batches))
            except StopIteration:
                micro_batches = None
                has_batch = 0

            active_ranks = all_reduce(has_batch, op="sum")
            if active_ranks != dp_size:
                if active_ranks:
                    local_dropped_tokens = (
                        count_loss_token(micro_batches)["foundation_tokens"].item() if has_batch else 0
                    )
                    dropped_tokens += all_reduce(local_dropped_tokens, op="sum")
                    dropped_rank_batches += active_ranks
                break

            self.trainer.micro_batches_token_len = count_loss_token(micro_batches)
            step_tokens = all_reduce(
                self.trainer.micro_batches_token_len["foundation_tokens"].item(),
                op="sum",
            )
            if step_tokens == 0:
                continue

            step_loss = 0.0
            step_loss_dict: Dict[str, float] = defaultdict(float)

            for micro_batch in micro_batches:
                loss, loss_dict, micro_batch_domain_stats = self._forward_eval_micro_batch(micro_batch)
                step_loss += loss
                for key, value in loss_dict.items():
                    step_loss_dict[key] += value
                for domain_name, item in micro_batch_domain_stats.items():
                    domain_stats[domain_name]["loss_sum"] += item["loss_sum"]
                    domain_stats[domain_name]["tokens"] += item["tokens"]

            step_loss = all_reduce(step_loss, group=get_parallel_state().fsdp_group)
            step_loss_dict = {
                key: all_reduce(value, group=get_parallel_state().fsdp_group) for key, value in step_loss_dict.items()
            }

            loss_numer += step_loss * step_tokens
            for key, value in step_loss_dict.items():
                loss_dict_numer[key] += value * step_tokens
            total_tokens += step_tokens
            eval_steps += 1

        return {
            "loss_numer": loss_numer,
            "loss_dict_numer": loss_dict_numer,
            "total_tokens": total_tokens,
            "dropped_tokens": dropped_tokens,
            "dropped_rank_batches": dropped_rank_batches,
            "eval_steps": eval_steps,
            "domain_stats": domain_stats,
        }

    def _build_metrics(self, stats: Dict[str, Any], prefix: str) -> Dict[str, float]:
        total_tokens = stats["total_tokens"]
        metrics = {f"{prefix}/total_loss": stats["loss_numer"] / total_tokens}
        for key, value in stats["loss_dict_numer"].items():
            metrics[f"{prefix}/{key}"] = value / total_tokens

        ppl_loss = metrics.get(f"{prefix}/foundation_loss", metrics[f"{prefix}/total_loss"])
        metrics[f"{prefix}/ppl"] = math.exp(ppl_loss) if ppl_loss < 100 else float("inf")
        metrics[f"{prefix}/tokens"] = total_tokens
        metrics[f"{prefix}/dropped_tokens"] = stats["dropped_tokens"]
        metrics[f"{prefix}/dropped_rank_batches"] = stats["dropped_rank_batches"]
        metrics[f"{prefix}/steps"] = stats["eval_steps"]
        return metrics

    def _build_domain_metrics(self, domain_stats: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        from torch import distributed as dist

        from veomni.distributed.parallel_state import get_parallel_state

        metrics = {}
        parallel_state = get_parallel_state()
        domain_names = sorted(domain_stats.keys())
        if dist.is_initialized():
            gathered_domain_names = [None for _ in range(dist.get_world_size(group=parallel_state.dp_group))]
            dist.all_gather_object(gathered_domain_names, domain_names, group=parallel_state.dp_group)
            domain_names = sorted({name for names in gathered_domain_names for name in names})

        for domain_name in domain_names:
            local_stats = domain_stats[domain_name]
            loss_sum = float(local_stats.get("loss_sum", 0.0))
            tokens = float(local_stats.get("tokens", 0.0))
            global_loss_sum, global_tokens = all_reduce((loss_sum, tokens), op="sum", group=parallel_state.dp_group)
            if global_tokens == 0:
                continue

            loss = global_loss_sum / global_tokens
            metric_name = self._metric_source_name(domain_name)
            metrics[f"eval/domain/{metric_name}/loss"] = loss
            metrics[f"eval/domain/{metric_name}/ppl"] = math.exp(loss) if loss < 100 else float("inf")
            metrics[f"eval/domain/{metric_name}/tokens"] = global_tokens
        return metrics

    @staticmethod
    def _metric_source_name(source_name: str) -> str:
        return source_name.replace("/", "_")

    @staticmethod
    def _normalize_micro_batches(micro_batches: Any) -> list[Dict[str, Any]]:
        if isinstance(micro_batches, Mapping):
            micro_batches = [micro_batches]

        normalized = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, Mapping):
                micro_batch = dict(micro_batch)
            if micro_batch.get("padding_flag", False):
                continue
            normalized.append(micro_batch)
        return normalized

    def _evaluate(self, state: TrainerState):
        if not self.eval_dataloaders:
            return

        from veomni.utils.device import synchronize

        was_training = self.trainer.model.training
        self.trainer.model.eval()
        synchronize()

        metrics = {}
        total_stats = {
            "loss_numer": 0.0,
            "loss_dict_numer": defaultdict(float),
            "total_tokens": 0.0,
            "dropped_tokens": 0.0,
            "dropped_rank_batches": 0.0,
            "eval_steps": 0,
        }
        total_domain_stats: Dict[str, Dict[str, float]] = defaultdict(lambda: {"loss_sum": 0.0, "tokens": 0.0})

        try:
            with torch.no_grad():
                for source_name, dataloader in self.eval_dataloaders:
                    source_stats = self._evaluate_dataloader(dataloader)
                    if source_stats["eval_steps"] == 0 or source_stats["total_tokens"] == 0:
                        logger.warning_rank0(f"Validation dataloader for source {source_name} produced no batches.")
                        continue

                    metrics.update(self._build_metrics(source_stats, f"eval/{self._metric_source_name(source_name)}"))

                    total_stats["loss_numer"] += source_stats["loss_numer"]
                    for key, value in source_stats["loss_dict_numer"].items():
                        total_stats["loss_dict_numer"][key] += value
                    total_stats["total_tokens"] += source_stats["total_tokens"]
                    total_stats["dropped_tokens"] += source_stats["dropped_tokens"]
                    total_stats["dropped_rank_batches"] += source_stats["dropped_rank_batches"]
                    total_stats["eval_steps"] += source_stats["eval_steps"]
                    for domain_name, item in source_stats["domain_stats"].items():
                        total_domain_stats[domain_name]["loss_sum"] += item["loss_sum"]
                        total_domain_stats[domain_name]["tokens"] += item["tokens"]
        finally:
            if was_training:
                self.trainer.model.train()
            synchronize()

        if total_stats["eval_steps"] == 0 or total_stats["total_tokens"] == 0:
            logger.warning_rank0("Validation dataloaders produced no batches; skip eval metrics.")
            return

        metrics = {**self._build_metrics(total_stats, "eval"), **metrics}
        metrics.update(self._build_domain_metrics(total_domain_stats))

        logger.info_rank0(
            "Validation metrics at step "
            f"{state.global_step}: " + ", ".join(f"{key}={value:.6g}" for key, value in metrics.items())
        )

        args = self.trainer.args
        if args.train.global_rank == 0 and args.train.wandb.enable:
            import wandb

            wandb.log(metrics, step=state.global_step)
