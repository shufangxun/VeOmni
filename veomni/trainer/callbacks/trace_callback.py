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
from typing import TYPE_CHECKING, Any, Dict, List

from tqdm import trange

from ...distributed.parallel_state import get_parallel_state
from ...utils import helper
from ...utils.dist_utils import all_reduce
from ...utils.logging import get_logger
from .base import Callback, TrainerState


logger = get_logger(__name__)


if TYPE_CHECKING:
    from ..base import BaseTrainer, VeOmniArguments


class MoERouterMonitorCallback(Callback):
    def __init__(self, trainer: "BaseTrainer") -> None:
        super().__init__(trainer)
        self.monitor = None
        self._maxvio_batch_sum = 0.0
        self._maxvio_batch_count = 0

        args: "VeOmniArguments" = self.trainer.args
        if not args.train.wandb.enable:
            logger.info_rank0("MoE router monitor disabled (wandb not enabled).")
            return
        if args.train.moe_load_balance_monitor_interval <= 0:
            logger.info_rank0("MoE router monitor disabled (moe_load_balance_monitor_interval=0).")
            return

        num_experts = self._get_num_experts(self.trainer.model_config)
        if num_experts is not None:
            from ...utils.moe_monitor import MoERouterMonitor, ensure_router_monitor_hooks, set_active_monitor

            num_hooks = ensure_router_monitor_hooks(self.trainer.model)
            self.monitor = MoERouterMonitor(num_experts)
            set_active_monitor(self.monitor)
            logger.info_rank0(
                f"MoE router monitor enabled: num_experts={num_experts}, "
                f"interval={args.train.moe_load_balance_monitor_interval}, router_hooks={num_hooks}"
            )
        else:
            logger.warning_rank0(
                "moe_load_balance_monitor_interval > 0 but model config has no 'num_experts' or "
                "'text_config.num_experts'. MoE router monitor not activated."
            )

    @staticmethod
    def _get_num_experts(config: Any) -> int | None:
        for candidate in (config, getattr(config, "text_config", None)):
            if candidate is None:
                continue
            num_experts = getattr(candidate, "num_experts", None)
            if num_experts is not None:
                return int(num_experts)
        return None

    @staticmethod
    def _build_wandb_metrics(maxvio_batch_sum: float, maxvio_batch_count: int) -> dict[str, float]:
        if maxvio_batch_count <= 0:
            return {}
        return {"moe/maxvio_batch": maxvio_batch_sum / maxvio_batch_count}

    def on_step_end(self, state: TrainerState, **kwargs) -> None:
        args: "VeOmniArguments" = self.trainer.args
        if not self.monitor:
            return

        load_matrix = self.monitor.get_load_matrix(current_step=state.global_step)
        num_layers = load_matrix.shape[0]
        if num_layers == 0:
            if args.train.global_rank == 0 and state.global_step % args.train.moe_load_balance_monitor_interval == 0:
                logger.warning_rank0(
                    f"Step {state.global_step}: MoE router monitor has no recorded data. "
                    "Check that router forward hooks are registered (e.g. PatchQwen3MoeTopKRouter)."
                )
            return

        from ...utils.moe_monitor import MoERouterMonitor

        maxvio_batch = MoERouterMonitor.compute_maxvio_batch(load_matrix).item()
        if not math.isnan(maxvio_batch):
            self._maxvio_batch_sum += maxvio_batch
            self._maxvio_batch_count += 1

        if state.global_step % args.train.moe_load_balance_monitor_interval != 0:
            return

        metrics = self._build_wandb_metrics(self._maxvio_batch_sum, self._maxvio_batch_count)
        self._maxvio_batch_sum = 0.0
        self._maxvio_batch_count = 0

        if args.train.global_rank == 0 and metrics:
            import wandb

            wandb.log(metrics, step=state.global_step)
            logger.info_rank0(
                f"Step {state.global_step}: logged MoE MaxVio_batch={metrics['moe/maxvio_batch']:.4f} "
                f"averaged across {num_layers} layers and the last "
                f"{args.train.moe_load_balance_monitor_interval} step(s)."
            )

    def on_train_end(self, state: TrainerState, **kwargs) -> None:
        if self.monitor is not None:
            from ...utils.moe_monitor import set_active_monitor

            set_active_monitor(None)
            self.monitor = None
            logger.info_rank0("MoE router monitor disabled.")


class WandbTraceCallback(Callback):
    def on_train_begin(self, state: TrainerState, **kwargs) -> None:
        args: "VeOmniArguments" = self.trainer.args
        if args.train.global_rank == 0 and args.train.wandb.enable:
            from dataclasses import asdict

            import wandb

            wandb.init(
                project=args.train.wandb.project,
                name=args.train.wandb.name,
                id=args.train.wandb.id,
                resume="allow" if args.train.wandb.id else None,
                config={**asdict(args.model), **asdict(args.data), **asdict(args.train)},
            )

    def on_step_end(self, state: TrainerState, **kwargs) -> None:
        args: "VeOmniArguments" = self.trainer.args

        if args.train.global_rank == 0 and args.train.wandb.enable:
            import wandb

            wandb.log(self.trainer.step_env_metrics, step=state.global_step)


class ProfileTraceCallback(Callback):
    def on_train_begin(self, state: TrainerState, **kwargs) -> None:
        args: "VeOmniArguments" = self.trainer.args
        if args.train.profile.this_rank:
            self.profiler = helper.create_profiler(
                start_step=args.train.profile.start_step,
                end_step=args.train.profile.end_step,
                trace_dir=args.train.profile.trace_dir,
                record_shapes=args.train.profile.record_shapes,
                profile_memory=args.train.profile.profile_memory,
                with_stack=args.train.profile.with_stack,
                with_modules=args.train.profile.with_modules,
                global_rank=args.train.global_rank,
            )
            self.profiler.start()

    def on_step_end(self, state: TrainerState, **kwargs) -> None:
        args: "VeOmniArguments" = self.trainer.args
        if args.train.profile.this_rank:
            if state.global_step <= args.train.profile.end_step:
                self.profiler.step()

            if state.global_step == args.train.profile.end_step:
                self.profiler.stop()


class EnvironMeterCallback(Callback):
    def __init__(self, trainer: "BaseTrainer") -> None:
        super().__init__(trainer)

        args: "VeOmniArguments" = self.trainer.args
        self.trainer.environ_meter = helper.EnvironMeter(
            config=trainer.model_config,
            global_batch_size=args.train.global_batch_size,
            empty_cache_steps=args.train.empty_cache_steps,
            enable_multisource=args.data.enable_multisource,
            dataloader=trainer.train_dataloader,
            data_path=args.data.train_path,
            gc_steps=args.train.gc_steps,
        )

    def on_step_begin(self, state: TrainerState, micro_batches: List[Dict[str, Any]] = None, **kwargs) -> None:
        for micro_batch in micro_batches:
            self.trainer.environ_meter.add(micro_batch)
        self.start_time = time.time()

    def on_step_end(
        self, state: TrainerState, loss: float, loss_dict: Dict[str, float], grad_norm: float, **kwargs
    ) -> None:
        delta_time = time.time() - self.start_time
        step_env_metrics = self.trainer.environ_meter.step(delta_time, global_step=state.global_step)

        step_train_metrics = {
            "total_loss": loss,
        }
        step_train_metrics.update(loss_dict)
        step_train_metrics["grad_norm"] = grad_norm

        # gather training_step_info from all ranks
        step_train_metrics = {
            f"training/{k}": all_reduce(v, group=get_parallel_state().fsdp_group)
            for k, v in step_train_metrics.items()
        }

        if self.trainer.lr_scheduler is not None:
            lr = max(self.trainer.lr_scheduler.get_last_lr())
            step_train_metrics["training/lr"] = lr

        step_env_metrics.update(step_train_metrics)
        step_env_metrics.update(getattr(self.trainer, "step_train_source_metrics", {}))

        self.trainer.step_train_metrics = step_train_metrics
        self.trainer.step_env_metrics = step_env_metrics


class TqdmCallback(Callback):
    def on_epoch_begin(self, state: TrainerState, **kwargs) -> None:
        args: "VeOmniArguments" = self.trainer.args
        self.data_loader_tqdm = trange(
            args.train_steps,
            desc=f"Epoch {state.epoch + 1}/{args.train.num_train_epochs}",
            total=args.train_steps,
            initial=self.trainer.start_step,
            disable=args.train.local_rank != 0,
        )

    def on_epoch_end(self, state: TrainerState, **kwargs) -> None:
        self.data_loader_tqdm.close()

    def on_step_end(self, state: TrainerState, **kwargs) -> None:
        def _format_metric(name: str, value: float) -> str:
            short_name = name.split("/", 1)[-1]
            if short_name == "lr" or short_name.endswith("_weighted"):
                return f"{short_name}: {value:.4g}"
            return f"{short_name}: {value:.2f}"

        postfix = ", ".join(_format_metric(k, v) for k, v in self.trainer.step_train_metrics.items())
        self.data_loader_tqdm.set_postfix_str(postfix)
        self.data_loader_tqdm.update()
