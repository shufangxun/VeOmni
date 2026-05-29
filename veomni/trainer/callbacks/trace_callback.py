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
    """Monitors MoE expert load distribution and logs heatmaps to wandb.

    Activation is gated only by ``moe_load_balance_monitor_interval > 0``; the
    monitor itself does not require wandb. Logging to wandb is gated by
    ``wandb.enable`` and ``global_rank == 0``.
    """

    def __init__(self, trainer: "BaseTrainer") -> None:
        super().__init__(trainer)
        self.monitor = None

        args: "VeOmniArguments" = self.trainer.args
        lb_config = args.train.moe_load_balance
        self.monitor_interval = lb_config.monitor_interval
        self.aux_free_update_interval = lb_config.aux_free_update_interval if lb_config.mode == "aux_free" else 0
        self.active_interval = min(
            interval for interval in (self.monitor_interval, self.aux_free_update_interval) if interval > 0
        ) if self.monitor_interval > 0 or self.aux_free_update_interval > 0 else 0

        if self.active_interval <= 0:
            logger.info_rank0("MoE router monitor disabled (moe_load_balance.monitor_interval=0).")
            return

        config = self.trainer.model_config
        num_experts = self._get_num_experts(config)
        if num_experts is None:
            logger.warning_rank0(
                "MoE load-balance monitoring requested but model config has no num_experts or n_routed_experts. "
                "MoE router monitor not activated."
            )
            return

        from ...utils.moe_monitor import MoERouterMonitor, set_active_monitor

        # Process groups are read lazily in on_train_begin once the device
        # mesh is guaranteed to be initialized.
        self.monitor = MoERouterMonitor(num_experts=num_experts)
        set_active_monitor(self.monitor)
        ps = get_parallel_state()
        logger.info_rank0(
            f"MoE router monitor created: num_experts={num_experts}, "
            f"monitor_interval={self.monitor_interval}, "
            f"aux_free_update_interval={self.aux_free_update_interval}, "
            f"mode={lb_config.mode}, "
            f"ep_size={ps.ep_size if ps.ep_enabled else 1}"
        )

    @staticmethod
    def _get_num_experts(config) -> int | None:
        for candidate in (config, getattr(config, "text_config", None)):
            if candidate is None:
                continue
            for attr in ("num_experts", "n_routed_experts"):
                value = getattr(candidate, attr, None)
                if value is not None:
                    return int(value)
        return None

    def on_train_begin(self, state: TrainerState, **kwargs) -> None:
        if self.monitor is None:
            return
        from ...utils.moe_monitor import attach_moe_router_monitor

        # fsdp_group is the dp_sp mesh dim — exactly the set of ranks that
        # hold distinct token slices. EP is intentionally not in this group;
        # see MoERouterMonitor.__init__ docstring.
        self.monitor.dp_group = get_parallel_state().fsdp_group

        attached = attach_moe_router_monitor(self.trainer.model, self.monitor)
        if attached == 0:
            logger.warning_rank0(
                "MoE router monitor: no recognized router modules found in the model. "
                "Disabling monitor. To add support for a new router class, register an "
                "extractor in veomni/utils/moe_monitor.py (see ROUTER_EXTRACTORS)."
            )
            from ...utils.moe_monitor import set_active_monitor

            set_active_monitor(None)
            self.monitor = None
        else:
            logger.info_rank0(f"MoE router monitor: attached to {attached} router module(s).")

    def on_step_end(self, state: TrainerState, **kwargs) -> None:
        args: "VeOmniArguments" = self.trainer.args
        if self.monitor is None or state.global_step % self.active_interval != 0:
            return

        lb_config = args.train.moe_load_balance
        # get_count_matrix runs an all-reduce across DP/SP groups, so every rank
        # must call it. Formatting and logging remain rank-0 only.
        count_matrix = self.monitor.get_count_matrix(current_step=state.global_step, reset=True)

        should_update = (
            lb_config.mode == "aux_free"
            and self.aux_free_update_interval > 0
            and state.global_step % self.aux_free_update_interval == 0
        )
        update_metrics = {}
        if should_update:
            update_metrics = self.monitor.update_aux_free_bias(
                count_matrix=count_matrix,
                update_rate=lb_config.aux_free_bias_update_rate,
            )

        should_log = self.monitor_interval > 0 and state.global_step % self.monitor_interval == 0
        metrics = (
            self.monitor.compute_metrics_from_counts(count_matrix=count_matrix, format_only_on=args.train.global_rank == 0)
            if should_log
            else {}
        )
        if args.train.global_rank != 0:
            return

        metrics.update(update_metrics)
        if not metrics:
            return

        if lb_config.log_to_console:
            self._log_console_summary(state.global_step, metrics)

        if args.train.wandb.enable:
            import wandb

            wandb_metrics = {}
            for k, v in metrics.items():
                if k.endswith("expert_load_heatmap"):
                    start, end = self.monitor._last_step_range
                    wandb_metrics[k] = wandb.Image(v, caption=f"Steps {start}-{end}")
                else:
                    wandb_metrics[k] = v
            wandb.log(wandb_metrics, step=state.global_step)

    def _log_console_summary(self, global_step: int, metrics: Dict[str, Any]) -> None:
        start, end = self.monitor._last_step_range
        parts = [
            f"Step {global_step}: MoE load balance (steps {start}-{end})",
            f"max_vio max={metrics.get('moe/max_vio/max', 0.0):.4f}",
            f"max_vio avg={metrics.get('moe/max_vio/avg', 0.0):.4f}",
            f"avg_vio max={metrics.get('moe/avg_vio/max', 0.0):.4f}",
            f"avg_vio avg={metrics.get('moe/avg_vio/avg', 0.0):.4f}",
            # f"min_vio max={metrics.get('moe/min_vio/max', 0.0):.4f}",
            # f"min_vio avg={metrics.get('moe/min_vio/avg', 0.0):.4f}",
        ]
        if "aux_free_bias/updated_layers" in metrics:
            parts.extend(
                [
                    f"aux_free updated_layers={metrics['aux_free_bias/updated_layers']:.0f}",
                    f"gamma={metrics['aux_free_bias/update_rate']:.2e}",
                    f"max_abs_bias={metrics['aux_free_bias/max_abs_bias']:.4f}",
                ]
            )
        logger.info_rank0(", ".join(parts) + ".")

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
        data_fetch_time = getattr(self.trainer, "step_data_fetch_time", 0.0)
        train_compute_time = delta_time
        step_wall_time = data_fetch_time + train_compute_time
        data_fetch_ratio = data_fetch_time / step_wall_time if step_wall_time else 0.0
        data_fetch_time, train_compute_time, step_wall_time, data_fetch_ratio = all_reduce(
            [data_fetch_time, train_compute_time, step_wall_time, data_fetch_ratio],
            op="max",
        )
        step_env_metrics.update(
            {
                "perf/data_fetch_time_max_s": data_fetch_time,
                "perf/train_compute_time_max_s": train_compute_time,
                "perf/step_wall_time_max_s": step_wall_time,
                "perf/data_fetch_ratio_max": data_fetch_ratio,
            }
        )

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
            if "loss" in short_name:
                return f"{short_name}: {value:.4f}"
            return f"{short_name}: {value:.2f}"

        postfix = ", ".join(_format_metric(k, v) for k, v in self.trainer.step_train_metrics.items())
        self.data_loader_tqdm.set_postfix_str(postfix)
        self.data_loader_tqdm.update()
