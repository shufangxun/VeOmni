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

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import yaml
from transformers import AutoTokenizer

from veomni.data import build_chat_template
from veomni.utils.constants import IGNORE_INDEX
from veomni.utils.multisource_utils import parse_multisource_config


def _parquet_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted(item for item in path.iterdir() if item.is_file() and item.suffix == ".parquet")
        if files:
            return files
    raise FileNotFoundError(f"No parquet files found under {path}")


def _data_paths(path: Path) -> list[Path]:
    if path.suffix in {".yaml", ".yml"}:
        return [Path(source) for source in parse_multisource_config(str(path))["sources"]]
    return [path]


def _all_parquet_files(paths: list[Path]) -> list[Path]:
    return [parquet_file for path in paths for parquet_file in _parquet_files(path)]


def _nested_get(config: dict[str, Any], keys: tuple[str, ...], default: Any = None) -> Any:
    value: Any = config
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def _load_config(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _resolve_arg(value: Any, config: dict[str, Any], keys: tuple[str, ...], default: Any = None) -> Any:
    return value if value is not None else _nested_get(config, keys, default)


def _new_stats() -> dict[str, float]:
    return {
        "samples": 0,
        "tokens": 0,
        "loss_tokens": 0,
        "max_tokens": 0,
    }


def _update_stats(stats: dict[str, float], tokens: int, loss_tokens: int) -> None:
    stats["samples"] += 1
    stats["tokens"] += tokens
    stats["loss_tokens"] += loss_tokens
    stats["max_tokens"] = max(stats["max_tokens"], tokens)


def _finalize_stats(stats: dict[str, float]) -> dict[str, float]:
    samples = stats["samples"]
    tokens = stats["tokens"]
    loss_tokens = stats["loss_tokens"]
    return {
        **stats,
        "avg_tokens": tokens / samples if samples else 0,
        "avg_loss_tokens": loss_tokens / samples if samples else 0,
    }


def _parse_weights(raw: str | None) -> dict[str, float]:
    if not raw:
        return {}
    if raw.strip().startswith("{"):
        return {str(k): float(v) for k, v in json.loads(raw).items()}

    weights = {}
    for item in raw.split(","):
        key, value = item.split("=", maxsplit=1)
        weights[key] = float(value)
    return weights


def _sampling_estimate(
    group_stats: dict[str, dict[str, float]],
    weights: dict[str, float],
    target_tokens: int,
) -> dict[str, dict[str, float]]:
    if not weights:
        return {}

    weight_sum = sum(weights.values())
    if weight_sum <= 0:
        raise ValueError("Sampling weights must sum to a positive value.")

    estimate = {}
    for group_name, weight in sorted(weights.items()):
        stats = group_stats.get(group_name, _new_stats())
        target_group_tokens = target_tokens * weight / weight_sum
        available_tokens = stats["tokens"]
        estimate[group_name] = {
            "weight": weight / weight_sum,
            "available_tokens": available_tokens,
            "target_tokens": target_group_tokens,
            "repeat_factor": target_group_tokens / available_tokens if available_tokens else math.inf,
        }
    return estimate


def _scaled_group_stats(
    group_stats: dict[str, dict[str, float]],
    scale: float,
) -> dict[str, dict[str, float]]:
    if scale == 1:
        return group_stats
    scaled_stats = {}
    for group_name, stats in group_stats.items():
        scaled = dict(stats)
        scaled["samples"] *= scale
        scaled["tokens"] *= scale
        scaled["loss_tokens"] *= scale
        scaled_stats[group_name] = scaled
    return scaled_stats


def _print_stats(name: str, stats_by_group: dict[str, dict[str, float]], limit: int) -> None:
    print(f"\n[{name}]")
    rows = sorted(stats_by_group.items(), key=lambda item: item[1]["tokens"], reverse=True)
    for group_name, stats in rows[:limit]:
        finalized = _finalize_stats(stats)
        print(
            f"{group_name}\t"
            f"samples={int(finalized['samples'])}\t"
            f"tokens={int(finalized['tokens'])}\t"
            f"loss_tokens={int(finalized['loss_tokens'])}\t"
            f"avg_tokens={finalized['avg_tokens']:.2f}\t"
            f"max_tokens={int(finalized['max_tokens'])}"
        )
    if len(rows) > limit:
        print(f"... {len(rows) - limit} more groups")


def _print_sampling_estimate(name: str, estimate: dict[str, dict[str, float]]) -> None:
    if not estimate:
        return

    print(f"\n[{name} sampling estimate]")
    for group_name, item in estimate.items():
        print(
            f"{group_name}\t"
            f"weight={item['weight']:.6f}\t"
            f"available_tokens={int(item['available_tokens'])}\t"
            f"target_tokens={int(item['target_tokens'])}\t"
            f"repeat_factor={item['repeat_factor']:.4f}"
        )


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    config = _load_config(args.config)
    data_path = Path(_resolve_arg(args.data_path, config, ("data", "train_path")))
    tokenizer_path = _resolve_arg(
        args.tokenizer_path,
        config,
        ("model", "tokenizer_path"),
        _nested_get(config, ("model", "model_path")),
    )
    chat_template_name = _resolve_arg(args.chat_template, config, ("data", "chat_template"), "chatml")
    text_key = _resolve_arg(args.text_key, config, ("data", "text_keys"), "messages")
    max_seq_len = int(_resolve_arg(args.max_seq_len, config, ("data", "max_seq_len"), 4096))
    global_batch_size = int(_resolve_arg(args.global_batch_size, config, ("train", "global_batch_size"), 1))
    bsz_warmup_ratio = float(_resolve_arg(args.bsz_warmup_ratio, config, ("train", "bsz_warmup_ratio"), 0.0))

    if tokenizer_path is None:
        raise ValueError(
            "tokenizer path is required. Pass --tokenizer-path or use a config with model.tokenizer_path/model_path."
        )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    chat_template = build_chat_template(chat_template_name, tokenizer)

    data_paths = _data_paths(data_path)
    parquet_files = _all_parquet_files(data_paths)
    total_rows = sum(pq.ParquetFile(path).metadata.num_rows for path in parquet_files)
    overall = _new_stats()
    group_stats = {key: defaultdict(_new_stats) for key in args.group_keys}
    processed = 0

    for parquet_file in parquet_files:
        columns = set(pq.ParquetFile(parquet_file).schema_arrow.names)
        read_columns = sorted({text_key, *args.group_keys} & columns)
        if text_key not in read_columns:
            raise KeyError(f"{text_key!r} not found in {parquet_file}. Available columns: {sorted(columns)}")

        for batch in pq.ParquetFile(parquet_file).iter_batches(batch_size=args.batch_size, columns=read_columns):
            data = batch.to_pydict()
            messages_list = data[text_key]
            for row_idx, messages in enumerate(messages_list):
                encoded = chat_template.encode_messages(messages, max_seq_len=max_seq_len)
                tokens = len(encoded["attention_mask"])
                loss_tokens = sum(1 for label in encoded["labels"] if label != IGNORE_INDEX)
                _update_stats(overall, tokens, loss_tokens)

                for group_key in args.group_keys:
                    values = data.get(group_key)
                    group_name = (
                        str(values[row_idx]) if values is not None and values[row_idx] is not None else "<missing>"
                    )
                    _update_stats(group_stats[group_key][group_name], tokens, loss_tokens)

                processed += 1
                if args.max_samples and processed >= args.max_samples:
                    break
            if args.max_samples and processed >= args.max_samples:
                break
        if args.max_samples and processed >= args.max_samples:
            break

    finalized_overall = _finalize_stats(overall)
    scanned_train_size = int(finalized_overall["tokens"])
    train_size = scanned_train_size
    scale = total_rows / processed if args.max_samples and processed else 1
    if args.max_samples:
        train_size = int(scanned_train_size * scale)
    train_steps = math.ceil(train_size * (1 + bsz_warmup_ratio / 2) / (global_batch_size * max_seq_len))

    weight_group_stats = _scaled_group_stats(group_stats.get(args.weight_group_key, {}), scale)
    weights = _parse_weights(args.weights)
    if args.equal_weights:
        weights = dict.fromkeys(weight_group_stats, 1.0)
    target_train_size = args.target_train_size or train_size
    sampling_estimate = _sampling_estimate(weight_group_stats, weights, target_train_size)

    result = {
        "data_path": str(data_path),
        "resolved_data_paths": [str(path) for path in data_paths],
        "tokenizer_path": tokenizer_path,
        "chat_template": chat_template_name,
        "max_seq_len": max_seq_len,
        "processed_samples": processed,
        "total_rows": total_rows,
        "overall": finalized_overall,
        "scanned_train_size": scanned_train_size,
        "suggested_train_size": train_size,
        "estimated_train_steps": train_steps,
        "groups": {
            key: {name: _finalize_stats(stats) for name, stats in values.items()}
            for key, values in group_stats.items()
        },
        "sampling_estimate": sampling_estimate,
    }

    print("[overall]")
    print(
        f"samples={int(finalized_overall['samples'])}\t"
        f"tokens={int(finalized_overall['tokens'])}\t"
        f"loss_tokens={int(finalized_overall['loss_tokens'])}\t"
        f"avg_tokens={finalized_overall['avg_tokens']:.2f}\t"
        f"max_tokens={int(finalized_overall['max_tokens'])}"
    )
    print(f"suggested data.train_size={train_size}")
    if args.max_samples:
        print(f"estimated from {processed}/{total_rows} rows; scanned tokens={scanned_train_size}")
    print(f"estimated train_steps={train_steps}")

    for group_key, stats in group_stats.items():
        _print_stats(group_key, stats, args.top_k)
    _print_sampling_estimate(args.weight_group_key, sampling_estimate)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize SFT parquet token counts and source/domain distribution.")
    parser.add_argument("--config", help="Optional VeOmni YAML config to fill defaults.")
    parser.add_argument("--data-path", help="Parquet file or directory. Defaults to data.train_path from --config.")
    parser.add_argument("--tokenizer-path", help="Tokenizer path. Defaults to model.tokenizer_path/model.model_path.")
    parser.add_argument("--chat-template", help="Chat template name. Defaults to data.chat_template or chatml.")
    parser.add_argument("--text-key", help="Conversation field. Defaults to data.text_keys or messages.")
    parser.add_argument("--max-seq-len", type=int, help="Truncation length. Defaults to data.max_seq_len or 4096.")
    parser.add_argument("--global-batch-size", type=int, help="Used to estimate train_steps.")
    parser.add_argument("--bsz-warmup-ratio", type=float, help="Used to estimate train_steps.")
    parser.add_argument("--batch-size", type=int, default=512, help="Rows to read per parquet batch.")
    parser.add_argument(
        "--max-samples", type=int, default=0, help="Limit rows for a quick estimate. 0 means full scan."
    )
    parser.add_argument("--group-keys", nargs="+", default=["source", "domain"], help="Columns to aggregate.")
    parser.add_argument("--top-k", type=int, default=50, help="Maximum groups to print per group key.")
    parser.add_argument("--weight-group-key", default="source", help="Group key used for sampling estimates.")
    parser.add_argument("--weights", help="Sampling weights as JSON or comma-separated key=value pairs.")
    parser.add_argument("--equal-weights", action="store_true", help="Estimate repeat factors for equal weights.")
    parser.add_argument("--target-train-size", type=int, default=0, help="Target tokens for sampling estimate.")
    parser.add_argument("--output-json", help="Optional path to write machine-readable stats.")
    args = parser.parse_args()

    summarize(args)


if __name__ == "__main__":
    main()
