#!/usr/bin/env python3
# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Convert OpenAI-style text messages to ShareGPT-style conversations."""

import argparse
from pathlib import Path
from typing import Any

from datasets import load_dataset


ROLE_TO_FROM = {
    "user": "human",
    "assistant": "gpt",
}


def convert_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    conversations = []
    for message in messages:
        role = message.get("role")
        if role == "system":
            continue
        if role not in ROLE_TO_FROM:
            raise ValueError(f"Unsupported OpenAI message role: {role}")
        content = message.get("content", "")
        if not isinstance(content, str):
            raise ValueError(f"Expected string message content, got {type(content)}")
        conversations.append({"from": ROLE_TO_FROM[role], "value": content})
    return conversations


def convert_split(input_path: Path, output_path: Path, num_proc: int | None) -> None:
    data_files = str(input_path / "*.parquet") if input_path.is_dir() else str(input_path)
    dataset = load_dataset("parquet", data_files=data_files, split="train")

    def convert_example(example: dict[str, Any]) -> dict[str, Any]:
        if "messages" not in example:
            raise ValueError("Input example is missing the messages field.")
        output = {key: value for key, value in example.items() if key != "messages"}
        output["conversations"] = convert_messages(example["messages"])
        return output

    dataset = dataset.map(convert_example, remove_columns=["messages"], num_proc=num_proc)
    output_path.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(str(output_path / "data-00000-of-00001.parquet"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-train", type=Path, required=True)
    parser.add_argument("--input-eval", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-proc", type=int, default=None)
    args = parser.parse_args()

    convert_split(args.input_train, args.output_dir / "train", args.num_proc)
    convert_split(args.input_eval, args.output_dir / "eval", args.num_proc)


if __name__ == "__main__":
    main()
