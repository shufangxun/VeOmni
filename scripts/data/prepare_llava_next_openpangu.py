#!/usr/bin/env python3
# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Download and prepare LLaVA-NeXT data for OpenPangu-VL training."""

import argparse
import json
import os
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download


REQUIRED_COLUMNS = {"id", "conversations", "image"}


def _parquet_files(path: Path) -> list[Path]:
    files = sorted(path.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {path}")
    return files


def _validate_schema(parquet_files: list[Path]) -> None:
    bad_files = []
    for parquet_file in parquet_files:
        schema = pq.read_schema(parquet_file)
        missing = sorted(REQUIRED_COLUMNS.difference(schema.names))
        if missing:
            bad_files.append((parquet_file, missing))

    if bad_files:
        details = "; ".join(f"{path}: missing {missing}" for path, missing in bad_files[:10])
        raise ValueError(f"LLaVA-NeXT parquet schema is not OpenPangu-VL compatible. {details}")


def _refresh_symlink(link: Path, target: Path, overwrite: bool) -> None:
    if link.exists() or link.is_symlink():
        if link.is_symlink() and Path(os.readlink(link)) == target:
            return
        if not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing path: {link}")
        link.unlink()
    link.symlink_to(target)


def _write_manifest(manifest_path: Path, output_dir: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        "\n".join(
            [
                "sources:",
                f"- {output_dir}",
                "",
                "names:",
                "- llava_next_qa",
                "",
                "preprocess:",
                "- conversation",
                "",
                "text_keys:",
                "- conversations",
                "",
                "image_keys:",
                "- image",
                "",
                "domains:",
                "- multimodal_qa",
                "",
                "schedule:",
                "- schedule_type: const",
                "  weights: [1.0]",
                "",
                "level: token",
                "stopping_strategy: never_exhausted",
                "upstream_sharded: true",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_metadata(output_dir: Path, repo_id: str, raw_dir: Path, parquet_files: list[Path], manifest_path: Path) -> None:
    metadata = {
        "repo_id": repo_id,
        "raw_dir": str(raw_dir),
        "output_dir": str(output_dir),
        "manifest_path": str(manifest_path),
        "num_shards": len(parquet_files),
        "required_columns": sorted(REQUIRED_COLUMNS),
    }
    (output_dir / "openpangu_vl_dataset_info.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def prepare_dataset(
    repo_id: str,
    raw_dir: Path,
    output_dir: Path,
    manifest_path: Path,
    max_workers: int,
    skip_download: bool,
    local_files_only: bool,
    overwrite_links: bool,
) -> None:
    if not skip_download:
        snapshot_download(
            repo_id,
            repo_type="dataset",
            local_dir=raw_dir,
            allow_patterns=["README.md", "data/*.parquet"],
            max_workers=max_workers,
            local_files_only=local_files_only,
        )

    raw_data_dir = raw_dir / "data"
    parquet_files = _parquet_files(raw_data_dir)
    _validate_schema(parquet_files)

    output_dir.mkdir(parents=True, exist_ok=True)
    for parquet_file in parquet_files:
        _refresh_symlink(output_dir / parquet_file.name, parquet_file.resolve(), overwrite_links)

    _write_manifest(manifest_path, output_dir)
    _write_metadata(output_dir, repo_id, raw_dir, parquet_files, manifest_path)
    print(f"Prepared {len(parquet_files)} shards")
    print(f"Raw data: {raw_dir}")
    print(f"Training source: {output_dir}")
    print(f"Manifest: {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="lmms-lab/LLaVA-NeXT-Data")
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--overwrite-links", action="store_true")
    args = parser.parse_args()

    prepare_dataset(
        repo_id=args.repo_id,
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        manifest_path=args.manifest_path,
        max_workers=args.max_workers,
        skip_download=args.skip_download,
        local_files_only=args.local_files_only,
        overwrite_links=args.overwrite_links,
    )


if __name__ == "__main__":
    main()
