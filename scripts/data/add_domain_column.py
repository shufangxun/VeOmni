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
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def _parquet_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(item for item in path.iterdir() if item.is_file() and item.suffix == ".parquet")
    raise FileNotFoundError(f"Input path does not exist: {path}")


def _output_path(input_root: Path, output_root: Path, input_file: Path) -> Path:
    if input_root.is_file():
        return output_root
    return output_root / input_file.name


def add_domain_column(input_path: str, output_path: str, domain: str) -> None:
    input_root = Path(input_path)
    output_root = Path(output_path)
    input_files = _parquet_files(input_root)

    if input_root.is_dir():
        output_root.mkdir(parents=True, exist_ok=True)
    else:
        output_root.parent.mkdir(parents=True, exist_ok=True)

    for input_file in input_files:
        table = pq.read_table(input_file)
        domain_column = pa.array([domain] * table.num_rows, type=pa.string())
        if "domain" in table.column_names:
            domain_index = table.column_names.index("domain")
            table = table.set_column(domain_index, "domain", domain_column)
        else:
            table = table.append_column("domain", domain_column)

        output_file = _output_path(input_root, output_root, input_file)
        pq.write_table(table, output_file)
        print(f"Wrote {output_file} with domain={domain!r} rows={table.num_rows}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Append or overwrite a constant domain column in parquet data.")
    parser.add_argument("--input", required=True, help="Input parquet file or directory containing parquet files.")
    parser.add_argument("--output", required=True, help="Output parquet file or directory.")
    parser.add_argument("--domain", required=True, help="Domain value to write to every row.")
    args = parser.parse_args()

    add_domain_column(args.input, args.output, args.domain)


if __name__ == "__main__":
    main()
