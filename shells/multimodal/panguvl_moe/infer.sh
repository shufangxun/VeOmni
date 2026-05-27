#!/usr/bin/env bash
set -euo pipefail

.venv/bin/python scripts/multimodal/openpangu_vl_30a3b/infer_hf_random.py \
  --model-dir /root/shufangxun/Verl-Pangu30B/models/pangu30b_vl_clean \
  --question "你好，简单介绍一下你自己" \
  --max-new-tokens 200 \
  --device cuda \
  --dtype bfloat16

.venv/bin/python scripts/multimodal/openpangu_vl_30a3b/infer_hf_random.py \
  --model-dir /root/shufangxun/Verl-Pangu30B/models/pangu30b_vl_clean \
  --image /root/shufangxun/VeOmni/tests/testdata/qwen-vl-demo.jpeg \
  --question "描述一下这张图" \
  --max-new-tokens 200 \
  --device cuda \
  --dtype bfloat16
