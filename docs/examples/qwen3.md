# Qwen3 LLM Training Guide

This guide uses Qwen3 as the reference dense LLM path in VeOmni. It covers pretraining, SFT, and mixed PT+SFT training with the unified text data pipeline.

## Environment

New model development targets transformers v5 and FSDP2:

```shell
uv sync --no-group transformers-stable --extra transformers5-exp --extra gpu --extra audio --dev
source .venv/bin/activate
```

The examples below use the 1.7B configs under `configs/text/qwen3/1.7b/`. They can be scaled to larger Qwen3 models by replacing `model.config_path`, `model.model_path`, and batch-size settings.

## Data Contract

Qwen3 text training uses `data_type: mixed_text` and `chat_template: mixed_chatml`. This lets one trainer consume both:

- `plaintext`: text pretraining records, usually a single `text` field.
- `conversation`: supervised chat records converted by the text preprocessor into ChatML-style supervised examples.

Multi-source YAML files define the source-level semantics explicitly. For example:

```yaml
sources:
  - /path/to/fineweb
  - /path/to/tulu-sft
names:
  - fineweb
  - tulu_sft
preprocess:
  - plaintext
  - conversation
text_keys:
  - text
  - messages
domains:
  - web
  - sft
```

`names` is used for source-level metrics and sampling statistics. `domains` is used for domain-level loss and perplexity logging.

## Pretraining

The scratch PT example uses FineWeb/FineWiki-style plaintext data:

```shell
bash shells/text/qwen3_1p7b/pretrain.sh
```

Main config:

```text
configs/text/qwen3/1.7b/fineweb_finewiki_scratch.yaml
```

Useful smoke-run overrides:

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 \
  shells/text/qwen3_1p7b/pretrain.sh \
  --train.max_steps 1 \
  --train.global_batch_size 4 \
  --train.micro_batch_size 1 \
  --train.wandb.enable false \
  --data.max_seq_len 256 \
  --data.dataloader.num_workers 1
```

## SFT

The SFT example uses conversation data:

```shell
bash shells/text/qwen3_1p7b/sft.sh
```

Main config:

```text
configs/text/qwen3/1.7b/tulu_sft.yaml
```

If an SFT dataset is stored in OpenAI `messages` format, keep that conversion offline instead of adding dataset-specific logic to the runtime transform. A small conversion utility is available:

```shell
python scripts/data/convert_openai_messages_to_sharegpt.py \
  --input-train /path/to/openai/train \
  --input-eval /path/to/openai/eval \
  --output-dir /path/to/sharegpt-output \
  --num-proc 4
```

The resulting records use ShareGPT-style `conversations` turns:

```json
{
  "conversations": [
    {"from": "human", "value": "Write a short summary."},
    {"from": "gpt", "value": "Here is a short summary."}
  ],
  "domain": "sft"
}
```

## Mixed PT + SFT

The mixed recipe combines plaintext PT sources and conversation SFT sources in one weighted multi-source run:

```shell
bash shells/text/qwen3_1p7b/mix_pt_sft.sh
```

Main config:

```text
configs/text/qwen3/1.7b/pt_sft.yaml
```

Use source/domain metrics to confirm the intended mixture:

- `source_loss/*`: loss, ppl, and token count per configured `names` entry.
- `domain_loss/*`: loss, ppl, and token count per configured `domains` entry.
- `data/*`: average sequence length and consumed token counters.
- `perf/*`: MFU and throughput.

## Key Config Fields

- `data.data_type: mixed_text`: selects the text transform that can handle plaintext, conversation, and mixed multi-source data.
- `data.chat_template: mixed_chatml`: applies ChatML-style formatting and label masking.
- `data.multisource_datasets_type: veomni_weighted_multisource`: enables weighted multi-source sampling.
- `train.accelerator.fsdp_config.fsdp_mode: fsdp2`: uses the primary FSDP2 path.
- `train.log_train_source_loss_steps` and `train.log_train_domain_loss_steps`: control source/domain loss logging.
