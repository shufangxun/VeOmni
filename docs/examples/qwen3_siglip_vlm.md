# Qwen3 SigLIP VLM Training Guide

This guide documents the Qwen3 SigLIP VLM path: a SigLIP vision tower, a pixel-shuffle MLP connector, and a Qwen3 language model. The recipe is designed for native-resolution image tokens and transformers v5/FSDP2 training.

## Environment

```shell
uv sync --no-group transformers-stable --extra transformers5-exp --extra gpu --extra audio --dev
source .venv/bin/activate
```

## Model Layout

The model is registered as `qwen3_siglip_vlm`.

- `text_config`: Qwen3 causal LM config.
- `vision_config`: SigLIP vision config.
- `vision_tower`: packed native-resolution SigLIP encoder.
- `connector`: pixel-shuffle downsampling followed by a two-layer MLP.
- `image_token_id`: placeholder token used to fill projected image features into the LLM embedding stream.

The base architecture config lives at:

```text
configs/model_configs/qwen3_siglip_vlm/qwen3_siglip_vlm_1p7b.json
```

The initialized checkpoint should contain both Qwen3 and SigLIP weights plus the randomly initialized connector. Use the compose script to build that checkpoint from source model checkpoints.

## Data Contract

Qwen3 SigLIP VLM uses a normalized runtime data format. Dataset-specific conversion should happen offline. See [Qwen3 SigLIP VLM Data Format](../key_features/qwen3_siglip_vlm_data_format.md) for exact schemas.

Each source declares:

```yaml
preprocess: plaintext | conversation | interleaved
text_keys: text | conversations | texts
image_keys: null | image | images
domain: caption | interleaved | qa | text_web | text_sft | ...
```

The current stage recipes mix:

- caption as single-image interleaved PT
- interleaved image-text PT
- ShareGPT/LLaVA-style multimodal QA
- optional text-only PT and SFT data

## Stage 1: Train Connector

Stage 1 freezes the SigLIP tower and Qwen3 LLM, and trains only the connector.

```shell
bash shells/multimodal/qwen3siglip/stage1_connector.sh
```

Main config:

```text
configs/multimodal/qwen3_siglip/1.7b/stage1_connector.yaml
```

Important settings:

```yaml
data:
  data_type: qwen3_siglip_vlm
  chat_template: siglip_qwen3

train:
  freeze_vit: true
  freeze_connector: false
  freeze_llm: true
  vit_lr: 0.0
  connector_lr: 1.0e-4
  llm_lr: 0.0
```

Smoke run:

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 \
  shells/multimodal/qwen3siglip/stage1_connector.sh \
  --train.max_steps 1 \
  --train.global_batch_size 4 \
  --train.micro_batch_size 1 \
  --train.wandb.enable false \
  --train.checkpoint.save_steps 0 \
  --data.max_seq_len 512 \
  --data.dataloader.num_workers 1
```

## Stage 2: Dense VLM Training

Stage 2 starts from a stage1 HF checkpoint and trains the vision tower, connector, and LLM together with separate learning rates.

```shell
bash shells/multimodal/qwen3siglip/stage2_dense.sh
```

Main config:

```text
configs/multimodal/qwen3_siglip/1.7b/stage2_dense.yaml
```

Important settings:

```yaml
model:
  config_path: /path/to/stage1-hf-checkpoint
  model_path: /path/to/stage1-hf-checkpoint
  tokenizer_path: /path/to/stage1-hf-checkpoint

train:
  freeze_vit: false
  freeze_connector: false
  freeze_llm: false
  vit_lr: 1.0e-6
  connector_lr: 2.0e-5
  llm_lr: 1.0e-5
```

Smoke run:

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 \
  shells/multimodal/qwen3siglip/stage2_dense.sh \
  --train.max_steps 1 \
  --train.global_batch_size 4 \
  --train.micro_batch_size 1 \
  --train.wandb.enable false \
  --train.checkpoint.save_steps 0 \
  --train.checkpoint.save_hf_weights false \
  --data.max_seq_len 512 \
  --data.dataloader.num_workers 1
```

## Checkpoint Conversion and Inference

Distributed checkpoints are saved in DCP format. Convert a DCP checkpoint to a HuggingFace-style checkpoint before HF inference:

```shell
python scripts/merge_dcp_to_hf.py \
  --load-dir exp/qwen3-siglip-1.7b-stage1-connector/checkpoints/global_step_100 \
  --save-dir /tmp/qwen3_siglip_hf_eval/global_step_100 \
  --model-assets-dir exp/qwen3-siglip-1.7b-stage1-connector/model_assets
```

Run image-text inference:

```shell
python scripts/multimodal/qwen3_siglip_1p7b/infer_hf.py \
  --model-dir /tmp/qwen3_siglip_hf_eval/global_step_100 \
  --image /path/to/image.jpeg \
  --question "Describe this image." \
  --device cuda \
  --dtype bfloat16
```

For text-only probing, prefer a raw prompt mode if the model was trained with base-model completion behavior instead of instruction-tuned chat behavior.

## Monitoring

Metrics are namespaced to keep W&B panels readable:

- `training/*`: loss, grad norm, and LR.
- `perf/*`: MFU and throughput, including per-GPU throughput.
- `data/*`: average sequence length and consumed token counters.
- `memory/*`: GPU and CPU memory.
- `source_loss/*`: source-level loss, ppl, and tokens.
- `domain_loss/*`: domain-level loss, ppl, and tokens.

MFU includes Qwen3 LLM FLOPs, SigLIP ViT FLOPs, and connector FLOPs. SigLIP image token counts are derived from `image_grid_hw`.

## Common Issues

- Placeholder mismatch: the number of `<image>` placeholders must equal the number of loaded images.
- FSDP hang on text-only micro-batches: VLM models need a zero-loss dummy vision forward when a rank has no real image path through the ViT.
- Slow dynamic-resolution batches: use data-side bucketing by image patch tokens before adding heavier all-to-all rebalancing.
- Empty or odd chat template outputs: keep the training chat template and HF inference chat template aligned.
