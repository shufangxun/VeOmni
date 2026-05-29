# OpenPangu-VL MoE Training Guide

This guide summarizes the OpenPangu-VL adaptation in VeOmni, including data
preparation, the stage-2 full-parameter training config, and checkpoint
conversion for HuggingFace-style inference.

## Environment

Run all commands from the repository root:

```bash
cd /root/shufangxun/VeOmni
```

Install the GPU development environment once:

```bash
uv sync --extra gpu --extra audio --dev
export PATH="${PWD}/.venv/bin:${PATH}"
python --version
```

VeOmni targets Python 3.11 and transformers v5. The training shell also
prepends `.venv/bin` to `PATH`, so using `export PATH=...` is enough even when
the virtualenv activation script is not present.

OpenPangu-VL can be trained with different recipes, such as connector-only
warmup, full-parameter multimodal training, frozen vision/text components, or
scratch versus checkpoint initialization. This guide uses stage-2
full-parameter training as the concrete example.

## Data Preparation

The stage-2 config points to:

```text
configs/multimodal/data/qwen3_siglip_stage2_full.yaml
```

That YAML is a local prepared-data manifest, not a download manifest. Each
`sources` entry must be a local file or a directory containing directly
readable `parquet`, `jsonl`, `json`, `csv`, or `arrow` files. If a downloaded
Hugging Face snapshot has nested directories, either point `sources` at the
leaf directory containing the data files or export normalized shards into a new
flat directory.

The default mixture expects these source semantics:

| Source name | Suggested public source | Runtime format | Required local fields |
| --- | --- | --- | --- |
| `llava_ov_mid_caption` | [`lmms-lab/LLaVA-OneVision-Mid-Data`](https://huggingface.co/datasets/lmms-lab/LLaVA-OneVision-Mid-Data) | `interleaved` | `texts`, `images` |
| `omnicorpus_interleaved` | [`OpenGVLab/OmniCorpus-CC`](https://huggingface.co/datasets/OpenGVLab/OmniCorpus-CC) or [`OpenGVLab/OmniCorpus-CC-210M`](https://huggingface.co/datasets/OpenGVLab/OmniCorpus-CC-210M) | `interleaved` | `texts`, `images` |
| `llava_next_qa` | [`lmms-lab/LLaVA-NeXT-Data`](https://huggingface.co/datasets/lmms-lab/LLaVA-NeXT-Data) | `conversation` | `conversations`, `image` |
| `fineweb_sample_10bt` | [`HuggingFaceFW/fineweb`](https://huggingface.co/datasets/HuggingFaceFW/fineweb), `sample/10BT` | `plaintext` | `text` |
| `finewiki_en` | [`HuggingFaceFW/finewiki`](https://huggingface.co/datasets/HuggingFaceFW/finewiki) | `plaintext` | `text` |
| `tulu_sft_sharegpt` | [`allenai/tulu-3-sft-mixture`](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) | `conversation` | `conversations` |

Download raw public data with the repository helper, for example:

```bash
python scripts/download_hf_data.py \
  --repo_id lmms-lab/LLaVA-OneVision-Mid-Data \
  --local_dir /data/raw/llava-onevision-mid

python scripts/download_hf_data.py \
  --repo_id OpenGVLab/OmniCorpus-CC-210M \
  --local_dir /data/raw/omnicorpus-cc-210m

python scripts/download_hf_data.py \
  --repo_id lmms-lab/LLaVA-NeXT-Data \
  --local_dir /data/raw/llava-next-data

python scripts/download_hf_data.py \
  --repo_id HuggingFaceFW/fineweb \
  --local_dir /data/raw/fineweb \
  --allow_patterns 'sample/10BT/*'

python scripts/download_hf_data.py \
  --repo_id HuggingFaceFW/finewiki \
  --local_dir /data/raw/finewiki

python scripts/download_hf_data.py \
  --repo_id allenai/tulu-3-sft-mixture \
  --local_dir /data/raw/tulu-3-sft-mixture
```

Normalize the raw data offline before training. OpenPangu-VL reuses the same
normalized runtime formats as the Qwen3-SigLIP VLM data path:

`plaintext` text pretraining:

```json
{"id": "sample-id", "text": "plain pretraining text"}
```

`conversation` QA/SFT data in ShareGPT/LLaVA style:

```json
{
  "id": "sample-id",
  "image": {"path": "/data/images/example.jpg", "bytes": null},
  "conversations": [
    {"from": "human", "value": "<image>\nWhat is shown?"},
    {"from": "gpt", "value": "A chart is shown."}
  ]
}
```

For text-only SFT, keep the same `conversations` field and omit `image`.
Raw Tulu records use `messages`; convert them offline to this ShareGPT
`conversations` schema, or the `conversation` transform will reject them.

For image QA/SFT, the `image` field can be either:

- a HuggingFace-style image struct, `{"bytes": <binary>, "path": null}`;
- a path-backed struct, `{"bytes": null, "path": "/path/to/image.jpg"}`;
- a plain local path string.

The number of `<image>` placeholders in `conversations` must match the number
of images loaded from `image` or `images`. During OpenPangu-VL preprocessing,
VeOmni normalizes the image entry to bytes or path, loads it as RGB, runs the
OpenPangu-VL image processor, and produces `pixel_values` plus
`image_grid_thw`. The chat template then replaces each `<image>` placeholder
with OpenPangu-VL vision tokens:

```text
<|vision_start|><|image_pad|>...<|image_pad|><|vision_end|>
```

The number of `<|image_pad|>` tokens comes from `image_grid_thw` and the image
processor merge size. The transform also builds `image_mask`, masks image token
labels with `IGNORE_INDEX`, and computes OpenPangu-VL 3D `position_ids`.

`interleaved` image-text pretraining:

```json
{
  "id": "sample-id",
  "images": [{"path": "/data/images/1.jpg", "bytes": null}, null],
  "texts": [null, "caption or text span"]
}
```

`images` and `texts` must have the same length, and exactly one of
`images[i]` or `texts[i]` must be non-null at every position. A caption sample
is just an interleaved sample with one image slot followed by one text slot.

After exporting the normalized shards, update the `sources` paths in
`configs/multimodal/data/qwen3_siglip_stage2_full.yaml`. Keep the per-source
`preprocess`, `text_keys`, `image_keys`, `domains`, and `schedule.weights`
aligned by index. `level: token` makes the weighted sampler compensate for
source length differences. `upstream_sharded: true` means VeOmni shards each
upstream streaming dataset by data-parallel rank before weighted mixing, so the
weighted mixer does not apply a second modulo shard.

## Training Configuration

This guide uses the OpenPangu-VL stage-2 full-parameter config:

```text
configs/multimodal/openpangu_vl/30a3b/stage2_full.yaml
```

Launch it with:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 \
  bash shells/multimodal/panguvl_moe/stage2_full.sh
```

`NPROC_PER_NODE` should match the number of visible GPUs. The default
`train.accelerator.ep_size: 2` must divide the total distributed world size.

Important fields:

- `model.config_path` and `model.tokenizer_path` point to the local
  OpenPangu-VL HuggingFace remote-code assets. These assets provide config,
  tokenizer, processor, image processor, and chat-template files.
- `model.model_path: null` initializes model weights from config. Set
  `model.model_path` to a local/HF checkpoint directory when continuing from
  pretrained or converted weights.
- `model.ops_implementation.attn_implementation: flash_attention_2`,
  `moe_implementation: fused_triton`, and
  `cross_entropy_loss_implementation: liger_kernel` keep the default optimized
  kernels enabled.
- `data.data_type: openpangu_vl` and `data.chat_template: openpangu_vl` select
  the OpenPangu-VL transform, image placeholder handling, and 3D position IDs.
- `data.max_seq_len` is the text sequence budget. If too many overlong samples
  are skipped, raise it or reduce the upstream text/image budgets.
- `data.mm_configs.image_max_pixels`, `image_min_pixels`, and `max_ratio`
  control dynamic image resize before OpenPangu-VL image patchification.
- `train.accelerator.ep_size` controls expert parallelism. It must divide the
  distributed world size used by `NPROC_PER_NODE` and node count.
- `train.accelerator.fsdp_config.fsdp_mode: fsdp2` with `full_shard: true`
  enables FSDP2 full sharding for the large language tower.
- `train.gradient_checkpointing.enable: true` trades compute for memory.
- `train.global_batch_size` and `train.micro_batch_size` determine gradient
  accumulation. With 4 GPUs, `global_batch_size: 4` and `micro_batch_size: 1`
  run one micro-batch per rank per optimizer step.
- `train.dyn_bsz_runtime: worker` enables dynamic batching around the
  `micro_batch_seq_length` budget.
- `train.freeze_vit`, `train.freeze_connector`, and `train.freeze_llm` select
  which components train. The stage-2 full config trains all three.
- `train.bsz_warmup_ratio` warms up effective batch size. Optimizer LR warmup
  is separately controlled by `train.optimizer.lr_warmup_ratio`.
- `train.checkpoint.output_dir` controls experiment output. Set
  `train.checkpoint.save_steps` for periodic DCP checkpoints and set
  `train.checkpoint.save_hf_weights: true` plus `hf_save_steps` or
  `hf_save_epochs` if you want VeOmni to export HF weights during training.

MoE load balancing is controlled by `train.moe_load_balance`. OpenPangu-V2
uses auxiliary-loss-free balancing by updating the expert-wise routing bias
after routing statistics are reduced:

```yaml
train:
  moe_load_balance:
    mode: aux_free
    monitor_interval: 1
    log_to_console: true
    aux_free_bias_update_rate: 1.0e-3
    aux_free_update_interval: 1
```

Set `mode: none` and `monitor_interval: 0` to disable collection. Positive
`monitor_interval` values aggregate router expert selections over that many
training steps and log MoE violation metrics to the console and to W&B when
W&B logging is enabled. Standard auxiliary-loss balancing remains available
for model families whose forward path emits `aux_loss` by setting
`mode: aux_loss`.

## Key Adaptation Items

- Model registration: `openpangu_vl` and `openpangu_vl_text` are registered
  through the normal VeOmni model loader path.
- Patchgen path: OpenPangu-VL and the nested OpenPangu-V2 text tower use
  transformers-v5 patchgen-generated GPU and NPU modeling files.
- Multimodal data path: `data_type: openpangu_vl` and
  `chat_template: openpangu_vl` handle image placeholders, image masks, and
  OpenPangu-VL position IDs.
- Expert parallelism: language-tower routed expert weights are covered by the
  OpenPangu-VL parallel plan and use VeOmni fused MoE dispatch when
  `model.ops_implementation.moe_implementation: fused_triton`.
- Optimized text ops: RMSNorm, SwiGLU MLP, text RoPE, cross entropy, and MoE
  expert kernels are dispatched through VeOmni op slots. Vision-specific and
  interleaved/multimodal RoPE paths stay on the eager path.
- FSDP text-only safety: text-only batches still touch the vision projection
  path when needed so mixed multimodal batches do not desynchronize FSDP
  collectives across ranks.
- Aux-free load balancing: OpenPangu-V2 routed experts expose
  `e_score_correction_bias`; VeOmni records selected experts per step, globally
  reduces expert counts, and updates the bias by `aux_free_bias_update_rate`
  without adding a differentiable router loss.

## Checkpoint Conversion and Inference

The default config keeps `save_steps: 0` and `save_hf_weights: false`, so a
long run will not produce a mid-run checkpoint unless you override checkpoint
settings. For evaluation, launch with periodic DCP saves:

```bash
bash shells/multimodal/panguvl_moe/stage2_full.sh \
  --train.checkpoint.save_steps 100 \
  --train.checkpoint.save_hf_weights false
```

Convert a saved DCP checkpoint to HuggingFace format:

```bash
python scripts/merge_dcp_to_hf.py \
  --load-dir exp/openpangu-vl-30a3b-sft/checkpoints/global_step_100 \
  --save-dir /tmp/openpangu-vl-30a3b-hf/global_step_100 \
  --model-assets-dir exp/openpangu-vl-30a3b-sft/model_assets
```

If `exp/openpangu-vl-30a3b-sft/model_assets` is not present, use the same
directory as `model.config_path`, for example:

```bash
python scripts/merge_dcp_to_hf.py \
  --load-dir exp/openpangu-vl-30a3b-sft/checkpoints/global_step_100 \
  --save-dir /tmp/openpangu-vl-30a3b-hf/global_step_100 \
  --model-assets-dir /root/shufangxun/Verl-Pangu30B/models/pangu30b_vl_clean
```

Always pass `--model-assets-dir` for OpenPangu-VL. The converter copies the
processor/tokenizer assets plus the root remote-code `.py` files required by
`trust_remote_code=True`.

`shells/multimodal/panguvl_moe/infer.sh` is a remote-code smoke test that
initializes random weights from the local model assets. It is useful for
checking the processor and prompt path, but it does not load trained weights.

For converted trained weights, point the HuggingFace loader at the converted
directory:

```python
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

model_dir = "/tmp/openpangu-vl-30a3b-hf/global_step_100"
image_path = "/path/to/image.jpg"

processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
).eval()

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Describe this image."},
        ],
    }
]
prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(
    text=[prompt],
    images=[Image.open(image_path).convert("RGB")],
    return_tensors="pt",
)
device = next(model.parameters()).device
inputs = {
    key: value.to(device, dtype=torch.bfloat16) if torch.is_floating_point(value) else value.to(device)
    for key, value in inputs.items()
}

with torch.inference_mode():
    output = model.generate(**inputs, max_new_tokens=128, do_sample=False)

prompt_len = inputs["input_ids"].shape[-1]
print(processor.tokenizer.decode(output[0, prompt_len:], skip_special_tokens=True))
```

For the 30A3B model, single-GPU inference may OOM. Use a multi-GPU
Transformers device map or your serving stack's tensor/expert-parallel
inference path if the converted checkpoint does not fit on one device.

## Monitoring

W&B is optional and controlled by `train.wandb.enable`. When enabled, VeOmni
logs coarse training metrics and finer-grained multimodal/MoE signals:

- `training/*`: total loss, foundation loss, gradient norm, and learning rate.
- `source_loss/*`: per-source loss, perplexity, and token counts. These map to
  the `names` entries in the multi-source data YAML.
- `domain_loss/*`: per-domain loss, perplexity, and token counts. These map to
  the `domains` entries such as `caption`, `interleaved`, `qa`, `text_web`,
  `text_knowledge`, and `text_sft`.
- `perf/*`: throughput and MFU-style performance counters.
- `memory/*`: GPU and CPU memory usage.
- MoE load-balance metrics: expert-load heatmaps plus `max_vio`,
  `min_vio`, and `avg_vio` summaries when
  `train.moe_load_balance.monitor_interval` is non-zero.

`train.moe_load_balance.monitor_interval` controls the monitor aggregation
cadence. Monitor collection does not require W&B; rank 0 logs scalar summaries
to the console, and heatmap/scalar logging is also emitted to W&B when W&B is
enabled. The expert-count all-reduce is small, with payload size roughly:

```text
num_moe_layers * n_routed_experts * sizeof(float32)
```

For a 25-layer, 256-expert model this is about 25.6 KB per training step, which
is negligible compared with FSDP and MoE token-exchange communication.

## TODO

- Add Muon optimizer support for OpenPangu-VL recipes.
- Add video data and video processor support.
- Add layer-wise learning-rate configuration for multimodal fine-tuning.
