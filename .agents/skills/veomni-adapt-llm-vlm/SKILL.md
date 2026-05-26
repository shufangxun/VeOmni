---
name: veomni-adapt-llm-vlm
description: "Adapt a dense LLM or SigLIP-style dense VLM training path in VeOmni using the Qwen3 and Qwen3-SigLIP workflow. Use when wiring model config, data_type transforms, canonical text/multimodal data formats, staged checkpoints, training shells, HF inference smoke tests, and docs for a new LLM/VLM experiment."
---

# VeOmni LLM/VLM Adaptation

Use this skill when adapting a model family through the same path as the Qwen3 LLM and Qwen3-SigLIP VLM work: canonicalize data, wire `data_type`, implement or reuse a model wrapper, define staged configs, run smoke training, and document the workflow.

If the task is to add a completely new HF architecture or patchgen model, use `/veomni-new-model` first. If the task is only bug diagnosis, use `/veomni-debug`.

## Ground Rules

- Keep `main` as an upstream mirror and put experiment logic on `dev` or a feature branch.
- Target transformers v5 and FSDP2.
- Do not edit generated files under `veomni/models/transformers/*/generated/`.
- Prefer existing VeOmni patterns: registry-based model loading, `DATA_TRANSFORM_REGISTRY`, multisource YAML, `MainCollator`, DCP checkpoints, and `merge_dcp_to_hf.py`.
- Make data routing explicit with `data.data_type`; do not infer transforms from `model_config.model_type`.

## First Pass

1. Read the current reference implementation:
   - LLM configs: `configs/text/qwen3/`
   - VLM configs: `configs/multimodal/qwen3_siglip/`
   - VLM model: `veomni/models/transformers/qwen3_siglip_vlm/`
   - VLM transform: `veomni/data/data_transform.py`, registry key `qwen3_siglip_vlm`
   - VLM template: `veomni/data/multimodal/multimodal_chat_template.py`, key `siglip_qwen3`
   - Checkpoint compose script: `scripts/compose_qwen3_siglip_vlm.py`
   - HF inference smoke script: `scripts/multimodal/qwen3_siglip_1p7b/infer_hf.py`
2. Inspect upstream/local branch state before changing files:
   - `git status --short --branch`
   - `git log --oneline --decorate -5`
3. Decide the adaptation category:
   - **LLM only**: text pretrain, SFT, or mixed PT+SFT.
   - **VLM wrapper**: existing LLM + vision tower + connector.
   - **VLM data-only extension**: new data source using existing model path.

## LLM Path

Use `data.data_type` as the contract:

- `plaintext`: pure PT text.
- `conversation`: SFT/chat data.
- `mixed_text`: PT + SFT mixture with per-source preprocess.

Checklist:

1. Put source-specific data details in `configs/text/data/*.yaml`.
2. Put model/training policy in `configs/text/<model>/<size>/**.yaml`.
3. For mixed PT/SFT, keep source lists aligned: `sources`, `names`, `preprocess`, `text_keys`, and optional `domains`.
4. Preserve domain metadata where available so domain loss/ppl logging keeps working.
5. Smoke the shell with tiny steps and W&B disabled before claiming success.

Typical checks:

```bash
.venv/bin/pytest tests/data/test_mixed_text_transform.py tests/data/test_domain_loss.py -q
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 WANDB_MODE=disabled WANDB_DISABLED=true \
  shells/text/qwen3_1p7b/pretrain.sh \
  --train.max_steps 1 --train.global_batch_size 4 --train.micro_batch_size 1 \
  --train.eval_steps 0 --train.eval_epochs 0 --train.checkpoint.save_steps 0 \
  --train.wandb.enable false --data.max_seq_len 256 --data.dataloader.num_workers 1
```

## VLM Model Path

For a SigLIP-style dense VLM, keep the final training model simple:

- `language_model`: decoder-only LLM.
- `vision_tower`: packed/native-resolution ViT.
- `connector`: pixel shuffle or MLP projector into LLM hidden size.
- `forward`: replace image placeholder embeddings, then reuse the LLM CE loss.

Do not add CoCa/SigLIP contrastive training to the normal VLM trainer unless the user explicitly asks. That is a separate ViT pretraining objective and requires extra text encoder/loss/collator support.

Checklist:

1. Add model package under `veomni/models/transformers/<name>/`:
   - `configuration_*.py`
   - `modeling_*.py`
   - `checkpoint_tensor_converter.py`
   - `__init__.py` with `MODEL_CONFIG_REGISTRY` and `MODELING_REGISTRY`
2. Add model config under `configs/model_configs/<name>/`.
3. Register imports in `veomni/models/transformers/__init__.py`.
4. Add compose/init script if merging independent LLM and ViT checkpoints.
5. Keep trainable module controls in `VLMTrainingArguments` and `VLMTrainer`:
   - `freeze_vit`
   - `freeze_connector`
   - `freeze_llm`
   - `vit_lr`
   - `connector_lr`
   - `llm_lr`
6. Ensure pure-text batches still trigger FSDP dummy vision forward when needed.
7. Preserve SP behavior: gather image features before connector replacement, then slice LLM inputs.

## VLM Data Path

Prefer canonical data categories:

- `plaintext`: pure text PT.
- `conversation`: ShareGPT-style multimodal QA with `conversations` and image placeholder alignment.
- `interleaved`: aligned `texts` and `images` lists, using `null` to mark the absent modality at each position.

For Qwen3-SigLIP style data, source YAML should drive preprocessing:

```yaml
sources:
  - /path/to/caption
  - /path/to/interleaved
  - /path/to/qa
names:
  - caption
  - interleaved
  - qa
preprocess:
  - interleaved
  - interleaved
  - conversation
text_keys:
  - texts
  - texts
  - conversations
image_keys:
  - images
  - images
  - image
```

Rules:

- Validate image placeholder count equals loaded image count for every multimodal sample.
- Keep image bytes in parquet for stable local training.
- Skip unreadable image samples with a warning; do not crash long runs for a single bad WebP.
- Let sample `domain`/`domain_name` win over YAML fallback when present.
- Avoid `max_image_nums` for static image interleaved training unless the user explicitly wants truncation.

## Training Stages

Use staged configs instead of one overloaded config:

1. **Stage 1 connector**:
   - freeze ViT and LLM.
   - train connector only.
   - data: caption/interleaved/QA multimodal sources.
2. **Stage 2 dense**:
   - train ViT, connector, and LLM.
   - data: multimodal sources plus canonical LLM data.
3. Optional **ViT+connector alignment**:
   - freeze LLM.
   - train ViT and connector with existing CE objective.
   - do not call this Kimi-style CoCa unless contrastive loss is implemented.

Common shell locations:

- `shells/multimodal/qwen3siglip/stage1_connector.sh`
- `shells/multimodal/qwen3siglip/stage2_dense.sh`
- `shells/multimodal/qwen3siglip/infer.sh`

## Checkpoint and HF Smoke

1. Compose initial checkpoints with a script like `scripts/compose_qwen3_siglip_vlm.py`.
2. Train to DCP checkpoints.
3. Convert selected DCP checkpoints to HF format with `scripts/merge_dcp_to_hf.py`.
4. Copy or save model assets: config, tokenizer files, processor/preprocessor files, and chat template if HF inference needs them.
5. Smoke both:
   - image + question inference
   - raw text prompt inference

If chat output degenerates, check whether the model expects raw Qwen3 base prompt formatting versus chat template formatting. Base LLMs may not behave correctly under chat-style prompts.

## Validation

Minimum validation before final response:

```bash
.venv/bin/ruff check veomni/data/data_transform.py veomni/trainer/vlm_trainer.py
.venv/bin/pytest tests/data/test_domain_loss.py -q
```

For Qwen3-SigLIP-like changes, also run:

```bash
.venv/bin/pytest tests/data/multimodal/test_qwen3_siglip_vlm_data.py tests/models/test_qwen3_siglip_vlm.py -q
```

For shell smoke tests, disable W&B and checkpoints unless the user asks otherwise.

## Documentation

Update docs whenever config fields, data formats, or checkpoint flow change:

- `docs/examples/<model>.md`
- `docs/key_features/*data_format*.md`
- `docs/index.md`
- relevant config examples under `configs/`

Keep docs operational: include exact config paths, shell paths, data format expectations, and checkpoint conversion commands.
