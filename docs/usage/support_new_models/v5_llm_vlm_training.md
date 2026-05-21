# Transformers v5 LLM/VLM Training Cookbook

This cookbook describes the minimum path for adding a new text-only LLM or VLM to VeOmni with transformers v5 and FSDP2. It complements the general [support-new-model checklist](./guide_and_checklist.md), which still contains older v4-era examples.

## 1. Classify the Model

Start from the HuggingFace assets:

- `config.json`: identify `model_type`, `architectures`, and nested configs.
- tokenizer files: identify the tokenizer class and chat template behavior.
- processor files: for VLMs, identify image/video/audio processors.
- modeling code: check attention type, RoPE, MoE, vision encoder layout, and multimodal feature injection.

Use the closest existing VeOmni model as the implementation reference:

- text-only dense Qwen-like LLM: `veomni/models/transformers/qwen3/`
- native multimodal Qwen-like VLM: `veomni/models/transformers/qwen3_vl/`
- custom SigLIP+LLM VLM: `veomni/models/transformers/qwen3_siglip_vlm/`

## 2. Register the Model

Create `veomni/models/transformers/<model_name>/` and register at import time. The registry key must match `config.json:model_type`.

Text-only models usually need only a modeling registration if the upstream config is already supported. VLMs usually need config and modeling registrations, and may need a processor registration for HF inference or runtime preprocessing.

Also import the new package from `veomni/models/transformers/__init__.py`; the registry is populated through import-time side effects.

## 3. Implement Training Behavior

For a text-only LLM:

- use `tasks/train_text.py` and `TextTrainer`
- set `data.data_type` to the matching text transform, usually `mixed_text`
- choose a chat template that matches the dataset supervision
- verify FlashAttention and SP behavior against the closest existing LLM

For a VLM:

- use `tasks/train_vlm.py` and `VLMTrainer`
- implement image/video feature extraction and fill-back into LLM embeddings
- provide `get_position_id_func` if the model needs custom multimodal position IDs
- add FSDP dummy forwards for vision/audio encoders so text-only or uneven batches do not hang in backward collectives
- support SP by padding/slicing vision sequences consistently with the collator
- add freeze controls and parameter groups if the recipe trains ViT, connector, and LLM with different learning rates

## 4. Define Data Semantics

Prefer normalized data formats and explicit source semantics over dataset-name-specific runtime branching.

For LLM training, keep PT and SFT as separate source semantics:

- `plaintext`: raw pretraining text
- `conversation`: supervised chat/instruction turns

For VLM training, keep the runtime transform small and explicit:

- `plaintext`: text-only PT
- `conversation`: text-only or multimodal QA/SFT
- `interleaved`: image-text interleaved PT, including caption as a single-image special case

Declare these in multi-source YAML with `preprocess`, `text_keys`, `image_keys`, and `domains`. Convert messy raw datasets offline.

## 5. Write Configs and Launchers

Add:

- a base architecture config under `configs/model_configs/<family>/`
- a train config under `configs/text/` or `configs/multimodal/`
- a multi-source data config if the run mixes sources
- a shell launcher for the common run command

Defaults for new recipes:

- transformers v5 environment
- `fsdp_mode: fsdp2`
- `init_device: meta`
- `attn_implementation: flash_attention_2`
- `datasets_type: iterable`
- small `max_steps=1` smoke-run overrides documented next to the full command

## 6. Test the Integration

Minimum checks before a model recipe is considered usable:

- model config loads through VeOmni registry
- tokenizer/processor assets save and reload
- one forward pass works on a tiny sample
- data transform validates placeholder/image alignment for VLMs
- one multi-GPU `max_steps=1` training smoke passes
- checkpoint save/load or DCP-to-HF conversion works if the recipe documents inference
- source/domain metrics show the intended data mixture

For VLMs, also test:

- image-only multimodal sample
- text-only conversation sample
- interleaved sample with multiple images
- SP smoke if the recipe claims SP support
- frozen-module modes such as connector-only and dense training

## 7. Operational Notes

- Do not edit files under `veomni/models/transformers/*/generated/` manually.
- Keep `model_type` and registry keys identical.
- Keep training chat templates and HF inference chat templates aligned.
- Keep W&B metrics namespaced: `training`, `perf`, `data`, `memory`, `source_loss`, and `domain_loss`.
- Convert checkpoints to HF format for standalone inference with `scripts/merge_dcp_to_hf.py`.
