# New Model Integration Playbook

This playbook describes the practical steps for adapting a new model to VeOmni.
It is intended to be the first document to read before adding a new text,
vision-language, MoE, diffusion, or omni-modal model.

VeOmni currently targets:

- Python `>=3.11,<3.12`
- `transformers==5.2.0`
- FSDP2 only
- patchgen-generated modeling files for HuggingFace transformers models

## 0. Decide Whether the Model Should Be Added

Before writing code, identify the smallest existing VeOmni model that is close
to the target model. Reuse its structure whenever possible.

Recommended references:

| Target model type | Reference implementation |
|---|---|
| Dense text-only LLM | `veomni/models/transformers/qwen3/` |
| Dense VLM with external vision tower | `veomni/models/transformers/qwen3_siglip_vlm/` |
| Native Qwen-style VLM | `veomni/models/transformers/qwen3_vl/` |
| Text MoE | `veomni/models/transformers/qwen3_moe/`, `qwen3_5_moe/` |
| VLM MoE | `veomni/models/transformers/qwen3_moe_siglip_vlm/` |
| Omni-modal MoE | `veomni/models/transformers/qwen3_omni_moe/` |
| Diffusion model | `veomni/models/diffusers/` and `configs/dit/` |

If the target can be represented as a config or training recipe of an existing
model package, prefer adding a config and launcher instead of creating a new
model package.

## 1. Analyze the Upstream HuggingFace Assets

Record the following before implementation:

- `config.json`
  - `model_type`
  - `architectures`
  - nested text, vision, audio, or MoE configs
  - `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`
  - MoE fields such as `num_experts`, `num_experts_per_tok`, router aux loss
  - multimodal placeholder token ids and projector settings
- tokenizer assets
  - tokenizer class
  - chat template
  - BOS/EOS/PAD behavior
  - special token ids used by multimodal placeholders
- processor assets
  - processor class name from `processor_config.json`
  - image/video/audio preprocessing behavior
  - min/max pixels, patch size, temporal patch size, and grid semantics
- upstream modeling source
  - attention implementation and RoPE variant
  - loss function and output class
  - multimodal feature injection point
  - MoE expert layout and router logits
  - generation-only branches that are irrelevant for training

Do not start patching until these are clear. Most integration bugs come from
assuming a Qwen-like detail that the target model does not actually share.

## 2. Choose the Runtime Entry Point

Pick the trainer and data path first.

| Model family | Runtime entry | Trainer | Typical config directory |
|---|---|---|---|
| Text LLM | `tasks/train_text.py` | `TextTrainer` | `configs/text/` |
| VLM | `tasks/train_vlm.py` | `VLMTrainer` | `configs/multimodal/` |
| Diffusion | `tasks/train_dit.py` | `DitTrainer` | `configs/dit/` |
| Omni/custom | model-specific task | model-specific trainer | model-specific directory |

The model package should match this trainer boundary. Avoid adding trainer
conditionals for a model when the same behavior can live in the model forward,
data transform, or model-specific config.

## 3. Create the Model Package

For a HuggingFace transformers model, create:

```text
veomni/models/transformers/<model_name>/
|-- __init__.py
|-- <model_name>_gpu_patch_gen_config.py
|-- <model_name>_npu_patch_gen_config.py        # if NPU is supported
|-- patches/
|   `-- <model_name>_gpu_patches.py
|-- generated/
|   `-- patched_modeling_<model_name>_gpu.py
`-- parallel_plan.py                            # if EP or custom sharding is needed
```

Optional files:

- `configuration_<model_name>.py` when upstream config needs a VeOmni wrapper.
- `processing_<model_name>.py` when processor behavior must be patched.
- converter scripts when checkpoints require non-trivial key or tensor layout
  conversion.

Also import the new package in:

```text
veomni/models/transformers/__init__.py
```

Registration happens through import-time side effects. If the package is not
imported there, `build_foundation_model()` can silently miss the registry entry.

## 4. Register Config, Modeling, and Processor Classes

Registration keys must match upstream metadata exactly.

| Registry | Key |
|---|---|
| `MODEL_CONFIG_REGISTRY` | `config.json:model_type` |
| `MODELING_REGISTRY` | `config.json:model_type` |
| `MODEL_PROCESSOR_REGISTRY` | processor class name from `processor_config.json` |

Rules:

- Register at import time in `__init__.py`.
- Do not delay registration behind helper functions.
- Ensure the model class returned by `MODELING_REGISTRY` matches
  `architectures[0]` or the class VeOmni should train.
- Keep `model_type` in saved configs identical to the registry key.

Minimal pattern:

```python
from ...loader import MODELING_REGISTRY


@MODELING_REGISTRY.register("<model_type>")
def register_modeling(architecture: str):
    from .generated.patched_modeling_<model_name>_gpu import <ModelClass>

    return <ModelClass>
```

Add config and processor registries only when the model needs them.

## 5. Implement Patchgen Patches

Never edit files under:

```text
veomni/models/transformers/*/generated/
```

Patch generated behavior by editing:

- `<model_name>_gpu_patch_gen_config.py`
- files under `patches/`

Then regenerate:

```bash
python -m veomni.patchgen.run_codegen veomni.models.transformers.<model_name>.<model_name>_gpu_patch_gen_config -v
```

Common patch types:

| Patch type | Typical use |
|---|---|
| `replace_class` | RMSNorm, attention, expert modules, model wrapper |
| `override_method` | attention forward, model forward, causal LM forward |
| `replace_function` | RoPE, loss helpers, load balancing loss helpers |
| `modify_init` | inject OpSlots or config fields |
| `add_post_import_block` | runtime bindings that must exist in generated code |
| `drop_import_names` | remove upstream imports replaced by VeOmni |

Patch only the behavior needed for training. Keep generation-only paths intact
unless they block training or checkpoint conversion.

## 6. Wire VeOmni OpSlots

Use VeOmni kernels through OpSlots instead of hard-coding backend choices inside
model code.

Common model-side slots:

- attention: FlashAttention with or without sequence parallel support
- RMSNorm
- rotary embedding
- causal LM cross entropy
- SwiGLU MLP
- fused MoE experts
- MoE load balancing loss

Config-side backend choices usually live under `model.ops_implementation`, for
example:

```yaml
model:
  ops_implementation:
    attn_implementation: flash_attention_2
    moe_implementation: fused_triton
    load_balancing_loss_implementation: triton
```

If the model reuses an existing language model as a submodule, re-export or bind
the same slots through the wrapper so nested modules see the configured kernels.

## 7. Add FSDP2 and Parallel Plans

VeOmni uses FSDP2. Do not add FSDP1 paths.

For dense models:

- identify transformer block classes for FSDP wrapping
- verify activation checkpointing boundaries
- keep parameter names stable for checkpoint compatibility

For MoE models:

- create `parallel_plan.py`
- shard expert parameters with `Shard(0)` on the EP mesh
- verify exact parameter names with `model.named_parameters()`
- ensure expert modules are listed in the EP no-shard module set when needed
- verify the EP gradient divide factor still represents the global data size

Typical MoE plan:

```python
from torch.distributed._tensor import Shard

from ....distributed.parallel_plan import ParallelPlan


def get_parallel_plan():
    ep_plan = {
        "model.layers.*.mlp.experts.gate_up_proj": Shard(0),
        "model.layers.*.mlp.experts.down_proj": Shard(0),
    }
    return ParallelPlan(extra_parallel_plan={"ep": ep_plan})
```

For wrapper models, prefix the base plan instead of duplicating it when
possible. For example, a VLM wrapper around a language model often prefixes
language-model paths with `language_model`.

## 8. Support Sequence Parallel Correctly

Sequence parallelism is not a simple all-gather mode. VeOmni Ulysses SP
exchanges sequence and heads around attention:

- before attention: local sequence, full heads -> full sequence, local heads
- after attention: full sequence, local heads -> local sequence, full heads

Model requirements:

- Keep `position_ids` aligned with the SP-sliced sequence.
- Compute FlashAttention varlen metadata from full position IDs before slicing
  position IDs in the collator.
- Do not recompute `cu_seqlens` inside every layer.
- Make sure outputs, labels, and loss reduction use SP-aware helpers.

VLM-specific requirements:

- Gather full sequence embeddings before filling image/video/audio features.
- Gather full visual/audio features if feature extraction is SP-sharded.
- Scatter multimodal features into the full sequence.
- Slice the LLM inputs back to the local SP chunk before entering the language
  model.
- Add dummy forward paths for text-only ranks under FSDP2 so all ranks
  participate in collectives.

MoE-specific requirements:

- Router logits are produced on the local SP sequence.
- Preserve a real-token router mask separately from the FlashAttention
  `attention_mask` when padding is present.
- Do not use an SP-padded all-one attention mask as the router aux-loss mask.
- Aggregate sufficient statistics across the SP group before applying the
  nonlinear load balancing formula:
  - expert counts
  - router probability sums
  - total real-token weight

Averaging per-rank router auxiliary losses is numerically wrong.

## 9. Implement Multimodal Data and Collation

For VLM or omni-modal models, define the data contract before writing the
model forward.

Typical work:

- add or reuse a transform in `veomni/data/multimodal/`
- use normalized source semantics such as `plaintext`, `conversation`, and
  `interleaved`
- preserve placeholder token ids and masks
- add model-specific collate entries for new tensor keys
- ensure image/video/audio tensors are shaped exactly as the model expects
- support text-only samples in the same dataloader if the recipe uses them

Collator rules:

- `attention_mask` is used by dynamic batching and FlashAttention behavior.
- `position_ids == 0` marks packed segment boundaries.
- labels must use `IGNORE_INDEX` for unsupervised tokens.
- new tensor keys must be added to collate info with correct padding and SP
  slicing behavior.
- SP collation ordering is load-bearing:
  1. shift labels
  2. pad and slice tensors
  3. compute FlashAttention kwargs from full position IDs
  4. slice position IDs last

For MoE router aux loss under SP, add a dedicated router mask key if needed.

## 10. Add Configs and Launchers

Add model architecture configs under:

```text
configs/model_configs/<family>/
```

Add training configs under:

```text
configs/text/
configs/multimodal/
configs/dit/
```

Config checklist:

- `model.config_path` points to the intended model config or pretrained assets.
- `model.model_path` or checkpoint path is documented.
- `model.trust_remote_code` is intentional.
- `model.ops_implementation` selects supported backends.
- `train.accelerator.fsdp_config.fsdp_mode: fsdp2`.
- `train.accelerator.ulysses_size` matches SP claims.
- `train.accelerator.ep_size` matches MoE EP claims.
- checkpoint save/load settings are explicit.
- W&B and experiment names do not contain hard-coded secrets.
- smoke-run overrides are documented for `max_steps=1`.

Shell launchers are useful, but keep them thin. Most behavior should be in YAML
or CLI overrides.

## 11. Handle Checkpoints and HF Conversion

Checkpoint compatibility must be considered during model design.

Rules:

- DCP checkpoint keys must match the model state dict.
- All ranks must participate in save and load.
- Renaming parameters after training starts breaks resume.
- EP-sharded expert tensors need a clear merge path back to HF format.
- If standalone HF inference is expected, add or update a conversion script and
  test it on a small checkpoint.

For MoE models, explicitly verify expert tensor layout:

- VeOmni fused expert layout
- EP local shard layout
- HF save layout
- HF load layout

Do not assume gate/up/down projection ordering without checking the upstream
state dict.

## 12. Testing Requirements

Minimum tests for every new model:

- registry import test
- config load test
- toy config under `tests/toy_config/<model>_toy/`
- single forward pass on toy inputs
- forward/backward parity or smoke in `tests/models/`
- `ruff check` and `ruff format --check`

Additional tests by feature:

| Feature | Required coverage |
|---|---|
| SP | 2-rank smoke or numerical parity test |
| EP | multi-GPU smoke proving expert tensors are sharded |
| MoE aux loss | mask, scaling, SP statistic aggregation, and gradient tests |
| VLM | image sample, text-only sample, interleaved or multi-image sample |
| FSDP2 dummy forward | asymmetric-rank multimodal batch test |
| checkpoint conversion | DCP save/load and HF export smoke |
| new collator key | padding, packing, and SP slicing unit tests |
| new op/kernel | eager reference parity and backend dispatch tests |

Useful commands:

```bash
.venv/bin/ruff check <changed files>
.venv/bin/ruff format --check <changed files>
.venv/bin/pytest tests/models/ -k <model>
.venv/bin/pytest tests/e2e/ -k <model>
.venv/bin/pytest tests/distributed/test_dummy_forward.py -k <model>
```

For a new training recipe, also run a real `max_steps=1` smoke with the intended
parallelism:

```bash
NPROC_PER_NODE=2 <launcher>.sh \
  --train.max_steps 1 \
  --train.global_batch_size 2 \
  --train.micro_batch_size 1 \
  --train.wandb.enable false \
  --train.checkpoint.save_steps 0
```

For MoE + SP, run a smoke with both enabled when the recipe claims support.

## 13. Documentation Requirements

Update docs when the integration adds or changes any user-visible workflow.

Common updates:

- `docs/examples/<model>.md` for runnable commands
- `docs/usage/support_new_models/` for reusable integration notes
- `docs/key_features/` when a new general mechanism is introduced
- `.agents/knowledge/constraints.md` only when a new hard constraint is found
- `README.md` supported-model table, if applicable

Do not document a command as supported until the corresponding smoke test has
passed.

## 14. Common Failure Modes

| Symptom | Likely cause | Fix |
|---|---|---|
| Model loads as vanilla HF class | `model_type` registry key mismatch | Match `config.json:model_type` exactly and import package in transformers `__init__.py` |
| Generated patch disappears after codegen | Edited `generated/` manually | Move the change into patchgen config or `patches/` |
| NCCL hang on text-only VLM batch | missing dummy forward under FSDP2 | Add dummy forward so all ranks enter collectives |
| SP loss changes with SP size | local loss averaged instead of globally reduced | use SP-aware loss reduction or aggregate sufficient statistics |
| Router aux loss changes with padding | FlashAttention mask reused for router loss | preserve a separate real-token router mask |
| EP has no memory effect | expert parameter path does not match | inspect `named_parameters()` and fix `parallel_plan.py` |
| Fused MoE mismatch | expert tensor layout is wrong | verify gate/up/down layout and contiguous shape conventions |
| FlashAttention varlen crash | invalid `position_ids` boundaries | ensure `position_ids == 0` marks packed segment starts |
| Checkpoint resume fails | parameter names or shapes changed | keep state dict stable or write migration logic |
| HF export cannot load | conversion misses wrapper or EP layout | add a conversion test with a tiny checkpoint |

## 15. Definition of Done

A new model integration is not complete until:

- The closest existing reference model is documented.
- Model registration works from `build_foundation_model()`.
- Patchgen can regenerate generated files without drift.
- FSDP2 wrapping works.
- Claimed SP and EP modes have tests or smoke runs.
- MoE router aux loss, if present, is globally scaled and SP-correct.
- Multimodal text-only and multimodal samples both work, if applicable.
- Checkpoint save/load works for the documented workflow.
- HF export works if standalone inference is documented.
- Docs include a runnable command and tested smoke overrides.
- No shell script contains credentials or machine-specific secrets.

When three reasonable approaches fail during an integration, stop and write
down the exact failure modes before trying a workaround. Most model integration
issues are caused by a wrong assumption about upstream config, tensor layout, or
parallel topology.
