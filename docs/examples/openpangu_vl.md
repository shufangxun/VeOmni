# OpenPangu-VL MoE Training Guide

This guide summarizes the OpenPangu-VL adaptation in VeOmni. The default
training entrypoint is:

```bash
bash shells/multimodal/panguvl_moe/sft.sh
```

The shell uses:

```text
configs/multimodal/openpangu_vl/30a3b/sft.yaml
```

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
  reduces expert counts, and updates the bias without adding a differentiable
  router loss.

## Load-Balance Strategy

Use `train.moe_load_balance_strategy` to select exactly one load-balancing
strategy:

```yaml
train:
  moe_load_balance_strategy: aux_free
  moe_aux_free_load_balance_update_rate: 0.001
```

Valid values:

- `auto`: preserve model defaults. Models with `router_aux_loss_coef` continue
  to use the existing differentiable router auxiliary loss.
- `none`: disable trainer-added MoE load balancing. If a HuggingFace-style
  model already added router aux loss to `outputs.loss`, VeOmni removes it and
  does not add it back.
- `aux_loss`: use the existing differentiable load-balancing loss path.
- `aux_free`: use per-step expert-bias updates and do not add router aux loss
  to the training objective.

The default OpenPangu-VL SFT config uses `aux_free`.

## Monitoring

`train.moe_load_balance_monitor_interval` controls MaxVio logging when W&B is
enabled. Aux-free balancing does not require W&B; the expert-count all-reduce is
small, with payload size roughly:

```text
num_moe_layers * n_routed_experts * sizeof(float32)
```

For a 25-layer, 256-expert model this is about 25.6 KB per training step, which
is negligible compared with FSDP and MoE token-exchange communication.
