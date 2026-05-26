# Qwen3-MoE SigLIP VLM Training Guide

This guide documents the `qwen3_moe_siglip_vlm` path: a native-resolution SigLIP vision tower, a pixel-shuffle MLP connector, and a Qwen3-MoE language model. It reuses the canonical Qwen3-SigLIP multimodal data transform and VeOmni's existing Qwen3-MoE fused expert implementation.

## Model Structure

The model is registered as `qwen3_moe_siglip_vlm`.

- `vision_tower`: packed native-resolution SigLIP encoder.
- `connector`: pixel-shuffle MLP projection into the LLM hidden size.
- `language_model`: patched `Qwen3MoeForCausalLM` from VeOmni's `qwen3_moe` path.

The wrapper only fills image features into `inputs_embeds`. MoE routing, fused expert forward, router aux loss, and log-prob outputs are handled by the underlying patched Qwen3-MoE language model.

## Compose Initial Checkpoint

Build the initial VLM checkpoint from a Qwen3-MoE checkpoint and a SigLIP checkpoint:

```bash
PATH=/root/shufangxun/VeOmni/.venv/bin:$PATH python   scripts/multimodal/qwen3_moe_siglip_30a3b/compose_qwen3_moe_siglip_vlm.py   --qwen-path Qwen/Qwen3-30B-A3B   --siglip-path google/siglip-so400m-patch14-384   --output-dir /root/shufangxun/checkpoints/qwen3_moe_siglip_30a3b_init
```

The compose script accepts original HF per-expert Qwen3-MoE weights. It converts them into VeOmni's fused expert layout under `language_model.*` while copying SigLIP weights and initializing the connector.

## EP Plan

Expert parallelism follows the existing Qwen3-MoE and Qwen3-VL-MoE design. The VLM wrapper adds the `language_model.` prefix to the text model's expert plan:

```python
{
    "language_model.model.layers.*.mlp.experts.gate_up_proj": Shard(0),
    "language_model.model.layers.*.mlp.experts.down_proj": Shard(0),
}
```

Use `model.ops_implementation.moe_implementation: fused_triton` for EP training. The eager expert path is intended for debugging and single-process tests, not EP throughput runs.

## Stage 1: Connector Alignment

Stage 1 freezes SigLIP and Qwen3-MoE, then trains only the connector:

```bash
bash shells/multimodal/qwen3moesiglip/stage1_connector.sh
```

Default config:

```text
configs/multimodal/qwen3_moe_siglip/30a3b/stage1_connector.yaml
```

## Stage 2: Joint Training

Stage 2 trains ViT, connector, and Qwen3-MoE together:

```bash
bash shells/multimodal/qwen3moesiglip/stage2_joint.sh
```

Default config:

```text
configs/multimodal/qwen3_moe_siglip/30a3b/stage2_joint.yaml
```

The config includes both EP and Ulysses SP fields. Adjust `ep_size`, `ulysses_size`, and global batch size to match the available GPUs.

## Data

The runtime data path is unchanged from Qwen3-SigLIP VLM:

```yaml
data:
  data_type: qwen3_siglip_vlm
  chat_template: siglip_qwen3
```

Use the existing canonical multimodal/text mix configs. Dataset-specific conversion should still happen offline before training.
