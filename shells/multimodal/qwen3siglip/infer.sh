# image inference
# .venv/bin/python scripts/multimodal/qwen3_siglip_1p7b/infer_hf.py \
#     --model-dir /tmp/qwen3_siglip_hf_eval/global_step_100 \
#     --image /tmp/qwen3_siglip_smoke.jpg \
#     --question "Describe this image." \
#     --max-new-tokens 512 \
#     --device cuda \
#     --dtype bfloat16

# text inference
.venv/bin/python scripts/multimodal/qwen3_siglip_1p7b/infer_hf.py \
     --model-dir /tmp/qwen3_siglip_hf_eval/global_step_100 \
     --question "who is leo messi?" \
     --raw-prompt \
     --max-new-tokens 100 \
     --device cuda \
     --dtype bfloat16