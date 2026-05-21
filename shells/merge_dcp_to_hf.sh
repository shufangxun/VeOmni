python scripts/merge_dcp_to_hf.py \
    --load-dir exp/qwen3-siglip-1.7b-stage1-connector/checkpoints/global_step_100 \
    --save-dir /tmp/qwen3_siglip_hf_eval/global_step_100 \
    --model-assets-dir /root/shufangxun/checkpoints/qwen3_siglip_1p7b_init \
    --shard-size 2000000000