# VeOmni 与 LLaMAFactory Qwen3-1.7B 预训练吞吐对比报告

## 目标

以 VeOmni 配置 `/root/shufangxun/VeOmni/scripts/qwen3_1p7b_pretrain/train.sh` 为参考，对比 VeOmni 和 LLaMAFactory 在 Qwen3-1.7B 预训练场景下的吞吐表现，并分析 LLaMAFactory 可能导致爆内存或 SSH 断连的原因。

本次对比重点区分三种口径：

- 训练循环吞吐：只看 trainer 进入训练循环后的 token throughput。
- 端到端到最后一个 train step：从进程启动到最后一个训练 step 完成，不包含 final save。
- 完整命令耗时：包含 final save 等收尾阶段。

## 实验设置

两边尽量保持一致：

| 项目 | 设置 |
| --- | --- |
| 模型 | `Qwen/Qwen3-1.7B-Base` |
| 训练方式 | from scratch / full pretrain |
| GPU | 4 卡 |
| sequence length | `4096` |
| global batch size | `32` |
| micro batch size | `1` |
| max steps | `30` |
| dtype | `bf16` |
| attention | FlashAttention 2 |
| FSDP | full shard |
| 数据 | fineweb + finewiki safe subset |
| 总 token 数 | 约 `3.93M` tokens |

配置文件：

- VeOmni: `.bench/qwen3_1p7b_throughput/configs/veomni_qwen3_1p7b_safe_bench.yaml`
- LLaMAFactory: `.bench/qwen3_1p7b_throughput/configs/llamafactory_qwen3_1p7b_safe_bench.yaml`

日志文件：

- VeOmni: `.bench/qwen3_1p7b_throughput/logs/veomni_safe_bench_sharded.log`
- LLaMAFactory: `.bench/qwen3_1p7b_throughput/logs/llamafactory_safe_bench_rerun.log`

## 结果汇总

| 对比口径 | VeOmni | LLaMAFactory | 结论 |
| --- | ---: | ---: | --- |
| 训练循环整体 | 约 `37.4k tok/s` | 约 `53.6k tok/s` | LLaMAFactory 更快 |
| 稳定段，不看前 10 step | 约 `44.4k tok/s` | 约 `54.1k tok/s` | LLaMAFactory 更快 |
| 进程启动到最后一个 train step | `126s`，约 `31.2k tok/s` | `174s`，约 `22.6k tok/s` | VeOmni 更快 |
| 包含 final save | 约 `126s` | 约 `195s`，约 `20.2k tok/s` | VeOmni 更快 |

## 主要结论

LLaMAFactory 的训练 loop 确实更快。如果只关心长时间稳定训练吞吐，LLaMAFactory 在这次 benchmark 中优于 VeOmni。

但按端到端冷启动口径，LLaMAFactory 明显更慢。主要额外开销来自：

1. 数据处理和 Arrow cache 构建。
2. 模型、trainer、distributed 初始化。
3. PT 阶段结束后的无条件 final save。

其中数据处理和初始化大多是一次性成本，长训时会被摊薄；final save 对长训也通常只占很小比例。但数据处理这部分不只是慢，还可能导致全量数据场景下的内存或磁盘压力，从而引发训练中断或 SSH 断连。

## LLaMAFactory 爆内存风险分析

LLaMAFactory 配置中虽然设置了：

```yaml
streaming: true
```

但对于本地 parquet 文件，当前 loader 逻辑并不是真正直接 streaming。代码中本地 file 数据会让 `streaming` 失效，先调用 `load_dataset(...)`，再转换成 iterable dataset：

```python
stream_loading = (
    data_args.streaming
    and dataset_attr.load_from != "file"
    and not data_args.streaming_offline_shuffle
)
...
dataset = load_dataset(..., streaming=stream_loading)
...
if (data_args.streaming and dataset_attr.load_from == "file") and convert_file_to_iterable:
    dataset = dataset.to_iterable_dataset(...)
```

对应文件：

`/root/shufangxun/LLaMA-Factory-private/src/llamafactory/data/loader.py`

这解释了观察到的现象：

- safe 小数据可以正常跑。
- 全量数据时可能在“Generating train split”或数据格式转换阶段消耗大量内存、磁盘和 CPU。
- 机器负载过高后，SSH 可能断开，看起来像服务器失联。

## Final save 影响

LLaMAFactory 的 PT workflow 会在训练结束后无条件保存 final model：

```python
train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
trainer.save_model(os.path.join(training_args.output_dir, "final"))
```

对应文件：

`/root/shufangxun/LLaMA-Factory-private/src/llamafactory/train/pt/workflow.py`

因此即使配置了：

```yaml
save_strategy: "no"
save_steps: 0
```

PT 阶段仍然会保存 final model。本次 LLaMAFactory final output 大约 `7.7G`，带来额外磁盘 IO 和收尾耗时。

## 如何公平解读结果

如果目标是比较纯训练性能，应该采用 warm-cache steady-state 口径：

- 先让数据 cache / 初始化完成。
- 正式统计时只看训练循环。
- 不把 final save 算进吞吐。
- 该口径下，LLaMAFactory 当前更快。

如果目标是评估真实全量预训练可用性，则必须采用 cold-start production 口径：

- 从空 cache 或真实线上状态启动。
- 统计数据处理、初始化、训练、保存的完整影响。
- 重点观察内存、磁盘、CPU、GPU 利用率和 SSH 稳定性。
- 该口径下，VeOmni 更稳，LLaMAFactory 当前存在本地 parquet 加载风险。

## 建议

短期建议：

- LLaMAFactory 全量数据不要直接沿用当前本地 parquet 配置长跑。
- 先用 `max_steps=30` 或 `max_steps=100` 做冷启动压力测试。
- 限制 `preprocessing_num_workers`，避免数据处理阶段并发过高。
- 将 cache 目录和 output 目录放到空间充足、IO 稳定的高速盘。
- 用 `nvidia-smi`、`free -h`、`df -h`、`iostat` 或类似工具观察数据处理阶段资源变化。

中期建议：

- 修改 LLaMAFactory 本地 parquet loader，使其支持真正 streaming，而不是先整体 `load_dataset` 成 Arrow/cache。
- 给 PT workflow 增加类似 `LF_SKIP_FINAL_SAVE` 的开关，避免 benchmark 或中间实验被 final save 干扰。
- 分别报告 `train-loop throughput`、`end-to-end before save`、`end-to-end with save`，避免混淆一次性成本和稳定训练吞吐。

最终判断：

LLaMAFactory 的训练核心吞吐更好，但当前本地 parquet 数据路径有明显全量 OOM 风险。VeOmni 端到端更稳，适合作为可靠 baseline。若能解决 LLaMAFactory 的真实 streaming 数据加载和 final save 控制问题，它在长时间预训练吞吐上会更有优势。
