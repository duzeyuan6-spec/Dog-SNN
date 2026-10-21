# SNN–VAE × OpenVLA-OFT / LIBERO

本分支归档 SNN–VAE 重建前置摄像头图像并接入 OpenVLA-OFT 的实验代码、配置、环境版本及测量结果。最后整理：2026-10-21。

## 最终结果

LIBERO-Spatial：10 个任务，每任务 50 个官方初始状态；五条件合计 2500 回合。同一 OpenVLA-OFT checkpoint，评估 seed=42。42/43/44 指三个独立训练的 SNN 模型种子。

| 输入 | 成功回合 | 成功率 | 平均周期耗时 (ms) | 板卡能耗 (J/周期) |
|---|---:|---:|---:|---:|
| 原始图像 | 490/500 | 98.0% | 204.225 | 47.884 |
| 官方未经微调 VAE | 490/500 | 98.0% | 431.866 | 133.358 |
| SNN seed42 | 492/500 | 98.4% | 417.540 | 118.625 |
| SNN seed43 | 491/500 | 98.2% | 416.768 | 118.771 |
| SNN seed44 | 494/500 | 98.8% | 413.198 | 118.563 |

**相对官方 VAE：SNN 成功率提高 0.2–0.8 个百分点，耗时降低 3.32%–4.32%，能耗降低 10.94%–11.09%。按用户明确的“有提升即成功”标准，本轮实验取得成功；不以10%作为当前成功门槛。** 不将重复测量中最好的一次或验证筛选结果当作最终验收。按实验委托要求，本轮 VLA 不计算置信区间；成功率差异仅作描述，不宣称统计显著优势。

### 成本测量范围

成本表来自 GPU0 的独占计算测试，共 3 轮 × 5 条件 = 15 块。每块为 10 个任务 × 3 周期。一个周期包含 8 次观测预处理、一次 VLA 动作块推理及动作后处理；重建、CPU/GPU 传输和进程间 pipe 通信计入。使用每任务官方初态0稳定后的固定观测重放，**不包含模拟器、模型加载、真实机器人时间或整机能耗**。板卡能耗来自 NVML 总能量差；系统 Xorg 显示服务不作为外来计算进程，但其能耗包含在读数中。各模型执行相同流程，完整原始数据保留。

共享 GPU 诊断单独保留，`acceptance_eligible=false`。历史共享结果目录名含 `gpu3`，实际物理 GPU 为2，见其中 `gpu_identity.json`；不要按目录名推断硬件。

## 方法及限制

- 仅 front camera 的 256×256 RGB 图像做确定性 `mode` 重建，再走官方 224 resize/crop；腕部图像和 proprio 不改。
- raw 条件不经过 VAE；vae 使用未经微调的官方 FLUX.2 VAE；SNN 使用对应种子的已训练编码器及配套 decoder。
- 所有 VAE 条件公平采用 decoder channels-last；encoder 布局保持原样。
- SNN 为单步二值 IF 加连续残差捷径和连续 decoder，**不是全脉冲网络，也不是事件驱动 GPU 实现**。
- 原有 ANN 对照更快且重建质量略好，不能把缩小模型或数据适配收益直接归因于脉冲。
- 重建 PSNR 改善但 LPIPS 略差（在预先指定容差内）；随机后验小样本表现较差，后验方向按要求停止，不能宣称完整生成能力等价。
- 历史冻结 VAE 按当时门槛得到的失败数据仍在 `runs/frozen_t1/paired_performance.json`，与本次 VLA 成本结果不同。

## 文件导航

- `src/`：SNN 编码器、训练阶段、三个种子、质量评估及资源队列的服务器源码快照。
- `vla/run_full_eval.py`：五条件真实任务评估与回合缓存。
- `vla/reconstruction_worker.py`：原 VAE 环境中的重建 worker。
- `vla/benchmark_pipeline.py`：固定观测重放的完整推理成本测量。
- `src/vla_pipeline_exclusive_queue.py`：独占计算资源检查及外来进程监视。
- `vla/runs/full_summary.json`：五组每任务结果，最终日志均未发现 `Episode error`。
- `vla/runs/pipeline_exclusive/1792517663508625690/`：`protocol.json`、`blocks.jsonl`、`result.json` 和 `summary.json`。
- `runs/frozen_t1/manifest.json`：冻结模型/源码/数据划分的原始 SHA256 身份。
- `vla/source_revisions.json`、`vla/model_identity.json`、`vla/environment.lock.txt`：外部源码、模型与环境版本。
- `ITERATIONS.md`：历史迭代记录，时间较早；最新结论以本 README 和结果 JSON 为准。

## 运行与复现

这是**已执行实验的归档快照**，不是包含模型数据的一键安装包。模型权重、数据集、视频、逐观测缓存、虚拟环境及第三方仓库没有上传。缺少训练后的 checkpoint 时不能直接复现 SNN 数值。

源码保留原始绝对目录 `/datasets2/dzy/snn_vae_20261015`，以保留测量时代码身份；`src/config.json` 中的 pilot 参数也是历史基础配置，最终模型实际 spec/timesteps 从 checkpoint 加载，不应误认为最终使用4步。移植时请在独立副本修改目录并记录新代码哈希，不要覆盖历史 manifest 或结果。

环境准备：

1. 准备原 VAE Python 环境 `ROOT/venv`（依赖见 `src/requirements.txt`）及官方模型、原始固定数据划分与训练 checkpoint。
2. 将 OpenVLA-OFT 和 LIBERO 官方仓库放到 `ROOT/vla/vendor/`，checkout `vla/source_revisions.json` 记录的提交。
3. 准备 `ROOT/tools/bin/uv`；审阅 `vla/prepare_vla.sh` 后创建独立 Python3.10 VLA 环境。官方 moojink transformers fork 不能替换成普通 transformers。
4. 按 `vla/model_identity.json` 下载固定版本 VLA；LIBERO_CONFIG_PATH 指向本机正确的 LIBERO 资源路径。
5. 成本测试额外需要 `nvidia-ml-py` 和 `psutil`。原始环境 lock 是历史快照，包含服务器本地 editable 路径，不能在其他机器直接当通用 requirements 使用。

在原服务器已配置环境中查看回合进度（只读）：

```bash
python3 /datasets2/dzy/snn_vae_20261015/src/vla_full_progress.py --once
```

所有既定回合及测量均已完成。**不要直接重启历史队列覆盖结果或启动已停止的原 VAE/ANN/后验训练方向。** 新实验必须使用独立输出目录，先核实 GPU 余量与其他用户进程；不得终止他人进程。正式能耗实验需要独占计算资源，不能把共享诊断冒充独占结果。

## 数据与许可

这里仅发布本实验代码与数值记录，不分发上游模型权重、LIBERO 数据或第三方源码。上游软件、模型及数据的许可和访问要求仍适用。冻结 manifest 中的服务器路径和哈希是历史溯源信息，不代表相应文件已随仓库发布。

## 结论口径更新

2026-10-21 用户明确：有提升即成功，不设10%提升门槛。结果 JSON 中的 `both_10pct_reduction_all_seeds=false` 和历史迭代中的10%判断仅记录当时的比较条件，不是当前成功判定，也不改变任何原始数值。这里的提升均相对官方VAE重建条件；相对直接输入原始图像，重建仍增加推理时间与能耗。
