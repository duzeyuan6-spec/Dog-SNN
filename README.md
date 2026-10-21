# SNN–VAE × OpenVLA-OFT / LIBERO

本分支归档 SNN–VAE 重建前置摄像头图像并接入 OpenVLA-OFT 的实验代码、配置、环境版本及测量结果。最后整理：2026-10-21。

## 第一部分：独立 SNN–VAE 重建实验

这部分先评估图像重建本身，不运行VLA。目标是在固定官方 FLUX.2 VAE 基线上，用紧凑混合SNN编码器重建图像，再比较质量、耗时和实测板卡能耗。当前结论按“有提升即成功”表述，同时保留质量退化和历史不理想结果。

### 模型原理

```text
256×256 RGB图像
  → PixelUnshuffle(4)
  → 卷积 + 含二值IF支路的残差编码器
  → 潜变量均值 μ / log variance
  → 配套连续FLUX.2 decoder
  → 256×256重建图像
```

`src/spike_encoder.py` 中的编码器宽度128、6个残差块；输出32通道均值和32通道log variance。IF支路将归一化电流逐步累积到膜电位，达到阈值发出0/1脉冲并减去阈值，再用平均发放率输出。训练采用代理梯度，残差捷径维持连续值。初期decoder冻结；joint_v4起联合微调**SNN模型配套的decoder**，官方验收基线的权重始终不变。

最终采用1时间步：保留二值IF激活，但没有多时间步积累；decoder和捷径仍连续。这里使用常规GPU稠密运算，不声称神经形态硬件或事件驱动加速。重建评估采用后验均值 `mode`，不等价于完整随机生成能力评估。

### 数据与训练路线

- 官方基线：`black-forest-labs/FLUX.2-klein-4B` 的VAE，固定版本见 `src/config.json`。
- 数据：固定版本 `lerobot/libero` 的front camera，256×256，按episode划分训练/验证/测试，防止同一episode跨集合。
- 训练期间使用固定400图验证子集；完整验证3018图；模型冻结后独立测试3072图。不同集合指标分别报告。
- 正式SNN种子42/43/44各按同一五阶段配方完成80,000步；43/44独立初始化，各阶段加载本种子前阶段最佳checkpoint。pilot作为早期探索，不计入这条80,000步配方。
- 训练损失包括潜变量蒸馏及图像重建项。v3起降低潜变量损失权重、加入图像MSE；joint_v4起联合训练decoder；完整实现见下表对应源码。

| 阶段 | 步数 | 主要变化 | 历史质量结果 |
|---|---:|---|---|
| pilot | 5,000 | 早期紧凑学生探索 | SNN在400图上PSNR28.086，质量不足 |
| encoder_v2 | 20,000 | 4步IF，decoder冻结，连续warmup后过渡到IF | 400图最佳PSNR33.938，质量不足 |
| encoder_v3 | 20,000 | 调整蒸馏/重建损失，decoder仍冻结 | 400图PSNR35.448，质量不足 |
| joint_v4 | 20,000 | 联合微调配套decoder | 3018图PSNR39.440、LPIPS0.004987 |
| joint_t2_v5 | 10,000 | IF从4步减少到2步，继续联合微调 | 最后400图PSNR39.397、LPIPS0.005569 |
| joint_t1_v6 | 10,000 | IF从2步减少到1步 | 3018图PSNR39.014、LPIPS0.006871 |

源码对应：`iterate_v2.py` → `iterate_v3.py` → `iterate.py` → `iterate_t2.py` → `iterate_t1.py`；多种子入口 `train_seed.py`。具体模型版本与时间步以checkpoint spec为准，不以历史pilot配置推断。

### 冻结独立测试：3072图

PSNR越高越好，LPIPS越低越好；以下为四舍五入值。

| 模型 | PSNR (dB) | LPIPS |
|---|---:|---:|
| 官方VAE | 38.035 | 0.005348 |
| 单步SNN seed42 | 38.973 | 0.006862 |
| 单步SNN seed43 | 38.977 | 0.006830 |
| 单步SNN seed44 | 38.964 | 0.006997 |

SNN的PSNR提高约0.93–0.94dB，LPIPS略差；符合预先指定的质量容差（PSNR下降不超过0.5dB、LPIPS增加不超过0.02）。**质量非劣容差通过不等于所有质量指标均改善。** 冻结身份见 `runs/frozen_t1/manifest.json`；历史质量不确定性分析见 `runs/quality_uncertainty/summary.json`，它早于后续VLA阶段“不计算置信区间”的要求。

### 脉冲消融与ANN对照

- joint_v4在400图上的正常PSNR为39.419；将脉冲支路置零降到21.402，表明该支路对重建有实际作用。二值脉冲及状态重置检查通过。
- 同规模、匹配训练流程的ANN在3018图上PSNR39.582、LPIPS0.004411，优于当时4步SNN的39.440/0.004987，而且更快。因此本实验不能证明脉冲本身优于ANN。
- 曾做过原始VAE微调辅助对照，后来按用户要求停止该方向；当前比较基线仍是未经微调官方VAE。历史对照源码只作归档，不代表需要重新训练。

### 独立VAE耗时与能耗迭代

下表均比较当轮SNN与同轮官方基线；不同版本/不同GPU的相对变化不能拼成同一次实验。测试范围是FP32、TF32关闭、batch1、GPU驻留的确定性重建forward；不是VLA完整流程或整机成本。

| 候选/实验 | 耗时变化 | 板卡能耗变化 | 解释 |
|---|---:|---:|---|
| 4步joint_v4 | 慢0.51% | 降低约26.0% | 重建质量改善，未观察到提速 |
| 2步joint_t2_v5 | 降低8.155% | 降低26.934% | 减少时间步后提速 |
| 原始单步T1，12块重复 | 降低约9.87% | — | 保留重复测量，不只引用首轮较高提速 |
| FastRateIF，seed42验证筛选 | 降低16.342% | 降低29.054% | 删除单步冗余计算；不是独立测试结论 |
| FastRateIF，冻结三种子验收 | 降低6.01%–11.44% | 降低约28% | 历史统计门槛未通过，原始记录保留 |
| 仅decoder channels-last，三种子验证 | 降低13.63%–13.90% | 降低22.71%–22.76% | 所有模型公平应用布局；验证筛选结果 |

FastRateIF仅优化eval、timesteps=1、blend=1的路径：省去零膜初始化、stack/mean及混合运算，保留归一化电流阈值和原权重。16张验证图编码器输出逐元素相同，并完成完整质量验证。全模型channels-last曾超出数值误差门槛，因此未采用；仅decoder布局探针最大偏差约2.3e-6，通过后才完整验证。全模型torch.compile也曾因误差超门槛失败，不能将其作为已成功优化。

独立VAE实验表明重建压缩模型有质量/效率改善；最终接入VLA后，必须重新测量整个推理流程，不能把上表13%–16%的VAE提速直接当成VLA提速。以下第二部分给出接入后的实际结果。

> 归档范围：本README汇总历史服务器实验记录；仓库包含训练/评估源码和部分原始结果，不包含全部历史checkpoint、日志或逐图评分。未上传的原始文件不能通过README链接获得。`ITERATIONS.md`保留早期叙述，不应当作仍待执行的任务清单。

## 第二部分：接入VLA后的结果

LIBERO-Spatial：10 个任务，每任务 50 个官方初始状态；五条件合计 2500 回合。同一 OpenVLA-OFT checkpoint，评估 seed=42。42/43/44 指三个独立训练的 SNN 模型种子。

| 输入 | 成功回合 | 成功率 | 平均周期耗时 (ms) | 板卡能耗 (J/周期) |
|---|---:|---:|---:|---:|
| 官方VAE | 490/500 | 98.0% | 431.866 | 133.358 |
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
