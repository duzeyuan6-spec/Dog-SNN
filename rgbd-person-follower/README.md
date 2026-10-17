# RGB-D 人跟随（Genisom L1）

原电脑：`nav@192.168.1.50`，目录 `/home/nav/ljx/rgbd-person-follower`。新 Jetson：`wheeltec@192.168.1.87`，目录 `/home/wheeltec/ljx/rgbd-person-follower`。

## 从 GitHub 获取

```bash
git clone --branch Dog-experiment --single-branch https://github.com/duzeyuan6-spec/Dog-SNN.git
cd Dog-SNN/rgbd-person-follower
bash setup.sh
```

本工程运行于 Ubuntu 22.04、Python 3.10，支持 x86_64 CPU 和 Jetson ARM64 CUDA，复用电脑已有的 PyTorch、OpenCV 和系统 pip。x86_64 的 `setup.sh` 补装 `requirements.txt`；ARM64 使用 `requirements-jetson.txt`，在项目虚拟环境内补齐 NumPy/科学计算依赖和固定版 Orbbec Python SDK，不替换系统的 CUDA/PyTorch/OpenCV。`yolo26n.pt` 和轻量人物重识别权重 `osnet_x0_25_msmt17.ts` 已随工程提供，运行时无需下载模型或安装新依赖。OSNet 来源、许可与导出方法见 `REID_MODEL.md`。

Genisom 原生 SDK 复用狗电脑现有安装，未打包进本仓库；上游见 [genisom_l1_sdk_old](https://github.com/zsibot/genisom_l1_sdk_old)。首次在其它机器部署时，修改 `config.json` 的 `robot.sdk_path`、机器与狗的 IP、`camera.serial`，匹配实际设备。克隆后可直接在本工程目录运行 `./run.sh`，无需移动到原部署目录。

## Jetson Orin NX 部署

新机 `wheeltec@192.168.1.87`，工程目录 `/home/wheeltec/ljx/rgbd-person-follower`。使用 Ubuntu 22.04 / JetPack 6.2 / Python 3.10，SDK 原生库必须是 `lib/zsl-1/aarch64`；x86_64 的 `.venv` 和 `.so` 不能直接复制。

新机 `config.json` 的 `vision.device` 设为 `"cuda:0"`，`robot.sdk_path` 设为 `/home/wheeltec/ljx/genisom_l1_sdk_old/lib/zsl-1/aarch64`。源码默认仍为 CPU，支持旧电脑。新机首次实测配置为跟随距离 1.5 m、最高前进速度 0.3 m/s、最高转向速度 0.5 rad/s。启动会先预热跟踪器和 OSNet、重置暖机跟踪状态，再打开相机和初始化控制进程；CUDA OSNet 关闭运行时形状优化，避免人数变化时出现秒级停顿。默认 Ultralytics 设置保存在工程的 `output/Ultralytics/`；若显式设置 `YOLO_CONFIG_DIR` 则尊重该路径。

在新机终端执行一次 `sudo bash /home/wheeltec/ljx/rgbd-person-follower/setup_dog_network.sh`，新增 `dog-sdk` 网络配置，绑定 `enP8p1s0` 为 `192.168.168.66/24`，无默认路由、网关或 DNS，保留 Wi-Fi 和原有连接配置；脚本不激活接口、不修改狗端。之后可用 `bash setup_dog_network.sh --check` 核对持久设置。相同 `.66` 的控制电脑不能同时接入同一个有线网络；正常停车退出后再换狗网线。

脚本还会把 SDK 固定端口 `43988` 合并进系统的 `net.ipv4.ip_local_reserved_ports`，保留已有范围并持久保存，防止动态端口服务在重启后抢占它。若该端口当前属于 `rpc.statd`，只重启 `rpc-statd.service` 释放端口，保留 NFS 服务和导出；其它占用者不会被停止。新机本次遇到的冲突来自默认动态端口的 NFS 状态监视服务，检查时无 NFS 挂载、客户端连接或 RPC 使用流量。`--check` 同时验证端口已保留；它不要求端口当前空闲，因为跟随运行时 SDK 会占用它。

新机狗端通信验收（2026-10-17）：有线接口自动使用 `.66/24`，到狗 `.168` 的 ping 平均约 0.44 ms；ARM SDK 成功连接并读取 4 次反馈，电量为 86%、86%、86%、85%，控制模式均为 0。全程只调用连接、状态和电量读取接口，没有调用 `move`、`standUp` 或 `lieDown`；SDK 正常析构后释放端口。修复后 NFS 状态服务和服务端均正常、原有 3 个导出保留。报告为新机 `output/deployment/sdk-feedback-acceptance.json`；这不等同于完成实机站立或行走验收。

机器狗 SDK 连接仍使用 `192.168.168.66:43988 -> 192.168.168.168`。完整跟随还需将头部相机 USB 接到新机，优先使用 USB 3.x 端口；SSH 热键录制使用 `ssh -t wheeltec@192.168.1.87`。

新机离线验收（2026-10-17）：ARM SDK 导入、Orbbec API、GPU NMS、控制/身份恢复/退出自检及模拟录制检查通过，未初始化真实机器狗 SDK。用原始 1069 帧彩色视频重新运行 CUDA YOLO+OSNet，并按 IoU 配对复用先前记录的躯干深度中位数（不含原始深度文件或实机测距）；约 31.90 秒重新认回目标，此后 113 帧保持跟随，处理延迟中位数 76.0 ms、P95 98.8 ms、最大 151.9 ms。回归检查同时确认暖机未改变首帧激活或初次锁定时机。

写盘压测使用原录像 RGB 和固定 2 m 的合成深度，以 30 Hz 生产 300 对帧；GPU 视觉约 11.95 FPS 时，保存 300 帧视频及 300 个深度文件，实测 30.014 FPS，零源帧丢失、零队列丢帧。此测试只验证并发写盘吞吐，未连接真相机，不用于评价实测深度或身份识别。报告及可重跑脚本在新机 `output/deployment/`；真相机同步、深度对齐、30 FPS 和真实站立跟随须接线后另验收。

新机真相机验收（2026-10-17）：头部 Gemini 336L 实际 USB 2 / 480 Mbps，640×480 同步 RGB-D。连续录制 40.148 秒，保存 1204 对帧，实测 29.964 FPS、最大帧间隔 33.374 ms，无源帧和队列丢失；视频逐帧解码数、深度文件数、CSV 行数均为 1204，全部深度 CRC 与对应采集帧一致。正常视觉约 22.3 FPS，额外增加每次 0.2 秒等待后约 3.5 FPS，两阶段录制均为 29.964 FPS。两路设备时间戳差最大 0.101 ms，深度非零比例中位数约 71.1%。已检查 RGB/软件 D2C 深度叠图；未使用实物标尺校准绝对测距精度。该相机吞吐测试画面无人，人物跟随结果见下面实机验收；此次全程未导入或初始化狗 SDK。报告及可重跑脚本为新机 `output/deployment/live-camera-acceptance.json`、`live_camera_acceptance.py`，原始测试片段路径在报告中，不上传图像、视频或深度文件。

新机正式脚本验收（2026-10-17）：真实终端运行 `./run.sh --dry-run`，验证 `1 → a → 1 → 连续三次 Ctrl+C`，分别完整保存 181 和 90 对 RGB-D，均为 29.964 FPS、无丢帧，视频解码数、深度文件数和 CSV 行数一致。录制中 Ctrl+C 约 1.14 秒退出，退出码 0，图像处理后 age 的 P95 为 92.7 ms。此测试不初始化狗 SDK；真实起立、跟随和趴下见下面实机验收。报告和可重跑脚本为新机 `output/deployment/live-cli-smoke.json`、`live_cli_smoke.py`。

新机实机跟随验收（2026-10-17）：正常启动起立并等待姿态稳定，锁定目标 ID 1，完成约 10 秒跟随后自动停车、趴下，退出码 0；现场用户确认“跟随平稳，已正常趴下”。134 帧有效目标均为同一 ID，仅短暂 1 帧漏检后重新确认恢复。速度命令峰值为前进 0.294 m/s、转向 0.374 rad/s，低于新机配置的 0.3 m/s 和 0.5 rad/s；图像处理后 age 的 P95 为 112.3 ms。同期保存 643 对 RGB-D，实测 29.964 FPS，无源帧或写盘丢帧，643 帧视频、643 个深度文件及 CSV 行逐一检查通过。电量记录为 84% → 83%，减少 1 个百分点。程序、SDK 端口和相机均已正常释放。报告及完整轨迹为新机 `output/deployment/live-motion-acceptance.json`、`live-motion-last-trace.json`，原始录制目录在报告中。该结果对应本次约 10 秒低速跟随，不代表所有场景和更高速度的验收。

## 一键启动

```bash
cd /home/nav/ljx/rgbd-person-follower
./run.sh
```

**上面命令会实际控制狗。** 启动时若已趴下，直接起立；若已站立，先零速停车、趴下并保持命令 3 秒，再起立。起立后等待至少 10 秒姿态稳定，才开始找人。目标站在头部相机正前方、5 米内，连续确认约 0.6 秒后锁定跟随。趴着看不到人不会阻止启动站立。

Ctrl+C、SIGTERM 和限时结束都会先零速停车，再发送趴下命令并保持 3 秒，之后退出；重复 Ctrl+C 不会打断这个收尾。正常收尾约 3.5 秒；SDK 尚未允许姿态切换时，最多额外重试 2 秒。无反馈或 SDK 拒绝会明确记录失败，不能保证通讯断开后的动作。`--dry-run` 不发起立或趴下命令。

主进程与控制进程的信号回调只标记退出请求，带锁的停止事件在正常循环中设置，避免 Ctrl+C 打断事件锁操作时发生重入死锁。`check.py` 包含持有事件锁时连续收到 SIGINT/SIGTERM 的回归检查，以及模拟 SDK 的趴下保持与重复信号测试。

退出修复验收（2026-10-12）：旧回调的持锁死锁已在无 SDK 的隔离进程中复现，修复后的持锁信号测试通过。真相机、真实终端、`--dry-run` 录制时连续发送 3 次 Ctrl+C，约 0.44 秒退出并保存 31 对 RGB-D，实测 29.964 FPS，无丢帧，退出码 0；记录为 `output/acceptance_ctrl_c_20261012.json`。起立等待及趴下保持期间收到 SIGINT/SIGTERM 的动作收尾通过模拟 SDK 验证，此次未重新进行实狗趴下测试。

连接狗后、起立前读取并显示启动电量；Ctrl+C 等正常收尾在趴下后再次读取结束电量，显示本次耗电，例如 `本次耗电：3 个百分点（启动 80% → 结束 77%）`。该差值包含启动和趴下过程，依据 SDK 的整数百分比，不是 Wh；不足 1 个百分点的消耗可能显示 0。电量上升会单独提示，不强行计为 0；无反馈、读取无效或模拟模式显示无法统计。启动值、结束值、差值同时写入 `output/last_run.json` 的 `battery_start_percent`、`battery_end_percent`、`battery_used_percent_points`（上升时差值为负，无有效读数时为 null）。

电量记录开关位于 `config.json` 的 `robot.record_battery`：`true` 开启（默认），`false` 关闭，修改后重新启动程序生效。关闭时跳过 SDK 电量读取，终端和本次新增日志不显示电量或耗电统计，`status.json`、`last_trace.json`、`last_run.json` 不写入电量字段。跟随、起立趴下和 RGB-D 录制照常运行；已有日志保留。旧配置未写该项时按开启处理。

默认跟随距离 3.5 米，按当前固定相机视角保留全身可见空间；前进上限 0.3 m/s，转向上限 0.5 rad/s。不倒退，大角度先转向。目标丢失、外观不符/有竞争目标、深度无效、超过 5.5 米或观测超过 0.45 秒时立即发零速。人物返场即使换了跟踪 ID，也会用 OSNet 比对原目标外观，连续确认至少 3 帧且约 0.6 秒后恢复；确认期间保持停车。原 ID 短暂丢帧（不超过 0.45 秒）且位置连续时，确认 3 帧恢复。换位置返场按连续候选的位置检查，不会一直卡在丢失前的位置。

只测相机、识人和模拟控制，完全不初始化狗 SDK：

```bash
./run.sh --dry-run --seconds 30
```

限制首次跟随开始后最多运行10秒（站立和选人等待不占这10秒）：`./run.sh --motion-seconds 10 --seconds 45`。

## 按键录制 RGB-D

在运行 `./run.sh` 的终端中，按 **`1`** 开始录制，按 **`a`** 停止并保存，均无需回车。录制期间再按 `1` 不会重复开启；看到 `Saved recording` 后可再按 `1` 录制下一段。按 `a` 只结束录制，跟随继续。Ctrl+C、SIGTERM、限时结束或异常退出也会收尾当前录制，停车和趴下流程照常执行。后台运行或没有交互终端时不启用按键；SSH 连接请使用 `ssh -t nav@192.168.1.50`。

每段保存到 `output/recordings/日期时间_唯一后缀/`，不会覆盖已有片段，也不会上传到 Git：

- `color.avi`：原始彩色画面，MJPG 编码，无检测框叠加。
- `depth/000000.npy` 起：与彩色视频逐帧对应的深度，已对齐到彩色图，`float32`，单位米，保留无效值 `0`/`NaN`。
- `timestamps.csv`：视频帧索引、相机帧序号、主机单调时钟时间、设备时间（均为秒），以及测距使用的 `fx/cx`。第 0 帧对应 `000000.npy`；单调时间不是日历时间。
- `metadata.json`：完成/失败状态、分辨率、相机序列号、帧数、实测帧率 `measured_fps`、相机缺帧间隔数 `source_dropped_frames`、写盘丢帧数 `dropped_pairs` 和帧率验收结果 `fps_verified`。

读取单帧深度：`depth_m = numpy.load("output/recordings/片段目录/depth/000000.npy", allow_pickle=False)`；像素 `(x, y)` 的距离为 `depth_m[y, x]` 米。

相机采集线程将每对已同步、对齐的 RGB-D 直接交给后台写盘，跟随推理只取最新帧，推理变慢不会减少录制采样。默认严格请求 640×480、30 FPS，无法启动这个模式时直接报错，不静默回退到 15 FPS 或较小彩色分辨率。AVI 使用 30 FPS 封装，真实帧率以设备时间戳为准，不复制或插值凑帧。

后台最多缓存 32 对帧（30 FPS 下约 1 秒）；持续写盘过慢导致队列溢出时会停止该片段、标记失败并触发程序正常停车收尾。验收标记 `fps_verified=true` 要求帧数完整、无相机缺帧间隔、实测帧率偏差不超过 2%、最大间隔不超过 1.5 帧；缺帧或过短无法验证的片段会明确提示。640×480、30 FPS 的未压缩深度约占 2.2 GB/分钟，另加视频空间。写盘失败无法挽回未写入的数据。

## 已实现

- 指定头部 Orbbec Gemini 336L（SN `CPC936300038`），640×480、30 FPS，彩色图与深度真实 D2C 对齐。按设备实时深度单位和内参计算距离与方位；跟随消费最新帧，录制消费每一对采集帧。
- 复用已有 `yolo26n.pt` 和 Ultralytics BoT-SORT，支持 CPU/CUDA 推理、相机运动补偿，首次选中央高置信目标，之后用 OSNet x0.25（MSMT17）验证人物外观，同时检查位置/深度连续性。首次外观锚点固定，最多保存 8 个可信近期视角；只有正常稳定跟随、较高置信、与其它人框无重叠且外观分数充分时更新，不在重新寻找期间学习。相同跟踪 ID 也必须通过外观比对。YOLO26 默认跳过 NMS；在分配 ID 前额外对人框做 IoU 0.5 去重，减少重复框造成的 ID 分裂，保持现有检测头。
- 独立进程 50 Hz 控制；视觉卡住不会持续维持旧速度。SDK最小非零速度以下发送零，停车绕过平滑。主进程退出后控制进程停车、趴下并退出。
- 接入现有 Genisom SDK，自动检查网卡/端口/反馈，输出实际电量（约每秒读取）。dry-run 的电量为 null。
- 输出 `output/latest.jpg`、`status.json`（含录制状态）、限大小运行日志、`last_run.json`、最近500帧的 `last_trace.json`，按键录制保存在 `output/recordings/`。相机与SDK只由本工程单一实例使用。

## 参数与现场校准

修改 `config.json`。`camera.serial` 必须是头部前视相机；不要切到腕部相机。`control.target_distance` 改跟随距离，`max_vx` 改速度上限，`kp_distance/kp_yaw` 改响应。`robot.lie_down_delay` 是趴下命令保持时间，默认及最低 3 秒，与现有 SDK 示例一致；`stand_delay` 最低 10 秒。

`vision.device` 控制 YOLO 和 OSNet 的统一推理设备，未设置时为 `"cpu"`；Jetson 设置为 `"cuda:0"`（`"cuda"` 等同于 `"cuda:0"`，多 GPU 可指定 `"cuda:1"` 等编号）。需要已经安装与设备匹配的 CUDA 版 PyTorch；指定的 CUDA 不可用时启动会明确报错，不自动退回 CPU。原 CPU 电脑无需修改配置。

`vision.reid_similarity` 默认 `0.7`，是重新认回目标的最低余弦相似度；已确认的同 ID 且位置/深度连续时，持续跟踪门限低 `0.05`（默认 `0.65`），避免在门限附近反复停走，换 ID 或失锁恢复仍使用完整门限。`vision.reid_margin` 默认 `0.1`，是最佳和次佳候选的最小分差。低于阈值显示 `target_unverified`，分差不足显示 `target_ambiguous`，确认中显示 `confirming_target`。这些分数不是识别正确率；不要为消除停车盲目降低阈值。当前初始外观锚点还要求至少 `0.5` 相似度，限制近期视角库漂移。旧配置缺少新参数时使用上述默认值，启动时必须能加载本地 ReID 权重。

速度配置按 SDK `move` 范围校验：`max_vx` 最大 3 m/s、`max_yaw` 最大 3 rad/s。仓库默认仍为 0.3 m/s、0.5 rad/s；SDK 接受的指令范围不能保证所有地面、转弯和负载条件下都能稳定达到。

首次真实步行应在空旷地面现场确认：人在左侧时狗左转、人在远处时狗前进、靠近后停止前进。相机朝向、狗身运动引起的遮挡和实际行走响应需要现场校准。当前方案没有加入障碍物路径规划。OSNet 外观比对能恢复换 ID 的原目标，但相似穿着、严重遮挡和多人交叉仍可能造成误识别或拒绝恢复，不能保证永远识别正确；高度重叠的人框也可能被去重抑制。

机器地址：本机 `192.168.168.66` → 狗 `192.168.168.168`，UDP `43988`。已有其它控制程序时先停止它；SDK优先级可能高于手柄。独立进程看门狗覆盖视觉故障，不代替硬件急停；SDK或操作系统整体故障仍依赖机器狗自身通讯保护。

## 自检与环境

```bash
.venv/bin/python check.py
.venv/bin/python check_recording.py
```

自检不会启动相机、导入真实狗 SDK 或发运动。依赖已部署；重建时运行 `bash setup.sh`，复用系统 PyTorch 与 pyorbbecsdk，不重复下载 CUDA。

`check_recording.py` 用模拟帧验证视频解码、深度逐帧对应、按键状态切换、30 FPS 生产与低频推理轮询解耦、慢盘溢出失败、相机缺帧标记和写盘错误；Linux 上还验证无需回车、Ctrl+C 信号及终端设置恢复。

返场恢复验收（2026-10-13）：使用用户已录制的 35.64 秒、1069 对 RGB-D，逐帧运行真实 YOLO + OSNet 推理。旧版人物返场后变成新 ID，持续等待旧 ID；修复版在视频约 31.94 秒完成重新锁定，此后 112 帧直到录像结束均有效跟随原人物，人物离开期间无有效跟随输出。整段未跟随其它跟踪 ID；另按约 10 FPS 抽帧重新运行检测/ReID，约 32.14 秒重新找回，最终判定逻辑在此后 36 个处理帧持续跟随。片段没有真实多人交叉，因此不代表多人识别准确率。推理延迟中位数 83.3 ms、P95 131.3 ms。报告 `output/reacquire_probe_20261013/recovery-acceptance.json`，带框回放 `recovery-preview.mp4`；仅离线回放，没有启相机或狗 SDK。硬件无关自检另覆盖新 ID、相同 ID 被不同外观替代、相似候选歧义、确认中断、陈旧推理不更新身份及位置跳变后的恢复。

真相机录制验收（2026-10-12）：头部相机 USB 2.1，640×480 同步 RGB-D 30 FPS；同一片段持续 30.169 秒，保存 905 对帧，实测 29.964 FPS，最大帧间隔 33.374 ms，相机缺帧间隔和写盘丢帧均为 0，`fps_verified=true`。正常 YOLO 推理约 26.69 FPS，增加每次 0.2 秒延迟后约 4.08 FPS；两阶段录制仍连续完整。视频解码帧数、深度文件数、CSV 行数均为 905，深度尺寸及类型逐帧检查通过。全程未初始化真实狗 SDK、未发运动命令。狗电脑验收记录为 `output/acceptance_recording_30fps_20261012.json`；测试片段保留在该记录指向的 `output/recordings/` 中，不上传图像或视频。该结果对应上述约 30 秒测试，长时间运行仍受存储空间与持续写盘性能限制。

实现沿用本地 `初步人识别` 的 YOLO+BoT-SORT 接口；跟随流程参考 [Human-tracker-IR](https://github.com/Q-QIQI/Human-tracker-IR)，相机接入依据狗电脑现有 pyorbbecsdk 的 `sync_align.py`。本工程是 RGB-D/Genisom 适配实现，不复现原项目的红外专用模型。Ultralytics及模型沿用其上游许可。

## 实机验收（2026-10-10）

完成真相机与Genisom SDK接管测试：自动站立、等待姿态稳定、锁定ID1、跟随窗口10秒、自动零速退出。总运行27.19秒，355帧，其中151帧为有效跟踪，目标ID保持1；观测延迟P95为58.3ms。现场用户确认“方向正确，动作平稳”。电量读取从10%到9%，测试后已退出并释放控制端口。狗电脑上的详细记录为 `output/acceptance_20261010.json`，运行日志及图像未上传；长期运行和其它环境仍需在充电后测试。

随后新增的“站立时先趴下再起立”和“退出自动趴下”已通过模拟 SDK 自检，尚未实机确认；上面的实机数据对应修改前流程。
