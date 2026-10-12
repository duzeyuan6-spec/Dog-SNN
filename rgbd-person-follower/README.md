# RGB-D 人跟随（Genisom L1）

部署目录：`/home/nav/ljx/rgbd-person-follower`，电脑 `nav@192.168.1.50`。

## 从 GitHub 获取

```bash
git clone --branch Dog-experiment --single-branch https://github.com/duzeyuan6-spec/Dog-SNN.git
cd Dog-SNN/rgbd-person-follower
bash setup.sh
```

本工程运行于 Ubuntu 22.04 x86_64、Python 3.10，复用狗电脑已有的 PyTorch、OpenCV、NumPy、`pyorbbecsdk` 和系统 pip；`setup.sh` 仅补装 `requirements.txt` 中的依赖。`yolo26n.pt` 已随工程提供。

Genisom 原生 SDK 复用狗电脑现有安装，未打包进本仓库；上游见 [genisom_l1_sdk_old](https://github.com/zsibot/genisom_l1_sdk_old)。首次在其它机器部署时，修改 `config.json` 的 `robot.sdk_path`、机器与狗的 IP、`camera.serial`，匹配实际设备。克隆后可直接在本工程目录运行 `./run.sh`，无需移动到原部署目录。

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

默认跟随距离 3.5 米，按当前固定相机视角保留全身可见空间；前进上限 0.3 m/s，转向上限 0.5 rad/s。不倒退，大角度先转向。原目标 ID 不在、深度无效、目标跳变、超过 5.5 米或观测超过 0.45 秒时立即发零速。失锁后不挑别的人；原 ID 连续确认 3 帧可恢复，无法恢复时停止脚本、让目标重新站好再启动。

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
- 复用已有 `yolo26n.pt` 和 Ultralytics BoT-SORT，CPU 推理、相机运动补偿，首次选中央高置信目标，之后固定 ID 加位置/深度连续性检查。
- 独立进程 50 Hz 控制；视觉卡住不会持续维持旧速度。SDK最小非零速度以下发送零，停车绕过平滑。主进程退出后控制进程停车、趴下并退出。
- 接入现有 Genisom SDK，自动检查网卡/端口/反馈，输出实际电量（约每秒读取）。dry-run 的电量为 null。
- 输出 `output/latest.jpg`、`status.json`（含录制状态）、限大小运行日志、`last_run.json`、最近500帧的 `last_trace.json`，按键录制保存在 `output/recordings/`。相机与SDK只由本工程单一实例使用。

## 参数与现场校准

修改 `config.json`。`camera.serial` 必须是头部前视相机；不要切到腕部相机。`control.target_distance` 改跟随距离，`max_vx` 改速度上限，`kp_distance/kp_yaw` 改响应。`robot.lie_down_delay` 是趴下命令保持时间，默认及最低 3 秒，与现有 SDK 示例一致；`stand_delay` 最低 10 秒。

速度配置按 SDK `move` 范围校验：`max_vx` 最大 3 m/s、`max_yaw` 最大 3 rad/s。仓库默认仍为 0.3 m/s、0.5 rad/s；SDK 接受的指令范围不能保证所有地面、转弯和负载条件下都能稳定达到。

首次真实步行应在空旷地面现场确认：人在左侧时狗左转、人在远处时狗前进、靠近后停止前进。相机朝向、狗身运动引起的遮挡和实际行走响应需要现场校准。当前方案没有加入障碍物路径规划，也没有外观重识别；多人交叉可能使跟踪器串 ID，连续性检查不能提供身份保证。

机器地址：本机 `192.168.168.66` → 狗 `192.168.168.168`，UDP `43988`。已有其它控制程序时先停止它；SDK优先级可能高于手柄。独立进程看门狗覆盖视觉故障，不代替硬件急停；SDK或操作系统整体故障仍依赖机器狗自身通讯保护。

## 自检与环境

```bash
.venv/bin/python check.py
.venv/bin/python check_recording.py
```

自检不会启动相机、导入真实狗 SDK 或发运动。依赖已部署；重建时运行 `bash setup.sh`，复用系统 PyTorch 与 pyorbbecsdk，不重复下载 CUDA。

`check_recording.py` 用模拟帧验证视频解码、深度逐帧对应、按键状态切换、30 FPS 生产与低频推理轮询解耦、慢盘溢出失败、相机缺帧标记和写盘错误；Linux 上还验证无需回车、Ctrl+C 信号及终端设置恢复。

真相机录制验收（2026-10-12）：头部相机 USB 2.1，640×480 同步 RGB-D 30 FPS；同一片段持续 30.169 秒，保存 905 对帧，实测 29.964 FPS，最大帧间隔 33.374 ms，相机缺帧间隔和写盘丢帧均为 0，`fps_verified=true`。正常 YOLO 推理约 26.69 FPS，增加每次 0.2 秒延迟后约 4.08 FPS；两阶段录制仍连续完整。视频解码帧数、深度文件数、CSV 行数均为 905，深度尺寸及类型逐帧检查通过。全程未初始化真实狗 SDK、未发运动命令。狗电脑验收记录为 `output/acceptance_recording_30fps_20261012.json`；测试片段保留在该记录指向的 `output/recordings/` 中，不上传图像或视频。该结果对应上述约 30 秒测试，长时间运行仍受存储空间与持续写盘性能限制。

实现沿用本地 `初步人识别` 的 YOLO+BoT-SORT 接口；跟随流程参考 [Human-tracker-IR](https://github.com/Q-QIQI/Human-tracker-IR)，相机接入依据狗电脑现有 pyorbbecsdk 的 `sync_align.py`。本工程是 RGB-D/Genisom 适配实现，不复现原项目的红外专用模型。Ultralytics及模型沿用其上游许可。

## 实机验收（2026-10-10）

完成真相机与Genisom SDK接管测试：自动站立、等待姿态稳定、锁定ID1、跟随窗口10秒、自动零速退出。总运行27.19秒，355帧，其中151帧为有效跟踪，目标ID保持1；观测延迟P95为58.3ms。现场用户确认“方向正确，动作平稳”。电量读取从10%到9%，测试后已退出并释放控制端口。狗电脑上的详细记录为 `output/acceptance_20261010.json`，运行日志及图像未上传；长期运行和其它环境仍需在充电后测试。

随后新增的“站立时先趴下再起立”和“退出自动趴下”已通过模拟 SDK 自检，尚未实机确认；上面的实机数据对应修改前流程。
