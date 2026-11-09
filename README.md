# xiancun

Linux NVIDIA GPU 空闲时自动进行矩阵计算，其他任务使用 GPU 时让出对应卡，空闲后恢复。用于需要保持 GPU 活跃的开发机；能否避免空闲关机取决于平台的判断规则，不能阻止平台强制回收或固定时限关机。

## 快速启动

在开发机**普通终端**执行，无需编辑代码、指定固定路径或提前安装 Supervisor/PyTorch：

```bash
git clone https://github.com/00xiaobiao/xiancun.git
cd xiancun
./xiancun start
```

首次运行会检查 GPU 和真实 CUDA 运算，自动寻找当前 Python、PATH、仓库 `.venv`、用户目录下的 Conda 环境。如果没有可用 CUDA PyTorch，自动在 `~/.cache/xiancun/runtime` 创建独立虚拟环境并下载依赖，不修改已有训练环境。首次下载可能需要数 GB 空间和数分钟，需要能访问 PyTorch 官方下载站。

默认最多使用 **1 张空闲 GPU、1 GiB 张量显存**，CUDA 上下文另有开销。后台运行后可关闭终端。GPU 连续空闲 30 秒后启动计算，再等待 PyTorch 初始化；用状态中的 `computing` 和最近计算心跳确认运行成功。

```bash
./xiancun status       # 状态、等待原因、GPU 计算心跳
./xiancun log          # 近期日志
./xiancun doctor       # 诊断设备、驱动、CUDA Python；不安装依赖
./xiancun pause 10     # 释放 GPU，暂停 10 分钟后继续
./xiancun stop         # 停止后台监测并释放 GPU
./xiancun start        # 恢复监测，可重复执行
```

## 基础环境

支持 Linux NVIDIA GPU，要求 Python 3.10+、可读的 `/proc`、已分配给当前环境的 GPU、正常的宿主机 NVIDIA 驱动及 `nvidia-smi`。这些基础条件通常由开发机平台镜像提供；脚本不能凭空分配 GPU，也不会在容器里重装宿主机驱动。

自动安装依赖使用 Python 的 `venv` 和 `pip`。如果极简 Debian/Ubuntu 镜像缺少 `venv`，先执行 `sudo apt-get install python3-venv`；其他系统使用对应软件包或平台 Python 镜像。无需另外下载完整 CUDA Toolkit。

## GPU 不可用时怎么处理

```bash
nvidia-smi
./xiancun doctor
```

`start` 在 GPU 检查失败时直接报错，不会启动一个假装保活的后台进程，也不会先下载大型依赖。诊断会给出对应原因：

| 现象 | 处理办法 |
| --- | --- |
| 普通终端正常，AI 沙箱里没有 `/dev/nvidia*` | 让执行工具获得沙箱外 GPU 访问权限，或直接在普通终端启动脚本。每个开发机和执行环境的权限需要分别授予，脚本本身不能修改沙箱策略。 |
| 容器没有 GPU 设备，但能读到 `/proc/driver/nvidia/version` | 宿主机驱动已加载；需要在平台上分配 GPU，或在容器启动时挂载 GPU。不要把这种情况误判为驱动损坏。 |
| `nvidia-smi` 不存在 | 使用包含 NVIDIA 用户态工具的 GPU 镜像，或在宿主机正确安装驱动。 |
| 设备存在但提示 driver/library version mismatch | 由平台管理员统一宿主机驱动与用户态库版本，必要时重启宿主机。 |
| `nvidia-smi` 正常但 CUDA 运算失败 | 检查是否误装 CPU 版 PyTorch、GPU 架构与 wheel 是否兼容，以及 `CUDA_VISIBLE_DEVICES`、容器 `NVIDIA_DRIVER_CAPABILITIES` 是否允许 `compute,utility`。 |

自管 Docker 需要先在**宿主机**安装驱动并配置 NVIDIA Container Toolkit，然后验证：

```bash
sudo docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
```

实际业务容器也需要通过 `--gpus all` 或对应平台的 GPU 分配选项启动。已有容器内的普通 Python 进程不能自行补授宿主机设备权限。参见 [NVIDIA 官方容器验证文档](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/sample-workload.html)。

## 参数与已有 Python 环境

```bash
# 最多两张空闲卡，每卡 5 GiB，24 小时轮换
./xiancun start 5 24 --gpu-count 2

# 自定义采样和空闲确认时间
./xiancun start 1 24 --interval 10 --idle-seconds 30 --gpu-count 1

# 前台运行，Ctrl+C 停止
./xiancun run

# 禁止自动下载依赖，适用于离线环境
./xiancun start --no-install

# 使用已激活的 CUDA Python，不需要修改代码
XIANCUN_PYTHON="$(command -v python)" ./xiancun start
```

`start` 和 `run` 的两个位置参数分别是每卡张量显存 GiB、轮换周期小时数。检测间隔最多 60 秒，空闲确认最多 300 秒。每次 `start` 保存参数；后续省略参数时沿用已有配置。旧版本保存的失效 Python 路径会自动重新检测。

首次自动安装默认使用官方 `cu128` wheel 源（驱动主版本 570+），更旧驱动尝试 `cu118`；安装后必须通过实际 CUDA 运算检查。不同 GPU 架构、Python 版本和老驱动不一定适配默认 wheel，可通过 `XIANCUN_TORCH_INDEX_URL` 指定兼容源，参考 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/) 及 [历史版本说明](https://pytorch.org/get-started/previous-versions/)。

其他环境变量：

| 变量 | 用途 |
| --- | --- |
| `XIANCUN_STATE_DIR` | 修改状态与日志目录；所有管理命令须使用相同设置。 |
| `XIANCUN_PYTHON` | 明确指定 CUDA Python；指定后检查失败会报错，不偷偷改用其他解释器。 |
| `XIANCUN_TORCH_INDEX_URL` | 修改独立环境安装 torch 时使用的 wheel 源。 |
| `XIANCUN_BACKEND=builtin` | 强制使用内置后台进程，便于没有 Supervisor 或无法创建 Unix socket 的容器。 |
| `XIANCUN_NVIDIA_SMI` | 指定不在 PATH 中的 `nvidia-smi`。 |

脚本遵守 `CUDA_VISIBLE_DEVICES` 指定的 GPU 索引或 GPU UUID。MIG 分区和非 NVIDIA GPU 尚未支持。

## 调度与后台运行

- 外部 GPU 利用率为 0% 时按空闲处理；只保留 CUDA 上下文或占用显存不会阻止启动。
- 每张卡独立判断，其他卡忙碌不会挡住空闲卡。剩余显存必须大于张量目标加 1 GiB 预留。
- 共用 GPU 时通过 `nvidia-smi pmon` 区分外部计算和自身计算；指标缺失时逐卡暂停自身计算约 2 秒来测量外部负载。
- 有可用 Supervisor 时启动独立实例，不影响平台已有的 Jupyter/SSH 服务；未安装或无法启动时自动使用内置后台进程。
- 工作进程退出或心跳超时后会重试。有 Supervisor 时调度器退出后也会重启；内置后台模式下调度器自身退出后，需要再次执行 `start`。
- 单实例锁、PID 和启动时间避免重复运行及 PID 复用误杀。运行后不要移动或删除仓库及所用 Python 环境。

`start` 保持后台运行，但**不会自动安装机器重启后的系统启动项**。确需重启恢复时，在平台实际支持的启动入口执行仓库里的 `xiancun ensure`；`stop` 后 `ensure` 不会重新拉起服务。

VS Code 可打开仓库后执行“终端 → 运行任务 → xiancun: 运行 GPU 保活”；该任务使用前台模式，需保留终端，关闭或 Ctrl+C 会停止。

## 测试与验证

```bash
python3 -m unittest discover -s tests -v
```

测试覆盖空闲判定、外部训练让出、采样失败重试、轮换、心跳恢复、PID 保护、设备隔离诊断、CPU PyTorch 拒绝、独立依赖安装、GPU 可见性和后台启停。单元测试不需要 GPU；后台进程集成测试需要同一 PID 命名空间下可读的 `/proc`，不满足时明确跳过。

当前开发机已在授权访问设备后通过 RTX 5090 实机验证：前台、内置后台、Supervisor 均进入 `computing`；两种后台模式均验证了重复启动、暂停恢复、参数修改和停止释放。33 项测试在可读 `/proc` 的环境全部通过。平台空闲关机规则、其他 GPU 型号、全新镜像依赖下载和整机重启恢复需要在目标环境中确认。

本项目在 [enterx-x/xiancun](https://github.com/enterx-x/xiancun) 的提交 `11409cf` 基础上增加通用启动、依赖准备、设备诊断和前台运行支持，保留原始 Git 历史。
