# xiancun

按实际 GPU 利用率自动启动显存占用和持续矩阵计算的 Linux 脚本。GPU 空闲时计算，其他任务开始使用算力时让出对应 GPU，空闲后自动恢复。

**利用率为 0% 就按空闲处理。** Jupyter/Python 保留 CUDA 上下文或显存不会阻止启动，但剩余显存必须足够。每张 GPU 独立判断，其他卡忙碌不会挡住空闲卡。

## 默认行为

| 参数 | 默认值 |
| --- | --- |
| GPU 检测间隔 | 30 秒 |
| 连续空闲确认时间 | 30 秒 |
| 外部 GPU 利用率空闲阈值 | 0% |
| 同时计算的 GPU 数 | 最多 2 张 |
| 每卡张量显存目标 | 5 GiB，驱动和 CUDA 上下文另有开销 |
| GPU 轮换周期 | 24 小时 |
| 计算心跳间隔 / 超时 | 10 秒 / 120 秒 |

外部计算停止后，通常约 30–60 秒启动工作进程，再完成 PyTorch/CUDA 初始化。原部署环境实测：进程和 CUDA 上下文一直保留，仅停止计算，约 60 秒恢复；同卡外部任务开始计算后约 24 秒让出。

## 环境要求

本仓库保存实际部署的版本，默认路径与原开发机环境一致：

- Linux、`/proc`、Python 3.10 或更新版本；调度器默认使用 `/usr/bin/python3`。
- NVIDIA GPU、可工作的驱动和 `nvidia-smi`。
- 支持当前 GPU 的 CUDA 版 PyTorch。工作进程默认使用 `~/miniforge/envs/gpu-keepalive/bin/python`，可通过 `XIANCUN_PYTHON` 指定现有环境。
- Supervisor。脚本中的 `SUPERVISORD`、`SUPERVISORCTL` 默认分别为 `/etc/aibox/bin/supervisord` 和 `/etc/aibox/bin/supervisorctl`。其他 Linux 环境请先把这两个常量改为本机 `command -v supervisord`、`command -v supervisorctl` 的结果。

监测器本身只使用 Python 标准库。PyTorch 只在 GPU 工作进程中导入；驱动暂时不可用时，监测会保留并重试。

## 安装与使用

准备好上述依赖后：

```bash
git clone https://github.com/enterx-x/xiancun.git
cd xiancun
mkdir -p "$HOME/bin"
install -m 755 xiancun "$HOME/bin/xiancun"
export PATH="$HOME/bin:$PATH"

# 若使用默认 Python 环境：
xiancun start

# 或指定已经安装 CUDA 版 PyTorch 的解释器（首次 start 时保存到配置）：
# XIANCUN_PYTHON=/path/to/environment/bin/python xiancun start
```

```bash
xiancun status       # 监测状态、等待原因、逐卡计算心跳
xiancun log          # 近期调度、计算和服务日志
xiancun start        # 启用 / 恢复监测，可重复执行
xiancun pause 10     # 立即释放自己的 GPU，暂停 10 分钟后继续监测
xiancun stop         # 彻底停止并禁用自动启动；之后需再次 start
```

自定义显存、轮换周期、检测间隔和卡数：

```bash
xiancun start 5 24 --interval 30 --idle-seconds 30 --gpu-count 2
```

`start` 的两个位置参数分别是每卡显存 GiB 和轮换小时数。检测间隔最多 60 秒，空闲确认时间最多 300 秒。需要立即占满显存的训练，可以先 `xiancun pause 10` 再启动；暂停结束后仍按各卡实际利用率判断是否恢复。

## 判断与恢复机制

1. 逐卡读取总利用率和剩余显存。外部利用率大于 0% 时，让出这张卡。
2. 其他进程与 xiancun 共用 GPU 时，用 `nvidia-smi pmon` 区分外部负载与自身负载。
3. 如果驱动不提供有效逐进程指标，逐卡暂停自身计算 2 秒，测量外部负载，然后恢复。双卡运行时另一张卡继续计算；单卡运行时会出现这段短暂停顿。
4. 独立 Supervisor 实例负责在调度器退出后重启。工作进程退出或计算心跳超时后，调度器会清理并重试。
5. PID、进程启动时间和环境标记用于识别自己的工作进程。单实例锁避免重复运行，日志按 5 MiB 轮转。

采样失败不视为空闲；已有计算保留，监测继续重试。显存不足时等待更多可用显存，默认启动一张卡需要约 6 GiB 空闲空间。

状态默认保存在 `~/.cache/xiancun/`，包括 `config.json`、`status.json`、`monitor.log`、`compute.log`、`service.log`。可通过 `XIANCUN_STATE_DIR` 改位置；使用自定义位置时，各次管理命令应保持相同设置。

## 机器启动后恢复

`xiancun start` 启动后台服务并保存启用标记，**不会自动安装系统启动项**。要在机器重启后恢复，将以下命令加入机器实际执行的用户启动入口：

```bash
"$HOME/bin/xiancun" ensure
```

`ensure` 只在已有启用标记时恢复服务，因此执行 `stop` 后不会被这个入口重新拉起。

原开发机的平台启动脚本会加载 `~/.bashrc`，使用了下面的启动段，放在 `.bashrc` 的非交互 `return` 之前：

```bash
if [[ "${BASH_SOURCE[1]:-}" == */entrypoint.sh || $- == *i* ]]; then
    if [ -x "$HOME/bin/xiancun" ] && [ -f "$HOME/.cache/xiancun/enabled" ]; then
        (nohup "$HOME/bin/xiancun" ensure >> "$HOME/.cache/xiancun/bootstrap.log" 2>&1 < /dev/null &)
    fi
fi
```

其他平台请使用实际支持的启动入口；只修改 `.bashrc` 不代表所有机器都能无人登录自启动。

## 测试

```bash
python3 -m unittest discover -s tests -v
```

22 项测试不需要 GPU 或 PyTorch，覆盖驻留 CUDA 进程、逐卡空闲判断、同卡外部计算让出、缺失指标回退、失败重试、心跳超时、轮换、单卡、显存不足和 PID 复用保护。

## 已知边界

- 空闲按采样时的利用率判断，训练较长的数据准备间隙也可能被视为空闲。
- Supervisor 能恢复退出的调度器，但其自身被杀或调度器卡住不退出时，没有独立定时巡检兜底。
- GPU/驱动损坏、环境缺失、磁盘写入失败等问题需要修复运行环境。
- 脚本不能控制平台强制关机或资源回收；完整机器重启未在原环境中实测，仅验证过启动入口恢复。
