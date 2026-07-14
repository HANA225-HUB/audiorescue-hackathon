# AudioRescue｜听清又听懂·智能音频急救台

两天 Vibe Coding 黑客松项目。P0 唯一主链路：

```text
噪声音频
  → 48kHz / mono / PCM16 标准化
  → DeepFilterNet 增强
  → 原轨与增强混合轨使用同一 Whisper 配置转写
  → A/B 播放、声谱图、文本差异与真实耗时
```

## 当前状态

- `v0.1-contract`：公共数据结构与 A 音频后端接口已经冻结；
- C 集成分支已实现 `process_audio` 编排、中文 CER、透明缓存、P1 隔离壳、结果持久化与集成测试；
- C 的自动测试使用确定性假后端，不代表 DeepFilterNet/Whisper 已完成真实推理验收；
- 真实 P0 闭环仍需合并 A 的音频模块与 B 的页面/可视化模块后，在 4090 演示环境执行；
- 本仓库当前为三人共享的公开仓库；只有受邀协作者可直接写入；
- 模型、外部数据集、用户音频和运行产物不会提交到 Git；
- `tests/fixtures/` 中的合成音频只用于联调，不作为比赛效果证据。

## 三人共享方式

三个人使用同一个仓库，各自在独立分支完成模块，通过 PR 合并：

| 成员 | 所有文件 |
|---|---|
| A | `core/audio_io.py`、`core/enhance.py`、`core/transcribe.py` 及对应测试 |
| B | `app.py`、`ui/`、`core/visualize.py`、`core/text_diff.py` 及对应测试 |
| C | `core/schemas.py`、`core/pipeline.py`、`core/metrics.py`、`core/cache.py`、配置、集成与文档 |

禁止两个人同时让 AI 修改同一个文件。公共契约只能由 C 修改。

## A 开工入口

1. 完整阅读 [`docs/A_BACKEND_CONTRACT_V1.md`](docs/A_BACKEND_CONTRACT_V1.md)；
2. 只从 [`core/schemas.py`](core/schemas.py) 导入公共类型与异常；
3. 使用 `tests/fixtures/dev_smoke_s01_fan.wav` 做第一次联调；
4. 先独立提交 A 的三个模块和测试，不修改 `pipeline.py`；
5. 返回调用示例、真实输出、耗时、模型位置、已知问题和回滚方式。

完整的两天执行手册已同步到 [`docs/handbook/`](docs/handbook/README.md)。若旧手册与实际代码契约冲突，以 `core/schemas.py`、`configs/app.yaml` 和 `docs/A_BACKEND_CONTRACT_V1.md` 为准。

## 服务器环境

推荐单卡 RTX 4090、8 vCPU、32GB RAM、Python 3.10、Ubuntu 22.04、PyTorch 2.5.1 + CUDA 11.8。RTX 5090 必须使用支持 Blackwell 的 PyTorch 2.7.1 + CUDA 12.8，并确认宿主驱动满足要求。

应用依赖：

```bash
apt-get update && apt-get install -y ffmpeg git libsndfile1
python -m pip install -r requirements.txt
python -m pip check
```

PyTorch 与 TorchAudio 应先按 GPU 单独安装，不写入通用 `requirements.txt`，避免 4090/5090 安装错误的 CUDA wheel。

本项目 4090 服务器已准备独立环境，登录后执行：

```bash
source /root/autodl-tmp/audiorescue_env.sh
cd "$AUDIORESCUE_ROOT"
```

需要下载学术资源或模型时，可在下载命令前执行 `source /etc/network_turbo`。该加速只影响当前 shell，不应写入项目代码。

## 配置与产物

- 冻结配置：[`configs/app.yaml`](configs/app.yaml)
- 每次任务：`outputs/{job_id}/`
- 正式本地数据：`data_local/`，不会提交
- 正式演示素材：仅经 C 审核后放入 `demo_assets/`
