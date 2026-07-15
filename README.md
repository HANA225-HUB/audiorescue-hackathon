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
- 当前冻结候选 heads：C PR #1 `5f5addfa504164c3463a512a98e8b6878c091de7`、A PR #8 `31b35aaa1bfd523c29b6d53312b883b6710a8caf`、B PR #5 `2650d5bd5d3c968012be28a70b2799bfe87da467`、协作文档 PR #7 `934c44ba7a542c26af4af12c4567e59c5c237c04`；
- 四方只读组合验收已通过：无冲突，targeted 185 OK / 1 skipped，full 285 OK / 1 skipped，`py_compile`、`git diff --check`、repo-safety、fixture 浏览器、隐私扫描和 UI-core 契约检查均通过；PR #5/#8/#7 仍未合并，PR #8 与公开文档 PR #9 仍为 draft，不能宣称 main 已包含或比赛 ready；
- 本仓库当前为三人共享的公开仓库；只有受邀协作者可直接写入；
- 模型、外部数据集、用户音频和运行产物不会提交到 Git；
- `tests/fixtures/` 中的合成音频只用于联调，不作为比赛效果证据。

## 三人共享方式

三个人使用同一个仓库，各自在独立分支完成模块，通过 PR 合并：

| 成员 | 所有文件 |
|---|---|
| A | `core/audio_io.py`、`core/enhance.py`、`core/transcribe.py` 及对应测试 |
| B | `app.py`、`ui/`、`core/visualize.py`、`core/text_diff.py` 及对应测试 |
| C | `core/schemas.py`、`core/pipeline.py`、`core/metrics.py`、`core/cache.py`、`scripts/`、配置、集成与文档 |

禁止两个人同时让 AI 修改同一个文件。公共契约只能由 C 修改。

## A 开工入口

1. 完整阅读 [`docs/A_BACKEND_CONTRACT_V1.md`](docs/A_BACKEND_CONTRACT_V1.md)；
2. 只从 [`core/schemas.py`](core/schemas.py) 导入公共类型与异常；
3. 使用不含人声的 `tests/fixtures/dev_smoke_s01_fan.wav` 验证音频 I/O/增强，再用私有台账中已授权的短样例验证双路 ASR；
4. 先独立提交 A 的三个模块和测试，不修改 `pipeline.py`；
5. 只返回版本、指纹前缀、耗时、通过状态、脱敏问题摘要和回滚方式；原始输出、路径、模型位置与素材信息仅记录在未跟踪的本地私有台账中。

完整的两天执行手册已同步到 [`docs/handbook/`](docs/handbook/README.md)。若旧手册与实际代码契约冲突，以 `core/schemas.py`、`configs/app.yaml` 和 `docs/A_BACKEND_CONTRACT_V1.md` 为准。

## 离线运行入口

比赛现场以本地、断网、生产 fixture 关闭为主链路。完整步骤见 [`docs/OFFLINE_DEMO_RUNBOOK.md`](docs/OFFLINE_DEMO_RUNBOOK.md)。

环境原则：

- Python 使用 3.10 或 3.11；
- PyTorch 与 TorchAudio 必须同版本、同 CPU/CUDA 来源安装；
- `requirements.txt` 只固定项目层依赖，torch/torchaudio 按机器环境单独准备；
- FFmpeg 必须在当前 shell 的 `PATH` 中可用；
- DeepFilterNet3、Whisper `base`、Gradio、soundfile 在联网准备阶段完成导入和预热；
- GPU 只作为环境加速选择，不是通用要求；不得把某台服务器配置写成现场必需项。

轻量安装检查：

```bash
python -m pip install -r requirements.txt
python -m pip check
ffmpeg -version
```

生产启动：

```bash
AUDIORESCUE_UI_FIXTURE=0 python app.py
```

开发 fixture 启动只用于 UI 状态和离线页面门禁，不代表真实 pipeline 效果：

```bash
AUDIORESCUE_UI_FIXTURE=1 python app.py
```

## 配置与产物

- 冻结配置：[`configs/app.yaml`](configs/app.yaml)
- 每次任务：`outputs/{job_id}/`
- 正式本地数据：private data root，不会提交
- 正式演示素材：仅经 C 审核后放入 `demo_assets/`

## 私有录音数据工作流

私有录音、参考文本、授权台账、评测划分、盲听配对和运行结果都只保存在本地私有工作区。公开仓库只保留中性原则：

- 使用 synthetic fixture 做公开测试和 UI 验证；
- approved short sample 只在本地私有台账确认后用于 smoke 或演示候选；
- reserved evaluation set 只在冻结后一次性使用，不用于调参、公开调试或补证；
- 公开报告只写安全摘要、测试命令和通过数量；
- 不公开私有文本、成员映射、真实文件名、样例数量、路径、哈希、答案表或逐样例指标。

具体私有数据命令由 C 在本地私有 runbook 中执行和审计，不复制到公开 PR、Issue、截图或日志。

## 合入前最低检查

不触发模型的轻量检查：

```bash
python -m py_compile app.py core/*.py ui/*.py scripts/*.py
python -m unittest discover -s tests -p "test_*.py" -q
git diff --check
```

需要预热模型和授权短样例的完整 smoke：

```bash
python scripts/smoke_audio_core.py --normalize-only
python scripts/smoke_audio_core.py --runs 2 --asr-model base --device auto
```

公开报告只记录版本、耗时、指纹摘要、测试数量和成功/失败状态。不得公开原始日志、私有文本、机器路径、服务器路径或模型缓存位置。历史本地 CPU smoke 只能作为技术门禁证据；服务器 GPU 补证仍为 provisional，不作为正式速度证据。真实用户样例、听感和效果验收仍需最终人工执行。

## M4 实时降噪与会议虚拟麦

这条独立链路不会改动上面的文件后处理流程：

```text
降噪模式：物理麦克风 → GTCRN 降噪 → 自动增益/限幅 → 48 kHz 声卡或虚拟麦
小声模式：物理麦克风 → 保真人声自动增益/限幅 → 48 kHz 声卡或虚拟麦
```

首次运行会下载并校验约 524 KB 的 GTCRN 模型。让会议中的其他人听到增强结果，需要安装一次 BlackHole，安装后重启 Mac：

```bash
conda activate audiorescue-real
brew install --cask blackhole-2ch
# 重启后
python scripts/live_gtcrn.py --list-devices
python scripts/live_gtcrn.py --virtual-mic
```

会议软件里把“麦克风”选成 `BlackHole 2ch`，把“扬声器”保留为真实耳机；首次对比时先关闭会议软件自带的自动增益和强降噪，避免双重处理。运行中按空格在原声与增强之间切换，按 `E` 只用增强，按 `R` 只用原声，按 `Q` 退出。只在自己的耳机中试听时，可把 `--output-device` 指向耳机。只验证声卡、不播放声音时可运行：

```bash
python scripts/live_gtcrn.py --duration 2 --gain 0
```

安静环境需要小声说话时，使用独立的小声增强模式。该模式保留原始人声后再做自动增益和限幅，不让 GTCRN 先把很小的语音当作噪声削弱：

```bash
python scripts/live_gtcrn.py --input-device 0 --virtual-mic --mode quiet
```

运行中按 `V` 进入小声增强，按 `E` 进入 GTCRN 降噪增强，按 `R` 切换原声。

### 实时会议提词

配置好 `DASHSCOPE_API_KEY` 后，可以把增强后的 16 kHz 音频同时送给 Fun-ASR，并由千问根据会前预设生成短提示：

```bash
conda activate audiorescue-real
python scripts/live_gtcrn.py \
  --input-device 1 \
  --virtual-mic \
  --meeting \
  --meeting-preset "黑客马拉松项目汇报；我负责实时降噪和会议助手，回答要简洁、准确。"
```

终端会显示实时字幕和 `[建议]`；按 `N` 可以立即生成下一句建议。转写只在一句结束后触发千问，不会让网络请求阻塞实时降噪。
