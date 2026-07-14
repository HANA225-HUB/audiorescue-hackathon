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
- A 的音频模块与 B 的页面/可视化已在候选 PR 中验证；最终组合复验已报告无 P0/P1，仍需 C Review、冻结与人工合并决策后才能宣称最终通过；
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
5. 返回调用示例、真实输出、耗时、模型位置、已知问题和回滚方式。

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
- 正式本地数据：`data_local/`，不会提交
- 正式演示素材：仅经 C 审核后放入 `demo_assets/`

## 私有录音数据工作流

下列工具只使用私有数据根目录，不会把母带、冻结集或运行结果加入 Git。公开文档不记录私有参考文本、成员映射、锁定样例名或录音文件名。

```bash
# 1. 幂等初始化：可重复执行，不覆盖已有台账或音频
python scripts/init_dataset.py --root <private-data-root>

# 2. 原始 M4A/AAC/WAV 先放入私有来源目录
#    然后在 recording_metadata.csv 填 source_original_path
#    录音进行中可只处理已就绪条目
python scripts/standardize_recordings.py --data-root <private-data-root> --available-only

# 3. 全部母带就绪后运行最终严格模式
python scripts/standardize_recordings.py --data-root <private-data-root>

# 4. 在 recording_metadata.csv 中逐条确认 consent_status=yes；
#    原始/标准化 SHA-256 必须已由上一步填写且保持匹配

# 5. 生成私有 manifest 和固定混音
python scripts/build_dataset.py \
  --dataset-root <private-data-root> \
  --consent-or-license team-approved-for-competition-evaluation

# 6. 校验 WAV、manifest、哈希、命名和授权状态
python scripts/validate_dataset.py <private-data-root>
```

开发集生成后，可批量运行：

```bash
python scripts/run_evaluation.py \
  --manifest <private-manifest> \
  --dataset-root <private-data-root> \
  --split dev \
  --output-dir <private-output-dir> \
  --force-recompute
```

只在模型、强度、代码、授权和私有 manifest 全部完成，且 Git 工作区干净后冻结：

```bash
python scripts/freeze_experiment.py
```

冻结测试集是一次性正式评测，不能用来试跑、调参或补证。需要修复崩溃时必须保留回执并创建明确的新实验版本，不得删除回执冒充首次评测。

生成三人盲听包时，先准备一个私有配对表。配对表、参考文本、授权台账和答案表都只保存在私有工作区。

然后生成三个随机、逐成员跨样例平衡的匿名试听包；每个人看到原轨位于 A/B 的次数差不超过 1。答案表仅由 C 保管：

```bash
python scripts/prepare_blind_ab.py \
  --pairs <private-pairs-csv> \
  --output <private-blind-output-dir>
```

该工具只做随机、平衡和逐字节复制，不替成员判断听感。盲听完成前不要打开答案表，也不要把私有参考文本、授权台账或答案表提交 Git。

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

公开报告只记录版本、耗时、指纹摘要、测试数量和成功/失败状态。不得公开原始日志、私有文本、机器路径、服务器路径或模型缓存位置。正式 DeepFilterNet/Whisper、GPU 性能和听感必须由最终组合复验与人工验收补齐。
