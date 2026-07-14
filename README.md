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
- A 的音频模块与 B 的页面/可视化已合入；真实 P0 还需在当前 SHA 上用已授权语音完成 4090 merged E2E 与页面验收；
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
3. 使用不含人声的 `tests/fixtures/dev_smoke_s01_fan.wav` 验证音频 I/O/增强，再用 `data_local` 内已授权的 S01/S02 录音验证双路 ASR；
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

## 正式录音数据工作流

下列工具只使用本地 `data_local/`，不会把母带、冻结集或运行结果加入 Git。

```bash
# 1. 幂等初始化：可重复执行，不覆盖已有台账或音频
python scripts/init_dataset.py --root data_local

# 2. 原始 M4A/AAC/WAV 先放入 data_local/source_original/
#    然后在 recording_metadata.csv 填 source_original_path
#    录音进行中可只处理已就绪条目
python scripts/standardize_recordings.py --data-root data_local --available-only

# 3. 15 条母带全部就绪后运行最终严格模式
python scripts/standardize_recordings.py --data-root data_local

# 4. 在 recording_metadata.csv 中逐条确认 consent_status=yes；
#    原始/标准化 SHA-256 必须已由上一步填写且保持匹配

# 5. 生成 27 条固定混音和 33 行私有 manifest
python scripts/build_dataset.py \
  --dataset-root data_local \
  --consent-or-license team-approved-for-competition-evaluation

# 6. 校验 42 个 WAV、33 行语义、哈希、命名和授权状态
python scripts/validate_dataset.py data_local
```

三段纯噪声补齐并生成开发集后，可批量运行：

```bash
python scripts/run_evaluation.py \
  --manifest data_local/manifest.csv \
  --dataset-root data_local \
  --split dev \
  --output-dir data_local/evaluations/dev_v1 \
  --force-recompute
```

只在模型、强度、代码、授权和 33 行 manifest 全部完成，且 Git 工作区干净后冻结：

```bash
python scripts/freeze_experiment.py

# locked_test 是一次性正式评测；这条命令不用来试跑
python scripts/run_evaluation.py \
  --manifest data_local/manifest.csv \
  --dataset-root data_local \
  --split locked_test \
  --output-dir data_local/evaluations/locked_v1 \
  --frozen-config data_local/config_frozen.json \
  --confirm-locked \
  --force-recompute
```

`locked_test` 运行前会再次核对 42 个 WAV / 33 行 manifest、音频 SHA-256、当前 Git commit、工作区、pipeline 配置和增强强度；9 条输入先复制到本次只读 staging，再创建消费回执。回执由冻结内容的规范身份生成并固定保存在 `data_local/.locked_receipts/`，复制、改名或重新排版 freeze JSON 不能获得第二次机会。需要修复崩溃时必须保留回执并创建明确的新实验版本，不得删除回执冒充首次评测。

生成三人盲听包时，先准备一个私有 `pairs.csv`，每行指定同一样例的标准原轨与默认增强混合轨。可以填写绝对路径；相对路径会从 `pairs.csv` 所在目录解析：

```csv
sample_id,original_path,enhanced_path
demo_01,/absolute/path/to/outputs/demo_01/original.wav,/absolute/path/to/outputs/demo_01/enhanced_mix.wav
```

然后生成三个随机、逐成员跨样例平衡的匿名试听包；每个人看到原轨位于 A/B 的次数差不超过 1。成员只拿各自的 `rater_A/B/C` 目录，答案表仅由 C 保管：

```bash
python scripts/prepare_blind_ab.py \
  --pairs data_local/blind_ab/pairs.csv \
  --output data_local/blind_ab/dev_v1
```

该工具只做随机、平衡和逐字节复制，不替成员判断听感。盲听完成前不要打开 `admin_keep_private/answer_key.csv`，也不要把私有参考文本、授权台账或答案表提交 Git。

## 合入前最低检查

```bash
python -m unittest discover -s tests -q
python -m py_compile scripts/*.py
git diff --check
```

轻量契约与离线数据测试也会在 GitHub Actions 的 Python 3.10/3.11 环境运行。正式 DeepFilterNet/Whisper、GPU 性能和听感仍必须在 4090 与真实样例上单独验收。
