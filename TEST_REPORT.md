# AudioRescue 测试报告

> 当前为滚动记录。没有真实运行证据的项目保持“待测”，不得填写估算值。

## 基线

| 项目 | 当前值 |
|---|---|
| 契约版本 | `v0.1-contract` |
| 规范音频 | 48kHz / mono / PCM16 WAV |
| 增强 | DeepFilterNet3，默认 strength 0.75 |
| ASR | multilingual Whisper `base`，`zh` |
| 默认增强轨 | `enhanced_mix.wav` |
| 开发样例 | `DEV_SMOKE_S01_FAN`（仅联调，不作效果证据） |

## 自动测试

| 时间 | 环境/提交 | 命令 | 结果 |
|---|---|---|---|
| 2026-07-14 | C 本地 macOS / Python 3.13.3 / `codex/c-integration` | `python3 -m unittest discover -s tests -v` | 64/64 通过；使用假后端，不等于真实模型验收 |
| 2026-07-14 | RTX 4090 / Ubuntu 22.04 / Python 3.10.8 / `1b05abb` | `python -m unittest discover -s tests -q` | 64/64 通过，工作树干净；使用假后端，不等于真实端到端验收 |
| 2026-07-14 | C 本地 macOS / Python 3.13.3 / `codex/c-integration` | `python3 -m unittest discover -s tests -q` | 144/144 通过；新增母带转码、逐条授权哈希、固定混音、严格校验、freeze v2、locked 规范身份回执、输入 staging、防冻结集重标、盲听平衡与仓库安全测试；未使用正式录音，不代表效果验收 |

## 4090 环境验收

| 项目 | 实测值 |
|---|---|
| GPU | NVIDIA GeForce RTX 4090 24GB |
| Python / PyTorch | Python 3.10.8 / PyTorch 2.5.1+cu118 |
| CUDA 可用性 | `torch.cuda.is_available() == True`；真实 FP16 矩阵乘法通过 |
| DeepFilterNet | DeepFilterNet3 `model_120.ckpt.best`，确认运行于 `cuda:0` |
| Whisper | multilingual `base`，语言 `zh`，FP16，确认运行于 CUDA |
| 开发样例 | `DEV_SMOKE_S01_FAN`，12.000 秒，输出增强轨为 48kHz / mono |

以上环境与模型测试均在目标 4090 服务器真实执行，不是估算值。开发样例是合成联调音频，不能据此宣称增强效果或比赛指标。

## 真实推理验收

| 项目 | 首次准备/冷启动 | 权重已缓存后的模型加载 | 已加载模型推理 | 结论 |
|---|---:|---:|---:|---|
| DeepFilterNet3 12秒样例 | 约 3.0 秒（含进程与模型加载） | 待补独立测量 | 0.27 秒，RTF 0.023 | 模型与 CUDA 跑通 |
| Whisper base 原轨 | 72 秒（包含 139MB 首次下载） | 1.327 秒 | 0.762 秒 | 模型与 CUDA 跑通 |
| Whisper base mixed轨 | 待测 | 待测 | 待测 | 待 A 增强模块合入 |
| 端到端 | 待测 | 待测 | 待测 | 待 A/B 模块合入 |

Whisper 开发样例实测文本为“今天下午三點,我們在實驗室討論與因處理項目的最終方案。”，与参考文本仍有字符差异。该结果只证明运行链路可用，也提醒正式样例必须做真实 Before/After 对照。

## 必须人工完成

- [ ] 三人盲听主演示样例，记录偏好与失真问题。
- [ ] WAV 与 MP3 各成功处理一次。
- [ ] 至少两个正式样例完整通过。
- [ ] 有人工参考文本时核对 Before/After CER；无参考时不显示 CER。
- [ ] 断网重启三次并完成固定样例处理。
- [ ] 1366x768 投屏、音量、下载和备用录屏验收。
- [ ] 从干净目录按 README 重现启动。
- [ ] 15 条母带完成授权登记并转码，`validate_dataset.py` 严格通过 42 个 WAV / 33 行 manifest。
- [ ] `dev` 18 条完成批量推理；只在最终冻结后消费 9 条 `locked_test`。
- [ ] 三人完成随机盲听表，答案揭盲后汇总偏好，不在试听前泄露 A/B 身份。
