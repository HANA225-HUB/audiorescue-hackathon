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
| 开发样例 | `DEV_SMOKE_S01_FAN`（纯数学信号，仅验证 I/O/增强，不作 ASR 或效果证据） |
| 模型指纹 | 4090 实测并写入配置的期望值：DeepFilterNet checkpoint `23b92884…6003` / config `415eb925…290`；Whisper base `ed3a0b6b…e34e`。自动加载时校验仍属于 A Round 2，未完成前不得声称强制冻结 |

## 自动测试

| 时间 | 环境/提交 | 命令 | 结果 |
|---|---|---|---|
| 2026-07-14 | C 本地 macOS / Python 3.13.3 / `codex/c-integration` | `python3 -m unittest discover -s tests -v` | 64/64 通过；使用假后端，不等于真实模型验收 |
| 2026-07-14 | RTX 4090 / Ubuntu 22.04 / Python 3.10.8 / `1b05abb` | `python -m unittest discover -s tests -q` | 64/64 通过，工作树干净；使用假后端，不等于真实端到端验收 |
| 2026-07-14 | C 本地 macOS / Python 3.13.3 / `codex/c-integration` | `python3 -m unittest discover -s tests -q` | 144/144 通过；新增母带转码、逐条授权哈希、固定混音、严格校验、freeze v2、locked 规范身份回执、输入 staging、防冻结集重标、盲听平衡与仓库安全测试；未使用正式录音，不代表效果验收 |
| 2026-07-14 | A/C 本地 / `efd83ac` + `feb5a96` | 全量 `unittest discover` | 191/191 通过，1 项可视化依赖缺失跳过；音频输入预检、模型/设备显式 ASR、线程锁和数据集级一次性回执通过 |
| 2026-07-14 | C 本地 macOS / 当前工作区 | 11 个关键模块定向 `unittest` | 124/124 通过；覆盖录音映射、模型哈希 cache identity、盲测随机化、UI fail-closed、公开音频授权门禁和 locked one-shot |
| 2026-07-15 | RTX 4090 / Python 3.10.8 / `db7eb92` | `python -m unittest discover -s tests -q` | 193/193 通过；同 SHA 的两次 GitHub Actions 均成功 |

## 4090 环境验收

| 项目 | 实测值 |
|---|---|
| GPU | NVIDIA GeForce RTX 4090 24GB |
| Python / PyTorch | Python 3.10.8 / PyTorch 2.5.1+cu118 |
| CUDA 可用性 | `torch.cuda.is_available() == True`；真实 FP16 矩阵乘法通过 |
| DeepFilterNet | DeepFilterNet3 `model_120.ckpt.best`，确认运行于 `cuda:0` |
| Whisper | multilingual `base`，语言 `zh`，FP16，确认运行于 CUDA |
| 开发样例 | 旧版 12 秒语音 fixture 曾完成真实推理；当前已替换为纯数学信号，新 SHA 的服务器复测待完成 |

以上环境与模型测试均在目标 4090 服务器真实执行，不是估算值。开发样例是合成联调音频，不能据此宣称增强效果或比赛指标。

## 真实推理验收

| 项目 | 首次准备/冷启动 | 权重已缓存后的模型加载 | 已加载模型推理 | 结论 |
|---|---:|---:|---:|---|
| DeepFilterNet3 历史 12秒样例 | 约 3.0 秒（含进程与模型加载） | 待补独立测量 | 0.27 秒，RTF 0.023 | 模型与 CUDA 跑通；当前 fixture 待复测 |
| Whisper base 历史原轨 | 72 秒（包含 139MB 首次下载） | 1.327 秒 | 0.762 秒 | 模型与 CUDA 跑通；不作为当前 fixture 的 ASR 证据 |
| 已授权 B-S01 clean WAV merged E2E | 5.22 秒总耗时 | enhancer 0.35 秒 / ASR 1.13 秒 | 增强 0.24 秒；双路 ASR 0.77 / 0.30 秒 | `success`，CUDA；CER 0.08 → 0.08 |
| 已授权 B-S01 clean M4A merged E2E | 5.28 秒总耗时 | enhancer 0.36 秒 / ASR 1.10 秒 | 增强 0.23 秒；双路 ASR 0.74 / 0.29 秒 | 首次发现并修复 ffprobe 参数后 `success`；CER 0.08 → 0.16 |
| B-S01 风扇真实带噪 M4A | 5.64 秒冷启动 | 热运行总耗时 1.69–1.79 秒 | strength 0.50 / 0.75 / 1.00 | CER 分别 0.20→0.16、0.20→0.32、0.20→0.20；该样例偏向轻度增强 |
| C-S02 键盘真实带噪 M4A | 5.90 秒冷启动 | 热运行总耗时 1.87 秒 | strength 0.50 / 0.75 | 固定繁简归一化后 CER 均为 0.04→0.00 |
| 生产 UI 构建与 presenter | 模型已缓存 | 不重新推理 | 读取真实 `result.json` | fixture 关闭；原轨、增强轨、波形、声谱图五项路径均通过安全根目录检查 |

上述 Whisper 文本来自已被替换的历史语音 fixture，只证明当时链路可用，不能归到当前纯数学 fixture，也不能作为比赛效果证据。当前双路 ASR 必须改用 `data_local` 内已授权的 S01/S02 真实录音。

真实开发样例说明：当前证据足以确认 merged WAV/M4A、CUDA、双路 ASR、可视化和 UI presenter 已跑通，但还不足以冻结默认 strength。风扇样例更偏向 0.50，键盘样例在 0.50/0.75 均改善；应等三段纯噪声生成 18 条 dev 控制混合并完成人工听感后再冻结，不能根据单条样例挑最好看的结果。

## 本轮真实录音入库

- 12/15 条母带已复制到私有 `data_local/source_original`，并标准化为 48kHz / mono / PCM16；原始 SHA 和标准化 SHA 已入私有台账。
- 已就绪：9 条 clean + 3 条真实同时带噪录音；实际 real 映射为 A/道路/S03、B/风扇/S01、C/键盘/S02。
- 仍缺：45–60 秒纯风扇、纯键盘、纯道路噪声各一条。
- A 的三条 clean 无满幅数字削波，但峰值约 0.99–1.00，触发高峰值警告；保留母带，不做隐式增益处理，混音阶段由固定峰值保护规则处理。

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
