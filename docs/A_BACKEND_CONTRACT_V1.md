# A 音频后端冻结契约 v0.1

> 冻结时间：2026-07-14
> 所有者：C
> 适用模块：`core/audio_io.py`、`core/enhance.py`、`core/transcribe.py`

这是今天首次生成的实际代码契约，不是声称此前已经存在。冻结文件为 `core/schemas.py`；如需变更，A 提交建议，由 C 修改后同步 A/B。

## 1. 公共 schema

A 只能从 `core.schemas` 导入：

```text
AudioLevelMetrics
AudioMeta
TranscriptResult
TranscriptSegment
EnhancementOutput
WarningItem
ErrorCode
InputAudioError
InputTooLongError
EnhancementError
OutputValidationError
ASRInferenceError
```

可空字段：

- `AudioMeta.rms_dbfs`、`silent_ratio`：无法可靠计算时可为 `None`；
- `AudioMeta.original_sample_rate`、`original_channels`、`source_format`：源信息不可取得时可为 `None`；
- `TranscriptResult.language`：Whisper 未返回时可为 `None`；
- `TranscriptResult.error`：成功或空文本时必须为 `None`；真正失败抛 `ASRInferenceError`；
- `ProcessResult` 中某阶段未产生的结果路径、转写、CER、图像均可为 `None`；
- `ProcessResult.original_levels` 与 `mixed_levels` 在对应轨道未生成或未完成体检时可为 `None`；
- `events`、`warnings`、`config_snapshot` 永不为 `None`，没有内容时为空集合。

## 2. A 层最终函数签名

```python
def normalize_audio(
    input_path: str,
    output_path: str,
    target_sr: int = 48_000,
    mono: bool = True,
) -> AudioMeta:
    ...


def inspect_audio(samples, sample_rate: int) -> dict:
    ...


def load_enhancer(model_dir: str | None = None):
    ...


def enhance_audio(
    input_wav: str,
    output_full_wav: str,
    output_mix_wav: str,
    strength: float = 0.75,
) -> EnhancementOutput:
    ...


def load_asr(model_name: str, device: str):
    ...


def transcribe_audio(
    audio_path: str,
    language: str = "zh",
    *,
    model_name: str = "base",
    device: str = "auto",
) -> TranscriptResult:
    ...
```

`EnhancementOutput` 固定包含：

```text
full_output_path
mixed_output_path
strength
runtime_seconds
model_name
warnings: list[WarningItem]
```

`TranscriptResult.segments` 只返回可 JSON 序列化的：

```json
[{"start": 0.0, "end": 2.4, "text": "大家好"}]
```

不得返回 Whisper 内部对象、Tensor 或模型对象。

## 3. 输出目录责任

确认由 C 的 pipeline 创建：

```text
outputs/{job_id}/
├── original.wav
├── enhanced_full.wav
├── enhanced_mix.wav
├── transcript_before.json
├── transcript_after.json
├── waveform_compare.png
├── spectrogram_compare.png
├── result.json
└── run.log
```

A 只按 pipeline 传入的绝对路径写：

- `original.wav`；
- `enhanced_full.wav`；
- `enhanced_mix.wav`。

A 不创建 `job_id`，不改文件名，不写 `result.json`。为健壮性可创建目标文件的 parent，但不得改变目录结构。写失败时不能覆盖源文件或生成伪成功空文件。

## 4. 错误码、异常与状态

| 情况 | A 的行为 | pipeline 警告码 | 最终状态 |
|---|---|---|---|
| 缺文件、空文件、不可解码、时长为0 | 抛 `InputAudioError` | `INPUT_INVALID` | `failed` |
| 时长超过60秒 | 抛 `InputTooLongError`，不静默截断 | `INPUT_TOO_LONG` | `failed` |
| 输入削波 | 返回结果并附 `INPUT_CLIPPED` | 同码 | 不单独降级 |
| 输入近静音 | 返回结果并附 `INPUT_NEAR_SILENT` | 同码 | 不单独降级；ASR空时再变 `partial` |
| 发生重采样/下混 | 返回结果并记录源属性 | `INPUT_RESAMPLED` / `INPUT_STEREO_DOWNMIXED` | 不降级 |
| DeepFilterNet 失败 | 抛 `EnhancementError` | `ENHANCE_FAILED` | `failed` |
| full/mixed 空、NaN、不可播放 | 抛 `OutputValidationError` | `OUTPUT_INVALID` | `failed` |
| ASR 推理失败 | 抛 `ASRInferenceError` | pipeline 按调用位置映射 Before/After | `partial`，前提是增强成功 |
| ASR 合法返回空文本 | 返回 `text=""`、`error=None` | `ASR_EMPTY` | `partial` |
| 可视化失败 | A 不处理 | `VIS_FAILED` | `partial` |
| CLAP 正常关闭 | A 不处理 | 无 | 不影响 P0 |
| CLAP 开启后失败 | A 不处理 | `EVENTS_SKIPPED` | 不影响 P0 |
| 缓存命中 | A 不处理 | `CACHE_USED` | 保留缓存中的原状态 |

额外规则：

- 原轨与增强轨 ASR 互不依赖；一路失败不能取消另一路；
- 增强失败时 pipeline 可以保留原轨与原轨转写，但不得伪造增强轨，状态仍是 `failed`；
- A 不把 traceback 写入面向用户的 message，完整异常只进入本地日志；
- `ASRInferenceError` 不知道 before/after，pipeline 根据调用位置映射为 `ASR_BEFORE_FAILED` 或 `ASR_AFTER_FAILED`。

## 5. 冻结配置

最终以 `configs/app.yaml` 为准：

| 项目 | v0.1 值 |
|---|---|
| 规范化 | WAV、48kHz、单声道、PCM16 |
| 输入时长 | 1–60秒；主演示10–15秒 |
| DeepFilterNet | DeepFilterNet3 |
| dry/wet | `mixed=(1-strength)*original+strength*enhanced` |
| 默认 strength | `0.75` |
| Whisper | 多语言 `base` |
| 语言/任务 | `zh` / `transcribe` |
| 设备 | `auto`；4090服务器解析为 `cuda`，本地无CUDA时为 `cpu` |
| 解码 | `temperature=0`、`condition_on_previous_text=false`、无 `initial_prompt` |
| 并发 | 1 |

原轨和增强轨必须使用同一模型、语言、设备类型和解码参数。参考台词只能进入 C 的 CER 模块，禁止进入 Whisper prompt。

## 6. A/B 响度公平与默认播放

P0 不在 A 的模型输出上做额外响度 DSP，也不引入临时 LUFS 依赖。冻结规则：

1. `original.wav`、`enhanced_full.wav`、`enhanced_mix.wav` 保持可审计；
2. 两个播放器使用相同播放器音量和同一系统音量；
3. 结果同时报告两轨 `rms_dbfs` 和 `peak_abs`；
4. 三人做盲听，不能把“更响”当“更清晰”；
5. 默认页面播放和 After ASR 均使用 `enhanced_mix.wav`；
6. `enhanced_full.wav` 仅作100%增强调试与下载；
7. `ProcessResult.enhanced_audio_path` 是兼容别名，永远等于 `mixed_output_path`。

响度证据的公共结构为：

```python
AudioLevelMetrics(
    peak_abs: float,        # 必须有限，范围 0.0..1.0
    rms_dbfs: float | None, # 非静音时必须有限；数字静音使用 None
)
```

`ProcessResult.original_levels` 严格对应 `original_audio_path`，
`ProcessResult.mixed_levels` 严格对应 `mixed_output_path`。两者是为了证明 A/B
没有靠放大音量制造效果，不是新的音频处理步骤。旧的 `ProcessResult(...)`
构造调用不传这两个字段仍合法，默认值均为 `None`。不得用 `NaN`、
`Infinity` 或 `-Infinity` 代表静音。

以后若加入 LUFS，只能生成新的播放副本，不覆盖上述三条标准轨，并需升级契约版本。

## 7. 缓存责任

确认：

- C 的 `core/cache.py` 负责任务结果缓存、缓存键、命中显示和失效；
- A 只负责进程内 enhancer/ASR 模型单例以及官方权重的本地缓存；
- A 不缓存 `ProcessResult`，不根据文件名绕过真实处理；
- 改变输入哈希、strength、模型、语言或代码/配置版本后不能误用旧结果。

## 8. 耗时字段与冷启动

公共字段在 `RuntimeStats`：

```text
decode_seconds
enhancer_load_seconds
enhancement_seconds
asr_load_seconds
asr_before_seconds
asr_after_seconds
visualization_seconds
event_seconds
persistence_seconds
total_seconds
cache_hit
cold_start
device
```

口径：

- `enhance_audio.runtime_seconds` 只计本次增强调用，不含模型加载；
- `TranscriptResult.runtime_seconds` 只计该轨音频加载、特征和解码，不含模型加载；
- 若模型在应用启动时预热，任务内 load 字段为0，`cold_start=false`；
- 若首次请求中懒加载，实际加载写入 load 字段，`cold_start=true`；
- `total_seconds` 从 `process_audio()` 入口到 `result.json` 持久化完成，包含本次请求内实际发生的模型加载；
- 测试报告分别记录一次冷启动和至少三次热运行，页面主数字使用热运行中位数并注明条件。

## 9. 首次联调样例

立即可用的仓库内样例：

```text
tests/fixtures/dev_smoke_s01_fan.wav
```

它是纯数学啁啾信号加固定种子的风扇状噪声，不含人声，只用于验证解码、增强、输出路径和错误处理；不能作为 Whisper 文本或 CER 证据。双路 ASR 首次联调改用 `data_local` 内已授权的 S01/S02 中文录音，真实录音不得提交 GitHub。

该公开 fixture 没有参考文本。

正式 dev 样例生成后固定为：

```text
data_local/controlled/dev/mix_spkA_s01_fan_snr000.wav
```

`S03` 属于 `locked_test`，配置冻结前不发给 A/B 调参。

首次预期输出：

```text
outputs/{job_id}/original.wav
outputs/{job_id}/enhanced_full.wav
outputs/{job_id}/enhanced_mix.wav
```

验收只要求三条文件可读取、48kHz/mono/PCM16、时长合理、无 NaN/Inf、增强不覆盖原轨；转写是否改善不作为合成 fixture 的验收条件。

## 10. CLAP P1

确认：A 只向 pipeline 提供标准化 `original.wav` 路径和 `AudioMeta`。`core/events.py` 只读原轨，不修改标准化、增强或 ASR 主链路。CLAP 正常关闭时不产生警告；只有显式开启后失败才追加 `EVENTS_SKIPPED`。
