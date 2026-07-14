# PROJECT_GUIDE

## 范围

P0 只做离线短音频标准化、DeepFilterNet、双轨 Whisper、A/B、可视化和文本差异。CLAP、实时流式、训练模型、账号系统和移动端不进入 P0。

## 协作规则

1. `main` 始终保持可运行；成员在自己的功能分支开发。
2. 同一时刻只有一个人和一个 AI 写同一个文件。
3. 公共契约变更必须先由 C 修改 `core/schemas.py` 和契约文档，再通知 A/B。
4. 每个提交只解决一个可验收问题。
5. 不提交模型、缓存、完整数据集、用户私密音频、日志或 `outputs/`。
6. 不把参考文本传入 Whisper prompt。
7. 不手改 Whisper 输出后再计算 CER。

## 文件所有权

- A：`core/audio_io.py`、`core/enhance.py`、`core/transcribe.py`、相应单元测试。
- B：`app.py`、`ui/`、`core/visualize.py`、`core/text_diff.py`、相应单元测试。
- C：`core/schemas.py`、`core/pipeline.py`、`core/metrics.py`、`core/cache.py`、`core/events.py`、配置和集成测试。

## 当前冻结

- 契约版本：`v0.1-contract`
- 配置文件：`configs/app.yaml`
- A 接口：`docs/A_BACKEND_CONTRACT_V1.md`
- 默认增强试听与 After ASR：`enhanced_mix.wav`
- `enhanced_full.wav`：100% 增强调试/下载轨
