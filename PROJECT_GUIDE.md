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
8. A/B 功能分支以 C 当日公布的 `origin/codex/c-integration` 提交为基线；已开工后不要强制改写历史，合入冲突交给 C 处理。
9. `locked_test` 只允许在冻结记录通过完整性校验后正式运行一次；崩溃也会留下消费回执，不删除回执重跑。

## 文件所有权

- A：`core/audio_io.py`、`core/enhance.py`、`core/transcribe.py`、相应单元测试。
- B：`app.py`、`ui/`、`core/visualize.py`、`core/text_diff.py`、相应单元测试。
- C：`core/schemas.py`、`core/pipeline.py`、`core/metrics.py`、`core/cache.py`、`core/events.py`、`scripts/` 数据与评测工具、配置、文档和集成测试。

## 本地数据边界

- 原始录音母带只放在 `data_local/source_original/`，统一 WAV 放在 `data_local/raw/`；两者都不提交 Git。
- 公开仓库只允许提交已声明授权的演示音频或合成测试音频；完整数据、参考文本、冻结集、模型和运行输出不得提交。
- 开发只使用 `dev` 和合成联调样例；`locked_test` 在最终代码、配置、manifest、授权与 Git 提交冻结前不可访问。
- 录音者授权或公开数据许可证记录在私有 manifest/台账；无法确认授权的素材不进入演示和评测。

## 当前冻结

- 契约版本：`v0.1-contract`
- 配置文件：`configs/app.yaml`
- A 接口：`docs/A_BACKEND_CONTRACT_V1.md`
- 默认增强试听与 After ASR：`enhanced_mix.wav`
- `enhanced_full.wav`：100% 增强调试/下载轨
