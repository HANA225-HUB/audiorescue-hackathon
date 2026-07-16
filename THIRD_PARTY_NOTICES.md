# Third-party notices

MeetingEverywhere / AudioRescue is distributed under the MIT License, but it
interoperates with third-party software, models and hosted services that keep
their own licenses and terms.

Nothing in this file replaces the authoritative license shipped by an
upstream project.

| Component | How it is used | Upstream terms |
|---|---|---|
| [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) | Online GTCRN inference and resampling support | Apache License 2.0 |
| [`gtcrn_simple.onnx`](https://github.com/k2-fsa/sherpa-onnx/releases/tag/speech-enhancement-models) | Downloaded at runtime from the sherpa-onnx speech-enhancement release; sourced from [Xiaobin-Rong/gtcrn](https://github.com/Xiaobin-Rong/gtcrn) and not stored here. Expected SHA-256: `e77603ac0c23dac3227dd2d7135b3a585cbee2679048aecfa886657d3ae1b534` | MIT License; release terms remain authoritative |
| [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) | Optional offline file enhancement | MIT or Apache License 2.0, at the user's option |
| [OpenAI Whisper](https://github.com/openai/whisper) | Optional offline speech transcription | MIT License |
| [Gradio](https://github.com/gradio-app/gradio) | Local web interface | Apache License 2.0 |
| [BlackHole](https://github.com/ExistentialAudio/BlackHole) | Optional macOS virtual audio driver; installed separately and not bundled | GPL-3.0 or a separate license offered by its author |
| [FFmpeg](https://ffmpeg.org/) | Optional system-level audio decoding and conversion dependency | License depends on how the local FFmpeg build was configured |
| [Alibaba Cloud DashScope](https://dashscope.aliyun.com/) | Optional hosted realtime ASR and Qwen inference | Alibaba Cloud service terms, model terms and usage fees |

The complete list of direct Python dependencies is recorded in
[`requirements.txt`](requirements.txt). Transitive dependencies may add
further notices.

The synthetic test fixture in `tests/fixtures/` contains no human speech and is
distributed under this project's MIT License. User recordings, private
meeting documents, downloaded model weights and generated outputs are not part
of the open-source distribution.
