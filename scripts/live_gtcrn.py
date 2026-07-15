#!/usr/bin/env python3
"""Run GTCRN microphone enhancement for headphones or a meeting virtual mic."""

from __future__ import annotations

import argparse
import os
import select
import sys
import time
from contextlib import contextmanager
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.live_denoise import (  # noqa: E402
    DEFAULT_GTCRN_MODEL_PATH,
    GtcrnDenoiser,
    LiveDenoiseEngine,
    QuietVoiceLeveler,
    VoiceLeveler,
    ensure_gtcrn_model,
)
from core.meeting_assistant import LiveMeetingAssistant  # noqa: E402
from core.realtime_asr import RealtimeAsrEvent  # noqa: E402


def _device(value: str):
    try:
        return int(value)
    except ValueError:
        return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M4 本地实时增强：麦克风 → GTCRN → 耳机或会议虚拟麦"
    )
    parser.add_argument("--list-devices", action="store_true", help="列出音频设备")
    parser.add_argument("--model", type=Path, default=DEFAULT_GTCRN_MODEL_PATH)
    parser.add_argument("--no-download", action="store_true", help="模型缺失时不下载")
    parser.add_argument("--input-device", type=_device, help="输入设备编号或名称")
    parser.add_argument("--output-device", type=_device, help="输出设备编号或名称")
    parser.add_argument(
        "--virtual-mic",
        action="store_true",
        help="自动把增强音频送到 BlackHole/Loopback 虚拟麦克风",
    )
    parser.add_argument("--duration", type=float, help="自动停止秒数；默认手动退出")
    parser.add_argument(
        "--stream-rate",
        type=int,
        choices=(16_000, 48_000),
        default=48_000,
        help="声卡/虚拟麦采样率，默认 48000",
    )
    parser.add_argument("--gain", type=float, default=1.0, help="监听增益，默认 1.0")
    parser.add_argument(
        "--queue-blocks", type=int, default=6, help="输出队列上限，默认 6 块"
    )
    parser.add_argument(
        "--mode",
        choices=("enhanced", "quiet", "raw"),
        default="enhanced",
        help="启动模式：降噪增强 / 安静环境小声增强 / 原声",
    )
    parser.add_argument(
        "--no-leveler", action="store_true", help="关闭小声自动增益和峰值保护"
    )
    parser.add_argument(
        "--allow-speakers",
        action="store_true",
        help="明确允许扬声器回放（有回声/啸叫风险）",
    )
    parser.add_argument(
        "--meeting-assistant",
        "--meeting",
        dest="meeting_assistant",
        action="store_true",
        help="启用百炼实时转写和千问会议提词",
    )
    parser.add_argument("--meeting-preset", help="会前角色、目标和背景预设")
    parser.add_argument(
        "--meeting-preset-file", type=Path, help="从 UTF-8 文本文件读取会前预设"
    )
    parser.add_argument(
        "--assistant-interval",
        type=float,
        default=8.0,
        help="普通发言自动生成下一句建议的最短间隔，默认 8 秒",
    )
    parser.add_argument(
        "--asr-final-silence-ms",
        type=int,
        default=600,
        help="判定一句话结束的静音毫秒数，默认 600",
    )
    parser.add_argument(
        "--asr-language",
        choices=("zh", "en", "auto"),
        default="zh",
        help="实时转写主要语言，默认中文",
    )
    return parser


def _sounddevice():
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError("缺少 sounddevice，请先安装 requirements.txt。") from exc
    return sd


def _list_devices(sd) -> None:
    default_input, default_output = sd.default.device
    print("可用音频设备：")
    for index, item in enumerate(sd.query_devices()):
        flags = []
        if index == default_input:
            flags.append("默认输入")
        if index == default_output:
            flags.append("默认输出")
        suffix = f" [{' / '.join(flags)}]" if flags else ""
        print(
            f"  {index}: {item['name']} | in={item['max_input_channels']} "
            f"out={item['max_output_channels']} rate={item['default_samplerate']:.0f}{suffix}"
        )


def _resolved_output_info(sd, device):
    if device is None:
        device = sd.default.device[1]
    return sd.query_devices(device, "output")


def _resolved_input_info(sd, device):
    if device is None:
        device = sd.default.device[0]
    return sd.query_devices(device, "input")


def _find_virtual_output(sd) -> int:
    for index, item in enumerate(sd.query_devices()):
        if item["max_output_channels"] >= 2 and _looks_like_virtual_audio(
            str(item["name"])
        ):
            return index
    raise SystemExit(
        "没有找到 BlackHole/Loopback 虚拟麦克风。请先安装 BlackHole 2ch 并重启 Mac。"
    )


def _looks_like_headphones(name: str) -> bool:
    normalized = name.casefold()
    return any(
        token in normalized
        for token in (
            "headphone",
            "headset",
            "airpods",
            "earbuds",
            "耳机",
            "耳麦",
            "耳塞",
        )
    )


def _looks_like_virtual_audio(name: str) -> bool:
    normalized = name.casefold()
    return any(token in normalized for token in ("blackhole", "loopback", "vb-cable"))


@contextmanager
def _terminal_keys():
    if not sys.stdin.isatty():
        yield False
        return
    import termios
    import tty

    descriptor = sys.stdin.fileno()
    previous = termios.tcgetattr(descriptor)
    try:
        tty.setcbreak(descriptor)
        yield True
    finally:
        termios.tcsetattr(descriptor, termios.TCSADRAIN, previous)


def _print_stats(engine: LiveDenoiseEngine) -> None:
    stats = engine.snapshot_stats()
    print(
        "运行统计："
        f"输入={stats.input_blocks} 增强={stats.enhanced_blocks} "
        f"输入丢块={stats.input_drops} 输出丢块={stats.output_drops} "
        f"重同步={stats.resyncs} 欠载={stats.output_underruns} "
        f"推理p95={stats.inference_p95_ms:.2f}ms RTF={stats.realtime_factor:.3f}"
    )


def _load_meeting_preset(args) -> str:
    if args.meeting_preset and args.meeting_preset_file:
        raise SystemExit("--meeting-preset 与 --meeting-preset-file 只能使用一个")
    if args.meeting_preset_file:
        try:
            return args.meeting_preset_file.expanduser().read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise SystemExit(f"无法读取会议预设文件：{exc}") from exc
    if args.meeting_preset:
        return args.meeting_preset.strip()
    return os.environ.get(
        "AUDIORESCUE_MEETING_PRESET",
        "普通工作会议。我的目标是准确理解问题，并给出简洁、自然、可直接说出口的回答。",
    ).strip()


def _print_live_transcript(event: RealtimeAsrEvent) -> None:
    if event.is_final:
        print(f"\r[转写] {event.text}" + " " * 12)
    else:
        print(f"\r[实时字幕] {event.text[:80]}" + " " * 8, end="", flush=True)


def _print_live_suggestion(text: str, is_final: bool) -> None:
    if is_final:
        print(f"[建议] {text}")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    sd = _sounddevice()
    if args.list_devices:
        _list_devices(sd)
        return 0
    if args.duration is not None and args.duration <= 0:
        raise SystemExit("--duration 必须大于 0")
    if args.assistant_interval < 0:
        raise SystemExit("--assistant-interval 必须为非负数")
    if not 200 <= args.asr_final_silence_ms <= 6000:
        raise SystemExit("--asr-final-silence-ms 必须在 200 到 6000 之间")
    if args.virtual_mic and args.output_device is not None:
        raise SystemExit("--virtual-mic 与 --output-device 只能使用一个")
    if args.virtual_mic:
        args.output_device = _find_virtual_output(sd)

    input_info = _resolved_input_info(sd, args.input_device)
    input_name = str(input_info["name"])
    if args.virtual_mic and _looks_like_virtual_audio(input_name):
        raise SystemExit(
            f"当前输入设备“{input_name}”是虚拟音频设备，会形成自读自写回路。"
            "请用 --input-device 选择真实麦克风。"
        )
    output_info = _resolved_output_info(sd, args.output_device)
    output_name = str(output_info["name"])
    output_is_unconfirmed = not (
        _looks_like_headphones(output_name) or _looks_like_virtual_audio(output_name)
    )
    if args.gain > 0 and output_is_unconfirmed and not args.allow_speakers:
        raise SystemExit(
            f"当前输出设备“{output_name}”未被识别为耳机，直接监听可能产生回声或啸叫。"
            "请先连接耳机；若已确认它确实是耳机，可加 --allow-speakers。"
        )

    model_path = ensure_gtcrn_model(args.model, download=not args.no_download)
    denoiser = GtcrnDenoiser(model_path, num_threads=1)
    leveler = None
    quiet_leveler = None
    if not args.no_leveler:
        leveler = VoiceLeveler(
            sample_rate=denoiser.sample_rate,
            block_size=denoiser.frame_shift_in_samples,
        )
        quiet_leveler = QuietVoiceLeveler(
            sample_rate=args.stream_rate,
            block_size=(
                args.stream_rate
                * denoiser.frame_shift_in_samples
                // denoiser.sample_rate
            ),
        )
    elif args.mode == "quiet":
        raise SystemExit("--mode quiet 不能与 --no-leveler 同时使用")
    meeting = None
    if args.meeting_assistant:
        meeting = LiveMeetingAssistant(
            preset=_load_meeting_preset(args),
            suggestion_interval=args.assistant_interval,
            final_silence_ms=args.asr_final_silence_ms,
            language_hint=None if args.asr_language == "auto" else args.asr_language,
            on_transcript=_print_live_transcript,
            on_suggestion=_print_live_suggestion,
        )
    engine = LiveDenoiseEngine(
        denoiser,
        stream_sample_rate=args.stream_rate,
        output_queue_blocks=args.queue_blocks,
        output_gain=args.gain,
        enhancement_processor=leveler,
        quiet_processor=quiet_leveler,
        enhanced_frame_sink=meeting.accept_audio if meeting is not None else None,
        sink_queue_blocks=64 if meeting is not None else 32,
    )
    engine.set_mode(args.mode)

    started = time.monotonic()
    meeting_error_reported = None
    try:
        if meeting is not None:
            print("正在连接百炼实时转写与会议助手……")
            meeting.start()
            print("会议助手已连接。")
        engine.start(
            input_device=args.input_device,
            output_device=args.output_device,
            latency="low",
        )
        print(
            f"实时降噪已启动：声卡 {engine.sample_rate} Hz / {engine.block_size} samples，"
            f"GTCRN {denoiser.sample_rate} Hz / {denoiser.frame_shift_in_samples} samples。"
        )
        if args.virtual_mic:
            print("会议模式：请在会议软件中把麦克风选为 BlackHole 2ch，扬声器仍选耳机。")
        controls = (
            "空格：原声/当前模式切换；E：降噪增强；"
            "V：安静环境小声增强；R：原声；Q：退出"
        )
        if meeting is not None:
            controls += "；N：立即生成下一句建议"
        print(controls + "。")
        with _terminal_keys() as interactive:
            while True:
                engine.raise_if_failed()
                if meeting is not None:
                    meeting_snapshot = meeting.snapshot()
                    if (
                        meeting_snapshot.error
                        and meeting_snapshot.error != meeting_error_reported
                    ):
                        meeting_error_reported = meeting_snapshot.error
                        print(f"\n[会议助手异常] {meeting_error_reported}")
                if args.duration is not None and time.monotonic() - started >= args.duration:
                    break
                if interactive and select.select([sys.stdin], [], [], 0.1)[0]:
                    key = sys.stdin.read(1).casefold()
                    if key == "q":
                        break
                    if key == " ":
                        engine.toggle_bypass()
                        mode_label = {
                            "raw": "原声",
                            "enhanced": "降噪增强",
                            "quiet": "安静环境小声增强",
                        }[engine.mode]
                        print("\r监听：" + mode_label + "      ")
                    elif key == "e":
                        engine.set_mode("enhanced")
                        print("\r监听：降噪增强      ")
                    elif key == "v":
                        if quiet_leveler is None:
                            print("\r小声增强已被 --no-leveler 关闭      ")
                        else:
                            engine.set_mode("quiet")
                            print("\r监听：安静环境小声增强      ")
                    elif key == "r":
                        engine.set_mode("raw")
                        print("\r监听：原声      ")
                    elif key == "n" and meeting is not None:
                        meeting.request_next_line()
                elif not interactive:
                    time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            engine.stop()
        finally:
            if meeting is not None:
                meeting.stop()
    _print_stats(engine)
    if meeting is not None:
        snapshot = meeting.snapshot()
        print(
            "会议助手统计："
            f"转写句数={len(snapshot.transcript)} "
            f"ASR丢包={snapshot.asr_dropped_packets} "
            f"状态={snapshot.status}"
        )
        if snapshot.error:
            print(f"会议助手最后错误：{snapshot.error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
