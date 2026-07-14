"""Isolated boundary for the optional P1 environment-event stage.

This module intentionally knows nothing about CLAP (or any other model
package).  P0 callers leave the stage disabled and get an empty result without
loading a detector.  A future P1 adapter can be injected explicitly after its
own dependencies and weights have been validated outside the main pipeline.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

from core.schemas import EventResult


EventDetector = Callable[..., Iterable[EventResult]]


class EventDetectionError(RuntimeError):
    """An explicitly enabled optional detector could not produce safe results."""


def detect_events(
    original_wav: str,
    labels: Sequence[str] | None = None,
    window_seconds: float = 2.0,
    hop_seconds: float = 1.0,
    *,
    enabled: bool = False,
    detector: EventDetector | None = None,
) -> list[EventResult]:
    """Run an injected detector against the original track only.

    The disabled path deliberately returns before validating inputs or touching
    ``detector``.  This keeps P0 startup and dependencies identical when P1 is
    off.  When enabled, failures are normalized to ``EventDetectionError`` so
    the pipeline can add ``EVENTS_SKIPPED`` and continue its P0 success path.

    The injected callable must accept the keyword arguments ``original_wav``,
    ``labels``, ``window_seconds`` and ``hop_seconds``, and return an iterable
    of the already-frozen ``EventResult`` schema.
    """

    if not enabled:
        return []

    if detector is None or not callable(detector):
        raise EventDetectionError(
            "环境事件检测已开启，但没有提供可调用的 detector；请跳过 P1 并继续 P0。"
        )

    normalized_labels = _normalize_labels(labels)
    if window_seconds <= 0:
        raise EventDetectionError("window_seconds 必须大于 0；请跳过 P1 并继续 P0。")
    if hop_seconds <= 0:
        raise EventDetectionError("hop_seconds 必须大于 0；请跳过 P1 并继续 P0。")

    try:
        raw_results = detector(
            original_wav=original_wav,
            labels=normalized_labels,
            window_seconds=float(window_seconds),
            hop_seconds=float(hop_seconds),
        )
        results = list(raw_results)
    except EventDetectionError:
        raise
    except Exception as exc:
        raise EventDetectionError(
            "环境事件检测执行失败；请记录 EVENTS_SKIPPED 并继续 P0。"
        ) from exc

    if any(not isinstance(item, EventResult) for item in results):
        raise EventDetectionError(
            "环境事件 detector 返回了非 EventResult 数据；请跳过 P1 并继续 P0。"
        )
    return results


def _normalize_labels(labels: Sequence[str] | None) -> list[str]:
    if labels is None:
        raise EventDetectionError(
            "环境事件检测已开启，但候选标签为空；请跳过 P1 并继续 P0。"
        )

    normalized = [label.strip() for label in labels if isinstance(label, str) and label.strip()]
    if not normalized:
        raise EventDetectionError(
            "环境事件检测已开启，但候选标签为空；请跳过 P1 并继续 P0。"
        )
    return normalized


__all__ = ["EventDetectionError", "EventDetector", "detect_events"]
