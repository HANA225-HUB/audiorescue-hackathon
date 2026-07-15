"""B-owned Gradio entry point.

Production mode calls only C's frozen `process_audio()` entry point. Fixture
mode is available only when the process explicitly opts in through
`AUDIORESCUE_UI_FIXTURE`.
"""

from __future__ import annotations

import json

from ui.file_delivery import FileDeliveryMiddleware
from ui.live_controller import snapshot_to_dict
from ui.layout import build_demo, strip_remote_html_resources


FLOATING_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AudioRescue 悬浮提示窗</title>
  <style>
    :root { color-scheme: light; }
    body {
      margin: 0;
      min-height: 100vh;
      background: #f2e4d0;
      color: #2b2118;
      font-family: "Songti SC", STSong, "Noto Serif CJK SC", Georgia, serif;
    }
    main {
      min-height: 100vh;
      padding: 18px;
      background:
        linear-gradient(rgba(255,255,255,.18) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,.14) 1px, transparent 1px),
        linear-gradient(145deg, rgba(250,231,203,.86), rgba(239,210,173,.72));
      background-size: 42px 42px, 42px 42px, auto;
    }
    .card {
      border: 1px solid rgba(139, 99, 60, .28);
      border-radius: 8px;
      padding: 18px;
      background: rgba(255, 240, 218, .72);
      box-shadow: 0 16px 38px rgba(96,66,39,.16);
    }
    header { color: #5b4635; font-size: 14px; margin-bottom: 14px; }
    h1 { margin: 0 0 14px; font-size: 24px; line-height: 1.35; }
    section { margin-top: 14px; }
    h2 { margin: 0 0 8px; font-size: 15px; color: #5b4635; }
    p { margin: 0; line-height: 1.65; white-space: pre-wrap; overflow-wrap: anywhere; }
    .dot {
      display: inline-block;
      width: 9px;
      height: 9px;
      border-radius: 50%;
      background: #7b5a35;
      margin-right: 7px;
      box-shadow: 0 0 0 6px rgba(123,90,53,.12);
    }
  </style>
</head>
<body>
<main>
  <div class="card">
    <header><span class="dot"></span><span id="status">连接中</span></header>
    <h1 id="suggestion">暂无建议</h1>
    <section>
      <h2>实时字幕</h2>
      <p id="partial">暂无实时字幕</p>
    </section>
    <section>
      <h2>正式记录</h2>
      <p id="transcript">暂无正式字幕</p>
    </section>
  </div>
</main>
<script>
async function refresh() {
  try {
    const response = await fetch("/audiorescue/live/state", { cache: "no-store" });
    const state = await response.json();
    document.getElementById("status").textContent = "会议助手 · " + state.meeting_status;
    document.getElementById("suggestion").textContent = state.suggestion || "暂无建议";
    document.getElementById("partial").textContent = state.partial_text || "暂无实时字幕";
    const transcript = (state.transcript || []).slice(-5).join("\\n");
    document.getElementById("transcript").textContent = transcript || "暂无正式字幕";
  } catch (error) {
    document.getElementById("status").textContent = "连接断开";
  }
}
refresh();
setInterval(refresh, 1000);
</script>
</body>
</html>"""


class LiveStateMiddleware:
    """Expose same-process live meeting snapshots for the floating helper window."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        path = scope.get("path") or ""
        if path == "/audiorescue/live/state":
            body = json.dumps(snapshot_to_dict(), ensure_ascii=False).encode("utf-8")
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [
                        (b"content-type", b"application/json; charset=utf-8"),
                        (b"cache-control", b"no-store"),
                        (b"content-length", str(len(body)).encode("ascii")),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": body})
            return
        if path == "/audiorescue/live/floating":
            body = FLOATING_HTML.encode("utf-8")
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [
                        (b"content-type", b"text/html; charset=utf-8"),
                        (b"cache-control", b"no-store"),
                        (b"content-length", str(len(body)).encode("ascii")),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": body})
            return
        await self.app(scope, receive, send)


class OfflineHtmlResourceMiddleware:
    """Filter Gradio default remote tags without touching non-HTML traffic."""

    _BODYLESS_STATUSES = {204, 304}

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http" or scope.get("method") == "HEAD":
            await self.app(scope, receive, send)
            return

        start_message = None
        should_filter = False
        body_parts: list[bytes] = []

        async def send_wrapper(message):
            nonlocal start_message, should_filter
            if message["type"] == "http.response.start":
                start_message = dict(message)
                headers = [
                    (key.lower(), value)
                    for key, value in start_message.get("headers", [])
                ]
                content_type_is_html = any(
                    key == b"content-type" and b"text/html" in value.lower()
                    for key, value in headers
                )
                should_filter = (
                    content_type_is_html
                    and int(start_message.get("status", 200))
                    not in self._BODYLESS_STATUSES
                )
                if not should_filter:
                    await send(message)
                return

            if message["type"] != "http.response.body" or not should_filter:
                await send(message)
                return

            body_parts.append(message.get("body", b""))
            if message.get("more_body", False):
                return

            raw_body = b"".join(body_parts)
            try:
                filtered_body = strip_remote_html_resources(
                    raw_body.decode("utf-8")
                ).encode("utf-8")
            except UnicodeDecodeError:
                filtered_body = raw_body

            headers = [
                (key, value)
                for key, value in start_message.get("headers", [])
                if key.lower() != b"content-length"
            ]
            headers.append((b"content-length", str(len(filtered_body)).encode("ascii")))
            start_message["headers"] = headers
            await send(start_message)
            await send({**message, "body": filtered_body, "more_body": False})

        await self.app(scope, receive, send_wrapper)


def offline_launch_app_kwargs():
    from starlette.middleware import Middleware

    return {
        "middleware": [
            Middleware(LiveStateMiddleware),
            Middleware(FileDeliveryMiddleware),
            Middleware(OfflineHtmlResourceMiddleware),
        ]
    }


def main() -> None:
    demo = build_demo()
    demo.launch(
        app_kwargs=offline_launch_app_kwargs(),
        enable_monitoring=False,
    )


if __name__ == "__main__":
    main()
