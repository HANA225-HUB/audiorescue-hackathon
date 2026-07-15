"""B-owned Gradio entry point.

Production mode calls only C's frozen `process_audio()` entry point. Fixture
mode is available only when the process explicitly opts in through
`AUDIORESCUE_UI_FIXTURE`.
"""

from __future__ import annotations

import json

from ui.file_delivery import FileDeliveryMiddleware
from ui.live_controller import LIVE_CONTROLLER, snapshot_to_dict
from ui.layout import build_demo, strip_remote_html_resources


class _RequestTooLarge(ValueError):
    pass


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
    label {
      display: block;
      margin-bottom: 7px;
      color: #5b4635;
      font-size: 14px;
      font-weight: 700;
    }
    textarea {
      box-sizing: border-box;
      width: 100%;
      min-height: 88px;
      resize: vertical;
      border: 1px solid rgba(139, 99, 60, .34);
      border-radius: 8px;
      padding: 10px 12px;
      background: rgba(255, 250, 242, .9);
      color: #2b2118;
      font: inherit;
      line-height: 1.5;
    }
    textarea:focus {
      outline: 2px solid rgba(123, 90, 53, .26);
      border-color: #7b5a35;
    }
    .quick-actions { display: grid; gap: 10px; }
    .action-button {
      min-height: 42px;
      border: 1px solid #6b4c2f;
      border-radius: 8px;
      padding: 9px 12px;
      background: #6b4c2f;
      color: #fffaf2;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
    }
    .action-button.secondary {
      background: rgba(255, 250, 242, .78);
      color: #4e3825;
    }
    .action-button:disabled { cursor: wait; opacity: .56; }
    .action-status {
      min-height: 24px;
      color: #5b4635;
      font-size: 13px;
    }
    .dot {
      display: inline-block;
      width: 9px;
      height: 9px;
      border-radius: 50%;
      background: #7b5a35;
      margin-right: 7px;
      box-shadow: 0 0 0 6px rgba(123,90,53,.12);
    }
    .home-link {
      display: inline-flex;
      align-items: center;
      min-height: 32px;
      margin: 0 0 12px;
      border: 1px solid rgba(139, 99, 60, .24);
      border-radius: 999px;
      padding: 5px 12px;
      background: rgba(255, 246, 230, .68);
      color: #4e3825;
      font-size: 13px;
      font-weight: 700;
      text-decoration: none;
      box-shadow: 0 8px 18px rgba(96,66,39,.08);
    }
    .home-link:hover {
      background: rgba(255, 235, 202, .84);
    }
  </style>
</head>
<body>
<main>
  <a class="home-link" id="return-main" href="/" target="_self">返回主界面</a>
  <div class="card">
    <header><span class="dot"></span><span id="status">连接中</span></header>
    <h1 id="suggestion">暂无建议</h1>
    <p id="suggestion-meta">建议状态：待生成</p>
    <section class="quick-actions">
      <h2>答辩快捷操作</h2>
      <button class="action-button secondary" id="next-section" type="button">
        立即生成下一段建议
      </button>
      <div>
        <label for="remote-question">对方刚刚说了什么？</label>
        <textarea
          id="remote-question"
          maxlength="2000"
          placeholder="输入评委或导师的问题，支持 Command/Ctrl + Enter 提交"
        ></textarea>
      </div>
      <button class="action-button" id="answer-question" type="button">
        根据会议资料生成回答
      </button>
      <p class="action-status" id="action-status" role="status" aria-live="polite">
        会议助手启动后可在这里直接操作。
      </p>
    </section>
    <section>
      <h2 id="verification">会议上下文</h2>
      <p id="sources">暂无资料引用</p>
    </section>
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
document.getElementById("return-main").addEventListener("click", (event) => {
  if (window.opener && !window.opener.closed) {
    event.preventDefault();
    try {
      window.opener.focus();
    } catch (error) {
      // Focusing the opener may be restricted in embedded browsers.
    }
    window.close();
  }
});

const actionButtons = [
  document.getElementById("next-section"),
  document.getElementById("answer-question")
];
const questionInput = document.getElementById("remote-question");
const actionStatus = document.getElementById("action-status");
let actionInFlight = false;
let refreshInFlight = false;

function setActionBusy(busy) {
  actionButtons.forEach((button) => {
    button.disabled = busy;
  });
}

async function runAction(action) {
  if (actionInFlight) {
    return;
  }
  const question = questionInput.value.trim();
  if (action === "answer" && !question) {
    actionStatus.textContent = "请先输入对方的问题。";
    questionInput.focus();
    return;
  }
  actionInFlight = true;
  setActionBusy(true);
  actionStatus.textContent = action === "answer"
    ? "正在提交问题……"
    : "正在请求下一段建议……";
  try {
    const response = await fetch("/audiorescue/live/action", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action, question })
    });
    const result = await response.json();
    if (!response.ok) {
      throw new Error(result.error || "操作失败。");
    }
    actionStatus.textContent = result.last_action || "操作已提交。";
    await refresh();
  } catch (error) {
    actionStatus.textContent = error instanceof Error ? error.message : "操作失败。";
  } finally {
    actionInFlight = false;
    setActionBusy(false);
  }
}

document.getElementById("next-section").addEventListener("click", () => {
  runAction("next_line");
});
document.getElementById("answer-question").addEventListener("click", () => {
  runAction("answer");
});
questionInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
    event.preventDefault();
    runAction("answer");
  }
});

async function refresh() {
  if (refreshInFlight) {
    return;
  }
  refreshInFlight = true;
  try {
    const response = await fetch("/audiorescue/live/state", { cache: "no-store" });
    if (!response.ok) {
      throw new Error("无法读取会议状态。");
    }
    const state = await response.json();
    document.getElementById("status").textContent = "会议助手 · " + state.meeting_status;
    document.getElementById("suggestion").textContent = state.suggestion || "暂无建议";
    const kindLabels = {
      answer: "问题回答",
      continue_section: "继续当前内容",
      next_section: "下一段提示",
      clarify: "澄清建议",
      correction: "修正提示",
      close: "收尾提示"
    };
    const kind = kindLabels[state.suggestion_kind] || "待生成";
    const confidence = Number(state.confidence || 0);
    document.getElementById("suggestion-meta").textContent = confidence > 0
      ? kind + " · 置信度 " + Math.round(confidence * 100) + "%"
      : kind;
    document.getElementById("verification").textContent = state.needs_verification
      ? "⚠ 这条建议需要核实"
      : "会议上下文";
    const sources = (state.suggestion_sources || []).join(" / ");
    document.getElementById("sources").textContent = sources || "未引用会议资料";
    document.getElementById("partial").textContent = state.partial_text || "暂无实时字幕";
    const transcript = (state.transcript || []).slice(-5).join("\\n");
    document.getElementById("transcript").textContent = transcript || "暂无正式字幕";
  } catch (error) {
    document.getElementById("status").textContent = "连接断开";
  } finally {
    refreshInFlight = false;
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

    @staticmethod
    async def _send_json(
        send,
        payload: dict,
        *,
        status: int = 200,
        extra_headers: tuple[tuple[bytes, bytes], ...] = (),
    ) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [
                    (b"content-type", b"application/json; charset=utf-8"),
                    (b"cache-control", b"no-store"),
                    (b"x-content-type-options", b"nosniff"),
                    (b"content-length", str(len(body)).encode("ascii")),
                    *extra_headers,
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})

    @staticmethod
    async def _read_json(receive, *, max_bytes: int = 16_384) -> dict:
        parts: list[bytes] = []
        size = 0
        while True:
            message = await receive()
            if message.get("type") == "http.disconnect":
                raise ValueError("请求连接已中断。")
            if message.get("type") != "http.request":
                raise ValueError("收到了无效请求消息。")
            chunk = message.get("body", b"")
            size += len(chunk)
            if size > max_bytes:
                raise _RequestTooLarge("请求内容过长。")
            parts.append(chunk)
            if not message.get("more_body", False):
                break
        try:
            payload = json.loads(b"".join(parts).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("请求内容不是有效 JSON。") from exc
        if not isinstance(payload, dict):
            raise ValueError("请求内容必须是 JSON 对象。")
        return payload

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        path = scope.get("path") or ""
        if path == "/audiorescue/live/state":
            await self._send_json(send, snapshot_to_dict())
            return
        if path == "/audiorescue/live/action":
            if scope.get("method") != "POST":
                await self._send_json(
                    send,
                    {"error": "只允许 POST 请求。"},
                    status=405,
                    extra_headers=((b"allow", b"POST"),),
                )
                return
            headers = {
                key.lower(): value
                for key, value in scope.get("headers", [])
            }
            content_type = headers.get(b"content-type", b"").decode(
                "latin-1", errors="ignore"
            )
            if not content_type.casefold().startswith("application/json"):
                await self._send_json(
                    send,
                    {"error": "请使用 application/json 提交。"},
                    status=415,
                )
                return
            try:
                payload = await self._read_json(receive)
                action = str(payload.get("action") or "").strip().casefold()
                if action == "next_line":
                    snapshot = LIVE_CONTROLLER.request_next_line()
                elif action == "answer":
                    question_value = payload.get("question")
                    if not isinstance(question_value, str):
                        raise ValueError("对方问题必须是文本。")
                    question = question_value.strip()
                    if not question:
                        raise ValueError("请先输入对方的问题。")
                    if len(question) > 2_000:
                        raise ValueError("对方问题不能超过 2000 字。")
                    snapshot = LIVE_CONTROLLER.request_answer(question)
                else:
                    raise ValueError("不支持的浮窗操作。")
            except _RequestTooLarge as exc:
                await self._send_json(send, {"error": str(exc)}, status=413)
                return
            except ValueError as exc:
                await self._send_json(send, {"error": str(exc)}, status=400)
                return
            except Exception as exc:
                LIVE_CONTROLLER.record_ui_error("悬浮窗操作失败", exc)
                await self._send_json(
                    send,
                    {"error": "操作失败，请查看会议状态后重试。"},
                    status=500,
                )
                return
            await self._send_json(
                send,
                {
                    "last_action": snapshot.last_action,
                    "state": snapshot_to_dict(snapshot),
                },
            )
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
