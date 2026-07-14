"""B-owned Gradio entry point.

Production mode calls only C's frozen `process_audio()` entry point. Fixture
mode is available only when the process explicitly opts in through
`AUDIORESCUE_UI_FIXTURE`.
"""

from __future__ import annotations

from ui.file_staging import default_staging_root
from ui.layout import build_demo, strip_remote_html_resources


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

    return {"middleware": [Middleware(OfflineHtmlResourceMiddleware)]}


def main() -> None:
    demo = build_demo()
    demo.launch(
        allowed_paths=[str(default_staging_root())],
        app_kwargs=offline_launch_app_kwargs(),
        enable_monitoring=False,
    )


if __name__ == "__main__":
    main()
