"""B-owned Gradio entry point.

The UI is fixture-first for offline development, and production mode calls only
C's frozen `process_audio()` entry point.
"""

from __future__ import annotations

from ui.layout import build_demo


def main() -> None:
    demo = build_demo()
    demo.launch()


if __name__ == "__main__":
    main()
