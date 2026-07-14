"""B-owned Gradio entry point.

Production mode calls only C's frozen `process_audio()` entry point. Fixture
mode is available only when the process explicitly opts in through
`AUDIORESCUE_UI_FIXTURE`.
"""

from __future__ import annotations

from ui.file_staging import default_staging_root
from ui.layout import build_demo


def main() -> None:
    demo = build_demo()
    demo.launch(allowed_paths=[str(default_staging_root())])


if __name__ == "__main__":
    main()
