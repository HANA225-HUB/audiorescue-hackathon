"""B-owned Gradio entry point.

Production mode calls only C's frozen `process_audio()` entry point. Fixture
mode is available only when the process explicitly opts in through
`AUDIORESCUE_UI_FIXTURE`.
"""

from __future__ import annotations

from ui.file_staging import default_staging_root
from ui.layout import build_demo, offline_launch_app_kwargs


def main() -> None:
    demo = build_demo()
    demo.launch(
        allowed_paths=[str(default_staging_root())],
        app_kwargs=offline_launch_app_kwargs(),
        enable_monitoring=False,
    )


if __name__ == "__main__":
    main()
