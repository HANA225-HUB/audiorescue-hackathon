import json
import tempfile
import unittest
from pathlib import Path

from scripts.dataset_spec import (
    DatasetSpecError,
    ensure_private_repo_root,
    load_example_dataset_spec,
    load_dataset_spec,
    safe_join,
    spec_content_hash,
)


def synthetic_spec_payload() -> dict:
    return {
        "dataset_version": "audiorescue-synthetic-template-v1",
        "sample_rate": 48_000,
        "channels": 1,
        "sample_width_bytes": 2,
        "protected_splits": ["holdout", "clean_control"],
        "consent_tokens": {
            "pending": "pending_local_authorization",
            "approved": "approved_for_evaluation",
        },
        "clean_recordings": [
            {
                "id": "clean_1",
                "speaker_id": "speaker_1",
                "sentence_id": "sentence_1",
                "reference_text": "synthetic reference one",
                "path": "raw/clean/speaker_1/sample_clean_1.wav",
                "split": "dev",
            },
            {
                "id": "clean_2",
                "speaker_id": "speaker_2",
                "sentence_id": "sentence_2",
                "reference_text": "synthetic reference two",
                "path": "raw/clean/speaker_2/sample_clean_2.wav",
                "split": "holdout",
            },
        ],
        "noise_recordings": [
            {
                "id": "noise_1",
                "noise_type": "noise_class_1",
                "path": "raw/noise/noise_class_1.wav",
            }
        ],
        "mixes": [
            {
                "sample_id": "sample_mix_1",
                "clean_id": "clean_1",
                "noise_id": "noise_1",
                "split": "dev",
                "snr_db": 0,
                "snr_label": "snr000",
                "path": "controlled/dev/sample_mix_1.wav",
                "is_locked": False,
                "is_demo_candidate": True,
            },
            {
                "sample_id": "sample_mix_2",
                "clean_id": "clean_2",
                "noise_id": "noise_1",
                "split": "holdout",
                "snr_db": -3,
                "snr_label": "snrm03",
                "path": "controlled/holdout/sample_mix_2.wav",
                "is_locked": True,
                "is_demo_candidate": False,
            },
        ],
        "clean_controls": [
            {
                "sample_id": "sample_clean_control_1",
                "clean_id": "clean_1",
                "split": "clean_control",
                "is_locked": True,
                "is_demo_candidate": False,
            }
        ],
        "real_recordings": [
            {
                "sample_id": "sample_real_1",
                "speaker_id": "speaker_2",
                "sentence_id": "sentence_2",
                "reference_text": "synthetic real reference",
                "noise_type": "noise_class_1",
                "path": "raw/real/sample_real_1.wav",
                "split": "real",
                "is_locked": False,
                "is_demo_candidate": False,
            }
        ],
    }


def write_synthetic_spec(path: Path, payload: dict | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload or synthetic_spec_payload(), indent=2) + "\n",
        encoding="utf-8",
    )
    return path


class DatasetSpecTest(unittest.TestCase):
    def test_requires_explicit_spec_unless_example_mode_is_requested(self) -> None:
        with self.assertRaisesRegex(DatasetSpecError, "explicit dataset spec"):
            load_dataset_spec()

        spec = load_example_dataset_spec()
        self.assertEqual(spec.dataset_version, "audiorescue-synthetic-template-v1")

    def test_loads_neutral_example_spec_without_private_shape(self) -> None:
        spec = load_example_dataset_spec()

        self.assertEqual(spec.dataset_version, "audiorescue-synthetic-template-v1")
        self.assertEqual(len(spec.clean_recordings), 2)
        self.assertEqual(len(spec.noise_recordings), 1)
        self.assertEqual(len(spec.mixes), 2)
        self.assertEqual(len(spec.clean_controls), 1)
        self.assertEqual(len(spec.real_recordings), 1)
        rendered = json.dumps(spec.to_public_dict(), ensure_ascii=False)
        self.assertIn("speaker_", rendered)
        self.assertIn("sentence_", rendered)
        self.assertIn("noise_class_", rendered)
        self.assertNotRegex(rendered, r"speaker_id\"\s*:\s*\"[A-Z]\"")

    def test_rejects_traversal_without_echoing_path_value(self) -> None:
        payload = synthetic_spec_payload()
        payload["clean_recordings"][0]["path"] = "../private/sample.wav"
        with tempfile.TemporaryDirectory() as temp_dir:
            spec_path = write_synthetic_spec(Path(temp_dir) / "spec.json", payload)

            with self.assertRaises(DatasetSpecError) as raised:
                load_dataset_spec(spec_path)

        rendered = str(raised.exception)
        self.assertIn("relative path", rendered)
        self.assertNotIn("private", rendered)
        self.assertNotIn("sample.wav", rendered)

    def test_rejects_duplicate_ids_and_unknown_references(self) -> None:
        duplicate_payload = synthetic_spec_payload()
        duplicate_payload["noise_recordings"].append(
            {
                "id": "noise_1",
                "noise_type": "noise_class_2",
                "path": "raw/noise/noise_class_2.wav",
            }
        )
        unknown_payload = synthetic_spec_payload()
        unknown_payload["mixes"][0]["clean_id"] = "missing_clean"
        with tempfile.TemporaryDirectory() as temp_dir:
            duplicate_path = write_synthetic_spec(
                Path(temp_dir) / "duplicate.json", duplicate_payload
            )
            unknown_path = write_synthetic_spec(
                Path(temp_dir) / "unknown.json", unknown_payload
            )

            with self.assertRaisesRegex(DatasetSpecError, "duplicate"):
                load_dataset_spec(duplicate_path)
            with self.assertRaisesRegex(DatasetSpecError, "unknown clean_id"):
                load_dataset_spec(unknown_path)

    def test_rejects_cross_type_id_collisions_without_echoing_values(self) -> None:
        payload = synthetic_spec_payload()
        payload["noise_recordings"][0]["id"] = payload["clean_recordings"][0]["id"]
        with tempfile.TemporaryDirectory() as temp_dir:
            spec_path = write_synthetic_spec(Path(temp_dir) / "collision.json", payload)

            with self.assertRaises(DatasetSpecError) as raised:
                load_dataset_spec(spec_path)

        rendered = str(raised.exception)
        self.assertIn("duplicate", rendered)
        self.assertNotIn(payload["clean_recordings"][0]["id"], rendered)

    def test_rejects_non_frozen_audio_format_values(self) -> None:
        for field_name, replacement in (
            ("sample_rate", 44_100),
            ("channels", 2),
            ("sample_width_bytes", 3),
        ):
            with self.subTest(field_name=field_name), tempfile.TemporaryDirectory() as temp_dir:
                payload = synthetic_spec_payload()
                payload[field_name] = replacement
                spec_path = write_synthetic_spec(Path(temp_dir) / "format.json", payload)

                with self.assertRaisesRegex(DatasetSpecError, "48 kHz mono PCM16"):
                    load_dataset_spec(spec_path)

    def test_protected_locked_and_demo_invariants_are_enforced(self) -> None:
        cases = [
            ("protected_without_lock", ("mixes", 1, "is_locked"), False),
            ("protected_demo", ("mixes", 1, "is_demo_candidate"), True),
            ("locked_demo", ("mixes", 0, "is_locked"), True),
            ("locked_demo", ("mixes", 0, "is_demo_candidate"), True),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            for label, (section, index, field_name), value in cases:
                payload = synthetic_spec_payload()
                payload[section][index][field_name] = value
                if label == "locked_demo":
                    payload["mixes"][0]["is_locked"] = True
                    payload["mixes"][0]["is_demo_candidate"] = True
                spec_path = write_synthetic_spec(
                    Path(temp_dir) / f"{label}_{field_name}.json", payload
                )

                with self.subTest(label=label, field_name=field_name):
                    with self.assertRaisesRegex(DatasetSpecError, "split invariant"):
                        load_dataset_spec(spec_path)

    def test_spec_hash_depends_on_content_not_location(self) -> None:
        payload = synthetic_spec_payload()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            first = write_synthetic_spec(temp / "one" / "spec.json", payload)
            second = write_synthetic_spec(temp / "two" / "renamed.json", payload)

            self.assertEqual(spec_content_hash(first), spec_content_hash(second))
            self.assertRegex(spec_content_hash(first), r"^[0-9a-f]{64}$")

    def test_safe_join_rejects_symlink_escape_before_access(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            root = temp / "dataset"
            outside = temp / "outside"
            root.mkdir()
            outside.mkdir()
            link = root / "raw"
            try:
                link.symlink_to(outside, target_is_directory=True)
            except (OSError, NotImplementedError) as exc:
                self.skipTest(f"directory symlink unavailable: {exc}")

            with self.assertRaisesRegex(DatasetSpecError, "unsafe path"):
                safe_join(root, Path("raw") / "escape.wav", must_exist=False)

    def test_safe_join_rejects_colon_in_any_path_segment(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for raw_path in ("raw/clean/a:b.wav", "raw/C:/sample.wav"):
                with self.subTest(raw_path=raw_path):
                    with self.assertRaisesRegex(DatasetSpecError, "unsafe path"):
                        safe_join(root, raw_path)

    def test_private_root_rejects_symlink_parent_before_resolve(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            outside = temp / "outside"
            outside.mkdir()
            link = temp / "linked_root"
            try:
                link.symlink_to(outside, target_is_directory=True)
            except (OSError, NotImplementedError) as exc:
                self.skipTest(f"directory symlink unavailable: {exc}")

            with self.assertRaisesRegex(DatasetSpecError, "unsafe path"):
                ensure_private_repo_root(link / "dataset", project_root=temp)
            self.assertFalse((outside / "dataset").exists())


if __name__ == "__main__":
    unittest.main()
