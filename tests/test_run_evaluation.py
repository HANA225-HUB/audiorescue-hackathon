import csv
import hashlib
import io
import json
import math
import random
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from scripts.build_dataset import APPROVED_CONSENT_TOKEN, MANIFEST_FIELDS, build_dataset
from scripts.dataset_spec import DatasetSpecError, load_dataset_spec
from scripts.run_evaluation import (
    EvaluationError,
    build_parser,
    locked_receipt_path,
    main,
    run_evaluation,
)

try:
    from tests.test_build_dataset import approve_sources, make_sources
    from tests.test_build_dataset import write_test_wav
    from tests.test_dataset_spec import synthetic_spec_payload, write_synthetic_spec
except ModuleNotFoundError:
    from test_build_dataset import approve_sources, make_sources
    from test_build_dataset import write_test_wav
    from test_dataset_spec import synthetic_spec_payload, write_synthetic_spec


def result_payload(
    status: str = "success",
    before: float | None = 0.5,
    after: float | None = 0.25,
) -> dict[str, object]:
    return {
        "job_id": "job",
        "status": status,
        "cer_before": None if before is None else {"cer": before},
        "cer_after": None if after is None else {"cer": after},
    }


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def build_approved_fixture(root: Path, spec_path: Path) -> list[dict[str, object]]:
    spec = load_dataset_spec(spec_path)
    for index, item in enumerate(spec.clean_recordings, start=1):
        clean = [
            round(5_000 * math.sin(2 * math.pi * (index + 3) * frame / 997))
            for frame in range(48_000)
        ]
        write_test_wav(root / item.path, clean)
    for index, item in enumerate(spec.noise_recordings, start=1):
        rng = random.Random(500 + index)
        noise = [rng.randint(-4_000, 4_000) for _ in range(96_000)]
        write_test_wav(root / item.path, noise)
    for index, item in enumerate(spec.real_recordings, start=1):
        real = [
            round(4_000 * math.sin(2 * math.pi * (index + 5) * frame / 991))
            for frame in range(48_000)
        ]
        write_test_wav(root / item.path, real)
    approve_sources(root, spec_path)
    return build_dataset(
        root,
        spec_path=spec_path,
        consent_or_license=APPROVED_CONSENT_TOKEN,
    )


class EvaluationSpecSelectionTest(unittest.TestCase):
    def test_parser_requires_dataset_spec_argument(self) -> None:
        option_strings = {
            option
            for action in build_parser()._actions
            for option in action.option_strings
        }
        self.assertIn("--dataset-spec", option_strings)

    def test_run_evaluation_requires_explicit_dataset_spec(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "data_local"
            root.mkdir()
            manifest = root / "manifest.csv"
            manifest.write_text("sample_id\n", encoding="utf-8")

            with self.assertRaisesRegex(EvaluationError, "explicit dataset spec"):
                run_evaluation(
                    manifest_path=manifest,
                    dataset_root=root,
                    split="dev",
                    output_dir=Path(temp_dir) / "out",
                    process_callable=lambda **_: result_payload(),
                )

    def test_cli_refusal_is_sanitized_when_spec_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "data_local"
            root.mkdir()
            manifest = root / "manifest.csv"
            manifest.write_text("sample_id\n", encoding="utf-8")
            output = io.StringIO()

            with redirect_stdout(output):
                exit_code = main(
                    [
                        "--manifest",
                        str(manifest),
                        "--dataset-root",
                        str(root),
                        "--split",
                        "dev",
                        "--output-dir",
                        str(root / "evaluation"),
                    ]
                )

            rendered = output.getvalue()
            self.assertEqual(exit_code, 2)
            self.assertIn("Evaluation refused", rendered)
            self.assertNotIn(temp_dir, rendered)
            self.assertNotIn(str(manifest), rendered)

    def test_dev_split_uses_explicit_spec_size_and_sanitized_output_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            root = temp / "data_local"
            spec_path = write_synthetic_spec(temp / "dataset_spec.json")
            rows = build_approved_fixture(root, spec_path)
            calls: list[dict[str, object]] = []

            def fake_process(**kwargs: object) -> dict[str, object]:
                calls.append(kwargs)
                return result_payload()

            index, summary = run_evaluation(
                manifest_path=root / "manifest.csv",
                dataset_root=root,
                dataset_spec=spec_path,
                split="dev",
                output_dir=root / "evaluation",
                process_callable=fake_process,
            )

            dev_rows = [row for row in rows if row["split"] == "dev"]
            self.assertEqual(len(calls), len(dev_rows))
            self.assertEqual(summary["sample_count"], len(dev_rows))
            encoded = json.dumps(index, ensure_ascii=False)
            self.assertNotIn(temp_dir, encoded)
            self.assertIn("controlled/dev/sample_mix_1.wav", encoded)

    def test_manifest_semantics_are_validated_with_same_spec_before_processing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            root = temp / "data_local"
            spec_path = write_synthetic_spec(temp / "dataset_spec.json")
            rows = [dict(row) for row in build_approved_fixture(root, spec_path)]
            rows[0]["clean_path"] = rows[1]["clean_path"]
            write_manifest(root / "manifest.csv", rows)  # type: ignore[arg-type]
            calls = 0

            def fake_process(**_: object) -> dict[str, object]:
                nonlocal calls
                calls += 1
                return result_payload()

            with self.assertRaisesRegex(EvaluationError, "dataset validation failed"):
                run_evaluation(
                    manifest_path=root / "manifest.csv",
                    dataset_root=root,
                    dataset_spec=spec_path,
                    split="dev",
                    output_dir=root / "evaluation",
                    process_callable=fake_process,
                )
            self.assertEqual(calls, 0)

    def test_spec_hash_is_recorded_without_spec_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            root = temp / "data_local"
            spec_path = write_synthetic_spec(temp / "dataset_spec.json")
            build_approved_fixture(root, spec_path)

            index, _summary = run_evaluation(
                manifest_path=root / "manifest.csv",
                dataset_root=root,
                dataset_spec=spec_path,
                split="dev",
                output_dir=root / "evaluation",
                process_callable=lambda **_: result_payload(),
            )

            self.assertEqual(
                index["dataset_spec"]["sha256"],
                hashlib.sha256(spec_path.read_bytes()).hexdigest(),
            )
            self.assertNotIn("path", index["dataset_spec"])


class EvaluationFreezeCompatibilityTest(unittest.TestCase):
    def test_locked_receipt_path_requires_the_same_explicit_spec(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            root = temp / "data_local"
            payload = synthetic_spec_payload()
            payload["mixes"][1]["split"] = "locked_test"
            payload["protected_splits"] = ["locked_test", "clean_control"]
            spec_path = write_synthetic_spec(temp / "dataset_spec.json", payload)
            build_approved_fixture(root, spec_path)
            freeze = temp / "freeze.json"
            freeze.write_text(
                json.dumps(
                    {
                        "dataset_version": load_dataset_spec(spec_path).dataset_version,
                        "manifest": {
                            "path": str((root / "manifest.csv").resolve()),
                            "sha256": hashlib.sha256(
                                (root / "manifest.csv").read_bytes()
                            ).hexdigest(),
                        },
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(EvaluationError, "explicit dataset spec"):
                locked_receipt_path(freeze)
            self.assertEqual(
                locked_receipt_path(freeze, dataset_spec=spec_path).suffix,
                ".json",
            )


if __name__ == "__main__":
    unittest.main()
