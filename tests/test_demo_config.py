import hashlib
import unittest
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DemoManifestContractTest(unittest.TestCase):
    def test_manifest_paths_hashes_and_defaults_are_valid(self) -> None:
        manifest_path = PROJECT_ROOT / "configs" / "demo.yaml"
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        samples = manifest["samples"]

        self.assertGreaterEqual(len(samples), 1)
        identifiers = [sample["id"] for sample in samples]
        self.assertEqual(len(identifiers), len(set(identifiers)))
        self.assertEqual(sum(bool(sample.get("default")) for sample in samples), 1)

        for sample in samples:
            with self.subTest(sample=sample["id"]):
                path = (PROJECT_ROOT / sample["file"]).resolve()
                self.assertTrue(path.is_relative_to(PROJECT_ROOT))
                self.assertTrue(path.is_file())
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                self.assertEqual(digest, sample["sha256"])
                self.assertTrue(sample["name"].strip())
                self.assertIn(
                    sample["intended_use"],
                    {"development_only", "main_demo", "boundary_demo", "p1_demo"},
                )

    def test_development_fixture_is_not_claimed_as_competition_evidence(self) -> None:
        manifest = yaml.safe_load(
            (PROJECT_ROOT / "configs" / "demo.yaml").read_text(encoding="utf-8")
        )
        smoke = next(
            sample for sample in manifest["samples"] if sample["id"] == "DEV_SMOKE_S01_FAN"
        )
        self.assertEqual(smoke["intended_use"], "development_only")


if __name__ == "__main__":
    unittest.main()
