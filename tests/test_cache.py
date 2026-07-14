import json
import math
import tempfile
import unittest
from pathlib import Path

from core.cache import LocalTaskCache, build_cache_key, canonical_json, sha256_file


class CacheKeyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.input_path = self.root / "first-name.wav"
        self.input_path.write_bytes(b"deterministic audio bytes")
        self.defaults = {
            "strength": 0.75,
            "model_name": "base",
            "language": "zh",
            "contract_version": "v0.1-contract",
            "config_version": "app-yaml-sha256",
            "code_version": "commit-abc123",
            "processing_config": {
                "enable_events": False,
                "temperature": 0.0,
                "reference_text": None,
            },
        }

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def make_key(self, path: Path | None = None, **overrides: object) -> str:
        arguments = dict(self.defaults)
        arguments.update(overrides)
        return build_cache_key(path or self.input_path, **arguments)

    def test_file_digest_depends_on_content_not_filename(self) -> None:
        renamed = self.root / "completely-different-name.mp3"
        renamed.write_bytes(self.input_path.read_bytes())

        self.assertEqual(sha256_file(self.input_path), sha256_file(renamed))
        self.assertEqual(self.make_key(), self.make_key(renamed))

    def test_input_content_change_changes_key(self) -> None:
        before = self.make_key()
        self.input_path.write_bytes(b"different audio bytes")
        self.assertNotEqual(before, self.make_key())

    def test_every_frozen_identity_component_changes_key(self) -> None:
        baseline = self.make_key()
        cases = {
            "strength": 0.5,
            "model_name": "tiny",
            "language": "en",
            "contract_version": "v0.2-contract",
            "config_version": "different-config",
            "code_version": "different-commit",
            "processing_config": {
                "enable_events": True,
                "temperature": 0.0,
                "reference_text": None,
            },
        }
        for field_name, changed_value in cases.items():
            with self.subTest(field_name=field_name):
                self.assertNotEqual(
                    baseline,
                    self.make_key(**{field_name: changed_value}),
                )

    def test_mapping_order_does_not_change_key(self) -> None:
        first = self.make_key(
            processing_config={"b": 2, "nested": {"y": 2, "x": 1}, "a": 1}
        )
        second = self.make_key(
            processing_config={"a": 1, "nested": {"x": 1, "y": 2}, "b": 2}
        )
        self.assertEqual(first, second)

    def test_non_finite_strength_or_config_is_rejected(self) -> None:
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(strength=value):
                with self.assertRaises(ValueError):
                    self.make_key(strength=value)

        with self.assertRaises(ValueError):
            self.make_key(processing_config={"threshold": math.nan})

        with self.assertRaises(TypeError):
            self.make_key(strength="0.75")

    def test_empty_version_identity_is_rejected(self) -> None:
        for field_name in (
            "model_name",
            "language",
            "contract_version",
            "config_version",
            "code_version",
        ):
            with self.subTest(field_name=field_name):
                with self.assertRaises(ValueError):
                    self.make_key(**{field_name: ""})

    def test_canonical_json_is_compact_and_ordered(self) -> None:
        self.assertEqual(canonical_json({"z": 1, "a": 2}), '{"a":2,"z":1}')

    def test_canonical_json_rejects_implicit_or_unordered_values(self) -> None:
        with self.assertRaises(TypeError):
            canonical_json({1: "integer keys are ambiguous"})
        with self.assertRaises(TypeError):
            canonical_json({"unordered": {"a", "b"}})


class LocalTaskCacheTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.cache = LocalTaskCache(self.root / "cache")
        self.cache_key = "a" * 64

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_missing_entry_is_a_miss_without_creating_directory(self) -> None:
        self.assertIsNone(self.cache.load(self.cache_key))
        self.assertFalse(self.cache.root.exists())

    def test_round_trip_is_json_only_and_deterministic(self) -> None:
        payload = {
            "status": "success",
            "runtime": {"cache_hit": False, "total_seconds": 1.25},
            "warnings": [],
        }
        path = self.cache.save(self.cache_key, payload)

        self.assertEqual(path.suffix, ".json")
        self.assertEqual(self.cache.load(self.cache_key), payload)
        self.assertEqual(
            list(self.cache.root.iterdir()),
            [self.cache.path_for_key(self.cache_key)],
        )
        raw = path.read_text(encoding="utf-8")
        self.assertEqual(raw, canonical_json(json.loads(raw)) + "\n")

    def test_atomic_replacement_updates_one_entry(self) -> None:
        self.cache.save(self.cache_key, {"version": 1})
        path = self.cache.save(self.cache_key, {"version": 2})

        self.assertEqual(self.cache.load(self.cache_key), {"version": 2})
        self.assertEqual(list(self.cache.root.glob("*.tmp")), [])
        self.assertEqual(list(self.cache.root.glob("*.json")), [path])

    def test_serialization_failure_preserves_existing_entry(self) -> None:
        self.cache.save(self.cache_key, {"version": "valid"})

        with self.assertRaises(ValueError):
            self.cache.save(self.cache_key, {"invalid": math.nan})

        self.assertEqual(self.cache.load(self.cache_key), {"version": "valid"})

    def test_malformed_or_truncated_json_is_a_miss(self) -> None:
        path = self.cache.path_for_key(self.cache_key)
        self.cache.root.mkdir(parents=True)
        for malformed in ('{"payload":', "not json", "\xff"):
            with self.subTest(malformed=malformed):
                path.write_text(malformed, encoding="utf-8")
                self.assertIsNone(self.cache.load(self.cache_key))

    def test_mismatched_envelope_is_a_miss(self) -> None:
        path = self.cache.path_for_key(self.cache_key)
        self.cache.root.mkdir(parents=True)
        cases = [
            [],
            {"cache_format_version": 999, "cache_key": self.cache_key, "payload": {}},
            {"cache_format_version": 1, "cache_key": "b" * 64, "payload": {}},
            {"cache_format_version": 1, "cache_key": self.cache_key, "payload": []},
            {
                "cache_format_version": 1,
                "cache_key": self.cache_key,
                "payload": {"number": float("nan")},
            },
        ]
        for envelope in cases:
            with self.subTest(envelope=envelope):
                path.write_text(json.dumps(envelope), encoding="utf-8")
                self.assertIsNone(self.cache.load(self.cache_key))

    def test_invalid_key_cannot_escape_cache_root(self) -> None:
        for invalid in ("../escape", "A" * 64, "a" * 63, "", "a/" * 32):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    self.cache.path_for_key(invalid)

    def test_payload_must_be_a_mapping(self) -> None:
        with self.assertRaises(TypeError):
            self.cache.save(self.cache_key, ["not", "a", "mapping"])  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
