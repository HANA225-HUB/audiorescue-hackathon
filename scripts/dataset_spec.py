"""Shared neutral dataset specification for public data tooling.

The repository tracks only a synthetic example spec. Private collection
matrices must be supplied by an explicit local spec path that is ignored by
Git.
"""

from __future__ import annotations

import json
import hashlib
import stat
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_SPEC_PATH = PROJECT_ROOT / "configs" / "dataset_spec.example.json"
DEFAULT_PENDING_CONSENT_TOKEN = "pending_local_authorization"
DEFAULT_APPROVED_CONSENT_TOKEN = "approved_for_evaluation"
FIXED_SAMPLE_RATE = 48_000
FIXED_CHANNELS = 1
FIXED_SAMPLE_WIDTH_BYTES = 2


class DatasetSpecError(ValueError):
    """Raised when a dataset spec is missing required generic structure."""


@dataclass(frozen=True, slots=True)
class ConsentTokens:
    pending: str = DEFAULT_PENDING_CONSENT_TOKEN
    approved: str = DEFAULT_APPROVED_CONSENT_TOKEN


@dataclass(frozen=True, slots=True)
class CleanRecording:
    id: str
    speaker_id: str
    sentence_id: str
    reference_text: str
    path: Path
    split: str


@dataclass(frozen=True, slots=True)
class NoiseRecording:
    id: str
    noise_type: str
    path: Path


@dataclass(frozen=True, slots=True)
class MixSpec:
    sample_id: str
    clean_id: str
    noise_id: str
    split: str
    snr_db: float
    snr_label: str
    path: Path
    is_locked: bool
    is_demo_candidate: bool


@dataclass(frozen=True, slots=True)
class CleanControlSpec:
    sample_id: str
    clean_id: str
    split: str
    is_locked: bool
    is_demo_candidate: bool


@dataclass(frozen=True, slots=True)
class RealRecording:
    sample_id: str
    speaker_id: str
    sentence_id: str
    reference_text: str
    noise_type: str
    path: Path
    split: str
    is_locked: bool
    is_demo_candidate: bool


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    dataset_version: str
    sample_rate: int
    channels: int
    sample_width_bytes: int
    protected_splits: tuple[str, ...]
    consent_tokens: ConsentTokens
    clean_recordings: tuple[CleanRecording, ...]
    noise_recordings: tuple[NoiseRecording, ...]
    mixes: tuple[MixSpec, ...]
    clean_controls: tuple[CleanControlSpec, ...]
    real_recordings: tuple[RealRecording, ...]

    @property
    def clean_by_id(self) -> dict[str, CleanRecording]:
        return {item.id: item for item in self.clean_recordings}

    @property
    def noise_by_id(self) -> dict[str, NoiseRecording]:
        return {item.id: item for item in self.noise_recordings}

    def master_audio_paths(self) -> dict[str, Path]:
        paths: dict[str, Path] = {}
        for item in self.clean_recordings:
            paths[item.id] = item.path
        for item in self.noise_recordings:
            paths[item.id] = item.path
        for item in self.real_recordings:
            paths[item.sample_id] = item.path
        return paths

    def required_audio_paths(self) -> set[Path]:
        paths = set(self.master_audio_paths().values())
        paths.update(item.path for item in self.mixes)
        return paths

    def manifest_primary_paths(self) -> set[Path]:
        paths = {item.path for item in self.mixes}
        clean_by_id = self.clean_by_id
        for item in self.clean_controls:
            paths.add(clean_by_id[item.clean_id].path)
        paths.update(item.path for item in self.real_recordings)
        return paths

    def split_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in (*self.mixes, *self.clean_controls, *self.real_recordings):
            split = item.split
            counts[split] = counts.get(split, 0) + 1
        return counts

    def locked_sample_ids(self) -> tuple[str, ...]:
        sample_ids = [
            item.sample_id
            for item in (*self.mixes, *self.clean_controls, *self.real_recordings)
            if item.is_locked
        ]
        return tuple(sorted(sample_ids))

    def to_public_dict(self) -> dict[str, Any]:
        def convert(value: Any) -> Any:
            if isinstance(value, Path):
                return value.as_posix()
            if isinstance(value, tuple):
                return [convert(item) for item in value]
            if isinstance(value, dict):
                return {key: convert(item) for key, item in value.items()}
            if hasattr(value, "__dataclass_fields__"):
                return convert(asdict(value))
            return value

        return convert(self)


def _default_payload() -> dict[str, Any]:
    return {
        "dataset_version": "audiorescue-synthetic-template-v1",
        "sample_rate": 48_000,
        "channels": 1,
        "sample_width_bytes": 2,
        "protected_splits": ["holdout", "clean_control"],
        "consent_tokens": {
            "pending": DEFAULT_PENDING_CONSENT_TOKEN,
            "approved": DEFAULT_APPROVED_CONSENT_TOKEN,
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


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DatasetSpecError(f"{label} must be an object")
    return value


def _require_list(payload: Mapping[str, Any], field_name: str) -> list[Any]:
    value = payload.get(field_name)
    if not isinstance(value, list):
        raise DatasetSpecError(f"{field_name} must be a list")
    return value


def _require_text(payload: Mapping[str, Any], field_name: str, label: str) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise DatasetSpecError(f"{label}.{field_name} must be a non-empty string")
    return value.strip()


def _require_bool(payload: Mapping[str, Any], field_name: str, label: str) -> bool:
    value = payload.get(field_name)
    if not isinstance(value, bool):
        raise DatasetSpecError(f"{label}.{field_name} must be true or false")
    return value


def _require_relative_path(payload: Mapping[str, Any], field_name: str, label: str) -> Path:
    raw = payload.get(field_name)
    if not isinstance(raw, str) or not raw.strip():
        raise DatasetSpecError(f"{label}.{field_name} must be a relative path")
    normalized = raw.replace("\\", "/").strip()
    if ":" in normalized.split("/", 1)[0]:
        raise DatasetSpecError(f"{label}.{field_name} must be a relative path")
    path = Path(normalized)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise DatasetSpecError(f"{label}.{field_name} must be a safe relative path")
    if path.suffix.lower() != ".wav":
        raise DatasetSpecError(f"{label}.{field_name} must point to a WAV file")
    return path


def _require_number(payload: Mapping[str, Any], field_name: str, label: str) -> float:
    value = payload.get(field_name)
    if not isinstance(value, (int, float)):
        raise DatasetSpecError(f"{label}.{field_name} must be numeric")
    return float(value)


def _check_unique(values: list[str], label: str) -> None:
    if len(values) != len(set(values)):
        raise DatasetSpecError(f"{label} contains duplicate values")


def _check_global_unique(values: list[str], label: str = "dataset identifiers") -> None:
    if len(values) != len(set(values)):
        raise DatasetSpecError(f"{label} contain duplicate values")


def _load_payload(path: str | Path | None, *, allow_example: bool) -> Mapping[str, Any]:
    if path is None:
        if not allow_example:
            raise DatasetSpecError(
                "explicit dataset spec is required; pass a local spec path "
                "or request explicit example mode"
            )
        if not EXAMPLE_SPEC_PATH.is_file():
            return _default_payload()
        path = EXAMPLE_SPEC_PATH
    try:
        raw = Path(path).read_text(encoding="utf-8")
        value = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DatasetSpecError("dataset spec could not be read as JSON") from exc
    return _require_mapping(value, "dataset spec")


def spec_content_hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_example_dataset_spec() -> DatasetSpec:
    return load_dataset_spec(None, allow_example=True)


def _load_integer(payload: Mapping[str, Any], field_name: str, default: int) -> int:
    value = payload.get(field_name, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise DatasetSpecError("audio format must be 48 kHz mono PCM16")
    return value


def _validate_fixed_audio_format(
    sample_rate: int,
    channels: int,
    sample_width_bytes: int,
) -> None:
    if (
        sample_rate != FIXED_SAMPLE_RATE
        or channels != FIXED_CHANNELS
        or sample_width_bytes != FIXED_SAMPLE_WIDTH_BYTES
    ):
        raise DatasetSpecError("audio format must be 48 kHz mono PCM16")


def _validate_split_invariants(spec: DatasetSpec) -> None:
    protected = set(spec.protected_splits)
    rows = (*spec.mixes, *spec.clean_controls, *spec.real_recordings)
    for row in rows:
        if row.split in protected and (not row.is_locked or row.is_demo_candidate):
            raise DatasetSpecError("split invariant violation")
        if row.is_locked and row.is_demo_candidate:
            raise DatasetSpecError("split invariant violation")


def load_dataset_spec(
    path: str | Path | None = None,
    *,
    allow_example: bool = False,
) -> DatasetSpec:
    payload = _load_payload(path, allow_example=allow_example)
    version = _require_text(payload, "dataset_version", "dataset_spec")
    sample_rate = _load_integer(payload, "sample_rate", FIXED_SAMPLE_RATE)
    channels = _load_integer(payload, "channels", FIXED_CHANNELS)
    sample_width_bytes = _load_integer(
        payload, "sample_width_bytes", FIXED_SAMPLE_WIDTH_BYTES
    )
    _validate_fixed_audio_format(sample_rate, channels, sample_width_bytes)
    protected_splits = tuple(
        _require_text(
            _require_mapping({"value": value}, "protected_splits[]"),
            "value",
            "protected_splits[]",
        )
        for value in _require_list(payload, "protected_splits")
    )
    _check_unique(list(protected_splits), "protected_splits")
    consent_payload = _require_mapping(payload.get("consent_tokens", {}), "consent_tokens")
    consent = ConsentTokens(
        pending=_require_text(consent_payload, "pending", "consent_tokens"),
        approved=_require_text(consent_payload, "approved", "consent_tokens"),
    )
    if consent.pending == consent.approved:
        raise DatasetSpecError("consent tokens must be distinct")

    clean_items: list[CleanRecording] = []
    for index, item in enumerate(_require_list(payload, "clean_recordings")):
        item = _require_mapping(item, f"clean_recordings[{index}]")
        label = f"clean_recordings[{index}]"
        clean_items.append(
            CleanRecording(
                id=_require_text(item, "id", label),
                speaker_id=_require_text(item, "speaker_id", label),
                sentence_id=_require_text(item, "sentence_id", label),
                reference_text=_require_text(item, "reference_text", label),
                path=_require_relative_path(item, "path", label),
                split=_require_text(item, "split", label),
            )
        )

    noise_items: list[NoiseRecording] = []
    for index, item in enumerate(_require_list(payload, "noise_recordings")):
        item = _require_mapping(item, f"noise_recordings[{index}]")
        label = f"noise_recordings[{index}]"
        noise_items.append(
            NoiseRecording(
                id=_require_text(item, "id", label),
                noise_type=_require_text(item, "noise_type", label),
                path=_require_relative_path(item, "path", label),
            )
        )

    clean_ids = [item.id for item in clean_items]
    noise_ids = [item.id for item in noise_items]
    _check_unique(clean_ids, "clean_recordings.id")
    _check_unique(noise_ids, "noise_recordings.id")
    _check_global_unique(clean_ids + noise_ids)
    clean_id_set = set(clean_ids)
    noise_id_set = set(noise_ids)

    mix_items: list[MixSpec] = []
    for index, item in enumerate(_require_list(payload, "mixes")):
        item = _require_mapping(item, f"mixes[{index}]")
        label = f"mixes[{index}]"
        clean_id = _require_text(item, "clean_id", label)
        noise_id = _require_text(item, "noise_id", label)
        if clean_id not in clean_id_set:
            raise DatasetSpecError(f"{label}.clean_id references unknown clean_id")
        if noise_id not in noise_id_set:
            raise DatasetSpecError(f"{label}.noise_id references unknown noise_id")
        mix_items.append(
            MixSpec(
                sample_id=_require_text(item, "sample_id", label),
                clean_id=clean_id,
                noise_id=noise_id,
                split=_require_text(item, "split", label),
                snr_db=_require_number(item, "snr_db", label),
                snr_label=_require_text(item, "snr_label", label),
                path=_require_relative_path(item, "path", label),
                is_locked=_require_bool(item, "is_locked", label),
                is_demo_candidate=_require_bool(item, "is_demo_candidate", label),
            )
        )

    clean_control_items: list[CleanControlSpec] = []
    for index, item in enumerate(_require_list(payload, "clean_controls")):
        item = _require_mapping(item, f"clean_controls[{index}]")
        label = f"clean_controls[{index}]"
        clean_id = _require_text(item, "clean_id", label)
        if clean_id not in clean_id_set:
            raise DatasetSpecError(f"{label}.clean_id references unknown clean_id")
        clean_control_items.append(
            CleanControlSpec(
                sample_id=_require_text(item, "sample_id", label),
                clean_id=clean_id,
                split=_require_text(item, "split", label),
                is_locked=_require_bool(item, "is_locked", label),
                is_demo_candidate=_require_bool(item, "is_demo_candidate", label),
            )
        )

    real_items: list[RealRecording] = []
    for index, item in enumerate(_require_list(payload, "real_recordings")):
        item = _require_mapping(item, f"real_recordings[{index}]")
        label = f"real_recordings[{index}]"
        real_items.append(
            RealRecording(
                sample_id=_require_text(item, "sample_id", label),
                speaker_id=_require_text(item, "speaker_id", label),
                sentence_id=_require_text(item, "sentence_id", label),
                reference_text=_require_text(item, "reference_text", label),
                noise_type=_require_text(item, "noise_type", label),
                path=_require_relative_path(item, "path", label),
                split=_require_text(item, "split", label),
                is_locked=_require_bool(item, "is_locked", label),
                is_demo_candidate=_require_bool(item, "is_demo_candidate", label),
            )
        )

    sample_ids = (
        [item.sample_id for item in mix_items]
        + [item.sample_id for item in clean_control_items]
        + [item.sample_id for item in real_items]
    )
    _check_unique(sample_ids, "manifest sample_id")
    _check_global_unique(
        clean_ids + noise_ids + [item.sample_id for item in real_items],
    )
    loaded_spec = DatasetSpec(
        version,
        sample_rate,
        channels,
        sample_width_bytes,
        protected_splits,
        consent,
        tuple(clean_items),
        tuple(noise_items),
        tuple(mix_items),
        tuple(clean_control_items),
        tuple(real_items),
    )
    _check_unique(
        [path.as_posix() for path in loaded_spec.required_audio_paths()],
        "audio paths",
    )
    _validate_split_invariants(loaded_spec)

    return loaded_spec


def _is_link_or_reparse_point(path: Path) -> bool:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    file_attributes = getattr(info, "st_file_attributes", 0)
    return path.is_symlink() or bool(reparse_flag and file_attributes & reparse_flag)


def _check_existing_absolute_components(path: Path) -> None:
    absolute = Path(path).expanduser()
    if not absolute.is_absolute():
        absolute = Path.cwd() / absolute
    current = Path(absolute.anchor) if absolute.anchor else Path()
    parts = absolute.parts[1:] if absolute.anchor else absolute.parts
    for part in parts:
        current = current / part
        if not current.exists():
            break
        if _is_link_or_reparse_point(current):
            raise DatasetSpecError("unsafe path component")


def _check_existing_components(root: Path, relative: Path, *, include_leaf: bool) -> None:
    current = root
    components = relative.parts if include_leaf else relative.parts[:-1]
    if current.exists() and _is_link_or_reparse_point(current):
        raise DatasetSpecError("unsafe path component")
    for part in components:
        current = current / part
        if current.exists() and _is_link_or_reparse_point(current):
            raise DatasetSpecError("unsafe path component")


def safe_join(
    root: str | Path,
    relative_path: str | Path,
    *,
    must_exist: bool = False,
) -> Path:
    raw = str(relative_path).replace("\\", "/").strip()
    if not raw:
        raise DatasetSpecError("unsafe path")
    raw_parts = raw.split("/")
    if (
        raw.startswith("/")
        or any(part in {"", ".", ".."} for part in raw_parts)
        or any(":" in part for part in raw_parts)
    ):
        raise DatasetSpecError("unsafe path")
    pure_relative = PurePosixPath(raw)
    if pure_relative.is_absolute():
        raise DatasetSpecError("unsafe path")
    relative = Path(*pure_relative.parts)

    root_path = Path(root).expanduser()
    _check_existing_absolute_components(root_path)
    root_resolved = root_path.resolve(strict=False)
    _check_existing_components(
        root_path,
        relative,
        include_leaf=must_exist or (root_path / relative).exists(),
    )
    candidate = (root_path / relative).resolve(strict=False)
    try:
        candidate.relative_to(root_resolved)
    except ValueError as exc:
        raise DatasetSpecError("unsafe path") from exc
    if must_exist and not candidate.is_file():
        raise DatasetSpecError("unsafe path")
    return candidate


def root_relative_label(root: str | Path, path: str | Path) -> str:
    root_resolved = Path(root).expanduser().resolve(strict=False)
    resolved = Path(path).expanduser().resolve(strict=False)
    try:
        return resolved.relative_to(root_resolved).as_posix()
    except ValueError:
        return "<external-path>"


def ensure_private_repo_root(
    dataset_root: str | Path,
    *,
    project_root: str | Path = PROJECT_ROOT,
) -> None:
    root_path = Path(dataset_root).expanduser()
    _check_existing_absolute_components(root_path)
    root = root_path.resolve(strict=False)
    project = Path(project_root).expanduser().resolve(strict=False)
    try:
        relative = root.relative_to(project)
    except ValueError:
        return
    check = subprocess.run(
        ["git", "check-ignore", "-q", "--", relative.as_posix()],
        cwd=project,
        check=False,
        capture_output=True,
    )
    if check.returncode != 0:
        raise DatasetSpecError(
            "dataset root inside the repository must be ignored by Git"
        )
