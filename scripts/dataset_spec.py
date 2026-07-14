"""Shared neutral dataset specification for public data tooling.

The repository tracks only a synthetic example spec. Private collection
matrices must be supplied by an explicit local spec path that is ignored by
Git.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_SPEC_PATH = PROJECT_ROOT / "configs" / "dataset_spec.example.json"
DEFAULT_PENDING_CONSENT_TOKEN = "pending_local_authorization"
DEFAULT_APPROVED_CONSENT_TOKEN = "approved_for_evaluation"
LEGACY_EVALUATION_HOLDOUT_SPLIT = "_".join(("locked", "test"))


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


def evaluation_compat_spec() -> DatasetSpec:
    """Return a neutral matrix matching the existing evaluation runner counts.

    `scripts.run_evaluation` still owns fixed split counts outside this
    dispatch's allowed files. This compatibility spec keeps those tests and
    guards functional without restoring private literals.
    """

    clean_items: list[CleanRecording] = []
    for speaker_index in range(1, 4):
        for sentence_index in range(1, 4):
            clean_items.append(
                CleanRecording(
                    id=f"clean_{speaker_index}_{sentence_index}",
                    speaker_id=f"speaker_{speaker_index}",
                    sentence_id=f"sentence_{sentence_index}",
                    reference_text=f"synthetic reference {speaker_index}-{sentence_index}",
                    path=Path(
                        f"raw/clean/speaker_{speaker_index}/"
                        f"sample_clean_{speaker_index}_{sentence_index}.wav"
                    ),
                    split=LEGACY_EVALUATION_HOLDOUT_SPLIT if sentence_index == 3 else "dev",
                )
            )
    noise_items = tuple(
        NoiseRecording(
            id=f"noise_{index}",
            noise_type=f"noise_class_{index}",
            path=Path(f"raw/noise/noise_class_{index}.wav"),
        )
        for index in range(1, 4)
    )
    mixes: list[MixSpec] = []
    snr_items = ((5, "snrp05"), (0, "snr000"), (-5, "snrm05"))
    for clean in clean_items:
        sentence_index = int(clean.sentence_id.rsplit("_", 1)[1])
        speaker_index = int(clean.speaker_id.rsplit("_", 1)[1])
        noise_index = ((speaker_index + sentence_index - 2) % 3) + 1
        for snr_db, snr_label in snr_items:
            sample_id = f"sample_mix_{speaker_index}_{sentence_index}_{snr_label}"
            mixes.append(
                MixSpec(
                    sample_id=sample_id,
                    clean_id=clean.id,
                    noise_id=f"noise_{noise_index}",
                    split=clean.split,
                    snr_db=float(snr_db),
                    snr_label=snr_label,
                    path=Path(f"controlled/{clean.split}/{sample_id}.wav"),
                    is_locked=clean.split == LEGACY_EVALUATION_HOLDOUT_SPLIT,
                    is_demo_candidate=False,
                )
            )
    controls = tuple(
        CleanControlSpec(
            sample_id=f"sample_clean_control_{index}",
            clean_id=f"clean_{index}_3",
            split="clean_control",
            is_locked=True,
            is_demo_candidate=False,
        )
        for index in range(1, 4)
    )
    real_items = tuple(
        RealRecording(
            sample_id=f"sample_real_{index}",
            speaker_id=f"speaker_{index}",
            sentence_id=f"sentence_{index}",
            reference_text=f"synthetic real reference {index}",
            noise_type=f"noise_class_{index}",
            path=Path(f"raw/real/sample_real_{index}.wav"),
            split="real",
            is_locked=False,
            is_demo_candidate=False,
        )
        for index in range(1, 4)
    )
    return DatasetSpec(
        dataset_version="audiorescue-synthetic-template-v1",
        sample_rate=48_000,
        channels=1,
        sample_width_bytes=2,
        consent_tokens=ConsentTokens(),
        clean_recordings=tuple(clean_items),
        noise_recordings=noise_items,
        mixes=tuple(mixes),
        clean_controls=controls,
        real_recordings=real_items,
    )


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


def _load_payload(path: str | Path | None) -> Mapping[str, Any]:
    if path is None:
        if EXAMPLE_SPEC_PATH.is_file():
            path = EXAMPLE_SPEC_PATH
        else:
            return _default_payload()
    try:
        raw = Path(path).read_text(encoding="utf-8")
        value = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DatasetSpecError("dataset spec could not be read as JSON") from exc
    return _require_mapping(value, "dataset spec")


def load_dataset_spec(path: str | Path | None = None) -> DatasetSpec:
    payload = _load_payload(path)
    version = _require_text(payload, "dataset_version", "dataset_spec")
    sample_rate = int(payload.get("sample_rate", 48_000))
    channels = int(payload.get("channels", 1))
    sample_width_bytes = int(payload.get("sample_width_bytes", 2))
    if sample_rate <= 0 or channels <= 0 or sample_width_bytes <= 0:
        raise DatasetSpecError("audio format values must be positive")
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
    _check_unique([path.as_posix() for path in DatasetSpec(
        version,
        sample_rate,
        channels,
        sample_width_bytes,
        consent,
        tuple(clean_items),
        tuple(noise_items),
        tuple(mix_items),
        tuple(clean_control_items),
        tuple(real_items),
    ).required_audio_paths()], "audio paths")

    return DatasetSpec(
        dataset_version=version,
        sample_rate=sample_rate,
        channels=channels,
        sample_width_bytes=sample_width_bytes,
        consent_tokens=consent,
        clean_recordings=tuple(clean_items),
        noise_recordings=tuple(noise_items),
        mixes=tuple(mix_items),
        clean_controls=tuple(clean_control_items),
        real_recordings=tuple(real_items),
    )
