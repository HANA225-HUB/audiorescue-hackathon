"""Text normalization and character error rate metrics.

The normalization policy is intentionally small and deterministic so that the
same reference/hypothesis pair produces the same CER in tests, reports, and
the demo UI.
"""

from __future__ import annotations

import unicodedata

from opencc import OpenCC

from core.schemas import CerResult


_TRADITIONAL_TO_SIMPLIFIED = OpenCC("t2s")


def normalize_zh_text(text: str) -> str:
    """Normalize text for multilingual character-level comparison.

    Policy:
    - apply Unicode NFKC normalization;
    - convert Traditional Chinese to Simplified Chinese with OpenCC ``t2s``;
    - lowercase Latin letters (and any other cased Unicode letters);
    - keep Unicode letters and numbers, including CJK, kana, and Hangul;
    - remove whitespace, punctuation, symbols, emoji, and control characters.

    Digits are deliberately retained because changing a meeting time, model
    number, or quantity is a real recognition error for this project.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    normalized = unicodedata.normalize("NFKC", text).lower()
    normalized = _TRADITIONAL_TO_SIMPLIFIED.convert(normalized)
    return "".join(character for character in normalized if character.isalnum())


def _edit_counts(reference: str, hypothesis: str) -> tuple[int, int, int]:
    """Return Levenshtein substitution, deletion, and insertion counts.

    When several optimal alignments exist, backtracking prefers substitution,
    then deletion, then insertion. The total edit distance is unaffected; the
    deterministic priority keeps the component counts reproducible.
    """

    reference_length = len(reference)
    hypothesis_length = len(hypothesis)

    distances = [list(range(hypothesis_length + 1))]
    for reference_index in range(1, reference_length + 1):
        row = [reference_index] + [0] * hypothesis_length
        previous_row = distances[reference_index - 1]
        for hypothesis_index in range(1, hypothesis_length + 1):
            if reference[reference_index - 1] == hypothesis[hypothesis_index - 1]:
                row[hypothesis_index] = previous_row[hypothesis_index - 1]
            else:
                row[hypothesis_index] = 1 + min(
                    previous_row[hypothesis_index - 1],  # substitution
                    previous_row[hypothesis_index],  # deletion
                    row[hypothesis_index - 1],  # insertion
                )
        distances.append(row)

    substitutions = 0
    deletions = 0
    insertions = 0
    reference_index = reference_length
    hypothesis_index = hypothesis_length

    while reference_index > 0 or hypothesis_index > 0:
        if (
            reference_index > 0
            and hypothesis_index > 0
            and reference[reference_index - 1] == hypothesis[hypothesis_index - 1]
            and distances[reference_index][hypothesis_index]
            == distances[reference_index - 1][hypothesis_index - 1]
        ):
            reference_index -= 1
            hypothesis_index -= 1
            continue

        current_distance = distances[reference_index][hypothesis_index]

        if (
            reference_index > 0
            and hypothesis_index > 0
            and current_distance
            == distances[reference_index - 1][hypothesis_index - 1] + 1
        ):
            substitutions += 1
            reference_index -= 1
            hypothesis_index -= 1
        elif (
            reference_index > 0
            and current_distance == distances[reference_index - 1][hypothesis_index] + 1
        ):
            deletions += 1
            reference_index -= 1
        elif (
            hypothesis_index > 0
            and current_distance == distances[reference_index][hypothesis_index - 1] + 1
        ):
            insertions += 1
            hypothesis_index -= 1
        else:  # pragma: no cover - protects against an inconsistent DP table
            raise RuntimeError("failed to backtrack Levenshtein alignment")

    return substitutions, deletions, insertions


def compute_cer(reference: str, hypothesis: str) -> CerResult:
    """Compute character error rate after the frozen normalization policy.

    A missing or normalization-empty reference has no meaningful CER
    denominator and therefore raises ``ValueError``. An empty hypothesis is
    valid and produces one deletion for every normalized reference character.
    """

    normalized_reference = normalize_zh_text(reference)
    normalized_hypothesis = normalize_zh_text(hypothesis)
    if not normalized_reference:
        raise ValueError("reference must contain at least one letter or number")

    substitutions, deletions, insertions = _edit_counts(
        normalized_reference,
        normalized_hypothesis,
    )
    cer = (substitutions + deletions + insertions) / len(normalized_reference)

    return CerResult(
        reference=reference,
        normalized_reference=normalized_reference,
        normalized_hypothesis=normalized_hypothesis,
        cer=cer,
        substitutions=substitutions,
        deletions=deletions,
        insertions=insertions,
    )


__all__ = ["compute_cer", "normalize_zh_text"]
