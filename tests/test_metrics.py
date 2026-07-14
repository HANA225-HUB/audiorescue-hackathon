import unittest

from core.metrics import compute_cer, normalize_zh_text
from core.schemas import CerResult


class NormalizeZhTextTest(unittest.TestCase):
    def test_nfkc_lowercase_whitespace_and_punctuation(self) -> None:
        text = " ＡＢＣ１２３， Hello—你 好！\n"
        self.assertEqual(normalize_zh_text(text), "abc123hello你好")

    def test_keeps_cjk_kana_hangul_and_digits(self) -> None:
        text = "中文・かな・カナ・한글 2026"
        self.assertEqual(normalize_zh_text(text), "中文かなカナ한글2026")

    def test_traditional_and_simplified_chinese_share_one_cer_form(self) -> None:
        traditional = "請記錄會議中的三個重點：數據來源、模型效果和系統穩定性。"
        simplified = "请记录会议中的三个重点：数据来源、模型效果和系统稳定性。"
        self.assertEqual(normalize_zh_text(traditional), normalize_zh_text(simplified))
        self.assertEqual(compute_cer(simplified, traditional).cer, 0.0)

    def test_removes_symbols_emoji_and_controls(self) -> None:
        self.assertEqual(normalize_zh_text("语音¥✅🎧\u200b处理"), "语音处理")

    def test_non_string_is_rejected(self) -> None:
        with self.assertRaisesRegex(TypeError, "text must be a string"):
            normalize_zh_text(123)  # type: ignore[arg-type]


class ComputeCerTest(unittest.TestCase):
    def test_identical_after_normalization(self) -> None:
        result = compute_cer("今天 ＡＢＣ，12！", "今天abc12")
        self.assertIsInstance(result, CerResult)
        self.assertEqual(result.normalized_reference, "今天abc12")
        self.assertEqual(result.normalized_hypothesis, "今天abc12")
        self.assertEqual(result.cer, 0.0)
        self.assertEqual(
            (result.substitutions, result.deletions, result.insertions),
            (0, 0, 0),
        )

    def test_substitution_and_insertion_counts(self) -> None:
        result = compute_cer("语音识别", "语因识别啊")
        self.assertEqual(result.substitutions, 1)
        self.assertEqual(result.deletions, 0)
        self.assertEqual(result.insertions, 1)
        self.assertAlmostEqual(result.cer, 0.5)

    def test_deletion_count(self) -> None:
        result = compute_cer("语音识别", "语音别")
        self.assertEqual(result.substitutions, 0)
        self.assertEqual(result.deletions, 1)
        self.assertEqual(result.insertions, 0)
        self.assertAlmostEqual(result.cer, 0.25)

    def test_empty_hypothesis_counts_all_deletions(self) -> None:
        result = compute_cer("会议3点", "")
        self.assertEqual(result.substitutions, 0)
        self.assertEqual(result.deletions, 4)
        self.assertEqual(result.insertions, 0)
        self.assertEqual(result.cer, 1.0)

    def test_empty_raw_reference_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "reference"):
            compute_cer("", "任意文本")

    def test_reference_empty_after_normalization_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "reference"):
            compute_cer(" ，！？\t🎧", "任意文本")


if __name__ == "__main__":
    unittest.main()
