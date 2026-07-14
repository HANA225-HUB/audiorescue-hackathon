from __future__ import annotations

import unittest

from core.text_diff import build_text_diff, tokenize_text


class TextDiffTests(unittest.TestCase):
    def test_tokenizer_keeps_latin_words_together(self) -> None:
        self.assertEqual(tokenize_text("测试 OpenAI ASR"), ["测", "试", "OpenAI", "ASR"])

    def test_diff_escapes_html(self) -> None:
        rendered = build_text_diff("<script>", "安全文本")
        self.assertNotIn("<script>", rendered)
        self.assertIn("&lt;", rendered)

    def test_same_text_is_not_called_improvement(self) -> None:
        rendered = build_text_diff("相同文本", "相同文本")
        self.assertIn("两次转写无变化", rendered)
        self.assertNotIn("提升", rendered)


if __name__ == "__main__":
    unittest.main()
