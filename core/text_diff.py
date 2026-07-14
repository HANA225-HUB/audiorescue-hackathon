"""Safe text-diff HTML rendering owned by B.

C's pipeline calls `build_text_diff()` and stores the returned HTML in
ProcessResult.text_diff_html. The UI only displays that field.
"""

from __future__ import annotations

import difflib
import html
import re


def tokenize_text(text: str) -> list[str]:
    """Tokenize Chinese by character while keeping Latin words together."""

    if not text:
        return []
    return re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]|[^\s]", text)


def _join(tokens: list[str]) -> str:
    return "".join(tokens)


def build_text_diff(before: str, after: str) -> str:
    before_tokens = tokenize_text(before or "")
    after_tokens = tokenize_text(after or "")
    if not before_tokens and not after_tokens:
        return "<div class='empty-state'>暂无文本差异。</div>"
    if before_tokens == after_tokens:
        return f"<div class='diff'><span>{html.escape(_join(before_tokens))}</span><p>两次转写无变化。</p></div>"

    matcher = difflib.SequenceMatcher(a=before_tokens, b=after_tokens)
    pieces: list[str] = ["<div class='diff'>"]
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        before_piece = html.escape(_join(before_tokens[i1:i2]))
        after_piece = html.escape(_join(after_tokens[j1:j2]))
        if tag == "equal":
            pieces.append(f"<span>{before_piece}</span>")
        elif tag == "delete":
            pieces.append(f"<del>{before_piece}</del>")
        elif tag == "insert":
            pieces.append(f"<ins>{after_piece}</ins>")
        elif tag == "replace":
            pieces.append(f"<span class='replace'><del>{before_piece}</del><ins>{after_piece}</ins></span>")
    pieces.append("</div>")
    return "".join(pieces)
