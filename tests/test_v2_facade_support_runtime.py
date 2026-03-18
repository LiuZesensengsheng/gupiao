from __future__ import annotations

import sys

from src.application.v2_facade_support_runtime import emit_progress


class _EncodingRejectingStdout:
    encoding = "gbk"

    def __init__(self) -> None:
        self.writes: list[str] = []

    def write(self, text: str) -> int:
        text.encode(self.encoding)
        self.writes.append(text)
        return len(text)

    def flush(self) -> None:
        return None


def test_emit_progress_falls_back_when_console_encoding_rejects_text(monkeypatch) -> None:
    fake_stdout = _EncodingRejectingStdout()
    monkeypatch.setattr(sys, "stdout", fake_stdout)

    emit_progress("trajectory", "progress € ok")

    assert "".join(fake_stdout.writes) == "[V2][trajectory] progress ? ok\n"
