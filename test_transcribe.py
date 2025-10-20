import pytest
import os
from batch_transcribe import split_audio, merge_segments
from format_transcription import format_dialog

def test_merge_segments_basic():
    results = [
        {"segments": [
            {"speaker": "A", "start": 0, "end": 1, "text": "Привет", "words": []},
            {"speaker": "B", "start": 1, "end": 2, "text": "Здравствуйте", "words": []}
        ]},
        {"segments": [
            {"speaker": "A", "start": 2, "end": 3, "text": "Как дела?", "words": []}
        ]}
    ]
    all_segments, speakers = merge_segments(results)
    assert len(all_segments) == 3
    assert speakers == {"A", "B"}

def test_format_dialog_basic():
    segments = [
        {"speaker": "A", "start": 0, "end": 1, "text": "Привет"},
        {"speaker": "B", "start": 1, "end": 2, "text": "Здравствуйте"},
        {"speaker": "A", "start": 2, "end": 3, "text": "Как дела?"}
    ]
    text = format_dialog(segments)
    assert "A (0.00 - 1.00): Привет" in text
    assert "B (1.00 - 2.00): Здравствуйте" in text
    assert "Длительность разговора:" in text
    assert "Соотношение длительности речи:" in text

def test_split_audio_mock(monkeypatch, tmp_path):
    # Мокаем ffprobe/ffmpeg
    def fake_check_output(cmd):
        return b"120.0"  # 2 минуты
    def fake_run(cmd, **kwargs):
        open(os.path.join(tmp_path, cmd[-1]), "wb").write(b"audio")
    monkeypatch.setattr("subprocess.check_output", fake_check_output)
    monkeypatch.setattr("subprocess.run", fake_run)
    chunks = split_audio("fake.mp3", 1, tmp_path)
    assert len(chunks) == 2
    assert all(os.path.exists(path) for path, _ in chunks)

if __name__ == "__main__":
    pytest.main()
