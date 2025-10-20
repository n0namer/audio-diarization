"""Batch transcription utilities for the audio diarization pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

LOG_FILE = Path(__file__).with_name("batch_transcribe.log")
logging.basicConfig(
    filename=str(LOG_FILE),
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
COG_DIR = PROJECT_ROOT / "cog-whisper-diarization"
sys.path.append(str(COG_DIR))

from predict import Predictor  # type: ignore

CHUNK_LENGTH_MIN = 10
DEBUG = os.environ.get("DEBUG", "0") == "1"


def debug_log(message: str) -> None:
    if DEBUG:
        print(f"[DEBUG] {message}")


def _ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _split_with_torchaudio(
    audio_path: Path, chunk_length_sec: int, out_dir: Path
) -> List[Tuple[Path, float]]:
    try:
        import torchaudio

        waveform, sample_rate = torchaudio.load(str(audio_path))
    except Exception as exc:  # pragma: no cover - защитный блок
        logger.error("split_audio (torchaudio) failed: %s", exc, exc_info=True)
        print(f"FATAL: split_audio (torchaudio) failed: {exc}")
        sys.exit(1)

    total_samples = waveform.shape[1]
    chunk_size_samples = chunk_length_sec * sample_rate
    chunks: List[Tuple[Path, float]] = []

    for idx, start_sample in enumerate(range(0, total_samples, chunk_size_samples)):
        end_sample = min(start_sample + chunk_size_samples, total_samples)
        start_sec = start_sample / sample_rate
        chunk_wave = waveform[:, start_sample:end_sample]
        out_file = out_dir / f"{audio_path.stem}_chunk_{idx}.wav"
        torchaudio.save(str(out_file), chunk_wave, sample_rate)
        logger.info("split_audio (torchaudio): created chunk %s", out_file)
        chunks.append((out_file, start_sec))

    logger.info("split_audio (torchaudio): total_chunks=%d", len(chunks))
    return chunks


def _split_with_ffmpeg(
    audio_path: Path, chunk_length_sec: int, out_dir: Path
) -> List[Tuple[Path, float]]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]
    logger.debug("ffprobe cmd: %s", " ".join(cmd))

    try:
        duration = float(subprocess.check_output(cmd).decode().strip())
    except Exception as exc:
        logger.error("Ошибка при получении длительности файла: %s", exc, exc_info=True)
        sys.exit(1)

    logger.info("split_audio: duration=%s", duration)
    chunks: List[Tuple[Path, float]] = []
    chunk_index = 0
    start = 0.0

    while start < duration:
        out_file = out_dir / f"{audio_path.stem}_chunk_{chunk_index}.mp3"
        length = min(chunk_length_sec, duration - start)
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(audio_path),
            "-ss",
            str(start),
            "-t",
            str(length),
            "-c",
            "copy",
            str(out_file),
        ]
        logger.debug("ffmpeg cmd: %s", " ".join(cmd))
        try:
            subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
        except Exception as exc:
            logger.error("Ошибка при создании чанка %s: %s", out_file, exc, exc_info=True)
            sys.exit(1)

        logger.info("split_audio: created chunk %s", out_file)
        chunks.append((out_file, start))
        chunk_index += 1
        start = chunk_index * chunk_length_sec

    logger.info("split_audio: total_chunks=%d", len(chunks))
    return chunks


def split_audio(
    audio_path: Path, chunk_length_min: int, out_dir: Path, backend: str = "torio"
) -> List[Tuple[Path, float]]:
    """Разбивает аудиофайл на последовательные чанки."""

    logger.info(
        "split_audio: start, audio_path=%s, chunk_length_min=%s, out_dir=%s, backend=%s",
        audio_path,
        chunk_length_min,
        out_dir,
        backend,
    )

    _ensure_directory(out_dir)
    chunk_length_sec = max(1, chunk_length_min * 60)

    if backend == "torchaudio":
        return _split_with_torchaudio(audio_path, chunk_length_sec, out_dir)

    return _split_with_ffmpeg(audio_path, chunk_length_sec, out_dir)


def transcribe_chunk(chunk_path: Path, offset: float, predictor: Predictor) -> dict:
    logger.info("transcribe_chunk: start, chunk_path=%s, offset=%s", chunk_path, offset)
    logger.debug(
        "TRACE: predictor.predict call with file=%s, language='ru', prompt=None, num_speakers=None",
        chunk_path,
    )

    try:
        result = predictor.predict(
            file=str(chunk_path),
            language="ru",
            prompt=None,
            num_speakers=None,
        )
    except Exception as exc:  # pragma: no cover - защитный блок
        logger.error(
            "Ошибка при транскрибации %s: %s", chunk_path, exc, exc_info=True
        )
        logger.error("TRACE: Exception traceback:\n%s", traceback.format_exc())
        return {"segments": []}

    if result is None:
        logger.warning("transcribe_chunk: %s, result is None!", chunk_path)
        return {"segments": []}

    data = result.dict() if hasattr(result, "dict") else dict(result)
    segments = data.get("segments") or []
    logger.info("transcribe_chunk: %s, segments=%d", chunk_path, len(segments))

    for segment in segments:
        start = float(segment.get("start", 0.0)) + offset
        end = float(segment.get("end", start)) + offset
        segment["start"] = start
        segment["end"] = end
        segment["language"] = "ru"

        words = segment.get("words") or []
        for word in words:
            word_start = float(word.get("start", 0.0)) + offset
            word_end = float(word.get("end", word_start)) + offset
            word["start"] = word_start
            word["end"] = word_end
            word["language"] = "ru"

    return {"segments": segments}


def merge_segments(results: Sequence[dict]) -> Tuple[List[dict], set]:
    debug_log(f"Merging {len(results)} chunk results")
    all_segments: List[dict] = []
    speakers = set()

    for result in results:
        for segment in result.get("segments", []):
            segment["language"] = "ru"
            for word in segment.get("words", []):
                word["language"] = "ru"

            all_segments.append(segment)
            speakers.add(segment.get("speaker", "UNKNOWN"))

    all_segments.sort(key=lambda seg: seg.get("start", 0.0))
    debug_log(
        f"Total segments: {len(all_segments)}, speakers: {sorted(speakers) if speakers else []}"
    )
    return all_segments, speakers


def check_torio_installation() -> str:
    """Проверяет наличие и работоспособность torio и его FFmpeg-расширений."""

    try:
        import importlib.util
        import tempfile
        import wave
        import contextlib

        spec = importlib.util.find_spec("torio")
        if spec is None:
            import subprocess as _subprocess

            logger.warning("torio не установлен. Пробую установить через pip...")
            print("torio не найден. Автоматическая установка через pip...")
            try:
                _subprocess.check_call([sys.executable, "-m", "pip", "install", "torio"])
                logger.info("torio успешно установлен через pip.")
            except Exception as exc:
                logger.error("Не удалось установить torio: %s", exc, exc_info=True)
                print(f"FATAL: Не удалось установить torio: {exc}")
                return "torchaudio"

        import torio

        torio_path = Path(torio.__file__).parent
        lib_dir = torio_path / "lib"
        if not lib_dir.exists():
            logger.warning("Папка torio/lib не найдена. Fallback на torchaudio.")
            print("torio установлен, но папка lib не найдена. Fallback на torchaudio.")
            return "torchaudio"

        has_extension = any(
            fname.startswith("libtorio_ffmpeg") and fname.endswith(".pyd")
            for fname in os.listdir(lib_dir)
        )
        if not has_extension:
            logger.warning(
                "В папке torio/lib нет libtorio_ffmpeg*.pyd — fallback на torchaudio."
            )
            print("torio установлен, но нет расширений. Fallback на torchaudio.")
            return "torchaudio"

        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                with wave.open(tmp.name, "w") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(16000)
                    wf.writeframes(b"\x00" * 32000)

                import torio.audio

                with contextlib.closing(torio.audio.open(tmp.name)) as file:
                    _ = file.read()
            logger.info("torio.audio.open() успешно прочитал WAV — расширения работают.")
            return "torio"
        except Exception as exc:
            logger.warning(
                "torio установлен, но расширения не работают: %s. Fallback на torchaudio.",
                exc,
                exc_info=True,
            )
            print(f"torio установлен, но расширения не работают: {exc}. Fallback на torchaudio.")
            return "torchaudio"
    except Exception as exc:  # pragma: no cover - защитный блок
        logger.error("Ошибка при проверке torio: %s", exc, exc_info=True)
        print(f"ОШИБКА: Ошибка при проверке torio: {exc}")
        return "torchaudio"


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch diarization helper")
    parser.add_argument("audio", help="Путь до аудиофайла для транскрибации")
    parser.add_argument(
        "--chunk-length",
        type=int,
        default=CHUNK_LENGTH_MIN,
        help="Длина чанка в минутах (по умолчанию 10)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("result"),
        help="Папка для сохранения результатов",
    )
    parser.add_argument(
        "--keep-chunks",
        action="store_true",
        help="Не удалять временные аудиофайлы после завершения",
    )
    return parser.parse_args(argv)


def transcribe_chunks(
    chunks: Iterable[Tuple[Path, float]], predictor: Predictor
) -> List[dict]:
    return [transcribe_chunk(path, offset, predictor) for path, offset in chunks]


def main(argv: Sequence[str] | None = None) -> None:
    logger.debug("Запуск batch_transcribe.py с debug-уровнем")
    args = parse_args(argv or sys.argv[1:])

    backend = check_torio_installation()
    if backend == "torio":
        logger.info("Используется torio для декодирования аудио.")
    elif backend == "torchaudio":
        logger.warning("torio не работает, fallback на torchaudio.")
        print("ВНИМАНИЕ: torio не работает, fallback на torchaudio.")
    else:  # pragma: no cover - защитный блок
        print("FATAL: Не удалось определить backend для аудио. Пайплайн остановлен.")
        sys.exit(1)

    audio_path = Path(args.audio).expanduser().resolve()
    if not audio_path.exists():
        print(f"Файл {audio_path} не найден")
        sys.exit(1)

    base = audio_path.stem
    result_dir = args.output_dir / base
    _ensure_directory(result_dir)

    print(f"Деление {audio_path} на чанки...")
    chunks = split_audio(audio_path, args.chunk_length, result_dir, backend=backend)
    print(f"Транскрибация чанков ({len(chunks)} шт.)...")

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Используется устройство: {device}")

    predictor = Predictor()
    predictor.setup()
    results = transcribe_chunks(chunks, predictor)

    print("Объединение сегментов...")
    all_segments, speakers = merge_segments(results)
    final = {
        "segments": all_segments,
        "language": "ru",
        "num_speakers": len(speakers),
    }

    out_name = result_dir / f"batch_result_{base}.json"
    try:
        with out_name.open("w", encoding="utf-8") as file:
            json.dump(final, file, ensure_ascii=False, indent=2)
        debug_log(f"Saved result to {out_name}")
    except Exception as exc:  # pragma: no cover - защитный блок
        print(f"Ошибка при сохранении результата: {exc}")
        if DEBUG:
            traceback.print_exc()
        sys.exit(1)

    print(f"Batch результат сохранён в {out_name}")

    if not args.keep_chunks:
        for chunk_file, _ in chunks:
            try:
                chunk_file.unlink(missing_ok=True)
                debug_log(f"Deleted chunk: {chunk_file}")
            except OSError as exc:  # pragma: no cover - защитный блок
                logger.warning("Не удалось удалить %s: %s", chunk_file, exc)

        print("Временные файлы удалены.")
    else:
        print("Временные файлы сохранены (--keep-chunks).")


if __name__ == "__main__":
    logger.debug("Старт main()")
    main()
    logger.debug("Завершение main()")
