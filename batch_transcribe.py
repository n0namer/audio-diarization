import os
import logging

# Настройка логирования
logging.basicConfig(
    filename='batch_transcribe.log',
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)
import subprocess
import json
import sys
import traceback
from multiprocessing import Pool

sys.path.append(os.path.join(os.path.dirname(__file__), "cog-whisper-diarization"))
from predict import Predictor

CHUNK_LENGTH_MIN = 10  # для максимального параллелизма
DEBUG = os.environ.get("DEBUG", "0") == "1"

def debug_log(msg):
    if DEBUG:
        print(f"[DEBUG] {msg}")

def split_audio(audio_path, chunk_length_min, out_dir, backend="torio"):
    """
    Делит аудиофайл на чанки. Если backend='torio' — стандартная логика через ffmpeg.
    Если backend='torchaudio' — fallback через torchaudio (wav/mp3 поддержка).
    """
    base = os.path.splitext(os.path.basename(audio_path))[0]
    logger.info(f"split_audio: start, audio_path={audio_path}, chunk_length_min={chunk_length_min}, out_dir={out_dir}, backend={backend}")
    if backend == "torchaudio":
        try:
            import torchaudio
            import torch
            waveform, sample_rate = torchaudio.load(audio_path)
            total_sec = waveform.shape[1] / sample_rate
            chunk_length_sec = chunk_length_min * 60
            chunks = []
            for i, start in enumerate(range(0, int(total_sec), chunk_length_sec)):
                end = min(start + chunk_length_sec, int(total_sec))
                chunk_wave = waveform[:, start * sample_rate:end * sample_rate]
                out_file = os.path.join(out_dir, f"{base}_chunk_{i}.wav")
                torchaudio.save(out_file, chunk_wave, sample_rate)
                logger.info(f"split_audio (torchaudio): created chunk {out_file}")
                chunks.append((out_file, start))
            logger.info(f"split_audio (torchaudio): total_chunks={len(chunks)}")
            return chunks
        except Exception as e:
            logger.error(f"split_audio (torchaudio) failed: {e}", exc_info=True)
            print(f"FATAL: split_audio (torchaudio) failed: {e}")
            sys.exit(1)
    else:
        cmd = [
            "ffprobe", "-v", "error", "-show_entries",
            "format=duration", "-of",
            "default=noprint_wrappers=1:nokey=1", audio_path
        ]
        logger.debug(f"ffprobe cmd: {' '.join(cmd)}")
        try:
            duration = float(subprocess.check_output(cmd).decode().strip())
            logger.info(f"split_audio: duration={duration}")
        except Exception as e:
            logger.error(f"Ошибка при получении длительности файла: {e}", exc_info=True)
            sys.exit(1)
        chunk_length_sec = chunk_length_min * 60
        chunks = []
        for i, start in enumerate(range(0, int(duration), chunk_length_sec)):
            out_file = os.path.join(out_dir, f"{base}_chunk_{i}.mp3")
            cmd = [
                "ffmpeg", "-y", "-i", audio_path,
                "-ss", str(start), "-t", str(chunk_length_sec),
                "-c", "copy", out_file
            ]
            logger.debug(f"ffmpeg cmd: {' '.join(cmd)}")
            try:
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                logger.info(f"split_audio: created chunk {out_file}")
            except Exception as e:
                logger.error(f"Ошибка при создании чанка {out_file}: {e}", exc_info=True)
                sys.exit(1)
            chunks.append((out_file, start))
        logger.info(f"split_audio: total_chunks={len(chunks)}")
        return chunks

def predictor_worker_init():
    global predictor
    predictor = Predictor()
    predictor.setup()
    debug_log("Predictor initialized in worker")

def transcribe_chunk(args, predictor):
    import traceback
    chunk_path, offset = args
    logger.info(f"transcribe_chunk: start, chunk_path={chunk_path}, offset={offset}")
    logger.debug(f"TRACE: transcribe_chunk input: chunk_path={chunk_path}, offset={offset}")
    try:
        logger.debug(f"TRACE: predictor.predict call with file={chunk_path}, language='ru', prompt=None, num_speakers=None")
        result = predictor.predict(
            file=chunk_path,
            language="ru",
            prompt=None,
            num_speakers=None
        )
        logger.debug(f"TRACE: predictor.predict returned: {repr(result)}")
        if result is None:
            logger.warning(f"transcribe_chunk: {chunk_path}, result is None!")
            return {"segments": []}
        logger.debug(f"TRACE: type(result)={type(result)}, dir(result)={dir(result)}")
        segments = getattr(result, "segments", None)
        logger.debug(f"TRACE: result.segments={segments}")
        num_segments = len(segments) if segments is not None else 0
        logger.info(f"transcribe_chunk: {chunk_path}, segments={num_segments}")
        if not segments:
            logger.warning(f"transcribe_chunk: {chunk_path}, пустой результат!")
            return {"segments": []}
    except Exception as e:
        logger.error(f"Ошибка при транскрибации {chunk_path}: {e}", exc_info=True)
        logger.error(f"TRACE: Exception traceback:\n{traceback.format_exc()}")
        return {"segments": []}
    for seg in result.segments:
        seg["start"] += offset
        seg["end"] += offset
        seg["language"] = "ru"
        for w in seg["words"]:
            w["start"] += offset
            w["end"] += offset
            w["language"] = "ru"
    logger.debug(f"TRACE: result.dict()={result.dict() if hasattr(result, 'dict') else str(result)}")
    return result.dict()

def merge_segments(results):
    debug_log(f"Merging {len(results)} chunk results")
    all_segments = []
    speakers = set()
    for res in results:
        for seg in res.get("segments", []):
            seg["language"] = "ru"
            for w in seg.get("words", []):
                w["language"] = "ru"
            all_segments.append(seg)
            speakers.add(seg.get("speaker", "UNKNOWN"))
    debug_log(f"Total segments: {len(all_segments)}, speakers: {speakers}")
    return all_segments, speakers

def check_torio_installation():
    """
    Проверяет наличие и работоспособность torio и его FFmpeg-расширений.
    Если torio не установлен — пытается установить через pip.
    Если расширения не работают — пытается fallback на torchaudio.
    Логирует подробности и возвращает строку: "torio", "torchaudio", либо завершает процесс с ошибкой.
    """
    try:
        import importlib.util
        import sys
        import os
        import subprocess

        # Проверка наличия torio
        spec = importlib.util.find_spec("torio")
        if spec is None:
            logger.warning("torio не установлен. Пробую установить через pip...")
            print("torio не найден. Автоматическая установка через pip...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "torio"])
                logger.info("torio успешно установлен через pip.")
            except Exception as e:
                logger.error(f"Не удалось установить torio: {e}", exc_info=True)
                print(f"FATAL: Не удалось установить torio: {e}")
                return "torchaudio"  # fallback

        import torio
        torio_path = os.path.dirname(torio.__file__)
        lib_dir = os.path.join(torio_path, "lib")
        found = False
        if os.path.isdir(lib_dir):
            for fname in os.listdir(lib_dir):
                if fname.startswith("libtorio_ffmpeg") and fname.endswith(".pyd"):
                    found = True
                    logger.info(f"Найдено расширение: {fname}")
            if not found:
                logger.warning("В папке torio/lib нет libtorio_ffmpeg*.pyd — fallback на torchaudio.")
                print("torio установлен, но нет расширений. Fallback на torchaudio.")
                return "torchaudio"
        else:
            logger.warning("Папка torio/lib не найдена. Fallback на torchaudio.")
            print("torio установлен, но папка lib не найдена. Fallback на torchaudio.")
            return "torchaudio"

        # Пробуем открыть dummy-файл через torio
        import tempfile
        import wave
        import contextlib
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                with wave.open(tmp.name, "w") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(16000)
                    wf.writeframes(b"\x00" * 32000)
                import torio.audio
                with contextlib.closing(torio.audio.open(tmp.name)) as f:
                    _ = f.read()
            logger.info("torio.audio.open() успешно прочитал WAV — расширения работают.")
            return "torio"
        except Exception as e:
            logger.warning(f"torio установлен, но расширения не работают: {e}. Fallback на torchaudio.", exc_info=True)
            print(f"torio установлен, но расширения не работают: {e}. Fallback на torchaudio.")
            return "torchaudio"
    except Exception as e:
        logger.error(f"Ошибка при проверке torio: {e}", exc_info=True)
        print(f"ОШИБКА: Ошибка при проверке torio: {e}")
        return "torchaudio"

def main():
    logger.debug("Запуск batch_transcribe.py с debug-уровнем")
    # --- GKY: автоматическая проверка torio ---
    backend = check_torio_installation()
    if backend == "torio":
        logger.info("Используется torio для декодирования аудио.")
    elif backend == "torchaudio":
        logger.warning("torio не работает, fallback на torchaudio.")
        print("ВНИМАНИЕ: torio не работает, fallback на torchaudio.")
    else:
        print("FATAL: Не удалось определить backend для аудио. Пайплайн остановлен.")
        sys.exit(1)
    # --- /GKY ---
    if len(sys.argv) < 2:
        print("Использование: python batch_transcribe.py <audiofile>")
        sys.exit(1)
    audio_path = sys.argv[1]
    if not os.path.exists(audio_path):
        print(f"Файл {audio_path} не найден")
        sys.exit(1)
    base = os.path.splitext(os.path.basename(audio_path))[0]
    result_dir = os.path.join("result", base)
    os.makedirs(result_dir, exist_ok=True)
    print(f"Деление {audio_path} на чанки...")
    chunks = split_audio(audio_path, CHUNK_LENGTH_MIN, result_dir, backend=backend)
    print(f"Транскрибация чанков ({len(chunks)} шт.)...")
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Используется устройство: {device}")
    if device == "cuda":
        # Обработка чанков последовательно на GPU
        predictor = Predictor()
        predictor.setup()
        results = [transcribe_chunk((chunk_path, offset), predictor) for chunk_path, offset in chunks]
    else:
        # Последовательная обработка на CPU (Pool отключён из-за несовместимости с Predictor)
        predictor = Predictor()
        predictor.setup()
        results = [transcribe_chunk((chunk_path, offset), predictor) for chunk_path, offset in chunks]
    print("Объединение сегментов...")
    all_segments, speakers = merge_segments(results)
    final = {
        "segments": all_segments,
        "language": "ru",
        "num_speakers": len(speakers)
    }
    out_name = os.path.join(result_dir, f"batch_result_{base}.json")
    try:
        with open(out_name, "w", encoding="utf-8") as f:
            json.dump(final, f, ensure_ascii=False, indent=2)
        debug_log(f"Saved result to {out_name}")
    except Exception as e:
        print(f"Ошибка при сохранении результата: {e}")
        if DEBUG:
            traceback.print_exc()
        sys.exit(1)
    print(f"Batch результат сохранён в {out_name}")

    for chunk_file, _ in chunks:
        if os.path.exists(chunk_file):
            os.remove(chunk_file)
            debug_log(f"Deleted chunk: {chunk_file}")
    print("Временные файлы удалены.")

if __name__ == "__main__":
    logger.debug("Старт main()")
    main()
    logger.debug("Завершение main()")
