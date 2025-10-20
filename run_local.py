import json
import os
import sys

# Для запуска из корня, корректно импортируем Predictor
sys.path.append(os.path.join(os.path.dirname(__file__), "cog-whisper-diarization"))
from predict import Predictor

if __name__ == "__main__":
    AUDIO_PATH = "test.mp3"
    predictor = Predictor()
    predictor.setup()
    result = predictor.predict(
        file=AUDIO_PATH,
        language="ru",
        prompt=None,
        num_speakers=None
    )
    # Проставляем язык для всех сегментов и слов
    for seg in result.segments:
        seg["language"] = "ru"
        for w in seg["words"]:
            w["language"] = "ru"
    out_name = f"result_{os.path.splitext(AUDIO_PATH)[0]}.json"
    with open(out_name, "w", encoding="utf-8") as f:
        json.dump(result.dict(), f, ensure_ascii=False, indent=2)
    print(f"Результат сохранён в {out_name}")
