import json
import sys
import os

def load_json(json_path):
    """
    Загружает JSON-файл, возвращает dict.
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Ошибка при загрузке {json_path}: {e}")
        sys.exit(1)

def format_dialog(segments):
    """
    Формирует текст диалога и статистику по длительности речи каждого спикера.
    """
    dialog = ""
    durations = {}
    total_duration = 0

    for seg in segments:
        speaker = seg.get("speaker", "UNKNOWN")
        start = seg.get("start", 0)
        end = seg.get("end", 0)
        text = seg.get("text", "")
        dialog += f"{speaker} ({start:.2f} - {end:.2f}): {text}\n"
        duration = end - start
        durations[speaker] = durations.get(speaker, 0) + duration

    total_duration = sum(durations.values())
    stats_text = f"Длительность разговора: {round(total_duration)} с.\n\nСоотношение длительности речи:\n"
    for speaker, dur in durations.items():
        ratio = (dur / total_duration * 100) if total_duration else 0
        stats_text += f"{speaker}: {ratio:.2f}% ({round(dur)} с. из {round(total_duration)} с.)\n"
    stats_text += "--------------ТЕКСТ ДИАЛОГА:\n\n"
    return stats_text + dialog

def save_txt(txt_path, text):
    """
    Сохраняет текст в файл txt_path.
    """
    try:
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)
    except Exception as e:
        print(f"Ошибка при сохранении {txt_path}: {e}")
        sys.exit(1)

def main():
    """
    Основной процесс: загрузка json, форматирование диалога, сохранение txt.
    """
    if len(sys.argv) < 2:
        print("Использование: python format_transcription.py result/test/batch_result_test.json")
        sys.exit(1)
    json_path = sys.argv[1]
    data = load_json(json_path)
    segments = data.get("segments", [])
    if not segments:
        print("Нет сегментов для форматирования.")
        sys.exit(1)
    text = format_dialog(segments)
    base = os.path.splitext(os.path.basename(json_path))[0].replace("batch_result_", "")
    os.makedirs("result", exist_ok=True)
    txt_path = os.path.join("result", f"{base}.txt")
    save_txt(txt_path, text)
    print(f"Диалог и статистика сохранены в {txt_path}")

if __name__ == "__main__":
    main()
