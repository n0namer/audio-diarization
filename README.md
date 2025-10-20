# audio-diarization

## Описание
Скрипты для аудио-диаризации с использованием внешней зависимости [cog-whisper-diarization](https://github.com/your-org/cog-whisper-diarization).

## Установка

1. Клонируйте этот репозиторий:
   ```
   git clone https://github.com/your-org/audio-diarization.git
   cd audio-diarization
   ```

2. Установите основные зависимости:
   ```
   pip install -r requirements.txt
   ```

3. Папка cog-whisper-diarization не входит в этот репозиторий и должна быть добавлена вручную (например, клонированием):
   ```
   git clone https://github.com/your-org/cog-whisper-diarization.git
   ```
   После этого установите зависимости для cog-whisper-diarization отдельно:
   ```
   cd cog-whisper-diarization
   pip install -r requirements.txt
   cd ..
   ```
   > **Важно:** cog-whisper-diarization полностью исключена из git-истории и .gitignore. Просто разместите папку рядом с основным проектом.

## Что делать, если папка отсутствует

- Скачайте или склонируйте cog-whisper-diarization вручную.
- Убедитесь, что она находится в корне рядом с этим проектом.
- Установите её зависимости отдельно.

## Использование

Пример запуска:
```
python batch_transcribe.py --input input/ --output result/
```

## Важно
- Папка cog-whisper-diarization не входит в этот репозиторий и должна быть установлена отдельно.
- Все временные и выходные файлы, а также cog-whisper-diarization, игнорируются через .gitignore.
