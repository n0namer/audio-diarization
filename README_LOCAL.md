# Локальные правила и структура для audio-diarization

## Как использовать cog-whisper-diarization как зависимость (vendor)
...
(остальной текст без изменений)
...

## Как запускать

- Все входные аудиофайлы должны лежать в папке `input/`.
- Для batch-обработки любого файла:  
  `python batch_transcribe.py input/<audiofile>`  
  Например:  
  `python batch_transcribe.py input/test.mp3`  
  Результат: `result/test/batch_result_test.json` и итоговый `result/test.txt`.

- Для форматирования результата:  
  `python format_transcription.py result/test/batch_result_test.json`  
  Итог: .txt-файл с диалогом и статистикой по спикерам.

## Troubleshooting

### Пустой результат (`"segments": []`)

- Проверьте, что входной файл корректный и не пустой.
- Убедитесь, что модель поддерживает ваш язык/формат.
- Проверьте логи: если есть строка  
  `WARNING transcribe_chunk: <chunk_path>, пустой результат!`  
  — значит, модель не вернула сегменты.
- Попробуйте другой аудиофайл или обновите модель.

### Ошибки ffmpeg/torio

- Если в логе есть ошибки вида  
  `RuntimeError: FFmpeg extension is not available.`  
  или  
  `FileNotFoundError: Could not find module 'libtorio_ffmpeg*.pyd'`  
  — это не критично для split_audio, но может повлиять на работу torio.
- Проверьте, что ffmpeg установлен и доступен в PATH.

### Пример лога с пустым результатом

```
INFO transcribe_chunk: start, chunk_path=result/test/test_chunk_0.mp3, offset=0
WARNING transcribe_chunk: result/test/test_chunk_0.mp3, пустой результат!
```

## Максимальное ускорение (GPU/многопроцессорность)
...
(остальной текст без изменений)
...
