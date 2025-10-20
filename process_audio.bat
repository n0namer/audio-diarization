@echo off
chcp 65001 >nul
REM === Укажите ниже список файлов для обработки (только имя, без input\) ===
REM Пример:
REM set FILES=test.mp3 psi_natalya_20251006.mp3 psi_natalya_20251013.mp3

set FILES=test.mp3

REM === Не менять ниже! ===
for %%F in (%FILES%) do (
    if exist input\%%F (
        echo Обработка файла: input\%%F
        set KMP_DUPLICATE_LIB_OK=TRUE
        python batch_transcribe.py input\%%F
        python format_transcription.py result\%%~nF\batch_result_%%~nF.json
        echo Готово: result\%%~nF.txt
    ) else (
        echo [ОШИБКА] Файл input\%%F не найден!
    )
)
echo Все файлы обработаны!
pause
