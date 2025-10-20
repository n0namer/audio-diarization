// Функция для форматирования транскрибации
function predictionFormat(transcription_obj) {
  try {
    console.log('predictionFormat: Starting format processing');
    
    // Проверяем наличие сегментов
    if (!transcription_obj) {
      console.error('predictionFormat: transcription_obj is null or undefined');
      return { 
        error: 'Invalid transcription data',
        formattedTranscription: 'Ошибка: Данные транскрибации отсутствуют или некорректны',
        callMetadata: {}
      };
    }
    
    if (!transcription_obj.output || !transcription_obj.output.segments) {
      console.error('predictionFormat: transcription_obj.output.segments is missing');
      console.error('predictionFormat: transcription_obj keys: ' + Object.keys(transcription_obj).join(', '));
      return { 
        error: 'Invalid transcription data structure',
        formattedTranscription: 'Ошибка: Структура данных транскрибации некорректна',
        callMetadata: transcription_obj.input || {}
      };
    }
    
    console.log('predictionFormat: Processing ' + transcription_obj.output.segments.length + ' segments');
    
    // #region ФОРМАТИРОВАНИЕ ТРАНСКРИБАЦИИ
    let dialog = ''; // итоговый текст
    let totalDuration = 0;
    let durations = {}; // длительность речи каждого спикера

    // Обрабатываем каждый сегмент
    transcription_obj.output.segments.forEach((item, index) => {
      console.log('predictionFormat: Processing segment ' + (index+1) + '/' + transcription_obj.output.segments.length);
      
      // Проверка наличия необходимых полей
      if (!item.speaker || item.start === undefined || item.end === undefined || !item.text) {
        console.warn('predictionFormat: Segment ' + (index+1) + ' has missing required fields');
        console.warn('predictionFormat: Segment data: ' + JSON.stringify(item));
      }
      
      // начитываем диалог
      dialog += item.speaker + ' (' + item.start.toFixed(2) + ' - ' + item.end.toFixed(2) + '): ' + item.text + '\n';

      // считаем длительность речи каждого из спикеров
      const duration = item.end - item.start;
    
      // Сохраняем продолжительность речи каждого спикера
      if (!durations[item.speaker]) {
          durations[item.speaker] = 0;
      }
      durations[item.speaker] += duration;
    });

    // Вычисляем общее время разговора
    totalDuration = Object.values(durations).reduce((a, b) => a + b, 0);
    console.log('predictionFormat: Total duration: ' + totalDuration + ' seconds');

    // Вычисляем соотношение длительности речи каждого спикера
    let stats_text = 'Длительность разговора: ' + Math.round(totalDuration) + ' с.\n\nСоотношение длительности речи:\r\n';
    console.log('predictionFormat: \nСоотношение длительности речи:');
    
    for (const speaker in durations) {
        const ratio = (durations[speaker] / totalDuration * 100).toFixed(2);
        stats_text += speaker + ': ' + ratio + '% (' + Math.round(durations[speaker]) + ' с. из ' + Math.round(totalDuration) + ' с.) \r\n';
        console.log('predictionFormat: ' + speaker + ': ' + ratio + '% (' + Math.round(durations[speaker]) + ' с. из ' + Math.round(totalDuration) + ' с.)');
    }
    
    stats_text += '--------------ТЕКСТ ДИАЛОГА:\r\n';
    const formattedText = stats_text + '\r\n' + dialog;
    // #endregion ФОРМАТИРОВАНИЕ ТРАНСКРИБАЦИИ
    
    console.log('predictionFormat: Completed formatting, final text length: ' + formattedText.length + ' characters');
    
    return { 
      formattedTranscription: formattedText,
      duration: Math.round(totalDuration),
      rawSegments: transcription_obj.output.segments,
      callMetadata: transcription_obj.input || {}
    };
  } catch (error) {
    console.error('Ошибка при форматировании транскрибации: ' + error.message);
    console.error('Стек ошибки: ' + error.stack);
    return { 
      error: error.message,
      formattedTranscription: 'Ошибка при форматировании транскрибации: ' + error.message,
      callMetadata: transcription_obj.input || {}
    };
  }
}

return predictionFormat($json);