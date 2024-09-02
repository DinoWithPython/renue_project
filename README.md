# Распознавание мусора на конвейерной ленте
<hr>

**Цель проекта:** Необходимо разработать решение для отслеживания и сортировки мусора на конвейере – выделять мусор в общем потоке предметов.
* Решение должно выдавать координаты центра обнаруженных объекта для каждого кадра;
* Скорость обработки должна быть не более 100 мс;
* Высокая метрика `MOTA` (Multiple Object Tracking Accuracy).

**Задачи:** 
* Ознакомиться с данными, форматами датасетов
* Изучить работу  трекеров, выбрать подходящий и создать базовое решение
* Обучить трекер, провести сравнительный анализ моделей и алгоритмов, предложить варианты улучшения решения
* Протестировать решение, проанализировать результат
* Подготовить доклад по результатам исследований

**В проекте использовались следующие трекеры:**
* Базовое решение от `ultralytics`
* Базовое решение от `ultralytics` с подбором гиперпараметров
* `Deep SORT` с перебором гиперпараметров
* `SMILEtrack`
* `SORT`

**Лучшие метрики:**
<table>
    <tr>
        <td><b>Трекер</b></td>
        <td><b>MOTA</b></td>
        <td><b>MOTP</b></td>
        <td><b>время на фрейм, мс</b></td>
        <td><b>среднее отклонение</b></td>
        <td><b>гиперпараметры</b></td>
    </tr>
    <tr>
        <td>ultralytics</td>
        <td>0.9124</td>
        <td>0.0929</td>
        <td>146.17</td>
        <td>287.58</td>
        <td>-</td>
    </tr>
    <tr>
        <td>ultralytics hyp</td>
        <td>0.9173</td>
        <td>0.9142</td>
        <td>113.04</td>
        <td>19.93</td>
        <td>`iou` 0.6 или 0.7, `conf` 0.5, `max_det` 50 или 300</td>
    </tr>
    <tr>
        <td>deep sort</td>
        <td>0.9319</td>
        <td>0.321</td>
        <td>50.17</td>
        <td>36.09</td>
        <td>`max_dis_iou` 0.8, `age` 1, `max_cos_dis` 0.3 или 0.1, `n_init` 6</td>
    </tr>
    <tr>
        <td>smile</td>
        <td>0.9197</td>
        <td>0.099</td>
        <td>199.10</td>
        <td>61.42</td>
        <td>-</td>
    </tr>
    <tr>
        <td>sort</td>
        <td>0.9148</td>
        <td>0.1812</td>
        <td>55.10</td>
        <td>60.32</td>
        <td>-</td>
    </tr>
</table>

Относительно двух трекеров был проведен визуальный анализ. 
**Для `deep sort`:**
Смазанные видео, итог:
* У некоторых объектов айди меняется под конец кадра (возможно это происходит из-за наложения одной рамки на другую)
* У некоторых объектов, у кого была такая же проблема – в конце айди вернулось как было, то есть эта проблема касается не всех подряд
* Попались и такие объекты, когда в начале, буквально на два кадра смог выделить объекты рамкой (причем только у одного из двух объектов был трекер)
* Происходит «наложение» рамок друг на друга из-за чего объект, который шел все время с одним айди, под конец кадра уходит совсем с другим
* Некоторые объекты находятся и выделяются рамкой только в самом конце
* А также некоторые объекты не находятся самим треком, трека нет
* Некоторые объекты плохо определяются (выделяются рамкой) – рамка оказалась меньше и захватила объект не полностью
* Бывает такое, что из-за неправильных размеров рамки, два объекта превращаются в один

Общий итог:
* Большинство объектов находятся, выделяются рамкой и рамка не пропадает до самого конца
* Большинство объектов трекаются нормально 

**Для `smile`:**
Положительные моменты:
* Объекты сразу выделяются рамкой по размеру, у `DeepSort` рамка в некоторых случаях сначала была гораздо больше, чем сам объект и только на середине кадра рамка находила оптимальный размер
* Почти все объекты находятся сразу и выделяются рамкой
* Трекаются объекты хорошо
* Уверенность отслеживания объектов – высокая
      
Отрицательные моменты:
* Периодические пропажи рамок объектов
* Рамка иногда захватывала объект не полностью
* Объект не находился и не трекался до самого конца кадра, только под самый конец объект находился и выделился в рамку
* Периодически мелкие объекты не обнаруживаются, и не выделяются рамкой
* Разделение одного крупного объекта на несколько маленьких

## Общие выводы и рекомендации заказчику:
* касательно метрики `MOTA`, скорости обработки кадра и среднего отклонения покаывает себя `Deep SORT`. Обратная сторона высокой `MOTA` это раздувание объекта в ширину и довольно выскоая метрика `MOTP`;
* со средним временем обработки кадра и так же хорошим отклонением показывает себя базовое решение от `YOLO` с лучшими гиперпараметрами `iou` равное 0.6 или 0.7, `conf` равное 0.5, `max_det` равное 300 или 50. Этот вариант так же показывает себя лучше в метрике `MOTP`;
* трекер `SORT` показал себя хуже двух предыдущих моделей как относительно `MOTA`, так и относительно `MOTP`;
* трекер `Smile TRACK` иногда показывает себя лучше двух лидеров, но основная проблема в потере мелких объектов и разделения крупных объектов на мелкие, что занижает ключевые метрики. При этом данный трекер в некоторых случаях показывает себя лучше при распозновании объектов среднего размера. 

**Рекомендации для заказчика:**    
1. Модернизировать конвейерную ленту, выбрать ребристую, чтобы уменьшить "катание"  мусора по ленте.
2. Попробовать изменить скорость движения ленты - возможно, незначительное замедление ленты даст лучшую скорость в сортировке мусора и снизит количество мусора, который повторно закидывается на ленту.
3. Улучшить освещение в зоне камеры, возможно, добавить боковой свет - а вдруг это повлияет на качество детекции объектов.
4. Провести обучение детектора на большем датасете.

В соответствии с поставленной задачей лучшим по метрике MOTA был выбран трекер - `Smile` , с результатом **`MOTA` = 0.9197**, поскольку так же показал хоршую метрику `MOTP`, практически на уровне базовой `YOLO`, но немного лучше. Так же приведены рекомендации касательно того, как можно улучшить работу системы распознавания и забора мусора с ленты.

## Инструкция по установке
Установить зависимости из файла requirements.txt:
```
pip install -r requirements.txt
```

Запустить главный файл*:
```
python main.py
```
* - исходя из того, `main.py` находится в одной папке с Datasets и Videos.
