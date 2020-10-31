# BAO

Evalutation instrument for instance segmentation markup

[Hackathon "Лидеры Цифровой Трансформации"](https://lk.hack2020.innoagency.ru/)

# Installation

```
pip install -r requirements.txt
pip install -e .
```

## Usage

## Training

### Gather metrics from markup

```bash
python bao/metrics/run_metrics.py --add_markup
```

### Split any dataframe with "fname" column

```python
from bao.utils import split_df

split_df(df)
```

## Как это работает
Полученная модель позволяет сравнивать два снимка сегментационной разметки. 
Общая идеалогия такая - подобрать и придумать репрезентативные признаки.
Обучить на этих признаках модель. Постпроцессинг предсказаний модели.

### Модель
Для обучения модель использовался фреймворк [https://lightgbm.readthedocs.io/en/latest/](lightGBM).
Так как данных мало использовалась [https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html](вложенная кросс-валидация).
[https://c.mql5.com/3/103/nested-k-fold.png](Вложенная кросс-валидация)

### Синтетика
Предположение - используем маски, которые врач оценил как 5 - то есть очень похожие на разметку врача.
Выбранные маски можно принять экспертными и сравнить их с двумя другими. 
Оценки в этом случае не поменяются.
![Синтетика](https://i.ibb.co/jDHnVsD/Untitled-Diagram.png)

### Признаки для модели
Разные болезни могут иметь разный геометрический вид разметки. Также разные типы болезней может быть сложнее найти. Для предсказания типа болезни был использован репозиторий [torchxrayvision](https://github.com/mlmed/torchxrayvision), который дает предсказания болезней на 15 классов на датасете NIH.

Если разметка находится не в зоне легких, то это грубая ошибка. Для сегментации зоны с легкими исопльзовался [lungs_finder](https://github.com/dirtmaxim/lungs-finder/tree/master/lungs_finder). Использовали два режима - один выделяет кажду область легкого, второй, объединяет две области в одну, заполняя промежуток между ними и дополняя до прямоугольника. 
Если легкие не были найдены, то маской с легкими считается весь снимок.
Для выделения областей не входящих в маску легких использовалась следущая последовательность побитовых операций:
```
diff_mask = XOR(lungs_mask, model_mask)
result = AND(diff_mask, model_mask)
```
Для нормализации метрики была посчитана площадь result и поделена на площадь первоначальной маски предсказания.

IOU

DICE

Hausdorff distance

F1 по объектам

**(c) Team "Бань Бао"**
