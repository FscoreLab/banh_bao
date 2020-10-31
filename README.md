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

### Синтетика
Предположение - используем маски, которые врач оценил как 5 - то есть очень похожие на разметку врача.
Выбранные маски можно принять экспертными и сравнить их с двумя другими. 
Оценки в этом случае не поменяются.
![Синтетика](https://i.ibb.co/jDHnVsD/Untitled-Diagram.png)

### Признаки для модели
Для 

**(c) Team "Бань Бао"**
