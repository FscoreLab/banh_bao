# BAO

Evalutation instrument for instance segmentation markup

[Hackathon "Лидеры Цифровой Трансформации"](https://lk.hack2020.innoagency.ru/)

# Installation

```
pip install -r requirements.txt
pip install -e .
```

# Usage

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

To be continued ...

**(c) Team "Бань Бао"**