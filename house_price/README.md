# House Price Prediction — Melbourne

Прогнозирование стоимости жилья в Мельбурне на основе данных об объектах недвижимости.

## Задача

Регрессия: предсказать цену дома (`Price`) по набору признаков (район, тип, количество комнат, площадь и т.д.).

## Данные

- **Источник:** Melbourne Housing Full (Kaggle)
- **Файл:** `ds/Melbourne_housing_FULL.csv`
- **Признаки:** Suburb, Type, Rooms, Price, Distance, Bedroom2, Bathroom, Car, Landsize, BuildingArea, YearBuilt, CouncilArea и др.
- **Целевая переменная:** `Price`

## Модели

### v1 — GradientBoostingRegressor (базовая)
- `n_estimators=150`, `learning_rate=0.1`, `max_depth=30`
- Кодирование: LabelEncoder
- Метрика: MAE

### v2 — GradientBoostingRegressor (оптимизация признаков)
- `n_estimators=250`, `max_depth=5`
- Frequency encoding вместо LabelEncoder для Suburb и CouncilArea
- Ручной маппинг для Type

### v3 — GridSearchCV + GradientBoosting
- Автоподбор гиперпараметров через `GridSearchCV`
- Параметры: n_estimators, max_depth, min_samples_split, learning_rate, loss

## Результаты

| Версия | Train MAE | Test MAE |
|--------|-----------|----------|
| v1     | —         | —        |
| v2     | —         | —        |
| v3     | —         | —        |

*(Заполни после запуска)*

## Запуск

```bash
cd house_price
python house_price_analyze.py      # v1
python house_price_analyze_v2.py   # v2
python house_price_analyze_v3.py   # v3 (GridSearch)
```

## Зависимости

```bash
pip install pandas scikit-learn joblib
```
