# California Housing Price Prediction

Предсказание медианной стоимости жилья в районах Калифорнии по характеристикам района.

## Данные

- **Источник:** `fetch_california_housing()` из sklearn (originally from StatLib)
- **20 640** районов, **8** признаков
- **Целевая переменная:** медианная стоимость дома (в сотнях тысяч $)

## Признаки

| Признак | Описание | Диапазон |
|---------|----------|----------|
| `MedInc` | Медианный доход (в десятках тысяч $) | 0.5 – 15 |
| `HouseAge` | Медианный возраст дома | 1 – 52 |
| `AveRooms` | Среднее число комнат | 0.8 – 142 |
| `AveBedrms` | Среднее число спален | 0.3 – 34 |
| `Population` | Население района | 3 – 35 682 |
| `AveOccup` | Среднее число жильцов на дом | 0.5 – 1 243 |
| `Latitude` | Широта | 32.5 – 34.8 |
| `Longitude` | Долгота | -124.4 – -114.3 |

## Модели

| Модель | Train RMSE | Val RMSE | Test RMSE |
|--------|-----------|----------|-----------|
| LinearRegression | 0.724 | 0.712 | 0.745 |
| Ridge + StandardScaler | подбор alpha | — | — |

RMSE в единицах таргета (×$100k). Ошибка ~$70k.

## Техники

- **StandardScaler** — приведение признаков к mean=0, std=1 (важно для Ridge)
- **Ручное разбиение 60/20/20** — train/val/test с перемешиванием
- **Ridge регуляризация** — подбор alpha по validation
- **Визуализация** — scatter plot predicted vs real

## Запуск

```bash
jupyter notebook california_housing.ipynb
```

## Структура

```
california_housing/
├── california_housing.ipynb   # Основной ноутбук
└── README.md
```
