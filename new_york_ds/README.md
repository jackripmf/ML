# NYC Airbnb Price Prediction

Предсказание стоимости аренды жилья на Airbnb в Нью-Йорке.

## Данные

- **Источник:** [AB_NYC_2019.csv](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data)
- **48 895** объявлений, **16** признаков
- **Целевая переменная:** `price` (стоимость за ночь, $)

## Признаки

| Признак | Описание | Тип |
|---------|----------|-----|
| `room_type` | Тип жилья (entire home, private room, shared room) | категория |
| `neighbourhood_group` | Район (Manhattan, Brooklyn, Queens, Bronx, Staten Island) | категория |
| `neighbourhood` | Квартал | категория |
| `latitude`, `longitude` | Координаты | число |
| `minimum_nights` | Минимум ночей для бронирования | число |
| `reviews_per_month` | Отзывов в месяц | число |
| `number_of_reviews` | Общее число отзывов | число |
| `availability_365` | Доступных дней в году | число |
| `calculated_host_listings_count` | Объявлений у хоста | число |

## Предобработка

1. Очистка названий колонок (lowercase, underscores)
2. Заполнение пропусков в `reviews_per_month` средним
3. One-Hot Encoding для `room_type`, `neighbourhood_group`, `neighbourhood`
4. Логарифмирование целевой переменной (`log1p`)

## Модели

| Модель | Train RMSE | Val RMSE | Test RMSE |
|--------|-----------|----------|-----------|
| LinearRegression | 0.467 | 0.471 | 0.490 |
| Ridge (alpha=10) | — | 0.469 | — |

RMSE в логарифмическом масштабе. В долларах ошибка ~$60-80.

## Запуск

```bash
jupyter notebook rent_ny.ipynb
```

## Структура

```
new_york_ds/
├── rent_ny.ipynb           # Основной ноутбук
├── ds/
│   └── AB_NYC_2019.csv     # Датасет
└── README.md
```
