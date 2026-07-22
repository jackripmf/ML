import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder

import joblib

df = pd.read_csv('ds/Melbourne_housing_FULL.csv')

print(df.head())
print(df.columns)

del df['Address']
del df['Method']
del df['SellerG']
del df['Date']
del df['Postcode']
del df['Lattitude']
del df['Longtitude']
del df['Regionname']
del df['Propertycount']

print(df.head())
print(df.columns)

df.dropna(axis=0, how='any', subset=None, inplace=True)
print(df.head())
# df = pd.get_dummies(df, columns=['Suburb', 'CouncilArea', 'Type'])
le_suburb = LabelEncoder()
df['Suburb'] = le_suburb.fit_transform(df['Suburb'])
le_area = LabelEncoder()
df['CouncilArea'] = le_area.fit_transform(df['CouncilArea'])
le_type = LabelEncoder()
df['Type'] = le_type.fit_transform(df['Type'])

print(df.head())

X = df.drop('Price', axis=1)
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)

# Модель на основе градиентного бустинга
model = ensemble.GradientBoostingRegressor(
    n_estimators=150, #количество деревьев
    learning_rate=0.1, #степень влияния дополнительных деревьев
    max_depth=30, #максимальное количество уровней(глубина) каждого дерева
    min_samples_split=4, #минимальноге количество образцов для выполнения бинарного разделения
    min_samples_leaf=6, #минимальное количество образцов для создания новой ветви
    max_features=0.6, #количество признаков, предъявляемых модели для наилучшего разбиения
    loss='huber', #коэфициент ошибок модели
)

model.fit(X_train, y_train)

joblib.dump(model, 'house_price_model_1.pkl')

Mae_train = mean_absolute_error(y_train, model.predict(X_train))
print('Training set MAE: ', Mae_train)
mae_test = mean_absolute_error(y_test, model.predict(X_test))
print('Test set MAE: ', mae_test)
