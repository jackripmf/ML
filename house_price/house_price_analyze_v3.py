import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV

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


model = ensemble.GradientBoostingRegressor()

hyperparameters = {
    'n_estimators': [200, 300],
    'max_depth': [4, 6],
    'min_samples_split': [3, 4],
    'min_samples_leaf': [5, 6],
    'learning_rate': [0.01, 0.02],
    'max_features': [0.8, 0.9],
    'loss': ['huber', 'ls', 'lad']
}

grid = GridSearchCV(model, param_grid=hyperparameters, n_jobs=4)
grid.fit(X_train, y_train)
grid.best_params_

joblib.dump(grid, 'house_price_model_3.pkl')

Mae_train = mean_absolute_error(y_train, grid.predict(X_train))
print('Training set MAE: ', Mae_train)
mae_test = mean_absolute_error(y_test, grid.predict(X_test))
print('Test set MAE: ', mae_test)
