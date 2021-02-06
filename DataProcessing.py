import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

raw_data = pd.read_csv('data/dataUpdated.csv')
print(raw_data.describe(include='all'))

print(raw_data.head())

print(raw_data.dtypes)

data = raw_data.copy()
data = data.apply(pd.to_numeric, errors='coerce')

print(data.describe(include='all'))

print(data.isna().sum())

na_list = data.columns[data.isna().any()].tolist()
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit(data[na_list])
data[na_list] = imp.transform(data[na_list])

print(data.isna().sum())

print(data.count())

print(data.dtypes)

print(data.describe(include='all'))

# Alternatively, we could adjust the column
# data['seniority in role(years)'] = np.where((
#     data['seniority in role(years)'] > data['seniority (years) in comapny']),
#     data['seniority (years) in comapny'], data['seniority in role(years)'])

data.drop('seniority in role(years)', axis='columns', inplace=True)
data['grade in last year review (0-10)'] = np.where(
    data['grade in last year review (0-10)'] < 0,
    0,
    data['grade in last year review (0-10)'])

data['monthly return on loan'] = np.where(
    data['monthly return on loan'] > data['monthly return on loan'].quantile(0.75),
    data['monthly return on loan'].quantile(0.75),
    data['monthly return on loan']
)


print(data.describe(include='all').to_markdown())


column_1 = 'seniority (years) in comapny'
column_2 = 'salary'

plt.style.use('seaborn-whitegrid')
plt.scatter(data[column_1], data[column_2])
plt.show()

column_names_to_normalize = [column_1, column_2]
real_data = data[column_names_to_normalize].values

min_max_scaler = preprocessing.MinMaxScaler()
scaled_data = min_max_scaler.fit_transform(real_data)
normalized_data = pd.DataFrame(scaled_data, columns=column_names_to_normalize)

plt.scatter(normalized_data[column_1], normalized_data[column_2])
plt.show()

X_train, X_test, y_train, y_test = train_test_split(real_data[:, 0].reshape(-1, 1), real_data[:, 1], test_size=0.2)

regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

df = pd.DataFrame({'Real Values': y_test, 'Predicted Values': y_pred})
print(df.head().to_markdown())


def all_metrics(y_test, y_pred):
    print('Mean absolute error: %.2f' % metrics.mean_absolute_error(y_test, y_pred))
    print('Mean squared error: %.2f' % metrics.mean_squared_error(y_test, y_pred))
    print('Coefficient of determination: %.2f' % metrics.r2_score(y_test, y_pred))


all_metrics(y_test, y_pred)

X_train, X_test, y_train, y_test = train_test_split(normalized_data[column_1].values.reshape(-1, 1),
                                                    normalized_data[column_2].values,
                                                    test_size=0.2)

regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

df = pd.DataFrame({'Real Values': y_test, 'Predicted Values': y_pred})
print(df.head().to_markdown())

all_metrics(y_test, y_pred)
