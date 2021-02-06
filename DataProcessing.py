import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn import preprocessing

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

# Alternatively we could adjust the column
# data['seniority in role(years)'] = np.where((
#     data['seniority in role(years)'] > data['seniority (years) in comapny']),
#     data['seniority (years) in comapny'], data['seniority in role(years)'])

data.drop('seniority in role(years)', axis='columns', inplace=True)

column_1 = 'seniority (years) in comapny'
column_2 = 'salary'

plt.style.use('seaborn-whitegrid')
plt.scatter(data[column_1], data[column_2])
plt.show()


column_names_to_normalize = [column_1, column_2]
x = data[column_names_to_normalize].values

min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
data[column_names_to_normalize] = pd.DataFrame(x_scaled)

plt.scatter(data[column_1], data[column_2])
plt.show()

print(data)
