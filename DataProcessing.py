from sklearn.preprocessing import MinMaxScaler
from matplotlib.colors import ListedColormap
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

# data = raw_data.copy()
# data.drop('seniority in role(years)', axis='columns', inplace=True)


# plt.style.use('seaborn-whitegrid')
# plt.scatter(data['seniority (years) in comapny'], data['salary'])
# # plt.show()


# # column_names_to_normalize = ['salary', 'seniority (years) in comapny']
# # x = data[column_names_to_normalize].values

# x = data.values  # returns a numpy array
# min_max_scaler = preprocessing.MinMaxScaler()
# x_scaled = min_max_scaler.fit_transform(x)
# data = pd.DataFrame(x_scaled)
# # test = data[['salary', 'seniority (years) in comapny']]
# # print(data)
# # MinMaxScaler.fit_transform(test.values)
# # df_temp = pd.DataFrame(
# #     x_scaled, columns=column_names_to_normalize, index=data.index)
# # data[column_names_to_normalize] = df_temp


# print(data.describe(include='all'))
