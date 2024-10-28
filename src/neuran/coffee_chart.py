import numpy as np
import pandas as pd
# import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

from .manipulate_input_data import get_input_data
from .db.sales import get_list_of_sales, get_sorted_sale_data, get_first_and_last_sale_dates
from .db.products import get_list_of_products
from .db.days import get_list_of_days

user_name = 'coffee_nyc'
time_division = 'daily'

first_date, last_date = get_first_and_last_sale_dates(user_name)

products_data = get_list_of_products(user_name)
days_array = get_list_of_days(user_name, first_date, last_date)
# input_data = get_input_data(days_array, products_data)[0].tolist()


days_by_products = get_input_data(days_array, products_data).tolist()
days_by_products = sum(days_by_products, [])
# print(len(days_by_products))
# print(days_by_products[:2])
sales_sum = []
total_sum = []
for product in products_data:
    list_of_sales = get_list_of_sales(user_name, product)
    sales_counts_per_day = get_sorted_sale_data(list_of_sales, first_date, last_date, time_division)
    total_sum.append([len(sale) for sale in sales_counts_per_day])
    sales_sum.extend([len(sale) for sale in sales_counts_per_day])
    
total = np.sum(total_sum, axis=0)
    
    
# print(len(sales_sum))
# print(len(total_sum))

for index, day in enumerate(days_by_products):
    day.append(sales_sum[index])

[day.append(total[index % len(total)]) for index, day in enumerate(days_by_products)]

    
adjust_input_data = [{
    'year': item[0],
    'month': item[1],
    'day': item[2],
    'weekday': item[3],
    'min_temperature': item[4],
    'max_temperature': item[5],
    'rain': item[6],
    'vacation': item[7],
    'holiday': item[8],
    'unusual': item[9],
    'product_index': item[10],
    'category_index': item[11],
    'sales_sum': item[12],
    'total': item[13],
} for item in days_by_products]

days_data = pd.DataFrame(adjust_input_data)
times = range(1, len(days_by_products) + 1)
days_data.bfill(inplace=True)
##########################################################################
# print(days_data.head())
# print(days_data.info())
# print(days_data['total'].value_counts())
# print(days_data[days_data['rain'] > 60]['rain'].count())
# print(days_data['max_temperature'].describe())
# print(days_data.describe(include='all'))
# print(days_data['min_temperature'].isnull().sum())
# print(days_data.isnull().sum())

# print(days_data.duplicated().sum())
# print(days_data[days_data.duplicated()])
# print(days_data.product_index.unique())
# print(len(days_data.product_index.unique()))

# print(sum(days_data.min_temperature < 0) / 29)
# Q1 = np.percentile(days_data['sales_sum'], 25)
# Q3 = np.percentile(days_data['sales_sum'], 75)
# IQR = Q3 - Q1
# print(Q1)
# print(Q3)
# print(IQR)
# print(len(days_data[(days_data['sales_sum'] < Q1 - 1.5 * IQR) | (days_data['sales_sum'] > Q3 + 1.5 * IQR)].index))
# sns.boxenplot(days_data.total)

# z_score = (days_data['sales_sum'] - days_data['sales_sum'].mean()) / days_data['sales_sum'].std()
# outliers = abs(z_score) > 3
# print(sum(outliers))
# print(min(days_data.sales_sum[outliers]))
# plt.scatter(days_data[outliers]['vacation'], days_data[outliers]['sales_sum']).set_facecolor('red')

# bins = [-100, 0, 10, 20, 30, 40, 100]
# labels = [0, 1, 2, 3, 4, 5]
# da = pd.cut(days_data['max_temperature'], bins, labels=labels)
# print(da.unique())

# ct1 = pd.crosstab(days_data['total'], days_data['vacation'])
# ct1.plot(kind='bar', figsize=(14, 5))
# ct2 = days_data['min_temperature'].value_counts().plot(kind='bar')

# fig = plt.figure(figsize=(18, 4))
# sns.boxplot(days_data.sales_sum, whis=3, color='orange')
# sns.boxplot(x='year', y='sales_sum', data=days_data)
# sns.violinplot(x='weekday', y='sales_sum', data=days_data)
# ct3 = pd.crosstab(days_data['sales_sum'], days_data['rain'])
# print(chi2_contingency(ct3))

# days_data.loc[days_data['sales_sum'] <= 8, 'sales_sum'] = 0
# days_data.loc[days_data['rain'] > 0, 'rain'] = 1

# sns.scatterplot(x='min_temperature', y='max_temperature', hue='rain', size='rain', sizes=(20, 10), data=days_data)

# ax = plt.axes(projection='3d')
# xdata = days_data.min_temperature
# ydata = days_data.max_temperature
# zdata = days_data.total
# ax.scatter3D(xdata, ydata, zdata)
# ax.scatter3D(xdata, ydata, zdata, c=zdata, depthshade=False)

# sns.pairplot(days_data[['min_temperature', 'max_temperature', 'total']])


# plt.scatter(x=days_data['min_temperature'],y=days_data['total'],c='r',marker='s',label='min_temperature')
# plt.scatter(x=days_data['max_temperature'],y=days_data['total'],c='b',marker='o',label='max_temperature')
# plt.scatter(x=days_data['rain'],y=days_data['total'],c='k',marker='*',label='rain')
# plt.legend(numpoints=1,loc=4)
# ipd = 'min_max'
# days_data[ipd] = (days_data['max_temperature'] * (days_data['max_temperature'] + days_data['vacation'] * 20))
# X = days_data[['min_temperature', 'max_temperature']].values.reshape(days_data[['min_temperature', 'max_temperature']].shape[0], 2)  # Feature
# y = days_data['total'].values.reshape(days_data['total'].shape[0], 1)  # Target
ipd = 'max_temperature'
X = days_data[[ipd]]  # Feature
y = days_data['total']  # Target
model = linear_model.LinearRegression(fit_intercept=False )
model.fit(X, y)

# plt.scatter(days_data[ipd], days_data['total'], c='k', marker='*', label='ipd')
# plt.plot(days_data[ipd], model.predict(X), color='blue', linewidth=3, label='Regression line')

# print("b1:",model.coef_)
# print("b0:",model.intercept_)

# plt.xlabel('IPD')
# plt.ylabel('Total')
# plt.legend()

def sse(Y, Y_HAT):  
    sse = sum([(y - y_hat)**2 for y, y_hat in zip(Y, Y_HAT)])
    return sse

Y = days_data['total'].tolist()
Y_HAT = model.predict(X).flatten()
mse = mean_squared_error(Y, Y_HAT)
SSE = mse * len(Y)

# print("SSE:", SSE / len(days_data))

r2 = r2_score(Y, Y_HAT)
# print("R^2 Score:", r2)










# plt.show()
##########################################################################
def show_in_chart():
    fig = plt.figure(figsize=(14, 5))
    fig1 = fig.add_subplot(1, 2, 1)
    fig2 = fig.add_subplot(1, 2, 2)
    
    fig1.hist(days_data['min_temperature'], bins=80)
    fig1.set_title('min_temperature histo')
    fig1.set_xlabel('min_temperature')
    fig1.set_ylabel('freque')
    
    fig2.hist(days_data['max_temperature'], bins=100)
    fig2.set_title('max_temperature histo')
    fig2.set_xlabel('max_temperature')
    fig2.set_ylabel('freque')
    
    # plt.plot(times, days_data['max_temperature'], label='LSTM')
    # plt.xlabel('Day')
    # plt.ylabel('Sales')
    # plt.title('Daily Sale')

    # plt.subplot(1, 2, 2)
    # plt.plot(times, sales, label='Ice Cream Sales', color='r')
    # plt.xlabel('Day of the Year')
    # plt.ylabel('Ice Cream Sales')
    # plt.title('Ice Cream Sales vs. Day of the Year')

    plt.tight_layout()
    plt.show()

# show_in_chart()
