# -*- coding: utf-8 -*-
# @Time    : 18-2-22 下午12:02
# @Author  : Storm
# @File    : my_o05.py


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import datetime
from datetime import date, timedelta
from dateutil.relativedelta import *
# machine lerning
from sklearn.model_selection import train_test_split

# test time stationary
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose

from statsmodels.tsa.stattools import acf, pacf


# 按星期分组统计
def drop_brand(df):
    cols = list(df.columns)
    if 'brand' in cols:
        df = df[["date", 'day_of_week', "cnt"]].groupby(['date', 'day_of_week'], as_index=False).sum()
    return df


def how_many_weeks(df):
    df = drop_brand(df)
    week_cnt = 0
    weekdays = []
    weekday = 0
    prev = -1
    for idx, row in df.iterrows():
        weekday += 1
        if row['day_of_week'] <= prev:
            weekdays.append(weekday)
            weekday = 0
            week_cnt += 1
        prev = row['day_of_week']
    weekdays = np.asarray(weekdays)
    return week_cnt, weekdays


# 将脱敏的日期变为正常日期
# 加入周数ID 与 月数ID
def add_actual_date(df, start_date=date(2003, 1, 1), week_cnt=1, month_cnt=1):
    df = drop_brand(df)
    df["real_date"] = start_date
    add_days = -1
    prev_week_days = 0
    prev = df.loc[0, "day_of_week"] - 1.0
    prev_month = month_cnt
    for idx, row in df.iterrows():

        if row['day_of_week'] <= prev:
            add_days += row['day_of_week'] + 7 - prev
            week_cnt += 1
        else:
            add_days += (row['day_of_week'] - prev)
        df.loc[idx, "real_date"] += timedelta(days=add_days)
        df.loc[idx, "week"] = week_cnt
        prev = row['day_of_week']
        if df.loc[idx, "real_date"].month != prev_month:
            month_cnt += 1
        df.loc[idx, "month"] = month_cnt
        prev_month = df.loc[idx, "real_date"].month

    df["real_date"] = df["real_date"].astype("datetime64[ns]")
    return df


# 给Weekly数据的
# 返回一行按周的日期
def add_week_date(df, start_date=date(2003, 1, 1)):
    df = df.copy()

    if "cnt" not in df.columns:
        df["cnt"] = 0
    df["weekdays"] = df[["cnt", "week"]].groupby(['week']).transform("count")
    df["cnt_ave_week"] = df[["cnt", "week"]].groupby(['week']).transform("mean")
    df = df.drop_duplicates(subset=['week'], keep='first')

    add_days = 0
    # df["week_date"]=0
    for idx, row in df.iterrows():
        df.loc[idx, "week_date"] = start_date + timedelta(days=add_days)
        add_days += 7
    df["week_date"] = df["week_date"].astype("datetime64[ns]")
    df = df.set_index('week_date')
    df = df[["cnt_ave_week"]]
    return df


# 给Monthly数据的
# 返回一行按月的日期
def add_month_date(df, start_date=date(2003, 1, 1)):
    df = df.copy()

    if "cnt" not in df.columns:
        df["cnt"] = 0
    df["monthdays"] = df[["cnt", "month"]].groupby(['month']).transform("count")
    df["cnt_ave_month"] = df[["cnt", "month"]].groupby(['month']).transform("mean")
    df = df.drop_duplicates(subset=['month'], keep='first')

    add_days = 0
    # df["week_date"]=0
    for idx, row in df.iterrows():
        df.loc[idx, "month_date"] = start_date
        start_date += relativedelta(months=1)
    df["month_date"] = df["month_date"].astype("datetime64[ns]")
    df = df.set_index('month_date')
    df = df[["cnt_ave_month"]]
    # df = df.loc[:,['week_date','cnt_ave_week']]
    return df


# 按星期 将dataframe分为7个
def split_by_day(df):
    week_day_list = []
    for i in range(1, 8):
        week_day_list.append(df.loc[df["day_of_week"] == i, :].reset_index(drop=True))

    return week_day_list


# time series helper function

def test_stationarity(timeseries):
    # Determing rolling statistics
    rolmean = timeseries.rolling(window=12, center=False).mean()
    rolstd = timeseries.rolling(window=12, center=False).std()

    # plt.figure(figsize=(20,10))
    # Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)

    # Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)


def plot_acf_pacf(ts):
    lag_acf = acf(ts, nlags=20)
    lag_pacf = pacf(ts, nlags=20, method='ols')

    plt.subplot(121)
    plt.plot(lag_acf)
    plt.axhline(y=0, linestyle='--', color='gray')
    plt.axhline(y=-1.96 / np.sqrt(len(ts)), linestyle='--', color='gray')
    plt.axhline(y=1.96 / np.sqrt(len(ts)), linestyle='--', color='gray')
    plt.title('Autocorrelation Function')

    # Plot PACF:
    plt.subplot(122)
    plt.plot(lag_pacf)
    plt.axhline(y=0, linestyle='--', color='gray')
    plt.axhline(y=-1.96 / np.sqrt(len(ts)), linestyle='--', color='gray')
    plt.axhline(y=1.96 / np.sqrt(len(ts)), linestyle='--', color='gray')
    plt.title('Partial Autocorrelation Function')
    plt.tight_layout()
    plt.show()


# ==============================================================================
# 读数据
data_train = pd.read_table('data/train_20171215.txt')
data_test_a = pd.read_table('data/test_A_20171225.txt')
data_test_b = pd.read_table('data/test_B_20171225.txt')

# 为数据加入日期
train_by_date = add_actual_date(data_train)
test_a_by_date = add_actual_date(data_test_a,
                                 start_date=train_by_date.real_date.max(),
                                 week_cnt=train_by_date.week.max(),
                                 month_cnt=train_by_date.month.max())
test_b_by_date = add_actual_date(data_test_b,
                                 start_date=test_a_by_date.real_date.max() + timedelta(days=1),
                                 week_cnt=test_a_by_date.week.max(),
                                 month_cnt=test_a_by_date.month.max())

# 按周一 到周日的 7个 dataframe.
data_train_day_lists = split_by_day(train_by_date)

# 给周的数据加时间
data_train_weekly = add_week_date(train_by_date)
test_a_weekly = add_week_date(test_a_by_date, start_date=data_train_weekly.index.max())

# 给月加数据
data_train_monthly = add_month_date(train_by_date)

'''
# 画图
plt.plot(data_train_monthly['cnt_ave_month'])
plt.title('trend-month')
plt.show()
plt.plot(data_train_weekly['cnt_ave_week'])
plt.title('trend-week')
plt.show()
'''

# ===========================================================================================
# 对周的数据进行forecast

# 测试stationary
ts_week = data_train_weekly["cnt_ave_week"]
ts_week_log = np.log(ts_week)

'''
# 画图
test_stationarity(ts_week)
test_stationarity(ts_week_log)

decomposition = seasonal_decompose(data_train_monthly, model='additive')
fig = decomposition.plot()
plt.show()

plot_acf_pacf(ts_week)
plot_acf_pacf(ts_week_log)
'''

model = ARIMA(ts_week, order=(1, 0, 1))
results_ARIMA = model.fit(disp=-1)
plt.plot(ts_week)
plt.plot(results_ARIMA.fittedvalues, color='red')
RSS = sum((results_ARIMA.fittedvalues - ts_week) ** 2)
MSE = RSS / len(ts_week)
plt.title('WEEK-RSS: %.4f,WEEK-MSE: %.4f' % (RSS, MSE))
plt.show()

'''
# 预测未来80周的数据
forecast_data = pd.DataFrame(results_ARIMA.forecast(steps=80)[0])
forecast_data.index = pd.date_range(ts_week.index.max(), periods=80, freq='W')
plt.plot(forecast_data)
plt.show()
model_season = SARIMAX(ts_week,
                       order=(1, 0, 1),
                       seasonal_order=(1, 0, 1, 52), enforce_invertibility=False)

results = model_season.fit(ts_week)
# 也是预测未来的80周
pred = results.get_prediction(start=170, end=250)
plt.plot(ts_week)
plt.plot(pred.predicted_mean)
plt.show()
'''

# =================================================================
# 对月的数据进行forecast

# 测试stationary
ts_month = data_train_monthly["cnt_ave_month"]
ts_month_log = np.log(ts_month)

'''
# 画图
test_stationarity(ts_month)
test_stationarity(ts_month_log)

plot_acf_pacf(ts_month)
'''

model = ARIMA(ts_month, order=(2, 0, 1))
results_ARIMA = model.fit(disp=-1)
plt.plot(ts_month)
plt.plot(results_ARIMA.fittedvalues, color='red')
RSS = sum((results_ARIMA.fittedvalues - ts_month) ** 2)
MSE = RSS / len(ts_week)
plt.title('MONTH-RSS: %.4f,MONTH-MSE: %.4f' % (RSS, MSE))
plt.show()

'''
forecast_data = pd.DataFrame(results_ARIMA.forecast(steps=30)[0])
forecast_data.index = pd.date_range(ts_month.index.max(), periods=30, freq='M')
plt.plot(forecast_data)

model_season = SARIMAX(ts_month,
                       order=(1, 0, 1),
                       seasonal_order=(1, 0, 1, 12), enforce_invertibility=False,
                       enforce_stationarity=False)

results = model_season.fit()

# 也是预测未来的20个月周
pred = results.get_prediction(start=12, end=60)
# pred.predicted_mean
plt.plot(ts_month)
plt.plot(pred.predicted_mean)
plt.show()
'''

model = ARIMA(ts_month, order=(1, 0, 1))
results_ARIMA = model.fit(disp=-1)
plt.plot(ts_month)
plt.plot(results_ARIMA.fittedvalues, color='red')
RSS = sum((results_ARIMA.fittedvalues - ts_month) ** 2)
MSE = RSS / len(ts_week)
plt.title('MONTH-RSS: %.4f,MONTH-MSE: %.4f' % (RSS, MSE))
plt.show()
# ==================================================================
'''
print(data_train_weekly.tail())
print(test_a_weekly.head())
data_train_weekly.info()
'''

# time series data
ts = data_train_weekly['cnt_ave_week']
ts_log = np.log(ts)

'''
test_stationarity(ts)
decomposition = seasonal_decompose(ts_log)
plt.plot(ts_log)
plt.show()
'''

model = ARIMA(ts_log, order=(1, 0, 1))
results_ARIMA = model.fit(disp=-1)
plt.plot(ts_log)
plt.plot(results_ARIMA.fittedvalues, color='red')
RSS = sum((results_ARIMA.fittedvalues - ts_log) ** 2)
MSE = RSS / len(ts_week)
plt.title('log WEEK-RSS: %.4f,WEEK-MSE: %.4f' % (RSS, MSE))
plt.show()

further = np.exp(results_ARIMA.forecast(steps=100)[0])
further = pd.Series(further, copy=True)
predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
# print(predictions_ARIMA_diff.head())

predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff
predictions_ARIMA_log = pd.Series(ts_log.ix[0], index=ts_log.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum, fill_value=0)
# print(predictions_ARIMA_log.head())
predictions_ARIMA = np.exp(predictions_ARIMA_diff)
# print(predictions_ARIMA.head(100))
# print(ts.head())
plt.plot(ts)
plt.plot(predictions_ARIMA)
plt.title('RMSE: %.4f' % (sum((predictions_ARIMA - ts) ** 2) / len(ts)))
plt.show()

# =======================================================================================

predict_test_a_weekly = pd.read_csv('result/predict_test_a_weekly.csv')
predict_test_a_weekly["week"] = predict_test_a_weekly["week"].astype("float64")
result_weekly_tbats = pd.merge(test_a_by_date, predict_test_a_weekly, on='week')
print(result_weekly_tbats.shape, test_a_by_date.shape, predict_test_a_weekly.shape)
# print(result_weekly_tbats)
# 周末上牌少
sum_by_day = data_train[["day_of_week", "cnt"]].groupby(['day_of_week'], as_index=False).median().sort_values(by='cnt',
                                                                                                              ascending=False)

mean_day = sum_by_day["cnt"].mean()
sum_by_day['ave'] = sum_by_day.transform(lambda row: row["cnt"] / mean_day, axis=1)

for idx, row in result_weekly_tbats.iterrows():
    result_weekly_tbats.loc[idx, "cnt"] = row["cnt"] * sum_by_day.loc[row["day_of_week"] - 1, "ave"]

result_weekly_tbats = result_weekly_tbats.drop(["day_of_week", "real_date", "week", "month"], axis=1)
result_weekly_tbats.cnt = result_weekly_tbats.cnt.astype("int64")
print(result_weekly_tbats.head(10))

# 画图
print('==============画图============')
plt.plot(train_by_date.date, train_by_date.cnt, 'b')
plt.plot(result_weekly_tbats.date, result_weekly_tbats.cnt, 'g')
now_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
plt.savefig('result/sub_o05{}_l.png'.format(now_time))
plt.show()
plt.plot(train_by_date.date, train_by_date.cnt, 'bo')
plt.plot(result_weekly_tbats.date, result_weekly_tbats.cnt, 'ro')

plt.savefig('result/sub_o05{}.png'.format(now_time))
result_weekly_tbats.to_csv('result/sub_o05{}.csv'.format(now_time), header=False, index=False)
result_weekly_tbats.to_csv('result/sub_o05{}.txt'.format(now_time), header=False, sep='\t',
                           index=False, float_format='%.0f')
''''''
