# -*- coding: utf-8 -*-
# @Time    : 18-2-24 上午11:28
# @Author  : Storm
# @File    : my_o03.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Activation

'''
train = pd.read_table('./data/train_20171215.txt')
test_A = pd.read_table('./data/test_A_20171225.txt')
sample_A = pd.read_table('./data/sample_A_20171225.txt', header=None)
nowtime = datetime.date(2015, 3, 4)
# print(starttime+datetime.timedelta(days=32))
timelist = []
l = np.shape(train)[0] - 1
i = 0
while i < l:
    d = [nowtime, train.loc[i, 'date']]
    timelist.append(d)
    while i < l and train.loc[i, 'day_of_week'] <= train.loc[i + 1, 'day_of_week']:
        if train.loc[i, 'day_of_week'] < train.loc[i + 1, 'day_of_week']:
            nowtime = nowtime + datetime.timedelta(days=1)
            d = [nowtime, train.loc[i + 1, 'date']]
            timelist.append(d)
        i = i + 1
    md = train.loc[i, 'day_of_week']
    while md < 7:
        nowtime = nowtime + datetime.timedelta(days=1)
        d = [nowtime, np.nan]
        timelist.append(d)
        md = md + 1
    i = i + 1
    if i >= l:
        break
    md = train.loc[i, 'day_of_week']
    k = 1
    while k < md:
        nowtime = nowtime + datetime.timedelta(days=1)
        d = [nowtime, np.nan]
        timelist.append(d)
        k = k + 1
    nowtime = nowtime + datetime.timedelta(days=1)
timelist = timelist[:-3]
# print(timelist)
r_timelist = []
for t in timelist:
    for br in [1, 2, 3, 4, 5]:
        temp = t.copy()
        temp.append(br)
        r_timelist.append(temp)
data = pd.DataFrame(r_timelist)
data.columns = ['r_date', 'date', 'brand']

train = pd.merge(data, train, how='right', on=['date', 'brand'])
data = pd.merge(data, train, how='left', on=['r_date', 'brand', 'date'])
week = data.groupby('r_date').day_of_week.median()
weeks = pd.DataFrame()
weeks['r_date'] = week.index.tolist()
weeks['day_of_week'] = week.values
data.drop('day_of_week', axis=1, inplace=True)
data = pd.merge(data, weeks, how='left', on=['r_date'])
for i in range(len(data)):
    if (~(data.loc[i, 'day_of_week'] > 0)):
        if (data.loc[i, 'r_date'] == data.loc[i - 1, 'r_date']):
            data.loc[i, 'day_of_week'] = data.loc[i - 1, 'day_of_week']
        else:
            if (data.loc[i - 1, 'day_of_week'] == 7):
                data.loc[i, 'day_of_week'] = 1
            else:
                data.loc[i, 'day_of_week'] = data.loc[i - 1, 'day_of_week'] + 1


def dorelax_day_mean(a):
    tmp = a.copy()
    flag = tmp[0] % 10
    d = pd.DataFrame(tmp)
    d.columns = ['cnt']
    d['is_relax_day'] = d['cnt'] % 10
    d['cnt'] = (d['cnt'] / 10).astype(int)
    d['cnt'].replace(0, np.nan, inplace=True)
    return np.mean(d.loc[d['is_relax_day'] == flag, 'cnt'])


def roll_day(data, day):
    return data['cnt'].rolling(window=day, min_periods=1).mean()


def roll_dorelax_day(data, day):
    result = pd.rolling_apply(data['dorelax_day_mean'], window=day, func=dorelax_day_mean, min_periods=1)
    # data.loc[data['is_relax_day'] == 0, 'dorelax_day_mean'] = pd.rolling_apply(data.loc[data['is_relax_day'] == 0, 'cnt'], window=day, func=dorelax_day_mean, min_periods=1)
    return result


relax_day = [6, 7]
data['is_relax_day'] = 0
data.loc[data.day_of_week.isin(relax_day), 'is_relax_day'] = 1
data.loc[~data.day_of_week.isin(relax_day), 'is_relax_day'] = 0
data['dorelax_day_mean'] = data['cnt'].fillna(0) * 10 + data['is_relax_day']
for br in [1, 2, 3, 4, 5]:
    data.loc[data['brand'] == br, 'weekmean'] = roll_day(data[data['brand'] == br], 7)
    data.loc[data['brand'] == br, 'monthmean'] = roll_day(data[data['brand'] == br], 30)
    data.loc[data['brand'] == br, 'dorelax_day_mean'] = roll_dorelax_day(data[data['brand'] == br], 30)
    # data.loc[data['brand'] == br,'brand_mean'] = data.loc[data['brand'] == br,'cnt'].mean()
    # for w in [1, 2, 3, 4, 5, 6, 7]:
    #     data.loc[(data['day_of_week'] == w)&(data['brand'] == br), 'week_day_brand_mean'] = data.loc[(data['day_of_week'] == w)&(data['brand'] == br), 'cnt'].mean()
# for w in [1,2,3,4,5,6,7]:
#     data.loc[data['day_of_week'] == w, 'week_day_mean'] = data.loc[data['day_of_week'] == br, 'cnt'].mean()

# for br in [1,2,3,4,5]
#     data.loc[data['brand'] == br, 'week_relax_mean'] = roll_dorelax_day(data[data['brand'] == br], 7)
#     data.loc[data['brand'] == br, 'month_relax_mean'] = roll_dorelax_day(data[data['brand'] == br], 30)
data['month'] = data['r_date'].apply(lambda x: x.month)
data['year'] = data['r_date'].apply(lambda x: x.year)

day_of_weeks = pd.get_dummies(data.day_of_week)
day_of_weeks.columns = ['day_of_week' + str(i + 1) for i in range(day_of_weeks.shape[1])]
data = pd.concat([data, day_of_weeks], axis=1)

months = pd.get_dummies(data.month)
months.columns = ['month' + str(i + 1) for i in range(months.shape[1])]
data = pd.concat([data, months], axis=1)

years = pd.get_dummies(data.year)
years.columns = ['year' + str(i + 1) for i in range(years.shape[1])]
data = pd.concat([data, years], axis=1)
brands = pd.get_dummies(data.brand)
brands.columns = ['brand' + str(i + 1) for i in range(brands.shape[1])]
data = pd.concat([data, brands], axis=1)
# data.drop(['month','year','day_of_week','brand'],inplace=True,axis=1)
data['r_date'] = pd.to_datetime(data['r_date'])
print(data.info())
data.to_csv('result/dataP_o03_data.csv', index=None)

# ================================================================================

import xgboost as xgb
from xgboost.sklearn import XGBRegressor

data = pd.read_csv('result/dataP_o03_data.csv')
data.drop('date', inplace=True, axis=1)
train = data[data['cnt'] > 0]
low = train[train['cnt'] < 80]
print(len(low))
print(np.mean(train['cnt']))
for i in range(5):
    train = pd.concat([low.copy(), train], axis=0)
y = train['cnt']
miny = min(y)
maxy = max(y)
y = (y - miny) / (maxy - miny)
X = train.drop(['cnt', 'r_date'], axis=1)
print(np.shape(X))
test = data[~(data['cnt'] > 0)]
test = test.drop(['cnt', 'r_date'], axis=1)
print(np.shape(test))
params = {'booster': 'gbtree',
          'objective': 'reg:logistic',
          'eval_metric': 'rmse',
          'gamma': 0.1,
          'min_child_weight': 2,
          'max_depth': 5,
          'lambda': 10,
          'alpha': 2.5,
          'silent': True,
          'subsample ': 0.8,
          'colsample_bytree': 0.5,
          'colsample_bylevel': 0.5,
          'eta': 0.01,
          'tree_method': 'gpu_exact',
          'seed': 50,
          'gpu_id': 0,
          # 'scale_pos_weight':10,
          'nthread': -1
          }


def loss(preds, dtrain):  # preds是结果（概率值），dtrain是个带label的DMatrix
    labels = dtrain.get_label()  # 提取label
    cha = (preds - labels) * (maxy - miny)
    trainmse = np.dot(cha, cha.T) / len(cha)
    return 'mse', trainmse


xgb1 = XGBRegressor(
    learning_rate=0.01,
    n_estimators=1000,
    max_depth=5,
    min_child_weight=2,
    gamma=0.1,
    reg_alpha=2.5,
    reg_lambda=5,
    subsample=0.8,
    colsample_bytree=0.5,
    objective='reg:logistic',
    nthread=-1,
    scale_pos_weight=1,
    silent=True,
    tree_method='gpu_exact',
    gpu_id=0,
    seed=0)
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=0)
result = xgb.cv(params, xgb.DMatrix(X, label=y), num_boost_round=1000, nfold=8, stratified=False, folds=None,
                maximize=False, early_stopping_rounds=10, as_pandas=True, verbose_eval=None, show_stdv=True,
                seed=0, callbacks=None, shuffle=True, feval=loss)
xgb1.set_params(n_estimators=result.shape[0])
xgb1.fit(X_train, Y_train)

preds = xgb1.predict(X_train) * (maxy - miny) + miny
# preds=preds+(preds-np.mean(preds))*0.5
cha = (preds - Y_train * (maxy - miny) - miny)
print('train', np.dot(cha, cha.T) / len(cha))

preds = xgb1.predict(X_test) * (maxy - miny) + miny
# preds=preds+(preds-np.mean(preds))*0.5
cha = (preds - Y_test * (maxy - miny) - miny)
print('test', np.dot(cha, cha.T) / len(cha))
print(np.min(preds), np.max(preds), np.mean(preds))

xgb1.fit(X, y)
preds = xgb1.predict(test) * (maxy - miny) + miny
data = pd.read_csv('result/dataP_o03_data.csv')
data.loc[~(data['cnt'] > 0), 'cnt'] = preds
print(data.info())
data.to_csv('result/dataP_o03_finaldata.csv', index=None)
print(np.mean(train['cnt']), np.mean(preds))
'''


# ======================================================================================
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def parseTest(values):
    weekday = 4
    result = []
    for i in range(len(values)):
        while weekday != values[i, 1]:
            result.append([values[i, 0] - 0.5, weekday])
            weekday = weekday + 1
            if weekday > 7:
                weekday = 1
        result.append([values[i, 0], weekday])
        weekday = weekday + 1
        if weekday > 7:
            weekday = 1
    print(result)
    return result


dataset = pd.read_csv('result/dataP_o03_finaldata.csv', header=0)
dataset = dataset[['r_date', 'date', 'cnt', 'day_of_week']]
for i in range(len(dataset)):
    if (dataset.loc[i, 'date'] > 0) == False:
        dataset.loc[i, 'date'] = dataset.loc[i - 1, 'date']
dataset = dataset.groupby(['r_date', 'date', 'day_of_week']).agg('sum').reset_index()
dataset.columns = ['r_date', 'date', 'day_of_week', 'cnt']
dataset = dataset[['date', 'day_of_week', 'cnt']]
dataset['cnt'] = np.log1p(dataset['cnt'])
t_y = dataset['cnt']
print(dataset.info())
values = dataset.values.astype('float32')
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(dataset.values)
reframed = series_to_supervised(scaled, 7, 1)
# reframed.drop(reframed.columns[[2,5,8,11,14,17,20,21]], axis=1, inplace=True)
values = reframed.values
# values=values[:-1,:]
trainsize = len(values)
test_A = pd.read_table('./data/test_A_20171225.txt')
test_A = np.array(parseTest(test_A.values))
test_A = test_A[1:, :]
# test_A=test_A.values
test_A_back = test_A
sample_A = pd.read_table('./data/sample_A_20171225.txt', header=None)
train = values
test = test_A
train_X, train_y = train[:, :-1], train[:, -1]
print(np.shape(test))
test = np.concatenate((test, np.zeros((np.shape(test)[0], 1))), axis=1)
test = scaler.transform(test)[:, :-1]
test_X = test

train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape)

from numpy.random import seed

seed(2)
from tensorflow import set_random_seed

set_random_seed(2)

model = Sequential()
model.add(LSTM(32, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))
model.add(Dropout(0.1))
model.add(LSTM(64, return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(32))
model.add(Dense(1))
model.add(Activation("tanh"))
model.compile(loss='mse', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=200, batch_size=100, verbose=2, shuffle=False)
# make a prediction
y = []
X = train_X[-1, :, :].reshape(1, train_X.shape[2])
X = np.concatenate((X, train_y[-1].reshape(1, 1)), axis=1)
for i in range(len(test)):
    X = np.concatenate((X[:, 3:], test[i].reshape(1, 2)), axis=1).reshape((1, 1, train_X.shape[2]))
    y.append(model.predict(X)[0, 0])
    X = np.concatenate((X.reshape(1, train_X.shape[2]), (y[-1]).reshape(1, 1)), axis=1)
print(y)
# yhat = model.predict(test_X)
# print(np.shape(yhat))
# test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# # invert scaling for forecast
inv_yhat = np.concatenate((test[:, 0:2], np.array(y).reshape(test.shape[0], 1)), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, -1]
inv_yhat = np.expm1(inv_yhat)
print(inv_yhat)
test_A_back = pd.DataFrame(test_A_back)
test_A_back.columns = ['date', 'value']
test_A_back['value'] = inv_yhat
# test_A_back.to_csv('result.csv', index=None)

test_A_back = pd.DataFrame(test_A_back)
test_A_back.columns = ['date', 'day_of_week']
test_A_back['value'] = inv_yhat
test_A_back = test_A_back[['date', 'value']]
sample_A.columns = ['date', 'value']
sample_A = sample_A[['date']]
sample_A = pd.merge(sample_A, test_A_back, on='date', how='inner')

plt.plot(inv_yhat, label='preds')
plt.legend()
plt.show()

now_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
sample_A.to_csv(r'result/sub_o03{}.csv'.format(now_time), header=None,
                index=False, float_format='%.0f')
sample_A.to_csv(r'result/sub_o03{}.txt'.format(now_time), header=False, sep='\t',
                index=False, float_format='%.0f')
