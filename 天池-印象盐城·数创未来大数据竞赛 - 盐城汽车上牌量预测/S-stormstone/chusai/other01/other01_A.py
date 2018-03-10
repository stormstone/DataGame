# -*- coding: utf-8 -*-
# @Time    : 2018-01-18 18:34
# @Author  : Storm
# @File    : other01_A.py

import pandas as pd
import matplotlib.pyplot as plt

dir = '../data/'
train = pd.read_table(dir + 'train_20171215.txt', engine='python')
test_A = pd.read_table(dir + 'test_A_20171225.txt', engine='python')
sample_A = pd.read_table(dir + 'sample_A_20171225.txt', engine='python', header=None)
sample_A.columns = ['date', 'day_of_week']

# 因为第一赛季只是预测与时间相关的cnt的数量
# 所以可以对数据以dat和dow进行数据合并
train = train.groupby(['date', 'day_of_week'], as_index=False).cnt.sum()
plt.plot(train['day_of_week'], train['cnt'], '*')
plt.savefig('../pic/星期几-分布图.png')
plt.show()

for i in range(7):
    tmp = train[train['day_of_week'] == i + 1]
    plt.subplot(7, 1, i + 1)
    plt.plot(tmp['date'], tmp['cnt'], '*')
plt.savefig('../pic/总量-星期几-趋势.png')
plt.show()

# 划分训练集测试集
xx_train = train[train['date'] <= 756]
xx_test = train[train['date'] > 756]
print('test shape', xx_test.shape)
print('train shape', xx_train.shape)
# test shape (276, 3)
# train shape (756, 3)

from sklearn.metrics import mean_squared_error

# 线下统计每周的均值数据，不加权
xx_train = xx_train.groupby(['day_of_week'], as_index=False).cnt.mean()
xx_result = pd.merge(xx_test, xx_train, on=['day_of_week'], how='left')
print('xx_result shape', xx_result.shape)
# xx_result shape (276, 4)
print(xx_result)
'''
date  day_of_week  cnt_x        cnt_y
757            6    314   387.715517
758            1   3309  2239.470085
759            2   1948  2372.838983
760            3   1722  2011.336134
761            4   1520  1592.033333
'''
print(mean_squared_error(xx_result['cnt_x'], xx_result['cnt_y']))
# 909407.288275

for i in range(7):
    tmp = xx_result[xx_result['day_of_week'] == i + 1]
    print('周%d' % (i + 1), mean_squared_error(tmp['cnt_x'], tmp['cnt_y']))
'''
周1 1256145.85465
周2 1462515.18676
周3 794863.108054
周4 652400.4243
周5 874054.284747
周6 172515.457498
周7 2318220.65784
'''

def xx(df):
    df['w_cnt'] = (df['cnt'] * df['weight']).sum() / sum(df['weight'])
    return df


xx_train = train[train['date'] <= 756]
xx_train['weight'] = ((xx_train['date'] + 1) / len(xx_train)) ** 6
xx_train = xx_train.groupby(['day_of_week'], as_index=False).apply(xx).reset_index()
xx_test = train[train['date'] > 756]
print('test shape', xx_test.shape)
print('train shape', xx_train.shape)
# test shape (276, 3)
# train shape (756, 6)

# #
from sklearn.metrics import mean_squared_error

# # 这里是加权的方案
xx_train = xx_train.groupby(['day_of_week'], as_index=False).w_cnt.mean()

xx_result = pd.merge(xx_test, xx_train, on=['day_of_week'], how='left')
print('xx_result shape', xx_result.shape)
# xx_result shape (276, 4)
print(xx_result)
'''
     date  day_of_week   cnt        w_cnt
0     757            6   314   419.121951
1     758            1  3309  2593.503011
2     759            2  1948  2615.940149
3     760            3  1722  2285.466506
4     761            4  1520  1839.909973
'''
print(mean_squared_error(xx_result['cnt'], xx_result['w_cnt']))
# 828419.30779

from pandas import DataFrame
from pandas import concat


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
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
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


time_cnt = list(train['cnt'].values)
# nin 前看 nout后看 这个题目需要前看
time2sup = series_to_supervised(data=time_cnt, n_in=276, dropnan=True)

import lightgbm as lgb

gbm0 = lgb.LGBMRegressor(
    objective='regression',
    num_leaves=64,
    learning_rate=0.05,
    n_estimators=10000)

print(time2sup.shape)
# (756, 277)
x_train = time2sup[time2sup.index < 755]
x_test = time2sup[time2sup.index > 755]
# 这个方式其实是最简单的，后面还可以很多改善，比如滚动预测一类
print(x_train.shape)
# (479, 277)
print(x_test.shape)
# (276, 277)

y_train = x_train.pop('var1(t)')
y_test = x_test.pop('var1(t)')

# 损失函数mse
gbm0.fit(x_train.values, y_train, eval_set=[(x_test.values, y_test)], eval_metric='mse', early_stopping_rounds=15)
print(gbm0.predict(x_test.values))

from sklearn.metrics import mean_squared_error

line1 = plt.plot(range(len(x_test)), gbm0.predict(x_test.values), label=u'predict')
line2 = plt.plot(range(len(y_test)), y_test.values, label=u'true')
plt.legend()
plt.show()
