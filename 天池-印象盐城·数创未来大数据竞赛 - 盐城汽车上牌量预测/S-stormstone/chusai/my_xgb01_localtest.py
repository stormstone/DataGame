# -*- coding: utf-8 -*-
# @Time    : 2018-01-20 16:35
# @Author  : Storm
# @File    : my_xgb01_localtest.py

'''
xgboost:
export PYTHONPATH=/stone/pythonPackage/xgboost/python-package
'''

import pandas as pd
import numpy as np
import time
import xgboost as xgb
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 读取数据
print('1.读取数据中...')
data_testA_result = pd.read_table("data/answer_A_20180225.txt", header=0, encoding='gb2312', delim_whitespace=True)

# data_path = 'result/'
# train = pd.read_csv(data_path + 'dataP_train03_poly.csv')
# test = pd.read_csv(data_path + 'dataP_test03_poly.csv')


data_path = 'resultAend/'
train = pd.read_csv(data_path + 'getdata_train_poly.csv')#没加testA的答案
test = pd.read_csv(data_path + 'getdata_testA_poly.csv')

# 构造训练集、标签、测试集
feature_columns = [x for x in train.columns if x not in ['cnt'] and train[x].dtype != object]
X_train, y = train[feature_columns], train['cnt']
X_test = test[feature_columns]

# 计算特征权重
corr = {}
for f in X_train.columns:
    data = X_train[f]
    corr[f] = pearsonr(data.values, y.values)[0]
feature = []
feature_v = []
for k, v in corr.items():
    if abs(v) >= 0:
        feature.append(k)
        feature_v.append(v)

# 保存特征和权重
feature_all = pd.DataFrame()
feature_all['特征'] = feature
feature_all['权重'] = feature_v
# 按权重排序

feature_all_sorted = feature_all.sort_values(by=['权重'], ascending=False)
# feature_all_sorted.plot(x='特征', y='权重', kind='bar')
# plt.show()
feature_all_sorted.to_csv('feature/feature_xgb_20180225.csv',
                          index=None,
                          encoding='utf-8')
''''''

# 保留权重大于0.1的特征
X_train = train[feature_columns]
X_test = test[feature_columns]

split_num = 750
trainX_split = X_train.iloc[:split_num, :]
testX_split = X_train.iloc[split_num:, :]
trainY_split = y[:split_num]
testY_split = y[split_num:]

# 构造XGBRegressor模型

'''
params = {'booster': 'gbtree',
          'objective': 'reg:linear',
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
          # 'tree_method': 'gpu_exact',
          # 'gpu_id': 0,
          'seed': 50,
          # 'scale_pos_weight':10,
          'nthread': -1
          }

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
    objective='reg:linear',
    nthread=-1,
    scale_pos_weight=1,
    silent=True,
    # tree_method='gpu_exact',
    # gpu_id=0,
    seed=0)

result = xgb.cv(params, xgb.DMatrix(X_train, label=y), num_boost_round=1000, nfold=8, stratified=False, folds=None,
                maximize=False, early_stopping_rounds=10, as_pandas=True, verbose_eval=None, show_stdv=True,
                seed=0, callbacks=None, shuffle=True)
xgb1.set_params(n_estimators=result.shape[0])

print('2.开始训练...')
xgb1.fit(trainX_split, trainY_split)

print('3.开始预测...')
y_predict = xgb1.predict(testX_split)

print('4.评测...')
# 对训练集本身评价
compare2 = pd.DataFrame()
compare2['true'] = testY_split
compare2['pred'] = y_predict
f2 = (1 / (len(compare2))) * sum((compare2['pred'] - compare2['true']) * (compare2['pred'] - compare2['true']))
print('评价结果：', f2)

# 预测
xgb1.fit(X_train, y)
y_predict_test = xgb1.predict(X_test)

# 画图
print('==============画图============')
# plt.plot(X_train.date, y, 'b')
plt.plot(testX_split.date, testY_split, 'b')
plt.title('All Count Law')
plt.xlabel('Date')
plt.ylabel('Cnt')
plt.plot(testX_split.date, y_predict, 'r')
plt.plot(X_test.date, y_predict_test, 'g')
plt.show()

'''
# model = xgb.XGBRegressor(n_estimators=10000, learning_rate=0.08, gamma=0, subsample=0.7,
#                          colsample_bytree=0.7, max_depth=8)
model = xgb.XGBRegressor(max_depth=10,
                         learning_rate=0.01,
                         n_estimators=10000,
                         silent=True,
                         objective='reg:linear',
                         nthread=-1,
                         gamma=0,
                         min_child_weight=1,
                         max_delta_step=0,
                         subsample=0.85,
                         colsample_bytree=0.7,
                         colsample_bylevel=1,
                         reg_alpha=0,
                         reg_lambda=1,
                         scale_pos_weight=1,
                         seed=1440,
                         missing=None,
                         )

print('2.开始训练...')
model.fit(trainX_split, trainY_split)

print('3.开始预测...')
y_predict = model.predict(testX_split)

print('4.评测...')
# 对训练集本身评价
compare2 = pd.DataFrame()
compare2['true'] = testY_split
compare2['pred'] = y_predict
f2 = (1 / (len(compare2))) * sum((compare2['pred'] - compare2['true']) * (compare2['pred'] - compare2['true']))
print('评价结果：', f2)

# 预测
model.fit(X_train, y)
y_predict_test = model.predict(X_test)
print('线下得分 resultA：    {}'.format(mean_squared_error(data_testA_result['cnt'], y_predict_test)))


# 画图
print('==============画图============')
# plt.plot(X_train.date, y, 'b')
plt.plot(testX_split.date, testY_split, 'b')
plt.title('All Count Law')
plt.xlabel('Date')
plt.ylabel('Cnt')
plt.plot(testX_split.date, y_predict, 'r')
plt.plot(X_test.date, y_predict_test, 'g')
plt.show()

