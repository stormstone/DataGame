# -*- coding: utf-8 -*-
# @Time    : 2018/2/27 22:57
# @Author  : 石头人m
# @File    : fusai_xgb
import datetime

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

data_path = 'getdata/'
train = pd.read_csv(data_path + 'getdata_train_ok.csv')
test = pd.read_csv(data_path + 'getdata_testA_ok.csv')

train = train.drop(['r_date'], axis=1)
test = test.drop(['r_date'], axis=1)

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
feature_all_sorted.to_csv('fusai_feature/feature_xgb.csv',
                          index=None,
                          encoding='utf-8')
''''''

# 保留权重大于0.1的特征
X_train = train[feature_columns]
X_test = test[feature_columns]

split_num = 8500
trainX_split = X_train.iloc[:split_num, :]
testX_split = X_train.iloc[split_num:, :]
trainY_split = y[:split_num]
testY_split = y[split_num:]

# 构造XGBRegressor模型
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
                         verbose=True
                         )

print('2.开始训练...')
model.fit(trainX_split, trainY_split)

print('3.开始预测...')
y_predict = model.predict(testX_split)

print('testA开始预测...')
testA_predict = model.predict(X_test)

# 处理小于0的情况
''''''
for i in range(len(y_predict)):
    if y_predict[i] < 0:
        # y_predict[i] = 0 - y_predict[i]
        y_predict[i] = 0
for i in range(len(testA_predict)):
    if testA_predict[i] < 0:
        # testA_predict[i] = 0 - testA_predict[i]
        testA_predict[i] = 0

print('4.评测...')
# 对训练集本身评价
compare2 = pd.DataFrame()
compare2['true'] = testY_split
compare2['pred'] = y_predict
f2 = (1 / (len(compare2))) * sum((compare2['pred'] - compare2['true']) * (compare2['pred'] - compare2['true']))
print('评价结果：', f2)

now_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
now_day = datetime.datetime.now().strftime('%Y%m%d')

# 画图
print('==============画图============')
plt.plot(X_train.date, y, 'bo')
plt.plot(testX_split.date, testY_split, 'bo')
plt.title('All Count Law')
plt.xlabel('Date')
plt.ylabel('Cnt')
plt.plot(testX_split.date, y_predict, 'ro')

plt.plot(X_test.date, testA_predict, 'go')
plt.savefig('fusai_result/' + now_day + '/sub_Xgb{}.png'.format(now_day))
plt.show()

# submission
submission = pd.DataFrame({'date': X_test.date, 'brand': X_test.brand, 'pred': testA_predict})
columns = ['date', 'brand', 'pred']  # 列名顺序
submission.to_csv(r'fusai_result/' + now_day + '/sub_Xgb{}.csv'.format(now_day), header=None,
                  index=False, columns=columns, float_format='%.0f')
submission.to_csv(r'fusai_result/' + now_day + '/sub_Xgb{}.txt'.format(now_day), header=False, sep='\t',
                  index=False, columns=columns, float_format='%.0f')

for i in range(10):
    train_brand = train.loc[train['brand'] == i + 1]
    testA_brand = submission.loc[submission['brand'] == i + 1]
    plt.title('brand-%d' % (i + 1))
    plt.xlabel('Date')
    plt.ylabel('Cnt')
    plt.plot(train_brand.date, train_brand.cnt, 'bo')
    plt.plot(testA_brand.date, testA_brand.pred, 'go')
    plt.savefig('fusai_result/' + now_day + '/brand{}.png'.format(i + 1))
    plt.show()
    plt.title('brand-%d' % (i + 1))
    plt.xlabel('Date')
    plt.ylabel('Cnt')
    plt.plot(train_brand.date, train_brand.cnt, 'b')
    plt.plot(testA_brand.date, testA_brand.pred, 'g')
    plt.savefig('fusai_result/' + now_day + '/brand{}_l.png'.format(i + 1))
    plt.show()
'''
'''
