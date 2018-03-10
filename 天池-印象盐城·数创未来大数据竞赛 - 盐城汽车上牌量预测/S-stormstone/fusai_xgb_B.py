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

'''
train = pd.read_csv(data_path + 'getdata_train_poly_addA.csv')
testA = pd.read_csv(data_path + 'getdata_testA_poly.csv')
testB = pd.read_csv(data_path + 'getdata_testB_poly.csv')
'''
train = pd.read_csv(data_path + 'getdata_train_ok_addA_adjust.csv')
testA = pd.read_csv(data_path + 'getdata_testA_ok.csv')
testB = pd.read_csv(data_path + 'getdata_testB_ok.csv')
train = train.drop(['r_date'], axis=1)
testA = testA.drop(['r_date'], axis=1)
testB = testB.drop(['r_date'], axis=1)

r_testA = pd.read_table('data_fusai/fusai_answer_a_20180307.txt')

# 构造训练集、标签、测试集
feature_columns = [x for x in train.columns if x not in ['cnt'] and train[x].dtype != object]
X_train, y = train[feature_columns], train['cnt']
X_test = testB[feature_columns]

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
feature_all_sorted.to_csv('fusai_feature/feature_xgb_b.csv',
                          index=None,
                          encoding='utf-8')
''''''

# 保留权重大于0.1的特征
X_train = train[feature_columns]
X_test = testB[feature_columns]

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
model.fit(X_train, y)

print('3.开始预测...')
print('testA开始预测...')
testA_predict = model.predict(testA[feature_columns])

print('testB开始预测...')
testB_predict = model.predict(X_test)

# 处理小于0的情况
''''''
for i in range(len(testA_predict)):
    if testA_predict[i] < 0:
        # testA_predict[i] = 0 - testA_predict[i]
        testA_predict[i] = 0
for i in range(len(testB_predict)):
    if testB_predict[i] < 0:
        # testB_predict[i] = 0 - testB_predict[i]
        testB_predict[i] = 0

print('4.评测...')
print('resultA：    {}'.format(mean_squared_error(r_testA.cnt, testA_predict)))

now_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
now_day = datetime.datetime.now().strftime('%Y%m%d')

# 画图
print('==============画图============')
plt.plot(X_train.date, y, 'bo')
plt.title('All Count Law')
plt.xlabel('Date')
plt.ylabel('Cnt')
plt.plot(testA.date, testA.cnt, 'bo')
plt.plot(testA.date, testA_predict, 'ro')
plt.plot(X_test.date, testB_predict, 'mo')
plt.savefig('fusai_result/' + now_day + '/xgb/sub_Xgb{}.png'.format(now_day))
plt.show()

# submission
submission = pd.DataFrame({'date': X_test.date, 'brand': X_test.brand, 'pred': testB_predict})
columns = ['date', 'brand', 'pred']  # 列名顺序

# 向下移动
for i in range(10):
    brand = submission.loc[submission['brand'] == i + 1]
    min1 = min(brand.pred)
    print(min1)
    brand.pred = brand.pred - min1 + 10
    submission.loc[submission['brand'] == i + 1] = brand
    brand2 = submission.loc[submission['brand'] == i + 1]
    min2 = min(brand2.pred)
    print(min2)

submission.to_csv(r'fusai_result/' + now_day + '/xgb/sub_Xgb{}.csv'.format(now_day), header=None,
                  index=False, columns=columns, float_format='%.0f')
submission.to_csv(r'fusai_result/' + now_day + '/xgb/sub_Xgb{}.txt'.format(now_day), header=False, sep='\t',
                  index=False, columns=columns, float_format='%.0f')

for i in range(10):
    train_brand = train.loc[train['brand'] == i + 1]
    testB_brand = submission.loc[submission['brand'] == i + 1]
    testA_brand_answer = r_testA.loc[r_testA['brand'] == i + 1]
    plt.title('brand-%d' % (i + 1))
    plt.xlabel('Date')
    plt.ylabel('Cnt')
    plt.plot(train_brand.date, train_brand.cnt, 'bo')
    plt.plot(testA_brand_answer.date, testA_brand_answer.cnt, 'go')
    plt.plot(testB_brand.date, testB_brand.pred, 'ro')
    plt.savefig('fusai_result/' + now_day + '/xgb/brand{}.png'.format(i + 1))
    plt.show()
    plt.title('brand-%d' % (i + 1))
    plt.xlabel('Date')
    plt.ylabel('Cnt')
    plt.plot(train_brand.date, train_brand.cnt, 'b')
    plt.plot(testA_brand_answer.date, testA_brand_answer.cnt, 'g')
    plt.plot(testB_brand.date, testB_brand.pred, 'r')
    plt.savefig('fusai_result/' + now_day + '/xgb/brand{}_l.png'.format(i + 1))
    plt.show()
'''
'''
