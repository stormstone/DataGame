# -*- coding: utf-8 -*-
# @Time    : 18-3-5 下午3:39
# @Author  : Storm
# @File    : fusai_lgb+xgb.py


import time
import datetime
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

data_path = 'getdata/'
train = pd.read_csv(data_path + 'getdata_train_ok.csv')
test = pd.read_csv(data_path + 'getdata_testA_ok.csv')
train = train.drop(['r_date'], axis=1)
test = test.drop(['r_date'], axis=1)

train_feat = train
test_feat = test
predictors = [f for f in test_feat.columns if f not in ['cnt']]

train_feat_cnt = train_feat['cnt']
test_feat_date = test_feat['date']
test_feat_brand = test_feat['brand']

r_lgb = pd.read_csv('fusai_result/20180228/sub_Lgb20180228.csv', header=-1)
r_xgb = pd.read_csv('fusai_result/20180303/sub_Xgb20180303.csv', header=-1)

for i in range(len(r_lgb)):
    if r_lgb.iat[i, 1] == 4 or r_lgb.iat[i, 1] == 9:
        r_lgb.iat[i, 2] = r_xgb.iat[i, 2]

y_predict_test = list(r_lgb.iloc[:, 2].values)

now_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
now_day = datetime.datetime.now().strftime('%Y%m%d')
# 画图
print('==============画图============')
plt.plot(train_feat.date, train_feat.cnt, 'bo')
plt.title('All Count Law')
plt.xlabel('Date')
plt.ylabel('Cnt')
plt.plot(test_feat.date, y_predict_test, 'go')
plt.savefig('fusai_result/' + now_day + '/sub_Lgb+xgb{}.png'.format(now_day))
plt.show()

# submission
submission = pd.DataFrame({'date': test_feat_date, 'brand': test_feat_brand, 'pred': y_predict_test})
columns = ['date', 'brand', 'pred']  # 列名顺序
submission.to_csv(r'fusai_result/' + now_day + '/sub_Lgb+xgb{}.csv'.format(now_day), header=None,
                  index=False, columns=columns, float_format='%.0f')
submission.to_csv(r'fusai_result/' + now_day + '/sub_Lgb+xgb{}.txt'.format(now_day), header=False, sep='\t',
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
