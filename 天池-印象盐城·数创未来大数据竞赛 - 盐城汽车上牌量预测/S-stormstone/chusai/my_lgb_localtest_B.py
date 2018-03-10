# -*- coding: utf-8 -*-
# @Time    : 2018-02-16 21:19
# @Author  : Storm
# @File    : my_lgb_localtest.py

import time
import datetime
import numpy as np
import pandas as pd
import lightgbm as lgb
from dateutil.parser import parse
from sklearn.cross_validation import KFold, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

data_path = 'resultAend/'
data_testA_result = pd.read_table("data/answer_A_20180225.txt", header=0, encoding='gb2312', delim_whitespace=True)

# train = pd.read_csv(data_path + 'getdata_train_resultA_poly.csv')
# test = pd.read_csv(data_path + 'getdata_testB_poly.csv')

train = pd.read_csv(data_path + 'getdata_train_resultA.csv')
test = pd.read_csv(data_path + 'getdata_testB_ok.csv')
train = train.drop(['r_date'], axis=1)
test = test.drop(['r_date'], axis=1)

train_feat = train
test_feat = test
predictors = [f for f in test_feat.columns if f not in ['cnt']]

train_feat_cnt = train_feat['cnt']
test_feat_date = test_feat['date']

split_num = len(train)-len(data_testA_result)
trainX_split = train_feat.iloc[:split_num, :]
testX_split = train_feat.iloc[split_num:, :]


def evalerror(pred, df):
    label = df.get_label().values.copy()
    score = mean_squared_error(label, pred)
    return ('mse', score, False)


print('开始训练...')

params = {
    'learning_rate': 0.01,
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'mse',
    'sub_feature': 0.5,
    'num_leaves': 256,
    'colsample_bytree': 0.7,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'min_data': 100,
    'min_hessian': 1,
    'verbose': -1,
    'is_unbalance': True
}

print('开始CV k折训练...')
k = 5
t0 = time.time()
train_preds = np.zeros(train_feat.shape[0])
test_preds = np.zeros((test_feat.shape[0], k))
kf = KFold(len(train_feat), n_folds=k, shuffle=True, random_state=520)
for i, (train_index, test_index) in enumerate(kf):
    print('第{}次训练...'.format(i))
    train_feat1 = train_feat.iloc[train_index]
    train_feat2 = train_feat.iloc[test_index]
    lgb_train1 = lgb.Dataset(train_feat1[predictors], train_feat1['cnt'])
    lgb_train2 = lgb.Dataset(train_feat2[predictors], train_feat2['cnt'])
    gbm = lgb.train(params,
                    lgb_train1,
                    num_boost_round=10000,
                    valid_sets=lgb_train2,
                    verbose_eval=100,
                    feval=evalerror,
                    early_stopping_rounds=100)
    feat_imp = pd.Series(gbm.feature_importance(), index=predictors).sort_values(ascending=False)
    train_preds[test_index] += gbm.predict(train_feat2[predictors])
    test_preds[:, i] = gbm.predict(test_feat[predictors])

# split test
y_predict_split = gbm.predict(testX_split[predictors])
y_predict_test = gbm.predict(test_feat[predictors])

print('==============评测============')
print('线下得分 kflod：    {}'.format(mean_squared_error(train_feat['cnt'], train_preds)))
print('线下得分 split：    {}'.format(mean_squared_error(testX_split['cnt'], y_predict_split)))
print('CV训练用时{}秒'.format(time.time() - t0))

# 画图
print('==============画图============')
plt.plot(train_feat.date, train_feat.cnt, 'b')
plt.plot(testX_split.date, testX_split.cnt, 'b')
plt.title('All Count Law')
plt.xlabel('Date')
plt.ylabel('Cnt')
plt.plot(testX_split.date, y_predict_split, 'r')
plt.plot(test_feat.date, y_predict_test, 'g')
# plt.plot(test_feat.date, y_predict_test, 'yo')
now_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

submission = pd.DataFrame({'date': test_feat_date, 'pred': test_preds.mean(axis=1)})

'''
plt.savefig('resultB/sub_myLgb{}_l.png'.format(now_time))

# submission
submission.to_csv(r'resultB/sub_myLgb{}.csv'.format(now_time), header=None,
                  index=False, float_format='%.0f')
submission.to_csv(r'resultB/sub_myLgb{}.txt'.format(now_time), header=False, sep='\t',
                  index=False, float_format='%.0f')
'''

print('线下得分 resultA：    {}'.format(mean_squared_error(data_testA_result['cnt'], y_predict_split)))

plt.show()
