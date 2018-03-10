# -*- coding: utf-8 -*-
# @Time    : 2018-01-18 17:33
# @Author  : Storm
# @File    : my_lgb_cycle.py

import time
import datetime
import numpy as np
import pandas as pd
import lightgbm as lgb
from dateutil.parser import parse
from sklearn.cross_validation import KFold, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA

data_path = 'result/'
train = pd.read_csv(data_path + 'dataP_train02_cycle.csv')
test = pd.read_csv(data_path + 'dataP_test02_cycle.csv')
train_feat = train
test_feat = test
predictors = [f for f in test_feat.columns if f not in ['cnt']]

train_feat_cnt = train_feat['cnt']
test_feat_date = test_feat['date']

# PCA
'''
train_feat_pca = train_feat.drop(['cnt'], axis=1)
pca_all_data = pd.concat(train_feat_pca,test_feat)
print(pca_all_data)

pca = PCA(n_components=0.98)
pca.fit(train_feat_pca)
train_feat_pca = pca.fit_transform(train_feat_pca)
test_feat_pca = test_feat
'''
'''
# 归一化
train_xuetang = train_feat['cnt']
train_feat = train_feat.drop(['cnt'], axis=1)
for i in train_feat.columns:
    train_feat[i] = (train_feat[i] - train_feat[i].min()) / (train_feat[i].max() - train_feat[i].min())
    test_feat[i] = (test_feat[i] - test_feat[i].min()) / (test_feat[i].max() - test_feat[i].min())
train_feat['cnt'] = train_xuetang
'''


# 取PCA结果
# train_feat = train_input
# test_feat = test_input


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
    'sub_feature': 0.7,
    'num_leaves': 256,
    'colsample_bytree': 0.7,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'min_data': 100,
    'min_hessian': 1,
    'verbose': -1,
}
'''
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'mse',
    'num_leaves': 256,
    'learning_rate': 0.1,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0,
}
'''
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
print('线下得分：    {}'.format(mean_squared_error(train_feat['cnt'], train_preds)))
print('CV训练用时{}秒'.format(time.time() - t0))

'''
submission = pd.DataFrame({'date': test_feat_date, 'pred': test_preds.mean(axis=1)})
submission.to_csv(r'result/sub_myLgb{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')), header=None,
                  index=False, float_format='%.0f')
'''
