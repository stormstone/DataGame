# -*- coding: utf-8 -*-
# @Time    : 2018-01-25 14:19
# @Author  : Storm
# @File    : my_poly_lgb.py

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn import metrics
import lightgbm as lgb

print('loading data ...')
df_train = pd.read_csv('./s_result/a_poly_train.csv', encoding='utf-8')
df_test = pd.read_csv('./s_result/a_poly_test.csv', encoding='utf-8')
test_old = pd.read_csv('./result/test.csv', encoding='utf-8')

target = 'orderType'
x_tags = [x for x in df_train.columns if x not in ['orderType']]
y_tag = target

X_train = np.array(df_train[x_tags])
y_train = np.array(df_train[y_tag])

X_test = np.array(df_test[x_tags])

lgb_train = lgb.Dataset(X_train, y_train)
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'num_trees': 100,
    'metric': {'auc', 'binary_logloss'},
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 1
}

print('lgb train...')
model_gbm = lgb.train(params, lgb_train)

# eval
print('score_AUC:',
      round(metrics.roc_auc_score(y_train, model_gbm.predict(X_train, num_iteration=model_gbm.best_iteration)), 5))

num_round = 1000
print('\n cv...')
lgb.cv(params, lgb_train, num_round, nfold=5)

print('best_iteration:', model_gbm.best_iteration)
test_predict = model_gbm.predict(X_test, num_iteration=model_gbm.best_iteration)

submission = pd.DataFrame()
submission['userid'] = test_old['userid']
submission['orderType'] = test_predict
submission.to_csv('s_result/poly_lgb_sub{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), index=False)
