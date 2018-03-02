# -*- coding: utf-8 -*-
# @Time    : 2018-01-21 21:42
# @Author  : Storm
# @File    : my_lgb.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn import preprocessing, ensemble, metrics, grid_search, model_selection, decomposition, linear_model
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from matplotlib.pyplot import rcParams
from sklearn.datasets import load_svmlight_file
import lightgbm as lgb

df_train = pd.read_csv('./result/count.csv', encoding='gb2312')
df_test = pd.read_csv('./result/test.csv', encoding='gb2312')

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
    'num_trees': 180,
    'metric': {'auc', 'binary_logloss'},
    'num_leaves': 120,
    'learning_rate': 0.08,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 1
}

model_gbm = lgb.train(params, lgb_train)

# eval
print('score_AUC:',
      round(metrics.roc_auc_score(y_train, model_gbm.predict(X_train, num_iteration=model_gbm.best_iteration)), 5))

# 0.9977


num_round = 10
lgb.cv(params, lgb_train, num_round, nfold=5)

test_predict = model_gbm.predict(X_test, num_iteration=model_gbm.best_iteration)
print(test_predict)

'''
submission = pd.DataFrame()
submission['userid'] = df_test['userid']
submission['orderType'] = test_predict
submission.to_csv('s_result/lgb_sub{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), index=False)
'''
