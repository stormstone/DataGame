# -*- coding: utf-8 -*-
# @Time    : 2018-01-21 22:20
# @Author  : Storm
# @File    : my_xgb.py

import pandas as pd
import numpy as np
from sklearn import metrics, model_selection
import xgboost as xgb
from xgboost.sklearn import XGBClassifier

print('=== loading data...')
df_train = pd.read_csv('result/count.csv', encoding='utf-8')
df_test = pd.read_csv('result/test.csv', encoding='utf-8')
print('=== read data complete...')

target = 'orderType'

x_tags = ['46', '47', '48', '49', '50', '51', '52', 'MR', 'time_span_mean', 'time_span_var', 'time_span_min',
          'time_span_tail1', 'time_span_tail2', 'time_span_tail3', 'time_span_tail4', 'time_last3_span_mean',
          'time_last3_span_var', 'time_first_span_var', 'order_product', '15mean', 'lastto2-4',
          'actiontime_span_last5_6', 'time_span_lastAction_lastOrder', 'lastto5', 'count_5_to_6', 'five_count',
          'eight_five_ratio', 'count_234_mean', 'timespan_last_to_last5', '6and5Ratio', '6count', 'lastto6', '59and245',
          'lastto1', 'lastto7_9', 'lastto79', 'lastday_count', 'lastdayto1_4and5', 'lastdayto5_6sum', 'lastdayto2_4sum',
          'lastdayto-2to234near', 'lastdayto6_9sum', 'last7dayto1_5max', 'last7dayto1_5min', 'last7dayto1_5sum',
          'last7dayto5_6max', 'last7dayto5_6sum', 'last7dayto5_6min', 'last7daytimeto1', 'last7daytimeto2_4',
          'last7daytimeto5', 'firstto1', 'firstto5', 'firstto6', 'firstto7_9', 'last1to2', 'last1to3', 'last1to4',
          'from', 'last678to-2', 'last1678', 'last1-9', 'lastto7']

y_tag = target
X_train = np.array(df_train[x_tags])
y_train = np.array(df_train[y_tag])

X_test = np.array(df_test[x_tags])
print('=== X_train、y_train、 X_test complete...')


def modelfit(alg, X_train, y_train, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_params()
        xgtrain = xgb.DMatrix(X_train, label=y_train)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'],
                          nfold=cv_folds, metrics='auc', early_stopping_rounds=early_stopping_rounds)
        print('best n_estimators:', cvresult.shape[0])
        alg.set_params(n_estimators=cvresult.shape[0])
    alg.fit(X_train, y_train, eval_metric='auc')
    dtrain_pred = alg.predict(X_train)
    dtrain_predprob = alg.predict_proba(X_train)[:, 1]
    print('\nModel Report:')
    print("AUC Score (Train): %.5g" % metrics.roc_auc_score(y_train, dtrain_predprob))
    print("Accuracy: %.5g" % metrics.accuracy_score(y_train, dtrain_pred))


np.random.seed(272)
model_xgb1 = XGBClassifier(
    learning_rate=0.1,
    n_estimators=3000,
    max_depth=5,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    nthread=8,
    scale_pos_weight=1,
    seed=272
)
print('=== begain fit...')
modelfit(model_xgb1, X_train, y_train)

model_xgb_final = model_xgb1
y_train_pred = model_xgb_final.predict(X_train)
y_train_predprob = model_xgb_final.predict_proba(X_train)[:, 1]
importances = model_xgb_final.feature_importances_
df_featImp = pd.DataFrame({'tags': x_tags, 'importance': importances})
df_featImp_sorted = df_featImp.sort_values(by=['importance'], ascending=False)
df_featImp_sorted.plot(x='tags', y='importance', kind='bar')
df_featImp.to_csv('feature/feat_20180120_c1.csv')
df_featImp_sorted.to_csv('feature/feat_20180120_c1.csv')

print('score_AUC:', round(metrics.roc_auc_score(y_train, y_train_predprob), 5))
print('score_precision:', round(metrics.accuracy_score(y_train, y_train_pred), 5))
# scores_cross = model_selection.cross_val_score(model_xgb_final, X_train, y_train, cv=5, scoring='roc_auc')
# print('score_cross:', round(np.mean(scores_cross), 5), 'std:', round(np.std(scores_cross), 5))

y_test_pred = model_xgb_final.predict_proba(X_test)[:, 1]
df_profile = pd.read_csv('data_train_test/userProfile_test.csv')
restable = pd.DataFrame(
    np.concatenate((np.array(df_profile['userid']).reshape((-1, 1)), y_test_pred.reshape((-1, 1))), axis=1))
restable.loc[:, 0] = restable.loc[:, 0].astype(np.int64)
# pd.DataFrame(restable).to_csv("s_result/orderFuture_test_20180125_c1.csv", header=['userid', 'orderType'], index=False)
