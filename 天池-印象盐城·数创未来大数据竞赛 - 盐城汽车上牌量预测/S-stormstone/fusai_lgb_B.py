# -*- coding: utf-8 -*-
# @Time    : 2018/2/27 22:28
# @Author  : 石头人m
# @File    : fusai_lgb

import time
import datetime
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

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

train_feat = train
test_feat = testB
predictors = [f for f in test_feat.columns if f not in ['cnt']]

train_feat_cnt = train_feat['cnt']
test_feat_date = test_feat['date']
test_feat_brand = test_feat['brand']


# train_feat = trainX_split


def evalerror(pred, df):
    label = df.get_label().values.copy()
    score = l(label, pred)
    return ('mse', score, False)


print('开始训练...')

params = {
    'learning_rate': 0.1,
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'mse',
    'sub_feature': 0.5,
    'num_leaves': 20,
    'colsample_bytree': 0.7,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'min_data': 100,
    'min_hessian': 1,
    'verbose': -1,
    'is_unbalance': True
}

print('开始CV k折训练...')
k = 3
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
y_predict_test = gbm.predict(testA[predictors])

# 处理小于0的情况
''''''
for i in range(len(y_predict_test)):
    if y_predict_test[i] < 0:
        # y_predict_test[i] = 0 - y_predict_test[i]
        y_predict_test[i] = 0
for i in range(len(test_preds)):
    for j in range(k):
        if test_preds[i:i + 1, j] < 0:
            # test_preds[i:i + 1, j] = 0 - test_preds[i:i + 1, j]
            test_preds[i:i + 1, j] = 0
for i in range(len(train_preds)):
    if train_preds[i] < 0:
        # train_preds[i] = 0 - train_preds[i]
        train_preds[i] = 0

print('==============评测============')
print('线下得分 kflod：    {}'.format(mean_squared_error(train_feat['cnt'], train_preds)))
print('resultA：    {}'.format(mean_squared_error(r_testA.cnt, y_predict_test)))
print('CV训练用时{}秒'.format(time.time() - t0))

now_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
now_day = datetime.datetime.now().strftime('%Y%m%d')
# 画图
print('==============画图============')
plt.plot(train_feat.date, train_feat.cnt, 'bo')
plt.title('All Count Law')
plt.xlabel('Date')
plt.ylabel('Cnt')
plt.plot(r_testA.date, r_testA.cnt, 'bo')
plt.plot(r_testA.date, y_predict_test, 'ro')
plt.plot(test_feat.date, test_preds.mean(axis=1), 'mo')
plt.savefig('fusai_result/' + now_day + '/lgb/sub_Lgb{}.png'.format(now_day))
plt.show()

# submission
submission = pd.DataFrame({'date': test_feat_date, 'brand': test_feat_brand, 'pred': test_preds.mean(axis=1)})
columns = ['date', 'brand', 'pred']  # 列名顺序
submission.to_csv(r'fusai_result/' + now_day + '/lgb/sub_Lgb{}.csv'.format(now_day), header=None,
                  index=False, columns=columns, float_format='%.0f')
submission.to_csv(r'fusai_result/' + now_day + '/lgb/sub_Lgb{}.txt'.format(now_day), header=False, sep='\t',
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
    plt.savefig('fusai_result/' + now_day + '/lgb/brand{}.png'.format(i + 1))
    plt.show()
    plt.title('brand-%d' % (i + 1))
    plt.xlabel('Date')
    plt.ylabel('Cnt')
    plt.plot(train_brand.date, train_brand.cnt, 'b')
    plt.plot(testA_brand_answer.date, testA_brand_answer.cnt, 'g')
    plt.plot(testB_brand.date, testB_brand.pred, 'r')
    plt.savefig('fusai_result/' + now_day + '/lgb/brand{}_l.png'.format(i + 1))
    plt.show()
'''
'''
