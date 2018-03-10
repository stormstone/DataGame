# -*- coding: utf-8 -*-
# @Time    : 2018-01-20 17:02
# @Author  : Storm
# @File    : my_xgb01_cycle.py
import datetime

import pandas as pd
import xgboost as xgb
from scipy.stats import pearsonr

# 读取数据
print('1.读取数据中...')
data_path = 'result/'
train = pd.read_csv(data_path + 'dataP_train02_cycle.csv')
test = pd.read_csv(data_path + 'dataP_test02_cycle.csv')

test_feat_date = test['date']

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
    if abs(v) >= 0.1:
        feature.append(k)
        feature_v.append(v)

# 保存特征和权重
# feature_all = pd.DataFrame()
# feature_all['特征'] = feature
# feature_all['权重'] = feature_v
# # 按权重排序
# feature_all_sorted = feature_all.sort_values(by=['权重'], ascending=False)
# feature_all_sorted.plot(x='特征', y='权重', kind='bar')
# feature_all_sorted.to_csv('feature/feature_xgb_20180120_cycle1-5.csv',
#                           index=None,
#                           encoding='utf-8')

# 保留权重大于0.1的特征
X_train = train[feature_columns]
X_test = test[feature_columns]

# 构造XGBRegressor模型
model = xgb.XGBRegressor(n_estimators=120, learning_rate=0.08, gamma=0, subsample=0.5,
                         colsample_bytree=0.9, max_depth=10)

print('2.开始训练...')
model.fit(X_train, y)

print('3.开始预测...')
test_preds = model.predict(X_test)
y_predict_local = model.predict(X_train)

print('4.评测...')
# 对训练集本身评价
compare2 = pd.DataFrame()
compare2['true'] = y
compare2['pred'] = y_predict_local
f2 = (1 / (len(compare2))) * sum((compare2['pred'] - compare2['true']) * (compare2['pred'] - compare2['true']))
print('评价结果：', f2)

submission = pd.DataFrame({'date': test_feat_date, 'pred': test_preds})
submission.to_csv(r'result/sub_myXgb{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')), header=None,
                  index=False, float_format='%.0f')
