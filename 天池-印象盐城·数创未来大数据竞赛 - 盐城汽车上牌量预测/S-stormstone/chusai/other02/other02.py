# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

dir = '../data/'
train = pd.read_table(dir + 'train_20171215.txt', engine='python')

actions1 = train.groupby(['date', 'day_of_week'], as_index=False)['cnt'].agg({'count1': np.sum})
# print(actions1)

df_train_target = actions1['count1'].values
print(type(df_train_target))
df_train_data = actions1.drop(['count1'], axis=1).values
print(type(df_train_data))

train_x, test_x, train_y, test_y = train_test_split(df_train_data, df_train_target, test_size=0.2, random_state=502)

gdbt = GradientBoostingRegressor()
gdbt.fit(train_x, train_y)
res = gdbt.predict(test_x)

print('本地cv:')
print(mean_squared_error(res, test_y))

'''
****************************test提交训练**************************************
'''
test_A = pd.read_table(dir + 'test_A_20171225.txt', engine='python')
gdbt.fit(df_train_data, df_train_target)
result = gdbt.predict(test_A)
pd_result = pd.DataFrame({'date': test_A["date"], 'cnt': result.astype(int)})
pd_result.to_csv('result_gdbt.txt', index=False, header=False, sep='\t', columns={'date', 'cnt'})
print('完成训练')
