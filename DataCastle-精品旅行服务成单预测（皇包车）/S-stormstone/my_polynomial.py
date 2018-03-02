# -*- coding: utf-8 -*-
# @Time    : 2018-01-25 14:01
# @Author  : Storm
# @File    : my_polynomial.py

import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

a_train = pd.read_csv('result/count.csv', encoding='utf-8')
a_test = pd.read_csv('result/test.csv', encoding='utf-8')
print(a_train.shape)
# (40307, 73)
print(a_test.shape)
# (10076, 72)

y_a_train = a_train['orderType']
x_a_train = a_train.drop(['orderType'], axis=1)

X_data = pd.concat([x_a_train, a_test])
print(X_data.shape)
# (50383, 72)

poly = PolynomialFeatures(interaction_only=True)  # 默认的阶数是２，同时设置交互关系为true
X_poly = poly.fit_transform(X_data)

print(X_poly.shape)
# (50383, 2629)

poly_X_train = X_poly[:len(y_a_train)]
poly_X_test = X_poly[len(y_a_train):]

print(poly_X_train.shape)
# (40307, 2629)
print(poly_X_test.shape)
# (10076, 2629)

poly_train = pd.DataFrame(poly_X_train)
poly_train['orderType'] = y_a_train
print(poly_train.shape)
# (40307, 2630)
poly_test = pd.DataFrame(poly_X_test)
print(poly_test.shape)
# (10076, 2629)

poly_train.to_csv("s_result/a_poly_train.csv", index=None, encoding='utf-8')
poly_test.to_csv("s_result/a_poly_test.csv", index=None, encoding='utf-8')
