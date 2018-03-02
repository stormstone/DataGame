## 第二届智慧中国杯首数据科学竞赛，首发皇包车（HI GUIDES）精品旅行服务成单预测竞赛

[精品旅行服务成单预测竞赛官网 ->](http://www.dcjingsai.com/common/cmpt/%E7%B2%BE%E5%93%81%E6%97%85%E8%A1%8C%E6%9C%8D%E5%8A%A1%E6%88%90%E5%8D%95%E9%A2%84%E6%B5%8B_%E7%AB%9E%E8%B5%9B%E4%BF%A1%E6%81%AF.html)

## 1 特征工程

用户标签：

- orderhistory
  - 购买过的旅游服务总数量
  - 购买过的普通旅游服务数量
  - 购买过的精品旅游服务数量
  - 精品+1/普通+1 or 精品订单数/（精品+普通）得出百分比
  - 最后一次订单的时间，时间戳最后一次订单的年份，月份
  - 一个用户的历史订单中去过每个大洲的次数
  - 一个用户的历史订单中去过每个国家的次数（training data中有51个国家）
- action
  - 一个用户action的总次数（用处不大）
  - 倒数第1-20个actionType（缺失的用0替代）
  - 倒数第1-20个actionTime（缺失的用0替代）
  - 每种actionType的次数占比
  - actionType1-5的时间间隔 timespan_1-5_
  - 最后一次每种actionType与下一个actionType之间的间隔，如1-5，5-6, e.g., timespanlast_5-6，timespanlast_6-7，timespanlast_7-8
  - actionType2-4与actionType5-9之间的比例
  - 倒数1个actionTime的年份，月份
  - 每个actionType最后一次出现的时间
  - actionType5-9的连续程度sequential_ratio（从最后一个5开始计算，没有5则为0，只有5，则为1，5之后只出现6，则为2，还出现了7则为3，出现8为4）
  - 最后20个action中，每个actionType出现的次数
  - 倒数3个actiontimespan：倒数第1个action与倒数第2个action之间的时间差，倒数第2个action与倒数第3个action之间的时间差……
  - 最后3个，5个，actiontimespan的均值
  - 最后3个，5个，actiontimespan的标准差
  - 第一个actionTime（用户第一次使用app的时间）action_firsttime
  - 最后一个actionTime与第一个actionTime的时间差，描述用户用了app总时间（是否是老用户）timespan_total
- comment
  - 一个用户在其所有订单中的平均打分数rating_average
  - 最低打分数rating_min
  - 最后一次打分数rating_last
- profile
  - 是否有男女信息
  - 是否是男
  - 是否是女
  - 是否有年龄信息
  - 是否60后，70后，80后，90后，00后
  - 用户所属省份
- 不同表之间的复合信息
  - 每个用户最后一个actionTime和最后一个orderTime之间的时间差 timespan_action_lastord
  - 每个actionType最后一次出现的时间与最后一个actionTIme之间的时间差 timespan_action1tolast, timespan_action2tolast, ...
  - ordertime_last_1与ordertime_last_2之间的时间差timespan_lastord_1_2，ordertime_last_2与ordertime_last_3之间的时间差timespan_lastord_2_3

## 2 模型选择与训练

首选 xgboost，初始参数列表为：

```python
model_xgb1 = XGBClassifier(
    learning_rate=0.03,
    n_estimators=3000,
    max_depth=5,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    nthread=4,
    scale_pos_weight=1,
    seed=314
    )
```
设置评价指标为 AUC，在模型 training 阶段使用 cross_validation(k_folds=5) 计算得到 AUC 值：

```python
model_xgb_final = model_xgb1
#model_xgb_final.fit(X_train, y_train)
y_train_pred = model_xgb_final.predict(X_train)
y_train_predprob = model_xgb_final.predict_proba(X_train)[:,1]
importances = model_xgb_final.feature_importances_
df_featImp = pd.DataFrame({'tags': x_tags, 'importance': importances})
df_featImp.sort_values(by=['importance'], ascending=False).plot(x='tags', y='importance', kind='bar')

print('score_AUC:', round(metrics.roc_auc_score(y_train, y_train_predprob), 5))
print('score_precision:', round(metrics.accuracy_score(y_train, y_train_pred), 5))
scores_cross = model_selection.cross_val_score(model_xgb_final, X_train, y_train, cv=5, scoring='roc_auc')
print('score_cross:', round(np.mean(scores_cross), 5), 'std:', round(np.std(scores_cross), 5))
```

