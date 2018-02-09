#!/usr/bin/python
# coding:utf8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import time
from sklearn.model_selection import train_test_split, KFold
from sklearn.grid_search import GridSearchCV
import lightgbm as lgb
from sklearn.metrics import mean_squared_error


def makeHistoryFeature(dfData, statCol, sumFunc=False, meanFunc=False, madFunc=False, medianFunc=False, minFunc=False,
                       maxFunc=False):
    """
    Desc：根据给定表和条件制作历史统计特征, 8-10月统计量做训练集输入, 8-11月统计量做预测用输入
    Params: 
        dfData: DataFrame格式数据, 数据中至少包含uid、month以及待统计的量
        statCol: 字符串, 欲统计的列名, 该列在dfData中是解脱敏的
        后续几个参数就代表, 需不需要这几个特征
    Return: 
        featureTrain: 训练用的输入特征, 返回结果中做了脱敏
        featurePred: 预测用的输入特征, 返回结果中做了脱敏
    """
    hisTrain = dfData.query('month != "11"')
    hisPred = dfData.copy()

    feaRec = []  # 记录需要哪些特征
    feaFunc = []
    if sumFunc == True:
        feaRec.append('Sum')
        feaFunc.append(np.sum)
    if meanFunc == True:
        feaRec.append('Mean')
        feaFunc.append(np.mean)
    if madFunc == True:
        feaRec.append('Mad')
        feaFunc.append(pd.Series.mad)
    if medianFunc == True:
        feaRec.append('Median')
        feaFunc.append(pd.Series.median)
    if minFunc == True:
        feaRec.append('Min')
        feaFunc.append(np.min)
    if maxFunc == True:
        feaRec.append('Max')
        feaFunc.append(np.max)
    # 拼接特征列名
    feaName = list(map(lambda x: statCol + x, feaRec))
    feaDict = dict(zip(feaName, feaFunc))

    featureTrain = hisTrain.groupby(['uid'])[statCol].agg(feaDict).reset_index()
    for i in feaName:
        featureTrain[i] = featureTrain[i].apply(lambda x: math.log(x + 1, 5))

    featurePred = hisPred.groupby(['uid'])[statCol].agg(feaDict).reset_index()
    for i in feaName:
        featurePred[i] = featurePred[i].apply(lambda x: math.log(x + 1, 5))

    return featureTrain, featurePred


def makeMonthFeature(dfData, statCol, sumFunc=False, meanFunc=False, madFunc=False, medianFunc=False, minFunc=False,
                     maxFunc=False):
    """
    Desc：根据给定表和条件制作单月统计特征, 10月统计量做训练集输入, 11月统计量做预测用输入
    Params: 
        dfData: DataFrame格式数据, 数据中至少包含uid、month以及待统计的量
        statCol: 字符串, 欲统计的列名, 该列在dfData中是解脱敏的
        后续几个参数就代表, 需不需要这几个特征
    Return: 
        featureTrain: 训练用的输入特征, 返回结果中做了脱敏
        featurePred: 预测用的输入特征, 返回结果中做了脱敏
    """
    monthTrain = dfData.query('month == "10"')
    monthPred = dfData.query('month == "11"')
    feaRec = []  # 记录需要哪些特征
    feaFunc = []
    if sumFunc == True:
        feaRec.append('Sum_a')
        feaFunc.append(np.sum)
    if meanFunc == True:
        feaRec.append('Mean_a')
        feaFunc.append(np.mean)
    if madFunc == True:
        feaRec.append('Mad_a')
        feaFunc.append(pd.Series.mad)
    if medianFunc == True:
        feaRec.append('Median_a')
        feaFunc.append(pd.Series.median)
    if minFunc == True:
        feaRec.append('Min_a')
        feaFunc.append(np.min)
    if maxFunc == True:
        feaRec.append('Max_a')
        feaFunc.append(np.max)

    # 拼接特征列名
    feaName = list(map(lambda x: statCol + x, feaRec))
    feaDict = dict(zip(feaName, feaFunc))

    featureTrain = monthTrain.groupby(['uid'])[statCol].agg(feaDict).reset_index()
    for i in feaName:
        featureTrain[i] = featureTrain[i].apply(lambda x: math.log(x + 1, 5))

    featurePred = monthPred.groupby(['uid'])[statCol].agg(feaDict).reset_index()
    for i in feaName:
        featurePred[i] = featurePred[i].apply(lambda x: math.log(x + 1, 5))

    return featureTrain, featurePred


def dealLoan(inputFile):
    """
    Desc：将借贷表中的数据取出，并把金额进行解脱敏处理，
    最后按uid和月份分组，进行加和，再脱敏处理后，新表输出到新文件。
    再做一些其他特征。
    """
    dfData = pd.read_csv(inputFile)
    # 取出日期格式里面的月份
    dfData['month'] = dfData['loan_time'].apply(lambda x: x.split('-')[1])
    # 解脱敏处理
    dfData['loan'] = dfData['loan_amount'].apply(lambda x: 5 ** x - 1)

    # !!!制作特征!!!
    # loan的单月sum特征
    loanTrain, loanPred = makeMonthFeature(dfData, 'loan', 1, 1, 1, 1, 1, 1)
    # loan的历史sum特征
    hisLoanTrain, hisLoanPred = makeHistoryFeature(dfData, 'loan', 1, 1, 1, 1, 1, 1)

    # loan_amount / plannum 特征, 简称为fen
    loanFen = dfData.copy()
    loanFen['fen'] = dfData['loan'] / dfData['plannum']
    # fen的单月sum特征
    fenTrain, fenPred = makeMonthFeature(loanFen, 'fen', 1, 0, 1, 0, 1, 1)
    # fen的历史sum特征
    hisFenTrain, hisFenPred = makeHistoryFeature(loanFen, 'fen', 1, 1, 1, 0, 1, 1)

    # --------------------------------------
    # loan_balance: 还需还款数额， 比如张三共借了1万，还了2000，还剩 8000，这个8000 就是 
    # remain_limit: 剩余可借贷款额度，在输入合并函数中加更方便，见 makeMergeInput()
    # 做训练时候用的输入, 不能含11月数据
    loanBalanceTrain = dfData.query('month != "11"')
    toMonth = 10  # 方便后面delta_字段表示过了几个月到10月
    loanBalanceTrain['delta_month'] = toMonth - loanBalanceTrain['month'].astype(int)
    loanBalanceTrain['delta_plannum'] = loanBalanceTrain['plannum'] - loanBalanceTrain['delta_month']
    # delta_plannum 字段表示剩余还需还几期
    loanBalanceTrain['delta_plannum'] = loanBalanceTrain['delta_plannum'].apply(lambda x: 0 if x < 0 else x)
    # 解脱敏处理
    loanBalanceTrain['loan_amount'] = loanBalanceTrain['loan_amount'].apply(lambda x: 5 ** x - 1)
    loanBalanceTrain['loan_balance'] = loanBalanceTrain['loan_amount'] * loanBalanceTrain['delta_plannum'] / \
                                       loanBalanceTrain['plannum']
    loanBalanceTrain = loanBalanceTrain.groupby('uid')['loan_balance'].sum().reset_index()
    # 脱敏处理
    loanBalanceTrain['loan_balance'] = loanBalanceTrain['loan_balance'].apply(lambda x: math.log(x + 1, 5))

    # 做预测时候用的输入
    loanBalancePred = dfData.copy()
    toMonth = 11  # 方便后面delta_字段表示过了几个月到10月
    loanBalancePred['delta_month'] = toMonth - loanBalancePred['month'].astype(int)
    loanBalancePred['delta_plannum'] = loanBalancePred['plannum'] - loanBalancePred['delta_month']
    # delta_plannum 字段表示剩余还需还几期
    loanBalancePred['delta_plannum'] = loanBalancePred['delta_plannum'].apply(lambda x: 0 if x < 0 else x)
    # 解脱敏处理
    loanBalancePred['loan_amount'] = loanBalancePred['loan_amount'].apply(lambda x: 5 ** x - 1)
    loanBalancePred['loan_balance'] = loanBalancePred['loan_amount'] * loanBalancePred['delta_plannum'] / \
                                      loanBalancePred['plannum']
    loanBalancePred = loanBalancePred.groupby('uid')['loan_balance'].sum().reset_index()
    # 脱敏处理
    loanBalancePred['loan_balance'] = loanBalancePred['loan_balance'].apply(lambda x: math.log(x + 1, 5))

    # --------------------------------------
    # 把分期数plannum的1,3,6,12当成四个种类，分别统计各用户历史操作每个种类的次数
    # 8-10月做训练集一部分输入特征
    planTrain = dfData.query('month != "11"')
    planTrain = planTrain.groupby(['uid', 'plannum'])['loan_time'].count().reset_index()
    planTrain['plannum'] = planTrain['plannum'].apply(lambda x: 'plannum_' + str(x))
    # 数据透视
    planTrain = planTrain.pivot(index='uid', columns='plannum', values='loan_time').reset_index()

    # 8-11月做预测时候的一部分输入特征
    planPred = dfData.copy()
    planPred = planPred.groupby(['uid', 'plannum'])['loan_time'].count().reset_index()
    planPred['plannum'] = planPred['plannum'].apply(lambda x: 'plannum_' + str(x))
    # 数据透视
    planPred = planPred.pivot(index='uid', columns='plannum', values='loan_time').reset_index()

    # --------------------------------------合并成训练集输入、预测用的输入
    mergeTrain = pd.merge(hisLoanTrain, loanBalanceTrain, how='outer', on='uid')
    mergeTrain = pd.merge(mergeTrain, loanTrain, how='outer', on='uid')
    mergeTrain = pd.merge(mergeTrain, fenTrain, how='outer', on='uid')
    mergeTrain = pd.merge(mergeTrain, hisFenTrain, how='outer', on='uid')
    mergeTrain = pd.merge(mergeTrain, planTrain, how='outer', on='uid')

    mergePred = pd.merge(hisLoanPred, loanBalancePred, how='outer', on='uid')
    mergePred = pd.merge(mergePred, loanPred, how='outer', on='uid')
    mergePred = pd.merge(mergePred, fenPred, how='outer', on='uid')
    mergePred = pd.merge(mergePred, hisFenPred, how='outer', on='uid')
    mergePred = pd.merge(mergePred, planPred, how='outer', on='uid')

    # -------------------------------------- 合并user表中的特征
    dfUser = pd.read_csv("../input/t_user.csv")
    # 去掉两端的字符用strip(), 这里需要借助replace替换掉所有'-'
    dfUser['active_date'] = dfUser['active_date'].apply(lambda x: x.replace('-', ''))

    # 拼接loan表得到的组合统计特征
    mergeTrain = pd.merge(dfUser, mergeTrain, how='outer', on='uid')
    # 加一个剩余可借额度特征
    mergeTrain['remain_limit'] = mergeTrain[['limit', 'loan_balance']].apply(
        lambda x: 5 ** x.limit - 5 ** x.loan_balance, axis=1)
    mergeTrain['remain_limit'] = mergeTrain['remain_limit'].apply(lambda x: 0 if x < 0 else x)
    mergeTrain['remain_limit'] = mergeTrain['remain_limit'].apply(lambda x: math.log(x + 1, 5))

    mergePred = pd.merge(dfUser, mergePred, how='outer', on='uid')
    mergePred['remain_limit'] = mergePred[['limit', 'loan_balance']].apply(lambda x: 5 ** x.limit - 5 ** x.loan_balance,
                                                                           axis=1)
    mergePred['remain_limit'] = mergePred['remain_limit'].apply(lambda x: 0 if x < 0 else x)
    mergePred['remain_limit'] = mergePred['remain_limit'].apply(lambda x: math.log(x + 1, 5))

    print(mergeTrain.shape)
    inputFileName = str(inputFile)[9:]
    mergeTrain.to_csv('dealed_trainInput_' + inputFileName, index=False)
    print(mergePred.shape)
    mergePred.to_csv('dealed_predInput_' + inputFileName, index=False)


def dealOrder(inputFile):
    """
    Desc：将订单信息表中的数据取出，并把金额进行解脱敏处理，计算每个用户每月购物花费
    另外加入其它特征。
    """
    dfData = pd.read_csv(inputFile)
    # 由于该表有缺失值，所以先要删除这些数据
    # dfData = dfData.dropna()
    # print(dfData[dfData.isnull().values == True]) 看空值出现的位置

    # 取出日期格式里面的月份
    dfData['month'] = dfData['buy_time'].apply(lambda x: x.split('-')[1])
    # 解脱敏处理
    dfData['price'] = 5 ** dfData['price'] - 1
    dfData['discount'] = 5 ** dfData['discount'] - 1
    dfData['cost'] = dfData['price'] * dfData['qty'] - dfData['discount']
    # 这样速度较慢 dfData['cost'] = dfData[['price', 'qty', 'discount']].apply(lambda x: x.price * x.qty - x.discount, axis = 1)
    dfData['cost'] = dfData['cost'].apply(lambda x: x if x > 0 else 0)

    # --------------------------------------
    # order花费金额cost的单月统计特征
    orderTrain, orderPred = makeMonthFeature(dfData, 'cost', 1, 0, 1, 1, 1, 1)

    # --------------------------------------
    # order花费金额cost的历史统计特征
    hisOrderTrain, hisOrderPred = makeHistoryFeature(dfData, 'cost', 1, 0, 1, 1, 1, 1)

    # --------------------------------------把购买的物品不同种类的个数做一个统计特征
    # conTable 作用是补全 cate_id 因为观察shape发现，有两件商品是11月之前没有出现过的
    conTable = []
    for i in range(1, 45):
        conTable.append([1, i, 0, '08'])
    conTable = pd.DataFrame(conTable, columns=['uid', 'cate_id', 'cost', 'month'])
    dfNew = dfData[['uid', 'cate_id', 'cost', 'month']]
    dfNew = pd.concat([conTable, dfNew])

    cateIdTrain = dfNew.query('month != "11"')
    cateIdTrain = cateIdTrain.groupby(['uid', 'cate_id'])['cost'].sum().reset_index()
    cateIdTrain['cost'] = cateIdTrain['cost'].apply(lambda x: math.log(x + 1, 5))
    cateIdTrain['cate_id'] = cateIdTrain['cate_id'].apply(lambda x: 'cate_id_' + str(x))
    cateIdTrain = cateIdTrain.pivot(index='uid', columns='cate_id', values='cost').reset_index()

    cateIdPred = dfNew.copy()
    cateIdPred = cateIdPred.groupby(['uid', 'cate_id'])['cost'].sum().reset_index()
    cateIdPred['cost'] = cateIdPred['cost'].apply(lambda x: math.log(x + 1, 5))
    cateIdPred['cate_id'] = cateIdPred['cate_id'].apply(lambda x: 'cate_id_' + str(x))
    cateIdPred = cateIdPred.pivot(index='uid', columns='cate_id', values='cost').reset_index()

    # 合并order处理出来的这些特征，并分别存储训练用的和预测用的数据
    mergeTrain = pd.merge(orderTrain, hisOrderTrain, how='outer', on='uid')
    mergeTrain = pd.merge(mergeTrain, cateIdTrain, how='outer', on='uid')
    print(mergeTrain.shape)
    inputFileName = str(inputFile)[9:]
    mergeTrain.to_csv('dealed_trainInput_' + inputFileName, index=False)

    mergePred = pd.merge(orderPred, hisOrderPred, how='outer', on='uid')
    mergePred = pd.merge(mergePred, cateIdPred, how='outer', on='uid')
    print(mergePred.shape)
    mergePred.to_csv('dealed_predInput_' + inputFileName, index=False)


def dealClick(inputFile):
    """
    Desc：将点击信息表中的数据取出，统计各用户三个月某pid-param点击次数
    """
    dfData = pd.read_csv(inputFile)
    # dfData = dfData.dropna()

    # 取出日期格式里面的月份
    dfData['month'] = dfData['click_time'].apply(lambda x: x.split('-')[1])
    dfData['click_time'] = 1
    # ------------------------------pid-param组合特征
    conTable = []
    for pid in range(1, 11):
        for par in range(1, 49):
            conTable.append([1, 0, pid, par, '09'])
    conTable = pd.DataFrame(conTable, columns=['uid', 'click_time', 'pid', 'param', 'month'])
    dfNew = pd.concat([conTable, dfData])

    # 8,9,10月做训练集一部分输入特征
    dfPidParTrain = dfNew.query('month != "11"')

    dfPidParTrain = dfPidParTrain.groupby(['uid', 'pid', 'param'])['click_time'].sum().reset_index()
    dfPidParTrain['pid_param'] = dfPidParTrain[['pid', 'param']].apply(
        lambda x: 'pid_param' + str(x.pid) + str(x.param), axis=1)
    dfPidParTrain = dfPidParTrain.pivot(index='uid', columns='pid_param', values='click_time').reset_index()

    # 8,9,10,11月做预测时一部分输入特征
    dfPidParPred = dfNew.copy()
    dfPidParPred = dfPidParPred.groupby(['uid', 'pid', 'param'])['click_time'].sum().reset_index()
    dfPidParPred['pid_param'] = dfPidParPred[['pid', 'param']].apply(lambda x: 'pid_param' + str(x.pid) + str(x.param),
                                                                     axis=1)
    dfPidParPred = dfPidParPred.pivot(index='uid', columns='pid_param', values='click_time').reset_index()

    print(dfPidParTrain.shape)
    print(dfPidParPred.shape)
    inputFileName = str(inputFile)[9:]
    dfPidParTrain.to_csv('dealed_trainInput_' + inputFileName, index=False)
    dfPidParPred.to_csv('dealed_predInput_' + inputFileName, index=False)


def makeMergeInput():
    """
    Desc：做训练输入数据和预测输入数据
    """
    # 拼接loan表得到的组合统计特征
    mergeTrain = pd.read_csv("dealed_trainInput_t_loan.csv")
    mergePred = pd.read_csv("dealed_predInput_t_loan.csv")

    # 拼接click表得到的组合统计特征
    dfClickTrain = pd.read_csv("dealed_trainInput_t_click.csv")
    mergeTrain = pd.merge(mergeTrain, dfClickTrain, how='outer', on='uid')
    dfClickPred = pd.read_csv("dealed_predInput_t_click.csv")
    mergePred = pd.merge(mergePred, dfClickPred, how='outer', on='uid')

    # 拼接order表得到的组合统计特征
    dfOrderTrain = pd.read_csv("dealed_trainInput_t_order.csv")
    mergeTrain = pd.merge(mergeTrain, dfOrderTrain, how='outer', on='uid')
    dfOrderPred = pd.read_csv("dealed_predInput_t_order.csv")
    mergePred = pd.merge(mergePred, dfOrderPred, how='outer', on='uid')

    print(mergeTrain.shape)
    print(mergePred.shape)
    mergeTrain.to_csv('trainInput.csv', index=False)
    mergePred.to_csv('predInput.csv', index=False)


def lgbTrainModel(inputFile):
    """
    Desc：用lightgbm来跑出回归模型，并预测看看
    """
    dfData = pd.read_csv(inputFile)

    # 去掉uid列,uid不是特征
    data = dfData.drop(labels=['uid'], axis=1)

    trainInput = data.fillna(0.0)
    print(trainInput.shape)

    # 11月借贷总额当做训练集结果
    dfAns = pd.read_csv('../input/t_loan_sum.csv')
    allUserAns = pd.merge(dfData, dfAns, how='outer', on='uid')
    allUserAns.fillna(0.0, inplace=True)
    loanAns = allUserAns['loan_sum']

    xTrain, xTest, yTrain, yTest = train_test_split(trainInput, loanAns, test_size=0.2, random_state=100)
    lgbTrain = lgb.Dataset(xTrain, yTrain)
    lgbEval = lgb.Dataset(xTest, yTest)
    lgbAll = lgb.Dataset(trainInput, loanAns)

    # 使用GridSearchCV调lgb的参
    param_grid = {
        'learning_rate': [0.3, 0.4, 0.5],
        'num_leaves': [30, 10, 20]
    }

    params = {
        'objective': 'regression',
        'boosting_type': 'gbdt',
        'metric': 'rmse',
        'seed': 0,
        'num_leaves': 30,
        'learning_rate': 0.05,
    }

    numRound = 10000  # 不会过拟合的情况下，可以设大一点
    modelTrain = lgb.train(params, lgbTrain, numRound, valid_sets=lgbEval, early_stopping_rounds=15)

    # 用分出的部分训练集测出的最佳迭代次数在，全体训练集中重新训练
    model = lgb.train(params, lgbAll, modelTrain.best_iteration)
    model.save_model('lgb.model')  # 用于存储训练出的模型

    # print(model.feature_importance()) # 看lgb模型特征得分
    dfFeature = pd.DataFrame()
    dfFeature['featureName'] = model.feature_name()
    dfFeature['score'] = model.feature_importance()
    dfFeature.to_csv('featureImportance.csv')


def lgbPredict(predictInput, modelFile):
    """
    Desc：借助已经跑出的模型，来预测12月借贷总额
    """
    model = lgb.Booster(model_file=modelFile)  # init model

    dfData = pd.read_csv(predictInput)
    # 去掉uid列,uid不是特征
    data = dfData.drop(labels='uid', axis=1)
    data.fillna(0, inplace=True)

    preds = model.predict(data)
    allUid = dfData['uid']  # 一个中括号是series类型，两个中括号是DataFrame类型
    res = pd.DataFrame()
    res['uid'] = allUid
    res['ans'] = preds
    print(res.shape)
    res.to_csv('lgbPredRes.csv', index=False, encoding='utf-8', header=False)


def lgbKFoldTrainModel(inputFile):
    """
    Desc：用lightgbm结合KFold来跑出回归模型，并预测看看
    """
    dfData = pd.read_csv(inputFile)

    # 去掉uid列,uid不是特征
    data = dfData.drop(labels=['uid'], axis=1)

    trainInput = data.fillna(0.0)

    print(trainInput.shape)
    # 11月借贷总额当做训练集结果
    dfAns = pd.read_csv('../t_loan_sum.csv')
    allUserAns = pd.merge(dfData, dfAns, how='outer', on='uid')
    allUserAns.fillna(0.0, inplace=True)
    loanAns = allUserAns['loan_sum']

    params = {
        'objective': 'regression',
        'boosting_type': 'gbdt',
        'metric': 'rmse',
        'seed': 0,
        'num_leaves': 30,
        'learning_rate': 0.05,
    }

    lgbAll = lgb.Dataset(trainInput, loanAns)
    # KFold交叉验证
    kf = KFold(n_splits=5, random_state=0)
    kf.get_n_splits(trainInput)
    print(kf)
    bestIterRecord = []  # 记录每次的最佳迭代次数
    rmseRecord = []  # 记录每次的最佳迭代点的rmse
    numRound = 10000  # 不会过拟合的情况下，可以设大一点
    for trainIndex, testIndex in kf.split(trainInput):
        print("Train Index:", trainIndex, ",Test Index:", testIndex)
        xTrain, xTest = trainInput.iloc[trainIndex], trainInput.iloc[testIndex]
        yTrain, yTest = loanAns.iloc[trainIndex], loanAns.iloc[testIndex]

        lgbTrain = lgb.Dataset(xTrain, yTrain)
        lgbEval = lgb.Dataset(xTest, yTest)
        evalRmse = {}  # 存储实时的rmse结果
        modelTrain = lgb.train(
            params=params,
            train_set=lgbTrain,
            num_boost_round=numRound,
            valid_sets=lgbEval,
            valid_names='get_rmse',
            evals_result=evalRmse,
            early_stopping_rounds=15)

        bestIterRecord.append(modelTrain.best_iteration)
        rmseRecord.append(evalRmse.get('get_rmse').get('rmse')[modelTrain.best_iteration - 1])

    bestIter = int(np.mean(bestIterRecord))  # 利用KFold求出的平均最佳迭代次数

    # 用分出的部分训练集测出的最佳迭代次数在，全体训练集中重新训练
    model = lgb.train(params, lgbAll, bestIter)
    model.save_model('lgb.model')  # 用于存储训练出的模型
    dfFeature = pd.DataFrame()
    dfFeature['featureName'] = model.feature_name()
    dfFeature['score'] = model.feature_importance()
    dfFeature.to_csv('featureImportance.csv')

    predTest = model.predict(trainInput)
    print('mean of rmse : ', np.mean(rmseRecord))
    print('best iteration : ', bestIter)
    print('self rmse : ', np.sqrt(mean_squared_error(loanAns, predTest)))


def makeProbaFeature(trainFile, predInput):
    '''
    Desc: 用lightgbm来跑出概率模型，并预测后充当回归模型的特征
    '''
    dfTrain = pd.read_csv(trainFile)

    # 去掉uid列,uid不是特征
    data = dfTrain.drop(labels=['uid'], axis=1)

    # 八九十月的借贷当做训练集输入
    trainInput = data.fillna(0.0)

    dfPred = pd.read_csv(predInput)
    # 去掉uid列,uid不是特征
    oldPredInput = dfPred.drop(labels='uid', axis=1)
    oldPredInput.fillna(0, inplace=True)

    newTrainInput = dfTrain.copy()
    newPredInput = dfPred.copy()

    # 11月借贷总额当做训练集结果
    dfAns = pd.read_csv('../input/t_loan_sum.csv')
    allUserAns = pd.merge(dfTrain, dfAns, how='outer', on='uid')
    allUserAns.fillna(0.0, inplace=True)
    allUserAns = allUserAns[['uid', 'loan_sum']]

    # 设置二分类阈值
    for threshold in [5, 6, 7]:
        tmpUserAns = allUserAns.copy()
        tmpUserAns['classify'] = tmpUserAns['loan_sum'].apply(lambda x: 0 if x < threshold else 1)
        loanAns = tmpUserAns['classify']

        xTrain, xTest, yTrain, yTest = train_test_split(trainInput, loanAns, test_size=0.2, random_state=50)
        lgbTrain = lgb.Dataset(xTrain, yTrain)
        lgbEval = lgb.Dataset(xTest, yTest)
        lgbAll = lgb.Dataset(trainInput, loanAns)

        params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': {'auc'},
            'num_leaves': 256,
            'learning_rate': 0.1,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': 0,
        }

        numRound = 10000  # 不会过拟合的情况下，可以设大一点
        modelTrain = lgb.train(params, lgbTrain, numRound, valid_sets=lgbEval, early_stopping_rounds=5)

        # 用分出的部分训练集测出的最佳迭代次数在，全体训练集中重新训练
        model = lgb.train(params, lgbAll, modelTrain.best_iteration)
        # print(model.feature_importance()) # 看lgb模型特征得分
        # model.save_model('lgbProba.model') # 用于存储训练出的模型

        # 原来特征放到二分类模型预测出的概率结果, 当新特征用
        preds = model.predict(oldPredInput)
        allUid = dfPred['uid']  # 一个中括号是series类型，两个中括号是DataFrame类型
        predNewFeature = pd.DataFrame()
        predNewFeature['uid'] = allUid
        predNewFeature['classify_' + str(threshold)] = preds
        print(predNewFeature.shape)
        # res.to_csv('lgbProbaAns.csv', index = False, encoding = 'utf-8', header = False)

        # 再跑一遍训练集输入, 得到训练集输入概率特征, 当新特征用
        trainNewAns = model.predict(trainInput)
        trainUid = dfTrain['uid']
        trainNewFeature = pd.DataFrame()
        trainNewFeature['uid'] = trainUid
        trainNewFeature['classify_' + str(threshold)] = trainNewAns

        # 把概率特征加入到新的训练输入和预测输入中，供回归使用
        newTrainInput = pd.merge(newTrainInput, trainNewFeature, how='outer', on='uid')
        newPredInput = pd.merge(newPredInput, predNewFeature, how='outer', on='uid')

    print(newTrainInput.shape)
    print(newPredInput.shape)
    newTrainInput.to_csv('newTrainInput.csv', index=False)
    newPredInput.to_csv('newPredInput.csv', index=False)


if __name__ == "__main__":
    startTime = time.time()

    '''
    dealLoan('../input/t_loan.csv')
    dealOrder('../input/t_order.csv')
    dealClick("../input/t_click.csv")
    makeMergeInput()
    '''

    # lgb跑概率模型并添加概率特征
    makeProbaFeature('trainInput.csv', 'predInput.csv')
    # lgb跑模型和预测, 有概率模型时
    lgbTrainModel('newTrainInput.csv')
    lgbPredict('newPredInput.csv', 'lgb.model')

    costTime = time.time() - startTime
    print("cost time:", costTime, "(s)......")
