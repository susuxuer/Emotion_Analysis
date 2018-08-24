# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

from sklearn.cluster import KMeans

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# df_data_all=pd.read_csv('allToneChange0803.csv')
df_price = pd.read_csv('D:/GFZQ/GFZQ/project/7_30_test/data/priceChange/priceChangeAll0802.csv')
df_similarity = pd.read_csv('D:/GFZQ/GFZQ/project/7_30_test/data/Tone/toneAll0801.csv')
# df_similarity=pd.read_csv('similarity0801.csv');

# print(df_price)
# print(df_similarity)


# df_all_data=pd.merge(df_price,df_similarity)
# print(df_all_data)

# indexs=df_all_data.notnull()
# df_all_data=df_all_data[indexs]


# indexs=df_price.notnull()
# df_prices=df_price[indexs]

# indexs=df_similarity.notnull()
# df_similarity=df_similarity[indexs]


df_price = df_price.fillna(0)  # 异常值先简单填0
df_similarity = df_similarity.fillna(0)

# print(df_price)
# print(df_similarity)
unrelated_price = ['date', 'code', 'short', 'full']
unrelated_sims = ['code', 'short']

valid_index = []

for i in df_price.index:
    label = 1

    for j in df_price.columns:
        if j in unrelated_price:
            continue
        if df_price.ix[i, j] < 0.0001 and df_price.ix[i, j] > -0.0001:
            label = 0
            continue

    for j in df_similarity.columns:
        if j in unrelated_sims:
            continue
        if df_similarity.ix[i, j] < 0.0001 and df_similarity.ix[i, j] > -0.0001:
            label = 0
            continue

    if label == 1:
        valid_index.append(i)

# print(valid_index)
# print(df_price.ix[valid_index])
# print(df_similarity.ix[valid_index])


for i in df_price.columns:
    if i in unrelated_price:
        continue
    # print(i)

    prices_1D = df_price.ix[valid_index, i].tolist()
    prices = []
    for p in prices_1D:
        prices.append([p])

    for j in df_similarity.columns:
        if j in unrelated_sims:
            continue

        # print(j)

        sims_1D = df_similarity.ix[valid_index, j].tolist()
        sims = []
        for s in sims_1D:
            sims.append([s])

        # 计算相关性
        data_combine = [prices_1D, sims_1D]
        array_data_combine = np.array(data_combine)
        p_corr = np.corrcoef(array_data_combine)
        print(array_data_combine)
        print(p_corr)
        print('\n\n')

        #
        #
        #
        # plt.figure()  # 实例化作图变量
        # plt.title(i + j)  # 图像标题
        # plt.xlabel('Tone')  # x轴文本
        # plt.ylabel('PriceChange')  # y轴文本
        # plt.axis([0, 1, -0.4, 0.4])
        # plt.grid(True)  # 是否绘制网格线
        #
        # X = np.mat(sims_1D)
        # m, n = X.shape  # 数据条数和维度
        # K = initial_centroids.shape[0]  # 类数
        # centroids = initial_centroids  # 记录当前类中心
        # previous_centroids = centroids  # 记录上一次类中心
        # idx = np.zeros((m, 1))  # 每条数据属于哪个类

        for i in range(max_iters):  # 迭代次数
            print('迭代计算次数：%d' % (i + 1))
            idx = findClosestCentroids(X, centroids)
            if plot_process:  # 如果绘制图像
                plt = plotProcessKMeans(X, centroids, previous_centroids)  # 画聚类中心的移动过程
                previous_centroids = centroids  # 重置
            centroids = computerCentroids(X, idx, K)  # 重新计算类中心
        if plot_process:  # 显示最终的绘制结果
            plt.show()

        train_p = prices[0:500]
        train_s = sims[0:500]

        test_p = prices[500:1000]
        test_s = sims[500:1000]

        # print(train_p)
        plt.plot(train_s, train_p, 'k.')

        model = LinearRegression()
        model.fit(train_s, train_p)

        predict_test_p = model.predict(test_s)

        plt.plot(test_s, predict_test_p, 'g-')
        plt.show()
