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
# print(df_price.ix[valid_index])             #清除掉有0的数据
# print(df_similarity.ix[valid_index])

for j in df_similarity.columns:
    if j in unrelated_sims:
        continue
    print(j)
