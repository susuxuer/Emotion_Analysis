""""
2018/08/03--V1.0
func:寻找股价波动与情感倾向的关系
"""
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import csv
import math
import pandas as pd
import numpy as np

def readFile(path):
    csvfile = open(path, 'r', encoding='UTF-8')
    reader = csv.reader(csvfile)
    return reader

def getInfomation():
    path1 ='D:/GFZQ/GFZQ/project/7_30_test/data/similarity/allSimilarityChange0803.csv'
    df_similarity = readFile(path1)
    path2 = 'D:/GFZQ/GFZQ/project/7_30_test/data/priceChange/allToneChange0803.csv'
    path ='D:/GFZQ/GFZQ/project/7_30_test/data/similarity/allcat0803.csv'
    df_tone = readFile(path)
    file =[]
    for item in df_tone:      #去除异常值
        if ''!= item[5] and '0.0' !=item[6] and '0.0' !=item[7] and '0.0' !=item[8] and '0.0' !=item[9] and '0.0' !=item[10] and '0.0' !=item[11] and '0.0' !=item[12]:
            file.append(item)
    # for item in file:
    #     print (item)
    # for item in df_tone:      #去除情感异常值
    #     if ''!= item[4] and '0.0' !=item[5] and '0.0' !=item[6] and '0.0' !=item[7] and '0.0' !=item[9] and '0.0' !=item[8] :
    #         file.append(item)
    # for item in file:
    #     print (item)
    return file

def calCorelation(file):
    toneQ = []
    toneA = []
    day5 =[]
    day10 =[]
    day20=[]
    day60=[]
    day120 = []
    day250 =[]
    day364 =[]
    for i in file:
        toneQ.append(i[4])
        toneA.append(i[5])
        day10.append(i[6])
        day5.append(i[7])
        day20.append(i[8])
        day60.append(i[9])
        day120.append(i[10])
        day250.append(i[11])
        day364.append(i[12])
    # # for i in toneQ:
    # #     print (i)
    # s1 = pd.Series(list(day10[2:]))
    # s2 = pd.Series(list((toneA[2:])))
    # data_combine=[s1,s2]
    # #print (type(data_combine))
    # array_data_combine=np.array(data_combine)
    # p_corr=np.corrcoef(array_data_combine)
    # #_corr = s1.corr(s2)
    # print(p_corr)
    return day10,toneA

def cal_():
    df_price = pd.read_csv('D:/GFZQ/GFZQ/project/7_30_test/data/Tone/toneAll0801.csv')

    df_price = df_price.fillna(0)  # 异常值先简单填0

    unrelated_price = ['date', 'code', 'short', 'full']
    unrelated_similarity = ['code', 'short', 'lsi0']
    unrelated_tone = ['code', 'short', 'allToneQ']

def calLine(sims,prices):
    plt.figure()  # 实例化作图变量
    plt.title(sims[0])  # 图像标题
    plt.xlabel('x')  # x轴文本
    plt.ylabel('y')  # y轴文本
    plt.axis([0, 0.2, -0.2, 0.2])
    plt.grid(True)  # 是否绘制网格线

    train_p = prices[0:1000]
    train_s = sims[0:1000]

    test_p = prices[1000:1300]
    test_s = sims[1000:1300]
    # print(train_p)
    plt.plot(train_s, train_p, 'k.')
    model = LinearRegression()
    model.fit(train_s, train_p)
    predict_test_p = model.predict(test_s)

    plt.plot(test_s, predict_test_p, 'g-')
    plt.show()



# 计算特征和类的平均值
def calcMean(x, y):
    sum_x =0
    for i in x:
        sum_x +=float(i)
    sum_y = 0
    for i in y:
        sum_y += float(i)
    print (sum_x)
    # sum_x = sum(x)
    # sum_y = sum(y)
    n = len(x)
    if n == 0:
        return 0, 0
    else:
        x_mean = float(sum_x + 0.0) / n
        y_mean = float(sum_y + 0.0) / n
        return x_mean, y_mean


# 计算Pearson系数
def calcPearson(x, y):
    x_mean, y_mean = calcMean(x, y)  # 计算x,y向量平均值
    n = len(x)
    sumTop = 0.0
    sumBottom = 0.0
    x_pow = 0.0
    y_pow = 0.0
    for i in range(n):
        sumTop += (float(x[i]) - x_mean) * (float(y[i]) - y_mean)
    for i in range(n):
        # x_pow += math.pow(x[i]-x_mean,2)
        x_pow += (float(x[i]) - x_mean) * (float(x[i]) - x_mean)
    for i in range(n):
        # y_pow += math.pow(y[i]-y_mean,2)
        y_pow += (float(y[i])- y_mean) * (float(y[i]) - y_mean)
    sumBottom = math.sqrt(x_pow * y_pow)
    p = sumTop / sumBottom
    return p


if __name__ == "__main__":
    file =getInfomation()
    day10, toneA = calCorelation(file)
    p =calcPearson(day10[1:],toneA[1:])