import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import csv
import statsmodels.api as sm

def get_wenben(path):
    csvfile = open(path, 'r', encoding='UTF-8')
    reader = csv.reader(csvfile)
    return reader

def get_all_code(): #获取全部2016举办业绩说明会的企业代码,时间
    path2 = 'D:/GFZQ/GFZQ/project/7_30_test/data/train/2016all_train0731full.csv'
    reader2 = get_wenben(path2)
    return reader2

def get_com_information(): #获取业绩说明会时间，简称，全称，股票代码
    reader2 = get_all_code()
    companies =[]
    for item in reader2:
        companies.append(item)
    return companies


def calT():
    df_single = pd.read_csv('D:/GFZQ/GFZQ/project/7_30_test/data/tone/tone0815Q.csv')
    df_all = pd.read_csv('D:/GFZQ/GFZQ/project/7_30_test/data/tone/tone0815divall.csv')
    code = df_single['code']
    short = df_single['short']
    pos = df_single['pos']
    neg = df_single['neg']
    num = df_all['allRes']
    POS =[]
    NEG =[]
    length = len(pos)
    for i in range(length):
         POS.append(pos[i]/num[i])
         NEG.append(neg[i] / num[i])
    return POS,NEG,code,short

def getPosNeg(pos):
    #df_tone = pd.read_csv('D:/GFZQ/GFZQ/project/7_30_test/data/tone/posneg0815A.csv')
    #pos = df_tone['negA']
    #pos = df_tone['allToneA']
    print (len(pos))
    length  = len(pos)
    print ("MAX:%f"%(max(pos)))
    print ('MIN:%f'%(min(pos)))
    print("MEAN：%f"%(np.mean(pos)))
    print ("MEDIAN：%f"%(np.median(pos)))
    print ("标准差:%f"%(np.std(pos, ddof=1)))

def calPearson():
    df_price = pd.read_csv('D:/GFZQ/GFZQ/project/7_30_test/data/tone/toneAll0801.csv')
    df_similarity = pd.read_csv('D:/GFZQ/GFZQ/project/7_30_test/data/tone/POSNEGQ.csv')

    df_price = df_price.fillna(0)  # 异常值先简单填0
    df_similarity = df_similarity.fillna(0)

    #print(df_price)
    # print(df_similarity)
    unrelated_price = [ 'code', 'short']
    unrelated_sims = ['code', 'short']

    valid_index = []
    for i in df_price.index:
        label = 1
        for j in df_price.columns:
            if j in unrelated_price:
                continue
            if df_price.ix[i, j] < 0.00001 and df_price.ix[i, j] > -0.00001:
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

        prices_1D = df_price.ix[valid_index, i].tolist()
        prices = []
        for p in prices_1D:
            prices.append([p])

        for j in df_similarity.columns:
            if j in unrelated_sims:
                continue
            sims_1D = df_similarity.ix[valid_index, j].tolist()
            sims = []
            for s in sims_1D:
                sims.append([s])

            # 计算相关性
            data_combine = [sims_1D, prices_1D]
            array_data_combine = np.array(data_combine)  # 转换为矩阵mat
            p_corr = np.corrcoef(array_data_combine)
            # print(array_data_combine)
            print (i,j)
            print(p_corr)



if __name__ =='__main__':
    start = time.clock()
    #getPosNeg()
    calPearson()
    #POS, NEG,code,short = calT()
    #getPosNeg(NEG)
    # predictions = ['code','short','POS','NEG']
    # df1 = pd.DataFrame({'code':code,'short':short,'POS':POS,'NEG':NEG},columns=predictions)
    # df1.to_csv('D:/GFZQ/GFZQ/project/7_30_test/data/Tone/POSNEGQ.csv', index=False)
    print ("TimeUse:%s"%(time.clock() - start))