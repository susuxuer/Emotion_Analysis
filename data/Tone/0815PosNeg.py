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

def getPosNeg():
    df_tone = pd.read_csv('D:/GFZQ/GFZQ/project/7_30_test/data/tone/tone0815Q.csv')
    pos = df_tone['pos']
    neg = df_tone['neg']
    # num_list  = pos[50:80]
    # num_list1 = neg[50:80]
    # length = len(num_list)
    # name_list = [i for i in range(length)]
    # # num_list = [1.5, 0.6, 7.8, 6]
    # # num_list1 = [1, 2, 3, 1]
    # plt.bar(range(len(num_list)), num_list, label='pos', fc='y')
    # plt.bar(range(len(num_list)), num_list1, bottom=num_list, label='neg', tick_label=name_list, fc='r')
    # plt.legend()
    # plt.show()
    length  = len(pos)
    posTone = []
    negTone = []
    for i in range(length):
        posTone.append(pos[i]/(pos[i]+neg[i]))
        negTone.append(neg[i]/(pos[i]+neg[i]))
    return posTone,negTone

def saveTone(companies,allToneQ,allToneA):
    short = []
    code =[]
    length = len(allToneQ)
    for i in range(length):
        short.append(companies[i][1])
        code.append(companies[i][3])
    predictions = ['code','short','allToneQ','allToneA']
    df1 = pd.DataFrame({'code':code,'short':short,'allToneQ':allToneQ,'allToneA':allToneA},columns=predictions)
    df1.to_csv('D:/GFZQ/GFZQ/project/7_30_test/data/Tone/posneg0815Q.csv', index=False)

if __name__ =='__main__':
    start = time.clock()
    companies = get_com_information()
    posTone, negTone = getPosNeg()
    saveTone(companies, posTone, negTone)
    print (time.clock() - start)