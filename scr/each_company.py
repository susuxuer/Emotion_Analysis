# -*- coding: utf-8 -*-
"""
author:suxue
date:2018/6/18
version-1.0

"""
import pandas as pd
import requests
import json
import time
import csv


"""
#获取每一家公司的业绩说明会基本信息
#title:业绩说明会标题
#speaktime:说明会举行时间
#speakcontent:说明会基本内容
"""
#获取所有的公司业绩说明会基本信息
def all_companies(path,top):
    trainfile = pd.read_csv(path)
    trainfile = trainfile[:top]
    PIDs = trainfile['company_PIDs']
    DATAs = trainfile['show_datas']
    TITLEs = trainfile['show_titles']
    return PIDs,DATAs,TITLEs

#获取每一家公司的业绩说明会问答内容
def get_each_company(page_nums,pid):      #num = pages,PID,公司ID
    #用于存储数据
    Questions = []
    Answers = []
    header = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.162 Safari/537.36",
            }
    url = "http://rs.p5w.net/roadshowLive/getNInteractionDatas.shtml"
    for page in range(page_nums):
            data = {
                    "roadshowId":  pid,                              #pid of each company
                    #"roadshowId": "0001454A03D2B32540C59F628FD99E6B0ED8",
                    #"roadshowId": "0001E0970E14E41E456691CBD9BE3261FCA5",
                    "isPagination": 0,                               # each value can success
                    "type": 2,                                       #1 reprents all, 2 reprents QA, we must set 2
                    "page": page,                                      #page location
                    "rows": 10,
                    }
            r = requests.post(url=url, headers=header, data=data)
            web_data = json.loads(r.text)
            #print (web_data)
            for i in web_data['rows']:
                #print(i.keys())
                #print ('Question:')
                Questions.append(i['speakContent'])
                Answers.append(i['replyList'][0]['speakContent'])
            print ("第%d页数据已抓取完成...." % (page+1))
            time.sleep(5)                           #延迟设置，防止被禁止
    return Questions,Answers

if __name__ == '__main__':
    path = './companies.csv'
    top =2000# 10                                        #小数量用于测试
    PIDs, DATAs, TITLEs = all_companies(path,top)
    #提取每一次业绩说明会的内容
    page_nums = 15
    for i in range(len(PIDs)) :
        Questions, Answers = get_each_company(page_nums, PIDs[i])
        question_column = pd.Series(Questions, name='Question')
        answer_column = pd.Series(Answers, name='Answer')
        predictions = pd.concat([question_column, answer_column], axis=1)
        df1 = pd.DataFrame({'Question': Questions, 'Answer': Answers})
        df1.to_csv('./%s.csv' %TITLEs[i])
        print("finish第%d家公司" % (i + 1))


    # Questions, Answers = get_each_company()
    # question_column = pd.Series(Questions,name='Question')
    # answer_column = pd.Series(Answers,name='Answer')
    # predictions = pd.concat([question_column, answer_column], axis=1)
    # df1 = pd.DataFrame({'Question':Questions ,'Answer':Answers})
    # df1.to_csv('questions.csv')
