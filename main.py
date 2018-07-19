#coding=UTF-8

"""
func:word2vec计算相似度
date:2018/7/17
"""
import re
import time
import csv
import sys
import os
import gensim.models.word2vec as w2v
import jieba.posseg as pseg
import jieba
import glob

def loadPoorEnt(path2 = 'G:/project/sentimation_analysis/data/stopwords.csv'):
    csvfile = open(path2,encoding='UTF-8')
    stopwords  = [line.strip() for line in csvfile.readlines()]
    return stopwords
stop_words = loadPoorEnt()

def cut(data):
    result=[]    #pos=['n','v']
    res = pseg.cut(data)
    list = []
    for item in res:
        #if item.word not in stop_words and (item.flag == 'n' or item.flag == 'a' or item.flag == 'v'):
        if item.word not in stop_words :
            list.append(item.word)
    result.append(list)
    return result

def get_all_content():
    #abel_dir = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
    all_files = glob.glob(r'D:/GFZQ/GFZQ/xuesu2018/xuesu/*.csv')
    return all_files

def get_wenben(path):
	csvfile = open(path,'r',encoding='UTF-8')
	reader = csv.reader(csvfile)
	return reader

def set_ourdict(all_files,length):
    dict_ask = []
    dict_ans = []

    QA_all =[]
    for i in range(length):
        print ("正在解析第%d家公司" %i)
        print ("%s"%all_files[i])
        file_path = all_files[i]
        wenben = get_wenben(file_path)
        QA1 = []
        for QA in wenben :
            seg_list1 = cut(QA[1])
            seg_list2 = cut (QA[2])
            seg_list =  seg_list1[0] +  seg_list2[0]
            dict_ask.append(seg_list1[0])
            dict_ans.append(seg_list2[0])
            QA1.append(seg_list)
        QA_all.append(QA1)
    return QA_all,QA1,dict_ans, dict_ask

def cal_time(time):
    if time < 60:
        return str(time) + 'secs'
    if time < 60*60:
        return str(time/60.0) + 'mins'
    if time< 60*60*60:
        return str(time/60.0) + 'hours'

if __name__ =='__main__':
    start = time.time()

    # all_files = get_all_content()  # 获取所有文件的路径
    # length = 1800 # len(all_files)
    # print ("统计了%d家公司" %length)
    # QA_all,myQA,answer,question = set_ourdict(all_files,length)
    # print ("分词完成")
    #
    # 初始化word2vec的参数
    features = 400  # 词向量的长度，数值越大，精确度越高，但是消耗时间越长
    min_word_count = 1 # 设置最小词频，低于这个频率的词语会被过滤，不进入后续计算
    context_size = 7  # 设置上下文窗口大小
    # 将分词结果写到本地, 每行是每句话的分词结果，用空格分割
    #
    # # index=1
    # # for item in QA_all:
    # #     print(index)
    # #     print("\n")
    # #     for word in item:
    # #         print(word)
    # #     print("\n")
    # #     index=index+1
    #
    # with open("G:/project/sentimation_analysis/data/corpus.csv", "w", encoding="utf-8") as f:
    #     for item  in QA_all:
    #         for word in item :
    #             f.write("{}\n".format(" ".join(word)))
    # 训练模型
    corpus = w2v.Text8Corpus('G:/project/sentimation_analysis/data/corpus.csv')
    book2vec = w2v.Word2Vec(corpus,
                            sg=1,
                            size=features,
                            min_count=min_word_count,
                            window=context_size)
    end = time.time()
    print('模型训练使用了:%s' %cal_time(end - start))

    # 查看模型结果
    ## 寻找关联词meis
    sim_word = book2vec.wv.most_similar(positive=['折扣'], topn=20)
    print('与折扣相近的词语: ')
    for word in sim_word:
        print (word)


    ## 用函数similarity()查看词语的相似度
    # sim = book2vec.wv.similarity('阿里', '朝鲜')
    # print('阿里和朝鲜的相似度为： ', sim)
