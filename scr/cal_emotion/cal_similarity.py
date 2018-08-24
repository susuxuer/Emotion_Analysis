"""
author:suxue
date:2018/6/20
version:1.0
2018/08/01--V2.0
copyright:suxue3@mail2.sysu.edu.cn
"""
"""
#1.提取问题和答案
#2.分词并对数据做简单清洗
#3.计算tiidf，提取关键词
#4.词袋向量化，对于每一对问答，计算出一个余弦相似度
#5.阈值判断，归一化处理
"""
import pandas as pd
import numpy as np
import csv
import time
import warnings
import gensim
from gensim import corpora, models, similarities
import jieba  # 引入结巴
import jieba.posseg as pseg  # 引入结巴词性标注

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
warnings.filterwarnings(action='ignore', category=UserWarning, module='pandas')

# 获取停用词库
def loadPoorEnt(path2='D:/GFZQ/GFZQ/project/7_30_test/data/model/stopwords.csv'):
    poor_ent = set([])
    # f = open(path, encoding='utf-8')
    with open(path2, 'r', encoding='utf-8') as ropen:
        lines = ropen.readlines()
        for line in lines:
            line = line.replace('\r', '').replace('\n', '')
            poor_ent.add(line)
    return poor_ent
stop_words = loadPoorEnt()

# 1.提取问题和答案
def extra_data(path, top):
    f = open(path, encoding='utf-8')
    trainFile = pd.read_csv(f)
    trainFile = trainFile[:top]
    Questions = trainFile['Question']
    Answers = trainFile['Answer']
    df = pd.DataFrame({})
    return Questions, Answers, trainFile

# 2.分词并做简单清洗
def cut(data):
    result = []  # pos=['n','v']
    for line in data:
        # line = line[0]
        res = pseg.cut(line)
        list = []
        for item in res:
            if item.word not in stop_words:
                list.append(item.word)
        result.append(list)
    return result

# 3.使用TF-IDF对语料库进行建模
def model(all_data):
    processed_corpus = [[token for token in text] for text in all_data]
    dictionary = corpora.Dictionary(processed_corpus)
    corpus = [dictionary.doc2bow(line) for line in processed_corpus]
    tfidf = models.TfidfModel(corpus)
    return tfidf

# 用于将词汇向量化
def list2vec(list, length):
    vector = np.zeros(length)
    for item in list:
        # print (item[0])
        # print (item[1])
        vector[item[0]] = item[1]
    return vector

# 计算余弦值
def cal_cos(vec_a, vec_b):
    # cosin =[]
    num = np.dot(vec_a, vec_b.T)
    denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    cos = num / denom  # 余弦值
    # cosin.append((cos, i))
    # cosin.append(cos)
    return cos

# 计算时间
def cal_time(time):
    if time < 60:
        return str(time) + 'secs '
    elif time < 60 * 60:
        return str(time / (60.0)) + ' mins '
    else:
        return str(time / (60 * 60.0)) + ' hours '

def similarity(Question, Answer):
    raw_data = []
    cut_question = cut(Question)
    cut_answer = cut(Answer)
    #print (cut_answer)
    raw_data = cut_question + cut_answer  # 不知道为啥不能用append和extend，气气气~
    #print("分词完成....")
    # tfidf = model(raw_data)
    processed_corpus = [[token for token in text] for text in raw_data]
    dictionary = corpora.Dictionary(processed_corpus)
    corpus = [dictionary.doc2bow(line) for line in processed_corpus]
    words = []
    for item in corpus:
        for word in item:
            words.append(word)
    tf_idf = models.TfidfModel(corpus)
    corpus_tfidf = tf_idf[corpus]
    lsi = models.LsiModel(corpus)
    corpus_lsi = lsi[corpus_tfidf]

    # ques_tfidf = [tfidf(line) for line in cut_question]
    # ans_tfidf = [tfidf(line1) for line1 in cut_answer]
    tfidfResult = []
    lsiResult =[]
    length = len(words)
    print (length)
    for i in range(len(Question)):
        q1 = dictionary.doc2bow(cut_question[i])
        a1 = dictionary.doc2bow(cut_answer[i])
        vec1 = tf_idf[q1]  # 必须是方括号,TF-IDF 值（前面一项为ID，后面一项是TFIDF值）
        vec2 = tf_idf[a1]
        vec_a = list2vec(vec1, length)
        vec_b = list2vec(vec2, length)
        cosin = cal_cos(vec_a, vec_b)
        vec1_ = lsi[q1]  # 必须是方括号,TF-IDF 值（前面一项为ID，后面一项是TFIDF值）
        vec2_ = lsi[a1]
        vec_a_ = list2vec(vec1_, length)
        vec_b_ = list2vec(vec2_, length)
        cosin_ = cal_cos(vec_a_, vec_b_)
        tfidfResult.append(cosin)
        lsiResult.append(cosin_)
    tfidf0 = sum(tfidfResult)/(len(Question))
    lsi0 = sum(lsiResult)/(len(Question))
    return tfidf0,lsi0

