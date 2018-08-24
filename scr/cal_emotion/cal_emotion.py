# -*- coding: utf-8 -*-
__author__ = "suxue"
"""
author = 'Su Xue'
2018/07/03--V1.0
2018/08/01--V2.0
reference:https://github.com/Azure-rong/Review-Helpfulness-Prediction/blob/master/main/Feature%20extraction%20module/Sentiment%20features/Sentiment%20dictionary%20features/pos%20neg%20senti%20dict%20feature.py
"""
import datetime
import text_processing as tp
import numpy as np
import pandas as pd
import  glob
import csv

# 1.读取情感词典和待处理文件

# 情感词典
print("reading...")
posdict = tp.read_lines("G:/project/sentimation_analysis/dict/emotion_dict/pos_all_dict.txt")
negdict = tp.read_lines("G:/project/sentimation_analysis/dict/emotion_dict/neg_all_dict.txt")

# 程度副词词典
mostdict = tp.read_lines('G:/project/sentimation_analysis/dict/degree_dict/most.txt')  # 权值为2
verydict = tp.read_lines('G:/project/sentimation_analysis/dict/degree_dict/very.txt')  # 权值为1.5
moredict = tp.read_lines('G:/project/sentimation_analysis/dict/degree_dict/more.txt')  # 权值为1.25
ishdict = tp.read_lines('G:/project/sentimation_analysis/dict/degree_dict/ish.txt')  # 权值为0.5
insufficientdict = tp.read_lines('G:/project/sentimation_analysis/dict/degree_dict/insufficiently.txt')  # 权值为0.25
inversedict = tp.read_lines('G:/project/sentimation_analysis/dict/degree_dict/inverse.txt')  # 权值为-1

# 情感级别
emotion_level1 = "悲伤。"
emotion_level2 = "愤怒。"
emotion_level3 = "淡定。"
emotion_level4 = "平和。"
emotion_level5 = "喜悦。"

# 情感波动级别
emotion_level6 = "情感波动很小。"
emotion_level7 = "情感波动较大。"


# 2.程度副词处理，根据程度副词的种类不同乘以不同的权值
def match(word, sentiment_value):
    if word in mostdict:
        sentiment_value *= 2.0
    elif word in verydict:
        sentiment_value *= 1.75
    elif word in moredict:
        sentiment_value *= 1.5
    elif word in ishdict:
        sentiment_value *= 1.2
    elif word in insufficientdict:
        sentiment_value *= 0.5
    elif word in inversedict:
        # print "inversedict", word
        sentiment_value *= -1
    return sentiment_value


# 3.情感得分的最后处理，防止出现负数
# Example: [5, -2] →  [7, 0]; [-4, 8] →  [0, 12]
def transform_to_positive_num(poscount, negcount):
    pos_count = 0
    neg_count = 0
    if poscount < 0 and negcount >= 0:
        neg_count += negcount - poscount
        pos_count = 0
    elif negcount < 0 and poscount >= 0:
        pos_count = poscount - negcount
        neg_count = 0
    elif poscount < 0 and negcount < 0:
        neg_count = -poscount
        pos_count = -negcount
    else:
        pos_count = poscount
        neg_count = negcount
    return (pos_count, neg_count)

# 求单条语句的情感倾向总得分
def single_review_sentiment_score(weibo_sent):
    single_review_senti_score = []
    all_word =[]
    cuted_review = tp.cut_sentence(weibo_sent)  # 句子切分，单独对每个句子进行分析
    for sent in cuted_review:
        seg_sent = tp.segmentation(sent)  # 分词

        seg_sent = tp.del_stopwords(seg_sent)[:]
        #print(seg_sent)
        i = 0  # 记录扫描到的词的位置
        s = 0  # 记录情感词的位置
        poscount = 0  # 记录该分句中的积极情感得分
        negcount = 0  # 记录该分句中的消极情感得分
        mark1_count = 0
        mark2_count = 0
        for word in seg_sent:  # 逐词分析
            all_word.append(word)
            if word in posdict:  # 如果是积极情感词
                # print "posword:", word
                poscount += 1  # 积极得分+1
                for w in seg_sent[s:i]:
                    poscount = match(w, poscount)
                # print "poscount:", poscount
                s = i + 1  # 记录情感词的位置变化

            elif word in negdict:  # 如果是消极情感词
                # print "negword:", word
                negcount += 1
                for w in seg_sent[s:i]:
                    negcount = match(w, negcount)
                # print "negcount:", negcount
                s = i + 1
            # 如果是感叹号，表示已经到本句句尾
            # elif word == "！" :
            elif word.encode('UTF-8') == "? " or word.encode('UTF-8') == " ？":
                mark1_count += 1

            elif word.encode('UTF-8') == "！" or word.encode('UTF-8') == "！":
                mark2_count += 1
                for w2 in seg_sent[::-1]:  # 倒序扫描感叹号前的情感词，发现后权值+2，然后退出循环
                    if w2 in posdict:
                        poscount += 1
                        break
                    elif w2 in negdict:
                        negcount += 1
                        break
            i += 1

        #print (negcount)
        single_review_senti_score.append(transform_to_positive_num(poscount, negcount))  # 对得分做最后处理
        #print("poscount,negcount,?, ！", poscount, negcount, mark1_count, mark2_count)
    #return  single_review_senti_score
    #print ("lenth:%d"%(len(all_word)))
    su = len(all_word)
    pos_result, neg_result = 0, 0  # 分别记录积极情感总得分和消极情感总得分
    sentlength = len(single_review_senti_score)
    #wordlength = len(all_word)
    #print ("该回答共有%d 分句,共有%d 个分词" %(sentlength,wordlength))
    #print ("该回答共有%d个词" %wordlength)
    pos_score =[]
    neg_score =[]
    for res1, res2 in single_review_senti_score:  # 每个分句循环累加
        pos_result += res1
        neg_result += res2

    pos_score.append(pos_result)
    neg_score.append(neg_result)

    # print pos_result, neg_result
    result1 = (pos_result - neg_result)     # 简单计算该语句的得分
    result2 =(pos_result + neg_result)
    try:
        result = result1 / result2          #利用林乐模型计算语调
        tone = round(result, 3)
        #print (tone)
        #return result
    except Exception as e:
        tone = 0
        #return result
    res =0
    if tone > 0.0:
        res = 1
    elif tone < 0.0:
        res = 2
    #print ("susu:%d xue:%d" %(pos_result,neg_result))
    return pos_result,neg_result,tone,su

def all_review_sentiment_score(senti_score_list):
    score = []
    for review in senti_score_list:
        score_array = np.array(review)
        Pos = np.sum(score_array[:,0])
        Neg = np.sum(score_array[:,1])
        AvgPos = np.mean(score_array[:,0])
        AvgNeg = np.mean(score_array[:,1])
        StdPos = np.std(score_array[:,0])
        StdNeg = np.std(score_array[:,1])
        score.append([Pos, Neg, AvgPos, AvgNeg, StdPos, StdNeg])
    return score

def get_all_content():
    #abel_dir = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
    all_files = glob.glob(r'D:/GFZQ/GFZQ/xuesu2018/xuesu/*.csv')
    return all_files

def get_wenben(path):
	csvfile = open(path,'r',encoding='UTF-8')
	reader = csv.reader(csvfile)
	return reader

def get_QA(wenben):
    Q_all =[]
    A_all =[]
    for QA in wenben :
        Q_all.append(QA[1])
        A_all.append(QA[2])
    return Q_all,A_all

def set_ourdict(all_files,length):
    Q_all =[]
    A_all =[]
    Q1 = []
    A1 = []
    for i in range(length):
        #print ("正在解析第%d家公司" %i)
        print ("%s"%all_files[i])
        file_path = all_files[i]
        wenben = get_wenben(file_path)
        for QA in wenben :
            Q1.append(QA[1])
            A1.append(QA[2])
        Q_all.append(Q1)
        A_all.append(A1)
    return Q_all,A_all

#计算准确率
def cal_AUC(path1,path2):
    file1 = get_wenben(path1)
    # file2 = get_wenben(path2)
    csvfile = open(path2, 'r',encoding='UTF-8')
    file2 = csv.reader(csvfile)
    right =0
    res2 =[]
    res1 =[]
    for item in file1:
        res1.append(item)
    for item in file2:
        res2.append(item)
    length = len(res2)
    for i in range(length):
        if res1[i][0] == res2[i][1]:
            right += 1
    all_len = length
    auc = float(right/all_len)
    return auc,right,all_len
