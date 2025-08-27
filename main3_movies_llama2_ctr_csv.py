import pickle
import pandas as pd
import os
import math
import torch
import argparse
from transformers import LlamaTokenizer, AdamW
from module4 import ContinuousPromptLearning # 这里要注意！！
from utils3 import DataLoader0, Batchify, now_time, ids2tokens, unique_sentence_percent, \
    feature_detect, feature_matching_ratio, feature_coverage_ratio, feature_diversity

# 根据LLM中index做CTR数据集 movies dataset


data_path = '/data/ggghhh/luyusheng/codeMC/CIKM20-NETE-Datasets/Amazon/MoviesAndTV/reviews.pickle'
data_path_price_brand = '/data/ggghhh/luyusheng/codeMC/CIKM20-NETE-Datasets/Amazon/MoviesAndTV/item.json'
index_dir = '/data/ggghhh/luyusheng/codeMC/CIKM20-NETE-Datasets/Amazon/MoviesAndTV/3'

words = int(5)

reviews = pickle.load(open(data_path, 'rb'))


###############################################################################
# Load data
###############################################################################

print(now_time() + 'Loading data')
bos = '<bos>' # 特殊字符串
eos = '<eos>'
pad = '<pad>'
tokenizer = LlamaTokenizer.from_pretrained('/data/ggghhh/luyusheng/huggingface/transformers/Llama-2-7b-hf', bos_token=bos, eos_token=eos, pad_token=pad) # 分词器，将单词转为对应的id，并转为词向量


# 获得两个字典 这里是DataLoader0不是DataLoader！！
corpus = DataLoader0(data_path,data_path_price_brand, index_dir, tokenizer, words) # 读取数据，加载划分训练集，验证集，测试集，用户特征user2feature，产品特征item2feature

hist_list = []
for review in reviews:

    user_id = corpus.user_dict.entity2idx[review['user']]
    item_id = corpus.item_dict.entity2idx[review['item']]
    if review['rating'] > 3:
        click = 1
    else:
        click = 0

    hist_list.append([user_id, item_id, click])


hist_df = pd.DataFrame(hist_list, columns= ['user_id','item_id','click'] )


hist_df.to_csv('movies_LLM_4reviews.csv',index=False)

