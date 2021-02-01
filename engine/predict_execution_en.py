import os
import logging
import argparse
from tqdm import tqdm, trange
import pymysql
import numpy as np
import torch
from tqdm import tqdm, trange
from sqlalchemy import create_engine
import sqlalchemy
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import AutoModelForSequenceClassification
import pandas as pd
from utils import init_logger, load_tokenizer
from eunjeon import Mecab
import kss
import json
from nltk import pos_tag
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nnsplit import NNSplit

stopwords = stopwords.words('english')
stopwords.extend(['https','http','@'])
stop_list = ['//']

class Predict:
    def __init__(self,keyword,channel):
        self.engine = create_engine(("mysql+pymysql://{}:{}@{}:{}/{}?charset=utf8mb4").format('root','robot369',
                                                                                                '1.221.75.76',3306,'datacast2'))
        self.keyword = keyword
        self.channel = channel
        self.splitter = NNSplit.load("en")

    # def nouns(self,phrase):
    #     """Nouns extractor."""
    #     verbs = ['VV']
    #     tagged = pos_tag(word_tokenize(phrase))
    #     return [s for s, t in tagged if t in verbs]
    #
    # def verbs(self,phrase):
    #     """Verbs extractor."""
    #     verbs = ['VV']
    #     tagged = self.nlp.pos(phrase)
    #     return [s for s, t in tagged if t in verbs]
    #
    # def adjs(self,phrase):
    #
    #     """Adjs extractor."""
    #     adjs = ['VA','IC']
    #     tagged = self.nlp.pos(phrase)
    #     return [s for s, t in tagged if t in adjs]

    def read(self):
        # conn = pymysql.connect(host='1.221.75.76', user='root', password='robot369', database='datacast')
        # curs = conn.cursor(pymysql.cursors.DictCursor)
        # sql_select_sentence = 'select * from analysis_sentence'
        # curs.execute(sql_select_sentence)
        # rows = curs.fetchall()
        ##pandas datatable 형태로 sentece 테이블 읽어들이기

        print('sql:',"SELECT ct.task_id,ct.channel,cc.contents_id,cc.text,cc.url from crawl_task as ct join crawl_contents as cc on ct.task_id=cc.task_id WHERE ct.keyword=\'%s\' and ct.channel =\'%s\'"%(self.keyword,self.channel))
        df_sentence_rows = pd.read_sql("SELECT ct.task_id,ct.channel,cc.contents_id,cc.text,cc.url from crawl_task as ct join crawl_contents as cc on ct.task_id=cc.task_id WHERE ct.keyword=\'%s\' and ct.channel ='\'%s\'"%(self.keyword,self.channel),self.engine)
        print('read finish')
        return df_sentence_rows


    def predict(self):
        chunck_size = 100
        df_cdata_rows = self.read()
        sid = SentimentIntensityAnalyzer()
        for idx,i in enumerate(tqdm(range(0, len(df_cdata_rows), chunck_size),desc="Nlp&Prediction")):
            try:
                df_cdata_chuncked_rows = df_cdata_rows[idx*chunck_size:(idx + 1) * chunck_size]
                ## crawl_contents 를 sentence 로 쪼개고 crawl_sentence 에 넣는 작업
                df_sentence_rows = pd.DataFrame()
                for index in df_cdata_chuncked_rows.index:
                    df_sentence_row= pd.DataFrame()
                    crawl_contents_id = df_cdata_chuncked_rows.at[index,'contents_id']
                    text = df_cdata_chuncked_rows.at[index,'text']
                    text = text.replace("[Music]","")
                    sentences = [str(sentence) for sentence in self.splitter.split([text])[0]]
                    seq = [i for i in range(0, len(sentences))]
                    df_sentence_row['contents_id'] = [crawl_contents_id]*len(sentences)
                    df_sentence_row['text'] = sentences
                    df_sentence_row['seq'] = seq
                    df_sentence_rows = df_sentence_rows.append(df_sentence_row, ignore_index=True)
                # chunk_size = 10000
                # list_df_sentence_rows = [df_sentence_rows[i:i+chunk_size] for i in range(0,df_sentence_rows.shape[0],chunk_size)]
                # for df_sentence_rows_to_read in list_df_sentence_rows:
                    ##모델이 읽을 수 있도록 데이터 형변환(to TensorDataset)
                for idx in df_sentence_rows.index:
                    sentence = df_sentence_rows.at[idx,'text'].lower()
                    nouns = [p[0] for p in pos_tag(word_tokenize(sentence),tagset='universal') if p[1] in ['NOUN']]
                    nouns = list(filter(lambda x: (x not in stopwords) and all(stop not in x for stop in stop_list),nouns))
                    nouns = json.dumps(nouns, ensure_ascii=False)


                    verbs = [p[0] for p in pos_tag(word_tokenize(sentence),tagset='universal') if p[1] in ['VERB']]
                    verbs = list(filter(lambda x: (x not in stopwords) and all(stop not in x for stop in stop_list),verbs))
                    verbs = json.dumps(verbs, ensure_ascii=False)

                    adjs = [p[0] for p in pos_tag(word_tokenize(sentence),tagset='universal') if p[1] in ['ADJ']]
                    adjs = list(filter(lambda x: (x not in stopwords) and all(stop not in x for stop in stop_list),adjs))
                    adjs = json.dumps(adjs, ensure_ascii=False)

                    pos = sid.polarity_scores(sentence)
                    pos = 1 if pos['compound']>=0 else 0


                    df_sentence_rows.at[idx,'nouns']= nouns
                    df_sentence_rows.at[idx, 'verbs'] = verbs
                    df_sentence_rows.at[idx, 'adjs'] = adjs
                    df_sentence_rows.at[idx, 'positiveness'] = pos

                # df_sentence_rows['sentiment_point'] = maximum_probs
                # df_sentence_rows.set_index('sentence_id',inplace=True)
                print(df_sentence_rows)
                ##analysis_sentence 테이블 sentiment update 해주는 작업
                ##가져온 데이터의 primary 키를 u 로 잡아준다.
                # chunk_size = 10000 #chunk row size
                # list_df_sentence_rows = [df_sentence_rows[i:i+chunk_size] for i in range(0,df_sentence_rows.shape[0],chunk_size)]
                # for index,df_sentence_rows in enumerate(list_df_sentence_rows):
                # print('{}번째 chunk'.format(index))
                # df_sentence_rows.to_sql('analysis_sentence_tmp',self.engine.connect(),if_exists='replace',index=False,chunksize=1)

                conn = self.engine.connect()
                trans = conn.begin()

                try:
                    # #delete those rows that we are going to "upsert"
                    # self.engine.execute("DELETE anal_s FROM analysis_sentence AS anal_s, analysis_sentence_tmp AS anal_st WHERE anal_s.sentence_id = anal_st.sentence_id")
                    # print('delete...ing')
                    # trans.commit()

                    #insert changed rows
                    df_sentence_rows.to_sql('crawl_sentence',self.engine,if_exists='append',index=False)
                    print('insert...ing')

                except Exception as e:
                    print(e)
                    trans.rollback()
                    raise
                print('prediction Done')
                conn.close()
            except Exception as e:
                print(e)
                continue