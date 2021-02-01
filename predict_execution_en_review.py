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
from db.almaden import Sql

stopwords = stopwords.words('english')
stopwords.extend(['https','http','@'])
stop_list = ['//']

class Predict:
    def __init__(self,keyword,channel,contents_id):
        self.engine = create_engine(("mysql+pymysql://{}:{}@{}:{}/{}?charset=utf8mb4").format('root','robot369',
                                                                                                    '1.221.75.76',3306,'datacast2'))
        self.db = Sql("datacast2")
        self.keyword = keyword
        self.channel = channel
        self.contents_id = contents_id
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

        print('sql:',
              "SELECT ct.channel,cc.contents_id,cs.text from crawl_task as ct join crawl_contents as cc on ct.task_id=cc.task_id JOIN crawl_sentence AS cs ON cs.contents_id = cc.contents_id "
              "WHERE cc.contents_id=\'%s\' and ct.keyword=\'%s\'" % (self.contents_id, self.keyword))
        df_sentence_rows = pd.read_sql(
            "SELECT ct.keyword,ct.channel,cc.contents_id as contents_id,cs.sentence_id as sentence_id, cs.text as sentence from crawl_task as ct join crawl_contents as cc on ct.task_id=cc.task_id JOIN crawl_sentence AS cs ON cs.contents_id = cc.contents_id "
            "WHERE cc.contents_id=\'%s\' and ct.keyword=\'%s\'" % (
            self.contents_id,self.keyword),
            self.engine)
        return df_sentence_rows


    def predict(self):
        df_sentence_data_rows = self.read()
        sid = SentimentIntensityAnalyzer()
        for idx in tqdm(df_sentence_data_rows.index,desc="sentence_anlysis&db_update"):
            try:
                sentence_id = df_sentence_data_rows.at[idx,'sentence_id']
                sentence = df_sentence_data_rows.at[idx,'sentence']
                korean = re.compile('[\u3131-\u3163\uac00-\ud7a3]+')
                sentence = re.sub(korean,"",sentence)
                sentence = sentence.lower()
                nouns = [p[0] for p in pos_tag(word_tokenize(sentence), tagset='universal') if p[1] in ['NOUN']]
                nouns = list(filter(lambda x: (x not in stopwords) and all(stop not in x for stop in stop_list), nouns))
                nouns = json.dumps(nouns, ensure_ascii=False)

                verbs = [p[0] for p in pos_tag(word_tokenize(sentence),tagset='universal') if p[1] in ['VERB']]
                verbs = list(filter(lambda x: (x not in stopwords) and all(stop not in x for stop in stop_list),verbs))
                verbs = json.dumps(verbs, ensure_ascii=False)

                adjs = [p[0] for p in pos_tag(word_tokenize(sentence),tagset='universal') if p[1] in ['ADJ']]
                adjs = list(filter(lambda x: (x not in stopwords) and all(stop not in x for stop in stop_list),adjs))
                adjs = json.dumps(adjs, ensure_ascii=False)

                pos = sid.polarity_scores(sentence)
                pos = 1 if pos['compound']>=0 else 0

                self.db.update_multi_column("crawl_sentence",
                                            update_dict={"nouns": nouns, "verbs": verbs, "adjs": adjs,
                                                         "positiveness": float(pos)},
                                            where_dict={"sentence_id": float(sentence_id)})
            except Exception as e:
                print(e)
                continue