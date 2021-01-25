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


logger = logging.getLogger(__name__)

from transformers import (
    BertConfig,
    DistilBertConfig,
    ElectraConfig,
    BertTokenizer,
    ElectraTokenizer,
    BertForSequenceClassification,
    DistilBertForSequenceClassification,
    ElectraForSequenceClassification
)
from tokenization_kobert import KoBertTokenizer

MODEL_CLASSES = {
    'kobert': (BertConfig, BertForSequenceClassification, KoBertTokenizer),
    'distilkobert': (DistilBertConfig, DistilBertForSequenceClassification, KoBertTokenizer),
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'kobert-lm': (BertConfig, BertForSequenceClassification, KoBertTokenizer),
    'koelectra-base': (ElectraConfig, ElectraForSequenceClassification, ElectraTokenizer),
    'koelectra-small': (ElectraConfig, ElectraForSequenceClassification, ElectraTokenizer),
}

class Pred_config:
    def __init__(self,
                 model_dir='./model',
                 batch_size=64,no_cuda=False):
        self.model_dir = model_dir
        self.batch_size = batch_size
        self.no_cuda = no_cuda
        self.args = self.get_args()
    def load_tokenizer(self):
        return MODEL_CLASSES[self.args.model_type][2].from_pretrained(self.args.model_name_or_path)

    def get_device(self):
        return "cuda" if torch.cuda.is_available() and not self.no_cuda else "cpu"

    def get_args(self):
        return torch.load(os.path.join(self.model_dir, 'training_args.bin'))

    def load_model(self, args, device):
        # Check whether model exists
        if not os.path.exists(self.model_dir):
            raise Exception("Model doesn't exists! Train first!")

        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                args.model_dir)  # Config will be automatically loaded from model_dir
            model.to(device)
            model.eval()
            logger.info("***** Model Loaded *****")
        except:
            raise Exception("Some model files might be missing...")

        return model

class Predict:
    def __init__(self, pred_config:Pred_config,task_id = None, keyword=None,channel=None):
        self.pred_config = pred_config
        self.engine = create_engine(("mysql+pymysql://{}:{}@{}:{}/{}?charset=utf8mb4").format('root','robot369',
                                                                                                '1.221.75.76',3306,'datacast2'))
        self.args = self.pred_config.get_args()

        ##쿠다, cpu 중 사용할 디바이스 설정
        self.device = self.pred_config.get_device()

        ##배치사이즈 설정(모델이 한번에 처리할 input 사이즈 크기)
        self.batch_size = self.pred_config.batch_size

        ##모델 가져오기
        self.model = self.pred_config.load_model(self.args, self.device)

        ##토크나이저 가져오기
        self.tokenizer = self.pred_config.load_tokenizer()
        self.nlp = Mecab()
        self.task_id = task_id
        self.keyword = keyword
        self.channel = channel
    def verbs(self,phrase):
        """Verbs extractor."""
        verbs = ['VV']
        tagged = self.nlp.pos(phrase)
        return [s for s, t in tagged if t in verbs]

    def adjs(self,phrase):

        """Adjs extractor."""
        adjs = ['VA','IC']
        tagged = self.nlp.pos(phrase)
        return [s for s, t in tagged if t in adjs]

    def read(self):
        # conn = pymysql.connect(host='1.221.75.76', user='root', password='robot369', database='datacast')
        # curs = conn.cursor(pymysql.cursors.DictCursor)
        # sql_select_sentence = 'select * from analysis_sentence'
        # curs.execute(sql_select_sentence)
        # rows = curs.fetchall()
        ##pandas datatable 형태로 sentece 테이블 읽어들이기

        print('sql:',"SELECT ct.task_id,ct.channel,cc.contents_id,cc.text,cc.url from crawl_task as ct join crawl_contents as cc on ct.task_id=cc.task_id WHERE ct.task_id=%s and ct.keyword=\'%s\' and ct.channel !='navershopping'"%(self.task_id,self.keyword))
        df_sentence_rows = pd.read_sql("SELECT ct.task_id,ct.channel,cc.contents_id,cc.text,cc.url from crawl_task as ct join crawl_contents as cc on ct.task_id=cc.task_id WHERE ct.task_id=%s and ct.keyword=\'%s\' and ct.channel !='navershopping'"%(self.task_id,self.keyword),self.engine)
        print('read finish')
        return df_sentence_rows

    def convert_input_sentence_to_tensor_dataset(self,df_sentence_rows,cls_token_segment_id=0,
                                             pad_token_segment_id=0,
                                             sequence_a_segment_id=0,
                                             mask_padding_with_zero=True):
        tokenizer = self.tokenizer
        args = self.args


        cls_token = tokenizer.cls_token
        sep_token = tokenizer.sep_token
        pad_token_id = tokenizer.pad_token_id

        all_input_ids = []
        all_attention_mask = []
        all_token_type_ids = []

        ###input file 읽어들이기
        ###input file 읽어서 tensordata type 으로 변환
        for index in df_sentence_rows.index:
            sentence = df_sentence_rows.at[index, 'text']

            tokens = tokenizer.tokenize(sentence)

            # Account for [CLS] and [SEP]
            special_tokens_count = 2
            #문장의 최대길이 보다 큰 문장에 대해서 길이 조정을 해준다.
            if len(tokens) > args.max_seq_len - special_tokens_count:
                tokens = tokens[:(args.max_seq_len - special_tokens_count)]

            # Add [SEP] token
            tokens += [sep_token]
            token_type_ids = [sequence_a_segment_id] *len(tokens)

            # Add [CLS] token
            tokens = [cls_token] + tokens
            token_type_ids = [cls_token_segment_id] + token_type_ids
            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 real tokens and 0 for padding tokens. Only real tokens are attended to.
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
            # Zero-pad up to the sequence length.
            padding_length = args.max_seq_len - len(input_ids)
            input_ids = input_ids+([pad_token_id] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

            all_input_ids.append(input_ids)
            all_attention_mask.append(attention_mask)
            all_token_type_ids.append(token_type_ids)

        # Change to Tensor
        all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
        all_attention_mask = torch.tensor(all_attention_mask, dtype=torch.long)
        all_token_type_ids = torch.tensor(all_token_type_ids, dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids)
        return dataset

    def predict(self):
        ##tuning 시 파라미터 정보가 들어있는 파일(training_args.bin)
        args = self.args

        ##쿠다, cpu 중 사용할 디바이스 설정
        device = self.device

        ##배치사이즈 설정(모델이 한번에 처리할 input 사이즈 크기)
        batch_size= self.batch_size

        ##모델 가져오기
        model = self.model
        logger.info(args)

        ##감성분석할 데이터 가져오기
        chunck_size = 100
        df_cdata_rows = self.read()
        for idx,i in enumerate(tqdm(range(0, len(df_cdata_rows), chunck_size),desc="Nlp&Prediction")):
            try:
                df_cdata_chuncked_rows = df_cdata_rows[idx*chunck_size+1:(idx + 1) * chunck_size]
                ## crawl_contents 를 sentence 로 쪼개고 crawl_sentence 에 넣는 작업
                df_sentence_rows = pd.DataFrame()
                for index in df_cdata_chuncked_rows.index:
                    df_sentence_row= pd.DataFrame()
                    crawl_contents_id = df_cdata_chuncked_rows.at[index,'contents_id']
                    text = df_cdata_chuncked_rows.at[index,'text']
                    sentences = kss.split_sentences(text)
                    seq = [i for i in range(0, len(sentences))]
                    df_sentence_row['contents_id'] = [crawl_contents_id]*len(sentences)
                    df_sentence_row['text'] = sentences
                    df_sentence_row['seq'] = seq
                    df_sentence_rows = df_sentence_rows.append(df_sentence_row, ignore_index=True)
                # chunk_size = 10000
                # list_df_sentence_rows = [df_sentence_rows[i:i+chunk_size] for i in range(0,df_sentence_rows.shape[0],chunk_size)]
                # for df_sentence_rows_to_read in list_df_sentence_rows:
                    ##모델이 읽을 수 있도록 데이터 형변환(to TensorDataset)
                dataset = self.convert_input_sentence_to_tensor_dataset(df_sentence_rows)

                # dataset 을 model 을 이용하여 output 도출
                # Predict
                sampler = SequentialSampler(dataset)
                data_loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
                preds = None
                probs = None
                print(type(data_loader),len(data_loader))
                for index,batch in enumerate(data_loader):
                    batch = tuple(t.to(device) for t in batch)
                    with torch.no_grad():
                        inputs = {"input_ids": batch[0],
                                  "attention_mask": batch[1],
                                  "labels": None}
                        if args.model_type != "distilkobert":
                            inputs["token_type_ids"] = batch[2]
                        outputs = model(**inputs)
                        logits = outputs[0]


                        if preds is None:
                            preds = logits.detach().cpu().numpy()
                            probs = np.exp(logits.detach().cpu().numpy())/ (1 + np.exp(logits.detach().cpu().numpy()))
                        else:
                            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                            probs = np.append(probs, np.exp(logits.detach().cpu().numpy())/ (1 + np.exp(logits.detach().cpu().numpy())), axis=0)
                print(len(preds),len(probs))
                preds = np.argmax(preds, axis=1).tolist()
                prob_max_index = np.argmax(probs, axis=-1)
                maximum_probs = probs[np.arange(probs.shape[0]), prob_max_index]
                # maximum_probs = maximum_probs.tolist()
                # maximum_probs = list([round(maximum_prob,2) if pred==1 else round(maximum_prob,2)*(-1) for pred,maximum_prob in zip(preds,maximum_probs)])
                for idx in df_sentence_rows.index:
                    sentence = df_sentence_rows.at[idx,'text']
                    nouns = list(set(self.nlp.nouns(sentence)))
                    nouns = json.dumps(nouns,ensure_ascii=False)

                    verbs = list(set(self.verbs(sentence)))
                    verbs = json.dumps(verbs,ensure_ascii=False)

                    adjs = list(set(self.adjs(sentence)))
                    adjs = json.dumps(adjs,ensure_ascii=False)

                    df_sentence_rows.at[idx,'nouns']= nouns
                    df_sentence_rows.at[idx, 'verbs'] = verbs
                    df_sentence_rows.at[idx, 'adjs'] = adjs
                df_sentence_rows['positiveness'] = preds
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


                logger.info("Prediction Done!")
                conn.close()
            except Exception as e:
                print(e)
                continue