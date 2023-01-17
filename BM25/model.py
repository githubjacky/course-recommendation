import os, sys
sys.path.append(os.path.abspath(f"{os.getcwd()}"))

from pathlib import Path
import json
import numpy as np
from rank_bm25 import BM25Okapi, BM25L, BM25Plus
from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger

from utils import (
    predict,
    predict_topic_from_course_no_secret
)

from metric import Metric

class bm25():
    def __init__(self, hparam):
        self.idx2int_topic = json.loads(Path('data/model_data/idx2int_topic.json').read_text())
        self.idx2int = json.loads(Path('data/model_data/idx2int.json').read_text())
        self.idx2course = json.loads(Path('data/model_data/idx2course.json').read_text())
        doc = list(json.loads(Path('data/model_data/course_feature.json').read_text()).values())
        if hparam['model'] == 'bm25okapi':
            self.bm25 = BM25Okapi(doc)
        elif hparam['model'] == 'bm25l':
            self.bm25 = BM25L(doc)
        elif hparam['model'] == 'bm25plus':
            self.bm25 = BM25Plus(doc)
        else:
            print('invalid model')
            sys.exit(1)

        self.course2subgroup = json.loads((hparam['data_dir']/'course2subgroup.json').read_text())
        self.metric = Metric('none')
        self.ws_driver  = CkipWordSegmenter(model="bert-base", device=0)
        self.pos_driver  = CkipPosTagger(model="bert-base", device=0)

        if hparam['stage'] == 'fit':
            if hparam['domain'] == 'seen_course' or hparam['domain'] == 'seen_topic':
                self.val_data = json.loads((hparam['data_dir']/'valseen.json').read_text())
            else:
                self.val_data = json.loads((hparam['data_dir']/'valunseen.json').read_text())
        else:
            if hparam['domain'] == 'seen_course' or hparam['domain'] == 'seen_topic':
                self.test_data = json.loads((hparam['data_dir']/'testseen.json').read_text())
            else:
                self.test_data = json.loads((hparam['data_dir']/'testunseen.json').read_text())

        self.hparam = hparam

    def clean(self, ws_res, pos_res):
        clean_instance = []
        for ws_res_i, pos_res_i in zip(ws_res, pos_res):
            clean_sentence = []
            block_pos = set(['Nep', 'Nh', 'Nb'])
            for i, j in zip(ws_res_i, pos_res_i):
                is_noun = j.startswith("N")  # retain noun
                is_not_block_pos = j not in block_pos  # retain some pos
                is_not_one_charactor = not (len(i) == 1)  # kick out one character 
            
                if is_noun and is_not_block_pos and is_not_one_charactor:
                    clean_sentence.append(i)
            clean_instance.append(clean_sentence)

        return clean_instance

    def fit_course(self, data):
        query = []
        for idx, sample in enumerate(data):
            int_topic = sample['interest_topic']
            inte = sample['interest']
            instance = [self.idx2int_topic[str(j)] for j in int_topic] + [self.idx2int[str(j)] for j in inte]
            s = ""
            for j in instance:
                s += j
                s += '_'
            query.append(s.strip('_'))
        
        ws_res = self.ws_driver(query)
        pos_res = self.pos_driver(ws_res, batch_size=128)
        clean_query = self.clean(ws_res, pos_res)

        pred_course = []
        for i in clean_query:
            pred_course.append(
                (self.bm25.get_scores(i).argsort())[::-1][:50]
            )
        return pred_course

    def fit(self, verbose=True):
        print(f'fit {self.hparam["model"]}...')
        pred_course = self.fit_course(self.val_data)
        pred_topic = predict_topic_from_course_no_secret(
            pred_course, self.course2subgroup
        )
        if self.hparam['domain'] == 'seen_course' or self.hparam['domain'] == 'seen_topic':
            course_score = self.metric.mapk_valseen_course(pred_course)
            topic_score = self.metric.mapk_valseen_topic(pred_topic)
        else:
            course_score = self.metric.mapk_valunseen_course(pred_course)
            topic_score = self.metric.mapk_valunseen_topic(pred_topic)
    
        if verbose:
            print('-----------------------------------')
            print('evaluation: ')
            print('(course)map@50:', course_score)
            print('(topic)map@50: ', topic_score)
            print('-----------------------------------')

    def pred_course(self):
        pred = self.fit_course(self.test_data)
        output_dir = self.hparam['pred_dir']/'course'
        output_dir.mkdir(parents=True, exist_ok=True)
        predict(
            result=[[self.idx2course[str(j)] for j in i] for i in pred], 
            path=f'{output_dir}/{self.hparam["model"]}.csv',
            user_id=[i['user_id'] for i in self.test_data],
            domain='course'
        )
        print(f'complete the process of {self.hparam["model"]} course prediction!')

    def pred_topic_from_course(self):
        pred_course = self.fit_course(self.test_data)
        pred = predict_topic_from_course_no_secret(
            pred_course, self.course2subgroup
        )
        output_dir = self.hparam['pred_dir']/'topic'
        output_dir.mkdir(parents=True, exist_ok=True)
        predict(
            result=[[self.idx2course[str(j)] for j in i] for i in pred], 
            path=f'{output_dir}/{self.hparam["model"]}.csv',
            user_id=[i['user_id'] for i in self.test_data],
            domain='course'
        )
        print('complete the process of {self.hparam["model"]} topic prediction!')