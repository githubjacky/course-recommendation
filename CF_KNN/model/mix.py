import sys
import json
from CF_KNN.model.base import CF_KNN
from CF_KNN.model.als import ALS
from CF_KNN.model.bpr import BPR
from CF_KNN.model.knn import KNN
from utils import(
    mix2_rearrange,
    mix3_rearrange,
    predict,
    predict_topic_from_course
)


class MIX2(CF_KNN):
    def __init__(self, hparam):
        super().__init__(hparam)
        if hparam['mix_base'] == 'als' and hparam['mix_reference1'] == 'bpr':
            self.base = ALS(hparam)
            self.ref = BPR(hparam)
        elif hparam['mix_base'] == 'als' and hparam['mix_reference1']== 'knn':
            self.base = ALS(hparam)
            self.ref = KNN(hparam)
        elif hparam['mix_base'] == 'bpr' and hparam['mix_reference1'] == 'als':
            self.base = BPR(hparam)
            self.ref = ALS(hparam)
        elif hparam['mix_base'] == 'bpr' and hparam['mix_reference1'] == 'knn':
            self.base = BPR(hparam)
            self.ref = KNN(hparam)
        elif hparam['mix_base'] == 'knn' and hparam['mix_reference1'] == 'als':
            self.base = KNN(hparam)
            self.ref = ALS(hparam)
        elif hparam['mix_base'] == 'knn' and hparam['mix_reference1'] == 'bpr':
            self.base = KNN(hparam)
            self.ref = BPR(hparam)
        else:
            print('invalid base model for MIX2 model')
            sys.exit(1)
        self.hparam = hparam

    def fit(self, verbose=True):
        self.base.fit(verbose=False)
        self.ref.fit(verbose=False)

        if self.hparam['domain'] == 'course':
            print(f'fit {self.hparam["mix_base"]}_{self.hparam["mix_reference1"]} for course prediction')
            pred_course = mix2_rearrange(
                self.base.pred_course_val,
                self.ref.pred_course_val
            )
            pred_topic = mix2_rearrange(
                self.base.pred_topic_val,
                self.ref.pred_topic_val
            )
            if verbose:
                print('-----------------------------------')
                print('evaluation: ')
                print('(course)map@50:', self.metric.val_metric(pred_course))
                print('(topic)map@50: ', self.metric.mapk_valseen_topic(pred_topic))
                print('-----------------------------------')
        else:
            print(f'fit {self.hparam["mix_base"]}_{self.hparam["mix_reference1"]} for topic prediction')
            pred_topic = mix2_rearrange(
                self.base.pred_topic_val,
                self.ref.pred_topic_val
            )
            if verbose:
                print('-----------------------------------')
                print('evaluation: ')
                print('(topic)map@50: ', self.metric.mapk_valseen_topic(pred_topic))
                print('-----------------------------------')

    def pred_course(self):
        test_data = json.loads((self.hparam['data_dir']/'testseen.json').read_text())
        model_name = f'{self.hparam["mix_base"]}_{self.hparam["mix_reference1"]}'

        self.base.pred_course(output=False)
        self.ref.pred_course(output=False)
        print(f'fit {model_name} for course prediction...')
        pred = mix2_rearrange(
            self.base.pred_course_test,
            self.ref.pred_course_test
        )
        
        output_dir = self.hparam['pred_dir']/'course'
        output_dir.mkdir(parents=True, exist_ok=True)
        predict(
            result=[[self.idx2course[str(j)] for j in i] for i in pred], 
            path=f'{output_dir}/{model_name}.csv',
            user_id=[i['user_id'] for i in test_data],
            domain='course'
        )
        print(f'complete the process of {model_name} course prediction!')

    def pred_topic_from_course(self):
        test_data = json.loads((self.hparam['data_dir']/'testseen.json').read_text())
        model_name = f'{self.hparam["mix_base"]}_{self.hparam["mix_reference1"]}'

        self.base.pred_topic_from_course(output=False)
        self.ref.pred_topic_from_course(output=False)
        print(f'fit {model_name} for course prediction...')
        pred = mix2_rearrange(
            self.base.pred_topic_test,
            self.ref.pred_topic_test
        )

        output_dir = self.hparam['pred_dir']/'course'
        output_dir.mkdir(parents=True, exist_ok=True)
        predict(
            result=[[self.idx2course[str(j)] for j in i] for i in pred], 
            path=f'{output_dir}/{model_name}.csv',
            user_id=[i['user_id'] for i in test_data],
            domain='course'
        )
        print(f'complete the process of {model_name} course prediction!')

    def pred_topic(self):
        test_data = json.loads((self.hparam['data_dir']/'testseen.json').read_text())
        model_name = f'{self.hparam["mix_base"]}_{self.hparam["mix_reference1"]}'

        self.base.pred_topic(output=False)
        self.ref.pred_topic(output=False)
        print(f'fit {model_name} for topic prediction...')
        pred = mix2_rearrange(
            self.base.pred_topic_test,
            self.ref.pred_topic_test
        )

        output_dir = self.hparam['pred_dir']/'course'
        output_dir.mkdir(parents=True, exist_ok=True)
        predict(
            result=[[self.idx2course[str(j)] for j in i] for i in pred], 
            path=f'{output_dir}/{model_name}.csv',
            user_id=[i['user_id'] for i in test_data],
            domain='course'
        )
        print(f'complete the process of {model_name} course prediction!')


class MIX3(CF_KNN):
    def __init__(self, hparam):
        super().__init__(hparam)
        if hparam['mix_base']== 'als':
            self.base = ALS(hparam)
            self.ref1 = BPR(hparam)
            self.ref2 = KNN(hparam)
        elif hparam['mix_base'] == 'bpr':
            self.base = BPR(hparam)
            self.ref1 = ALS(hparam)
            self.ref2 = KNN(hparam)
        elif hparam['mix_base'] == 'knn':
            self.base = KNN(hparam)
            self.ref1 = ALS(hparam)
            self.ref2 = BPR(hparam)
        else:
            print('invalid base model for MiX3 model')
            sys.exit(1)
        self.hparam = hparam

    def fit(self, verbose=True):
        self.base.fit(verbose=False)
        self.ref1.fit(verbose=False)
        self.ref2.fit(verbose=False)

        if self.hparam['domain'] == 'course':
            print(f'fit {self.hparam["mix_base"]}_{self.hparam["mix_reference1"]}_{self.hparam["mix_reference2"]} for course prediction')
            pred_course = mix3_rearrange(
                self.base.pred_course_val,
                self.ref1.pred_course_val,
                self.ref2.pred_course_val
            )
            pred_topic = mix3_rearrange(
                self.base.pred_topic_val,
                self.ref1.pred_topic_val,
                self.ref2.pred_topic_val
            )
            if verbose:
                print('-----------------------------------')
                print('evaluation: ')
                print('(course)map@50:', self.metric.val_metric(pred_course))
                print('(topic)map@50: ', self.metric.mapk_valseen_topic(pred_topic))
                print('-----------------------------------')
        else:
            print(f'fit {self.hparam["mix_base"]}_{self.hparam["mix_reference1"]}_{self.hparam["mix_reference2"]} for topic prediction')
            pred_topic = mix3_rearrange(
                self.base.pred_topic_val,
                self.ref1.pred_topic_val,
                self.ref2.pred_topic_val
            )
            if verbose:
                print('-----------------------------------')
                print('evaluation: ')
                print('(topic)map@50: ', self.metric.mapk_valseen_topic(pred_topic))
                print('-----------------------------------')

    def pred_course(self):
        test_data = json.loads((self.hparam['data_dir']/'testseen.json').read_text())
        model_name = f'{self.hparam["mix_base"]}_{self.hparam["mix_reference1"]}_{self.hparam["mix_reference2"]}'

        self.base.pred_course(output=False)
        self.ref1.pred_course(output=False)
        self.ref2.pred_course(output=False)
        print(f'fit {model_name} for course prediction...')
        pred = mix3_rearrange(
            self.base.pred_course_test,
            self.ref1.pred_course_test,
            self.ref2.pred_course_test
        )
        
        output_dir = self.hparam['pred_dir']/'course'
        output_dir.mkdir(parents=True, exist_ok=True)
        predict(
            result=[[self.idx2course[str(j)] for j in i] for i in pred], 
            path=f'{output_dir}/{model_name}.csv',
            user_id=[i['user_id'] for i in test_data],
            domain='course'
        )
        print(f'complete the process of {model_name} course prediction!')

    def pred_topic_from_course(self):
        test_data = json.loads((self.hparam['data_dir']/'testseen.json').read_text())
        model_name = f'{self.hparam["mix_base"]}_{self.hparam["mix_reference1"]}_{self.hparam["mix_reference2"]}'

        self.base.pred_topic_from_course(output=False)
        self.ref1.pred_topic_from_course(output=False)
        self.ref2.pred_topic_course(output=False)
        print(f'fit {model_name} for course prediction...')
        pred = mix3_rearrange(
            self.base.pred_topic_test,
            self.ref1.pred_topic_test,
            self.ref2.pred_topic_test
        )
        
        output_dir = self.hparam['pred_dir']/'course'
        output_dir.mkdir(parents=True, exist_ok=True)
        predict(
            result=[[self.idx2course[str(j)] for j in i] for i in pred], 
            path=f'{output_dir}/{model_name}.csv',
            user_id=[i['user_id'] for i in test_data],
            domain='course'
        )
        print(f'complete the process of {model_name} course prediction!')

    def pred_topic(self):
        test_data = json.loads((self.hparam['data_dir']/'testseen.json').read_text())
        model_name = f'{self.hparam["mix_base"]}_{self.hparam["mix_reference1"]}_{self.hparam["mix_reference2"]}'

        self.base.pred_topic(output=False)
        self.ref1.pred_topic(output=False)
        self.ref2.pred_topic(output=False)
        print(f'fit {model_name} for topic prediction...')
        pred = mix3_rearrange(
            self.base.pred_topic_test,
            self.ref1.pred_topic_test,
            self.ref2.pred_topic_test
        )
        
        output_dir = self.hparam['pred_dir']/'course'
        output_dir.mkdir(parents=True, exist_ok=True)
        predict(
            result=[[self.idx2course[str(j)] for j in i] for i in pred], 
            path=f'{output_dir}/{model_name}.csv',
            user_id=[i['user_id'] for i in test_data],
            domain='course'
        )
        print(f'complete the process of {model_name} course prediction!')