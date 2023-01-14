import sys
import implicit
from implicit.gpu.als import AlternatingLeastSquares
import json

from model.base import CF_KNN
from model.utils import (
    predict,
    predict_topic_from_course
)


class ALS(CF_KNN):
    def __init__(self, hparam):
        super().__init__(hparam)
    def fit(self, verbose=True):
        if self.hparam['domain'] == 'course':
            if self.use_gpu:
                self.model_course = AlternatingLeastSquares(
                    factors=self.hparam['als_course_factors'],
                    regularization=self.hparam['als_course_regularization'],
                    alpha=self.hparam['als_course_alpha'],
                    iterations=self.hparam['als_course_iterations'],
                    random_state=self.hparam['random_state']
                )
            else:
                self.model_course = implicit.cpu.als.AlternatingLeastSquares(
                    factors=self.hparam['als_course_factors'],
                    regularization=self.hparam['als_course_regularization'],
                    alpha=self.hparam['als_course_alpha'],
                    iterations=self.hparam['als_course_iterations'],
                    random_state=self.hparam['random_state']
                )
            
            val_data = json.loads((self.hparam['data_dir']/'valseen.json').read_text())
            user_idx = [self.user_id2user_idx[i['user_id']] for i in val_data]

            print('fit ALS for course prediction...')
            self.model_course.fit(self.user_item_data)
            pred_course, score_course = self.model_course.recommend(
                user_idx, self.user_item_data[user_idx], N=50
            )
            pred_course_, score_course_ = self.model_course.recommend(
                user_idx, self.user_item_data[user_idx], filter_already_liked_items=False, N=50
            )
            pred_topic = predict_topic_from_course(
                pred_course_, user_idx, self.known_topic, self.course2subgroup
            )
            if verbose:
                print('-----------------------------------')
                print('evaluation: ')
                print('(course)map@50:', self.metric.val_metric(pred_course))
                print('(topic)map@50: ', self.metric.mapk_valseen_topic(pred_topic))
                print('-----------------------------------')

            self.pred_course_val = pred_course
            self.pred_topic_val = pred_topic
        elif self.hparam['domain'] == 'topic':
            if self.use_gpu:
                self.model_topic = AlternatingLeastSquares(
                    factors=self.hparam['als_topic_factors'],
                    regularization=self.hparam['als_topic_regularization'],
                    alpha=self.hparam['als_topic_alpha'],
                    iterations=self.hparam['als_topic_iterations'],
                    random_state=self.hparam['random_state']
                )
            else:
                self.model_topic = implicit.cpu.als.AlternatingLeastSquares(
                    factors=self.hparam['als_topic_factors'],
                    regularization=self.hparam['als_topic_regularization'],
                    alpha=self.hparam['als_topic_alpha'],
                    iterations=self.hparam['als_topic_iterations'],
                    random_state=self.hparam['random_state']
                )
            
            val_data = json.loads((self.hparam['data_dir']/'valseen.json').read_text())
            user_idx = [self.user_id2user_idx[i['user_id']] for i in val_data]

            print('fit ALS for topic prediction...')
            self.model_topic.fit(self.user_item_data)
            pred_topic, score_topic = self.model_topic.recommend(
                user_idx, self.user_item_data[user_idx], filter_already_liked_items=False, N=50
            )
            if verbose:
                print('-----------------------------------')
                print('evaluation: ')
                print('(topic)map@50: ', self.metric.val_metric(pred_topic))
                print('-----------------------------------')

            self.pred_topic_val = pred_topic
        else:
            print('wrong domain')
            sys.exit(1)


    def pred_course(self, output=True):
        if self.hparam['domain'] != 'course':
            print('wrong domain')
            sys.exit(1)

        test_data = json.loads((self.hparam['data_dir']/'testseen.json').read_text())
        user_idx = [self.user_id2user_idx[i['user_id']] for i in test_data]

        self.fit(verbose=False)
        pred, score = self.model_course.recommend(
            user_idx, self.user_item_data[user_idx], N=50
        )

        if output: 
            output_dir = self.hparam['pred_dir']/'course'
            output_dir.mkdir(parents=True, exist_ok=True)
            predict(
                result=[[self.idx2course[str(j)] for j in i] for i in pred], 
                path=f'{output_dir}/als.csv',
                user_id=[i['user_id'] for i in test_data],
                domain='course'
            )
        else:
            self.pred_course_test = pred
        print('complete the process of ALS course prediction!')
    
    def pred_topic_from_course(self, output=True):
        if self.hparam['domain'] != 'course':
            print('wrong domain')
            sys.exit(1)

        test_data = json.loads((self.hparam['data_dir']/'testseen.json').read_text())
        user_idx = [self.user_id2user_idx[i['user_id']] for i in test_data]

        self.fit(verbose=False)
        pred_course, score_course = self.model_course.recommend(
            user_idx, self.user_item_data[user_idx], filter_already_liked_items=False, N=50
        )
        pred_topic = predict_topic_from_course(
            pred_course, user_idx, self.known_topic, self.course2subgroup
        )
        
        if output:
            output_dir = self.hparam['pred_dir']/'topic'
            output_dir.mkdir(parents=True, exist_ok=True)
            predict(
                result=pred_topic, 
                path=f'{output_dir}/als_from_course.csv',
                user_id=[i['user_id'] for i in test_data],
                domain='topic'
            )
        else:
            pred_topic_test = pred_topic
        print('complete the process of ALS topic prediction(from course)!')
    
    def pred_topic(self, output=True):
        if self.hparam['domain'] != 'topic':
            print('wrong domain')
            sys.exit(1)

        test_data = json.loads((self.hparam['data_dir']/'testseen.json').read_text())
        user_idx = [self.user_id2user_idx[i['user_id']] for i in test_data]

        
        self.fit(verbose=False)
        pred, score = self.model_topic.recommend(
            user_idx, self.user_item_data[user_idx], filter_already_liked_items=False, N=50
        )

        if output:
            output_dir = self.hparam['pred_dir']/'topic'
            output_dir.mkdir(parents=True, exist_ok=True)
            predict(
                result=pred, 
                path=f'{output_dir}/als.csv',
                user_id=[i['user_id'] for i in test_data],
                domain='topic'
            )
        else:
            pred_topic_test = pred
        print('complete the process of ALS topic prediction!')