from sklearn.neighbors import NearestNeighbors
from cuml.neighbors import NearestNeighbors as cuNearestNeighbors
import json
import sys

from model.base import CF_KNN
from model.utils import (
    knn_predict_course,
    predict_topic_from_course,
    knn_predict_topic,
    predict
)


class KNN(CF_KNN):
    def __init__(self, hparam):
        super().__init__(hparam)
        self.hparam = hparam

    def fit(self, verbose=True):
        if self.hparam['domain'] == 'course':
            print('fit KNN for course prediction...')
            if self.use_gpu:
                self.model_course = cuNearestNeighbors(
                    n_neighbors=self.hparam['knn_course_k'],
                    algorithm=self.hparam['knn_course_algorithm']
                ).fit(self.user_item_data.toarray())
                distances, indices = self.model_course.kneighbors(
                    self.user_item_data.toarray()
                )
            else:
                self.model_course = NearestNeighbors(
                    n_neighbors=self.hparam['knn_course_k'],
                    algorithm=self.hparam['knn_course_algorithm']
                ).fit(self.user_item_data)
                distances, indices = self.model_course.kneighbors(
                    self.user_item_data
                )

            val_data = json.loads((self.hparam['data_dir']/'valseen.json').read_text())
            user_idx = [self.user_id2user_idx[i['user_id']] for i in val_data]

            pred_course = knn_predict_course(indices, user_idx, self.known_course)
            pred_topic = predict_topic_from_course(
                pred_course, user_idx, self.known_topic, self.course2subgroup
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
            print('fit KNN for topic prediction...')
            if self.use_gpu:
                self.model_topic = cuNearestNeighbors(
                    n_neighbors=self.hparam['knn_topic_k'],
                    algorithm=self.hparam['knn_topic_algorithm']
                ).fit(self.user_item_data.toarray())
                distances, indices = self.model_topic.kneighbors(
                    self.user_item_data.toarray()
                )
            else:
                self.model_topic = NearestNeighbors(
                    n_neighbors=self.hparam['knn_topic_k'],
                    algorithm=self.hparam['knn_topic_algorithm']
                ).fit(self.user_item_data)
                distances, indices = self.model_topic.kneighbors(
                    self.user_item_data
                )
                
            val_data = json.loads((self.hparam['data_dir']/'valseen.json').read_text())
            user_idx = [self.user_id2user_idx[i['user_id']] for i in val_data]

            pred_topic = knn_predict_topic(indices, user_idx, self.known_topic)
            if verbose:
                print('-----------------------------------')
                print('evaluation: ')
                print('(topic)map@50: ', self.metric.mapk_valseen_topic(pred_topic))
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
        distances, indices = self.model_course.kneighbors(
                self.user_item_data
        )
        pred = knn_predict_course(indices, user_idx, self.known_course)

        if output:
            output_dir = self.hparam['pred_dir']/'course'
            output_dir.mkdir(parents=True, exist_ok=True)
            predict(
                result=[[self.idx2course[str(j)] for j in i] for i in pred], 
                path=f'{output_dir}/knn.csv',
                user_id=[i['user_id'] for i in test_data],
                domain='course'
            )
        else:
            self.pred_course_test = pred
        print('complete the process of KNN course prediction!')

    def pred_topic_from_course(self, output=True):
        if self.hparam['domain'] != 'course':
            print('wrong domain')
            sys.exit(1)

        test_data = json.loads((self.hparam['data_dir']/'testseen.json').read_text())
        user_idx = [self.user_id2user_idx[i['user_id']] for i in test_data]

        self.fit(verbose=False)
        distances, indices = self.model_course.kneighbors(
                self.user_item_data
        )
        pred_course = knn_predict_course(indices, user_idx, self.known_course)
        pred_topic = predict_topic_from_course(
                pred_course, user_idx, self.known_topic, self.course2subgroup
        )

        if output:
            output_dir = self.hparam['pred_dir']/'topic'
            output_dir.mkdir(parents=True, exist_ok=True)
            predict(
                result=pred_topic, 
                path=f'{output_dir}/knn_from_course.csv',
                user_id=[i['user_id'] for i in test_data],
                domain='topic'
            )
        else:
            self.pred_topic_test = pred_topic
        print('complete the process of KNN topic prediction(from course)!')

    def pred_topic(self, output=True):
        if self.hparam['domain'] != 'topic':
            print('wrong domain')
            sys.exit(1)

        test_data = json.loads((self.hparam['data_dir']/'testseen.json').read_text())
        user_idx = [self.user_id2user_idx[i['user_id']] for i in test_data]

        self.fit(verbose=False)
        distances, indices = self.model_topic.kneighbors(
                self.user_item_data
        )
        pred = knn_predict_topic(indices, user_idx, self.known_topic)

        if output:
            output_dir = self.hparam['pred_dir']/'topic'
            output_dir.mkdir(parents=True, exist_ok=True)
            predict(
                result=pred, 
                path=f'{output_dir}/knn.csv',
                user_id=[i['user_id'] for i in test_data],
                domain='topic'
            )
        else:
            self.pred_topic_test = pred
        print('complete the process of KNN topic prediction!')