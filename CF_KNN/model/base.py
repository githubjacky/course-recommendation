import numpy as np
import json
import torch
from scipy.sparse import csr_matrix


from model.metric import Metric


class CF_KNN():
    def __init__(self, hparam):
        self.use_gpu = True if torch.cuda.is_available() else False

        train_data = json.loads((hparam['data_dir']/'train.json').read_text())
        if hparam['domain'] == 'course':
            num_record = 0
            row_ind = []
            col_ind = []
            user_id2user_idx = dict()
            known_topic = []
            known_course = []
            for idx, sample in enumerate(train_data):
                num_purch = len(sample['course'])
                num_record += num_purch
                row_ind += [idx] * num_purch
                col_ind += sample['course']
                user_id2user_idx[sample['user_id']] = idx
                known_topic.append(sample['topic'])
                known_course.append(sample['course'])

            self.user_id2user_idx = user_id2user_idx
            self.known_topic = known_topic
            self.known_course = known_course
            self.user_item_data = csr_matrix(
                (np.ones(num_record),
                (row_ind, col_ind)),
                shape=(hparam['num_train_sample'], hparam['num_course'])
            )
            self.metric = Metric('seen_course')
        elif hparam['domain'] == 'topic':
            num_record = 0
            row_ind = []
            col_ind = []
            user_id2user_idx = dict()
            known_topic = []
            for idx, sample in enumerate(train_data):
                num_purch = len(sample['topic'])
                num_record += num_purch
                row_ind += [idx] * num_purch
                col_ind += sample['topic']
                user_id2user_idx[sample['user_id']] = idx
                known_topic.append(sample['topic'])
            self.user_id2user_idx = user_id2user_idx
            self.known_topic = known_topic
            self.user_item_data = csr_matrix(
                (np.ones(num_record),
                (row_ind, col_ind)),
                shape=(hparam['num_train_sample'], hparam['num_group'])
            )
            self.metric = Metric('seen_topic')
        else:
            print('invalid domain')

        self.hparam = hparam
        self.idx2course = json.loads((hparam['data_dir']/'idx2course.json').read_text())
        self.course2subgroup = json.loads((hparam['data_dir']/'course2subgroup.json').read_text())