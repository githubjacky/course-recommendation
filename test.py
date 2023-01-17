import numpy as np
import csv
from CF_KNN.model.als import ALS
from CF_KNN.model.bpr import BPR
from CF_KNN.model.knn import KNN
from CF_KNN.model.mix import MIX2, MIX3
from CF_KNN.hparam import parse_hparam


if __name__ == '__main__':
    hparam = vars(parse_hparam())
    hparam['domain'] = 'topic'
    # hparam['bpr_topic_factors'] = 250
    # hparam['bpr_topic_lr'] = 0.000622
    # hparam['bpr_topic_regularizations'] = 51.02939 
    # hparam['als_topic_factors'] = 150
    k = [int(i) for i in np.linspace(100, 2000, 100)]
    # factors = [int(i) for i in np.linspace(10, 500, 50)]
    # lrs= np.linspace(1e-5, 1e-2, 50)
    course_score_val = []
    topic_score_val = []
    # for i in lrs:
    for i in k:
        # or j in factors:
        hparam['knn_topic_k'] = i
            # hparam['bpr_topic_lr'] = i
            # hparam['bpr_topic_factors'] = j

        model = KNN(hparam)
        model.fit(verbose=False)

        # course_score_val.append(model.course_score)
        topic_score_val.append(model.topic_score)


    # head = ["course_val", "topic_val"]
    head = ["topic_val"]
    with open('test.csv', "w") as f:
        writer = csv.writer(f)
        writer.writerow(head)

        # for i, j in zip(course_score_val, topic_score_val):
        for i in topic_score_val:
            writer.writerow([i])

