import numpy as np
import csv
from model.als import ALS
from model.bpr import BPR
from model.knn import KNN
from model.mix import MIX2, MIX3
from hparam import parse_hparam


if __name__ == '__main__':
    hparam = vars(parse_hparam())
    hparam['domain'] = 'topic'
    # hparam['als_course_factors'] = 489.7961
    # hparam['als_course_regularizations'] = 370
    # hparam['als_topic_regularizations'] = 489.7961
    # hparam['als_topic_factors'] = 370
    # iterations = [int(i) for i in np.linspace(30, 300, 20)]
    factors = [int(i) for i in np.linspace(10, 500, 50)]
    regularizations = np.linspace(0.01, 500, 50)
    # course_score_val = []
    topic_score_val = []
    for i in regularizations:
    # for i in iterations:
        for j in factors:
        # hparam['als_course_iterations'] = i
        # hparam['als_course_factors'] = j
            hparam['als_topic_regularizations'] = i
            hparam['als_topic_factors'] = j

        model = ALS(hparam)
        model.fit(verbose=False)

        # course_score_val.append(model.course_score)
        topic_score_val.append(model.topic_score)


    # head = ["course_val", "topic_val"]
    head = ["topic_val"]
    with open('test/als_test.csv', "w") as f:
        writer = csv.writer(f)
        writer.writerow(head)

        # for i, j in zip(course_score_val, topic_score_val):
        for i in topic_score_val:
            writer.writerow([i])

