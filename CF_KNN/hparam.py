import argparse
from pathlib import Path

from numpy import float64


def parse_hparam():
    hparam = argparse.ArgumentParser()
    hparam.add_argument(
        '--domain',
        type=str,
        default='course'
    )
    hparam.add_argument(
        '--model',
        type=str,
        default='als'
    )
    hparam.add_argument(
        '--stage',
        type=str,
        default='fit'
    )
    hparam.add_argument(
        '--data_dir',
        type=Path,
        default='data/model_data'
    )
    hparam.add_argument(
        '--pred_dir',
        type=Path,
        default='pred/CFF_KNN'
    )
    hparam.add_argument(
        '--num_train_sample',
        type=int,
        default=59737
    )
    hparam.add_argument(
        '--num_course',
        type=int,
        default=728
    )
    hparam.add_argument(
        '--num_group',
        type=int,
        default=92
    )
    hparam.add_argument(
        '--als_course_factors',
        type=int,
        default=60
    )
    hparam.add_argument(
        '--als_course_regularization',
        type=float64,
        default=51.02939
    )
    hparam.add_argument(
        '--als_course_alpha',
        type=int,
        default=1
    )
    hparam.add_argument(
        '--als_course_iterations',
        type=int,
        default=312
    )
    hparam.add_argument(
        '--random_state',
        type=int,
        default=11112224
    )
    hparam.add_argument(
        '--als_topic_factors',
        type=int,
        default=40
    )
    hparam.add_argument(
        '--als_topic_regularization',
        type=float,
        default=51.02939
    )
    hparam.add_argument(
        '--als_topic_alpha',
        type=int,
        default=1
    )
    hparam.add_argument(
        '--als_topic_iterations',
        type=int,
        default=425
    )
    hparam.add_argument(
        '--bpr_course_factors',
        type=int,
        default=30
    )
    hparam.add_argument(
        '--bpr_course_lr',
        type=float,
        default=0.001437
    )
    hparam.add_argument(
        '--bpr_course_iterations',
        type=int,
        default=300
    )
    hparam.add_argument(
        '--bpr_topic_factors',
        type=int,
        default=250
    )
    hparam.add_argument(
        '--bpr_topic_lr',
        type=float,
        default=0.00062 
    )
    hparam.add_argument(
        '--bpr_topic_iterations',
        type=int,
        default=389
    )
    hparam.add_argument(
        '--knn_course_k',
        type=int,
        default=1980
    )
    hparam.add_argument(
        '--knn_course_algorithm',
        type=str,
        default='brute'
    )
    hparam.add_argument(
        '--knn_topic_k',
        type=int,
        default=1980
    )
    hparam.add_argument(
        '--knn_topic_algorithm',
        type=str,
        default='brute'
    )
    hparam.add_argument(
        '--mix_base',
        type=str,
        default='bpr'
    )
    hparam.add_argument(
        '--mix_reference1',
        type=str,
        default='als'
    )
    hparam.add_argument(
        '--mix_reference2',
        type=str,
        default='knn'
    )
    return hparam.parse_args()