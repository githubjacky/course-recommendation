import argparse
from pathlib import Path


def parse_hparam():
    hparam = argparse.ArgumentParser()
    hparam.add_argument(
        '--domain',
        type=str, 
        default='seen_course'
    )
    hparam.add_argument(
        '--find_lr',
        action='store_true'
    )
    hparam.add_argument(
        '--resume',
        action='store_true'
    )
    hparam.add_argument(
        '--resume_model',
        default=Path
    )
    hparam.add_argument(
        '--data_dir',
        type=Path,
        default='data/model_data'
    )
    hparam.add_argument(
        '--log_dir',
        type=Path,
        default='DNN/log'
    )
    hparam.add_argument(
        '--ckpt_dir',
        type=Path,
        default='DNN/ckpt'
    )
    hparam.add_argument(
        '--test_dir',
        type=Path,
        default='DNN/test'
    )
    hparam.add_argument(
        '--proj_name',
        type=str,
        default='howhow-challenge'
    )
    hparam.add_argument(
        '--trial_label',
        type=str, 
    )
    hparam.add_argument(
        '--check_val_every_n_epoch',
        type=int, 
        default=3
    )
    hparam.add_argument(
        '--log_every_n_steps',
        type=int, 
        default=100
    )
    hparam.add_argument(
        '--num_epoch',
        type=int, 
        default=200
    )
    hparam.add_argument(
        '--batch_size',
        type=int, 
        default=16
    )
    hparam.add_argument(
        '--lr',
        type=float,
        default=3e-5
    )
    return hparam.parse_args()