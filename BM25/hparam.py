import argparse
from pathlib import Path


def parse_hparam():
    hparam = argparse.ArgumentParser()
    hparam.add_argument(
        '--domain',
        type=str,
        default='unseen_course'
    )
    hparam.add_argument(
        '--model',
        type=str,
        default='bm25okapi'
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
        default='pred/BM25'
    )
    return hparam.parse_args()