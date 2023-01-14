import pandas as pd
from pathlib import Path
import argparse


from util import DataUtil

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument(
        '--raw_data_dir',
        type=Path,
        default='data/raw_data'
    )
    args.add_argument(
        '--target_data_dir',
        type=Path,
        default='data/model_data'
    )
    return args.parse_args()



def main(args):
    args.target_data_dir.mkdir(parents=True, exist_ok=True)
    data = DataUtil(
        user = args.raw_data_dir / 'users.csv',
        course = args.raw_data_dir / 'courses.csv',
        course_chapter = args.raw_data_dir / 'course_chapter_items.csv',
        topic = args.raw_data_dir / 'subgroups.csv',
        train = args.raw_data_dir / 'train.csv',
        train_topic = args.raw_data_dir / 'train_group.csv',
        valseen = args.raw_data_dir / 'val_seen.csv',
        valseen_topic = args.raw_data_dir / 'val_seen_group.csv',
        testseen = args.raw_data_dir / 'test_seen.csv',
        testseen_topic = args.raw_data_dir / 'test_seen_group.csv',
        valunseen = args.raw_data_dir / 'val_unseen.csv',
        valunseen_topic = args.raw_data_dir / 'val_unseen_group.csv',
        testunseen = args.raw_data_dir / 'test_unseen.csv',
        testunseen_topic = args.raw_data_dir / 'test_unseen_group.csv',
    )

    data.encode_customer_feature_train('train', args.target_data_dir)
    data.encode_customer_feature_train('valseen', args.target_data_dir)
    data.encode_customer_feature_train('valunseen', args.target_data_dir)
    data.encode_customer_feature('testseen', args.target_data_dir)
    data.encode_customer_feature('testunseen', args.target_data_dir)

    data.output_utils(args.target_data_dir)


if __name__ == "__main__":
    args = parse_args()
    main(args)