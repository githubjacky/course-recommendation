import os, sys
sys.path.append(os.path.abspath(f"{os.getcwd()}"))

from BM25.model import bm25
from BM25.hparam import parse_hparam
import sys


def main(hparam):
    model = bm25(hparam)

    if hparam['stage'] == 'fit':
        model.fit()
    elif hparam['stage'] == 'test':
        if hparam['domain'] == 'seen_course' or hparam['domain'] == 'unseen_course':
            model.pred_course()
        elif hparam['domain'] == 'seen_topic' or hparam['domain'] == 'unseen_topic':
            model.pred_topic_from_course()
        else:
            print('invalid domain')
            sys.exit(1)
    else:
        print('invalid stage')
        sys.exit(1)

if __name__ == "__main__":
    hparam = vars(parse_hparam())
    main(hparam)