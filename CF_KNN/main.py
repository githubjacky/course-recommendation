import os, sys
sys.path.append(os.path.abspath(f"{os.getcwd()}"))

from CF_KNN.hparam import parse_hparam
from CF_KNN.model.als import ALS
from CF_KNN.model.bpr import BPR
from CF_KNN.model.knn import KNN
from CF_KNN.model.mix import MIX2, MIX3


def main(hparam):
    if hparam['model'] == 'als':
        model = ALS(hparam)
    elif hparam['model'] == 'bpr':
        model = BPR(hparam)
    elif hparam['model'] == 'knn':
        model = KNN(hparam)
    elif hparam['model'] == 'mix2':
        model = MIX2(hparam)
    elif hparam['model'] == 'mix3':
        model = MIX3(hparam)
    else:
        print('invalid model for CF_KNN class')
        sys.exit(1)
    
    if hparam['stage'] == 'fit':
        model.fit()
    elif hparam['stage'] == 'test_course':
        model.pred_course()
    elif hparam['stage'] == 'test_topic' and hparam['domain'] == 'course':
        model.pred_topic_from_course()
    elif hparam['stage'] == 'test_topic' and hparam['domain'] == 'topic':
        model.pred_topic()
    else:
        print('invalid stage for CF_KNN class')
        sys.exit(1)


if __name__ == '__main__':
    hparam = vars(parse_hparam())
    main(hparam)