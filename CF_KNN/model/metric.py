from pathlib import Path
import json
import numpy as np

class Metric():
    def __init__(self, domain, data_dir ='data/model_data'):
        self.domain = domain
        self.data_dir = data_dir

    def apk(self, actual, predicted, k=50):
        """
        Computes the average precision at k.

        This function computes the average prescision at k between two lists of
        items.

        Parameters
        ----------
        actual : list
                A list of elements that are to be predicted (order doesn't matter)
        predicted : list
                    A list of predicted elements (order does matter)
        k : int, optional
            The maximum number of predicted elements

        Returns
        -------
        score : double
                The average precision at k over the input lists

        """
        if len(predicted)>k:
            predicted = predicted[:k]

        score = 0.0
        num_hits = 0.0

        for i,p in enumerate(predicted):
            if p in actual and p not in predicted[:i]:
                num_hits += 1.0
                score += num_hits / (i+1.0)

        if not actual:
            return 0.0

        return score / min(len(actual), k)

    def mapk(self, actual, predicted, k=50):
        """
        Computes the mean average precision at k.

        This function computes the mean average prescision at k between two lists
        of lists of items.

        Parameters
        ----------
        actual : list
                A list of lists of elements that are to be predicted 
                (order doesn't matter in the lists)
        predicted : list
                    A list of lists of predicted elements
                    (order matters in the lists)
        k : int, optional
            The maximum number of predicted elements

        Returns
        -------
        score : double
                The mean average precision at k over the input lists

        """
        return np.mean([self.apk(a,p,k) for a,p in zip(actual, predicted)])


    def mapk_train_course(self, pred):
        samples = json.loads(Path(f'{self.data_dir}/train.json').read_text())
        actual = [sample['course'] for sample in samples]
        return self.mapk(actual, pred)

    def mapk_train_topic(self, pred):
        samples = json.loads(Path(f'{self.data_dir}/train.json').read_text())
        actual = [sample['topic'] for sample in samples]
        return self.mapk(actual, pred)
    
    def mapk_valseen_course(self, pred):
        samples = json.loads(Path(f'{self.data_dir}/valseen.json').read_text())
        actual = [sample['course'] for sample in samples]
        return self.mapk(actual, pred)

    def mapk_valseen_topic(self, pred):
        samples = json.loads(Path(f'{self.data_dir}/valseen.json').read_text())
        actual = [sample['topic'] for sample in samples]
        return self.mapk(actual, pred)

    def mapk_valunseen_course(self, pred):
        samples = json.loads(Path(f'{self.data_dir}/valunseen.json').read_text())
        actual = [sample['course'] for sample in samples]
        return self.mapk(actual, pred)

    def mapk_valunseen_topic(self, pred):
        samples = json.loads(Path(f'{self.data_dir}/valunseen.json').read_text())
        actual = [sample['topic'] for sample in samples]
        return self.mapk(actual, pred)

    def train_metric(self, pred):
        if self.domain == 'seen_course' or self.domain == 'unseen_course':
            return self.mapk_train_course(pred)
        else:
            return self.mapk_train_topic(pred)

    def val_metric(self, pred):
        if self.domain == 'seen_course':
            return self.mapk_valseen_course(pred)
        elif self.domain == 'seen_topic':
            return self.mapk_valseen_topic(pred)
        elif self.domain == 'unseen_course':
            return self.mapk_valunseen_course(pred)
        else:
            return self.mapk_valunseen_topic(pred)