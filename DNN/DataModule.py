from pathlib import Path
import json
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from pytorch_lightning import LightningDataModule


# base data set module
class MLCDataset(Dataset):
    def __init__(self, data_path):
        self.data = json.loads(Path(data_path).read_text())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


# pytorch lightning data module
class MLCLSDataModule(LightningDataModule):
    def __init__(self, domain, data_path, idx2cat_path, batch_size):
        super().__init__()
        self.domain = domain
        self.data_path = data_path
        self.idx2cat_path = idx2cat_path
        self.batch_size=batch_size

    def prepare_data(self):
        self.dataset = {
            split: MLCDataset(data_path)
            for split, data_path in self.data_path.items()
        }
        self.idx2cat = {
            split: json.loads(Path(idx2cat_path).read_text())
            for split, idx2cat_path in self.idx2cat_path.items()
        }

        self.num_gender = len(self.idx2cat['gender'])
        self.num_interest_topic = len(self.idx2cat['interest_topic'])
        self.num_interest = len(self.idx2cat['interest'])
        self.num_feature = (self.num_gender-1) + (self.num_interest_topic-1) + (self.num_interest-1)

        if self.domain == 'seen_course' or self.domain == 'unseen_course':
            self.num_label = len(self.idx2cat['course'])
        else:
            self.num_label = len(self.idx2cat['topic'])
        
    def setup(self, stage):
        if stage == 'seen_course' or stage == 'seen_topic' :
            self.val_dataset = self.dataset['valseen']
            self.test_dataset = self.dataset['testseen']
        if stage == 'unseen_course' or stage == 'unseen_topic':
            self.val_dataset = self.dataset['valunseen']
            self.test_dataset = self.dataset['testunseen']

    def onehot_encode(self, sample, n):
        tensor = torch.zeros(n-1)
        if isinstance(sample, list):
            for i in sample:
                tensor[i] += 1
        else:
            tensor[sample] += 1
        return tensor
    
    def label_encode(self, sample, n):
        tensor = torch.zeros(n)
        for i in sample:
            tensor[i] += 1
        return tensor

    def collate_fn_fit(self, samples):
        features = torch.zeros((self.batch_size, self.num_feature))
        labels = torch.zeros((self.batch_size, self.num_label))
        for idx, sample in enumerate(samples):
            oh_gender = self.onehot_encode(sample['gender'], self.num_gender)
            oh_interest_topic = self.onehot_encode(sample['interest_topic'], self.num_interest_topic)
            oh_interest = self.onehot_encode(sample['interest'], self.num_interest)
            features[idx, :] = torch.cat(
                (oh_gender, oh_interest_topic, oh_interest), dim=0
            )
    
            if self.domain == 'seen_course' or self.domain == 'unseen_course':
                oh_course = self.label_encode(sample['course'], self.num_label)
                labels[idx, :] = oh_course
            else:
                oh_topic = self.label_encode(sample['topic'], self.num_label)
                labels[idx, :] = oh_topic
        return {
            'feature': features,
            'label': labels
        }

    def collate_fn_test(self, samples):
        features = torch.zeros((256, self.num_feature))
        for idx, sample in enumerate(samples):
            oh_gender = self.onehot_encode(sample['gender'], self.num_gender)
            oh_interest_topic = self.onehot_encode(sample['interest_topic'], self.num_interest_topic)
            oh_interest = self.onehot_encode(sample['interest'], self.num_interest)
            features[idx, :] = torch.cat(
                (oh_gender, oh_interest_topic, oh_interest), dim=0
            )
        return {'feature': features}

    def train_dataloader(self):
        return DataLoader(
            self.dataset['train'],
            batch_size=self.batch_size,
            collate_fn=self.collate_fn_fit
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn_fit
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=256,
            collate_fn=self.collate_fn_test
        )