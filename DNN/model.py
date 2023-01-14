import torch
from torch import nn
from torch.nn import Dropout, functional as F

from pytorch_lightning import LightningModule


class Sequential(nn.Module):
    def __init__(self, num_feature, num_label):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(num_feature, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, num_label),
            nn.Sigmoid()
        )
    def forward(self, X):
        output = self.seq(X)
        return output
  

# encoder decoder framework
class DropoutNet(nn.Module):
    def __init__(self, num_feature, num_label):
        super().__init__()
        self.input_dropout = nn.Dropout(p=0.8)
        self.seq= nn.Sequential(
            nn.Linear(num_feature, 128),
            nn.ReLU(),
            
            nn.Linear(128, 256),
            nn.ReLU(),
            
            nn.Linear(256, 512),
            nn.ReLU(),
            
            nn.Linear(512, num_label)
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, X):
       h0 = self.input_dropout(X)
       h1 = self.seq(h0)
       return self.sigmoid(h1)

class MLCLS(LightningModule):
    def __init__(self, hparam):
        super().__init__()
        self.hparam = hparam
        # model should be defined in __init__
        self.model = self.hparam['model']
    
    def forward(self, X):
        return self.model(X)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparam['lr'])

    def training_step(self, train_batch, batch_idx):
        X, y = train_batch['feature'], train_batch['label']
        y_hat = self.model(X)
        loss = F.binary_cross_entropy(y_hat, y)
        self.log('train_loss', loss)

        sort, indices = torch.sort(y_hat, descending=True)
        return {'loss': loss, 'recomm': indices[:, :50]}

    def training_epoch_end(self, outputs):
        pred = torch.cat([output['recomm'] for output in outputs], dim=0).cpu().detach().numpy()
        train_mapk = self.hparam['metric'].train_metric(pred)
        self.log('train_mapk', train_mapk)

    def validation_step(self, val_batch, batch_idx):
        X, y = val_batch['feature'], val_batch['label']
        y_hat = self.model(X)
        loss = F.binary_cross_entropy(y_hat, y)
        self.log('val_loss', loss)

        sort, indices = torch.sort(y_hat, descending=True)
        return {'loss': loss, 'recomm': indices[:, :50]}

    def validation_epoch_end(self, outputs):
        pred = torch.cat([output['recomm'] for output in outputs], dim=0).cpu().detach().numpy()
        val_mapk = self.hparam['metric'].val_metric(pred)
        self.log('val_mapk', val_mapk)