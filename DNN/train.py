import os, sys
sys.path.append(os.path.abspath(f"{os.getcwd()}"))

from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger

from DNN.DataModule import MLCLSDataModule
from DNN.hparam import parse_hparam
import DNN.model
from metric import Metric


def main(hparam):
    hparam['ckpt_dir'].mkdir(parents=True, exist_ok=True)
    hparam['log_dir'].mkdir(parents=True, exist_ok=True)

    data_path = {
        'train' : hparam['data_dir'] / 'train.json',
        'valseen': hparam['data_dir'] / 'valseen.json',
        'testseen': hparam['data_dir'] / 'testseen.json',
        'valunseen': hparam['data_dir'] / 'valunseen.json',
        'testunseen': hparam['data_dir'] / 'testunseen.json'
    }
    idx2cat_path = {
        'gender': hparam['data_dir'] / 'idx2gender.json',
        'interest_topic': hparam['data_dir'] / 'idx2int_topic.json',
        'interest': hparam['data_dir'] / 'idx2int.json',
        'course': hparam['data_dir'] / 'idx2course.json'
    }

    # data module
    dm = MLCLSDataModule(hparam['domain'], data_path, idx2cat_path, hparam['batch_size'])
    dm.prepare_data()
    dm.setup(stage=hparam['domain'])
    

    # pytorch lightning and trainer
    hparam['model'] = model.DropoutNet(dm.num_feature, dm.num_label)
    hparam['metric'] = Metric(hparam['domain'])
    dnn = model.MLCLS(hparam)

    # callbacks
    ckpt = ModelCheckpoint(
        dirpath=hparam['ckpt_dir'] / hparam['domain'] / hparam['trial_label'],
        filename='{epoch}-{val_mapk:.3f}',
        monitor='val_mapk',
        mode='max',
        save_top_k=3,
        save_last=True,
    )
    pbar = TQDMProgressBar(refresh_rate=0)
    callbacks=[ckpt, pbar]

    # logger
    logger = WandbLogger(
        log_model='all',  # Log model checkpoints as they get created during training:
        project=hparam['proj_name'],
        save_dir=hparam['log_dir'],
        name=hparam['trial_label'],
    )
    logger.watch(dnn)

    # trainer
    trainer = Trainer(
        # fast_dev_run=True,
        # auto_scale_batch_size=None,
        accelerator='auto',
        callbacks=callbacks,
        devices=1,
        logger=logger,
        log_every_n_steps=hparam['log_every_n_steps'],
        max_epochs=hparam['num_epoch'],
        check_val_every_n_epoch=hparam['check_val_every_n_epoch']
    )

    # learning rate finder
    if hparam['find_lr']:
        lr_finder = trainer.tuner.lr_find(dnn, dm)
        print(lr_finder.suggestion())
    elif hparam['resume']:
        trainer.fit(
            dnn,
            datamodule=dm,
            ckpt_path=hparam['resume_model']
        )
    else:
        trainer.fit(dnn, dm)


if __name__ == '__main__':
    hparam = vars(parse_hparam())
    main(hparam)