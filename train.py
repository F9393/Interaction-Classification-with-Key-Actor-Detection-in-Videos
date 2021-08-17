import os
import numpy as np
import timeit
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter

from omegaconf import DictConfig, OmegaConf
import hydra
import hydra.compose, hydra.initialize

from utils.checkpoint import CheckPointer, get_save_directory
from utils.display import display_sample_images, display_model_graph
from utils.training_utils import repeat_k_times
from models.models import Model1, Model2, Model3, Model4
from datasets.sbu.train_test_split import train_sets, test_sets # generates K-fold train and test sets
from datasets.sbu.sbu_dataset import  M1_SBU_Dataset, M2_SBU_Dataset, M3_SBU_Dataset, M4_SBU_Dataset

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning import Callback
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint


class KeyActorDetection(pl.LightningModule):
    def __init__(self, CFG):
        super(KeyActorDetection, self).__init__()
        if CFG.training.model == 'model1':
            self.model = Model1(**CFG.model1)
        elif CFG.training.model == 'model2':
            self.model = Model2(**CFG.model2)
        elif CFG.training.model == 'model3':
            self.model = Model3(**CFG.model3)
        elif CFG.training.model == 'model4':
            self.model = Model4(**CFG.model4)

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        out = self.model(x)
        return out

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        *inps, y = batch
        y = y.view(-1,)
        out = self.model(*inps)
        loss = F.cross_entropy(out, y)
        
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        *inps, y = batch
        y = y.view(-1,)
        out = self.model(*inps)
        loss = F.cross_entropy(out, y).item()
        
        y_pred = torch.argmax(out, axis=1)
        correct_pred = torch.sum(y_pred == y).item()
        total_pred = y.size(0)

        return loss, correct_pred, total_pred

    def validation_epoch_end(self, validation_step_outputs):
        all_losses, all_correct_pred,all_pred = zip(*validation_step_outputs)

        mean_loss = np.mean(all_losses)
        sum_correct_pred = sum(all_correct_pred)
        sum_all_pred = sum(all_pred)

        val_accuracy = sum_correct_pred * 1.0 / sum_all_pred
        
        self.log('step', self.trainer.current_epoch)
        self.log_dict(
            {"val_accuracy" :val_accuracy, "val_loss": mean_loss}, prog_bar=True
        )


    def configure_optimizers(self):
        return self.model.optimizer()



def main():
    hydra.initialize(config_path="configs")
    DEF_CFG = hydra.compose(config_name="config")
    CFG = DEF_CFG.dataset

    
    model = KeyActorDetection(CFG)
 

    tb_logger = pl_loggers.TensorBoardLogger("logs/")

    checkpoint_callback = ModelCheckpoint(monitor='val_accuracy',
                                          save_top_k=1,
                                          save_last=True,
                                          save_weights_only=True,
                                          filename='checkpoint/{epoch:02d}-{valid_score:.4f}',
                                          verbose=False,
                                          mode='max')

    trainer = Trainer(
        max_epochs=CFG.training.num_epochs,
        gpus=[0],
        accumulate_grad_batches=1,
        precision=32,
        callbacks=[checkpoint_callback],
        logger=tb_logger,
        weights_summary='top',
        log_every_n_steps=1,
        # accelerator = "ddp"
    )

    folds_acc = []
    for fold_no in CFG.training.folds:
        if CFG.training.model == 'model1':
            train_set = M1_SBU_Dataset(train_sets[fold_no], CFG.training.select_frame, mode='train', resize=CFG.model1.resize, fold_no = fold_no+1)
            valid_set = M1_SBU_Dataset(test_sets[fold_no], CFG.training.select_frame, mode='valid', resize=CFG.model1.resize, fold_no = fold_no+1)
        elif CFG.training.model == 'model2':
            train_set = M2_SBU_Dataset(CFG.model2.pose_coord, train_sets[fold_no], CFG.training.select_frame, mode='train', resize=CFG.model1.resize, fold_no = fold_no+1)
            valid_set = M2_SBU_Dataset(CFG.model2.pose_coord, test_sets[fold_no], CFG.training.select_frame, mode='valid', resize=CFG.model1.resize, fold_no = fold_no+1)
        elif CFG.training.model == 'model3':
            train_set = M3_SBU_Dataset(CFG.model3.pose_coord, train_sets[fold_no], CFG.training.select_frame, mode='train', resize=CFG.model1.resize, fold_no = fold_no+1)
            valid_set = M3_SBU_Dataset(CFG.model3.pose_coord, test_sets[fold_no], CFG.training.select_frame, mode='valid', resize=CFG.model1.resize, fold_no = fold_no+1)
        elif CFG.training.model == 'model4':
            train_set = M4_SBU_Dataset(CFG.model3.pose_coord, train_sets[fold_no], CFG.training.select_frame, mode='train', resize=CFG.model1.resize, fold_no = fold_no+1)
            valid_set = M4_SBU_Dataset(CFG.model3.pose_coord, test_sets[fold_no], CFG.training.select_frame, mode='valid', resize=CFG.model1.resize, fold_no = fold_no+1)
        else:
            raise Exception(f"invalid model name - {CFG.training.model}! Must be one of model1, model2, model3, model4")

        params = CFG.training.dataloader
#         print(params)
        train_loader = data.DataLoader(train_set, **params)
        valid_loader = data.DataLoader(valid_set, **params)
        
        trainer.fit(model, train_dataloader=train_loader, val_dataloaders=valid_loader)
    
#         result_metrics = train(CFG = CFG, train_set = train_set, valid_set = valid_set, save_model_subdir = save_model_subdir, fold_no = fold_no + 1)
#         folds_acc.append(result_metrics)

#         if CFG.training.save_checkpoint:
#             with open(os.path.join(CFG.training.save_model_path, save_model_subdir, f"fold={fold_no+1}", "average_results.txt"), "w") as f:
#                 for key,val in result_metrics.items():
#                     f.write(f"{key} : {val}\n")

   
#     folds_avg_metrics = {key:0 for key in folds_acc[0].keys()}
#     for fold_result in folds_acc:
#         for key,val in fold_result.items():
#             folds_avg_metrics[key] += val
#     for key in folds_avg_metrics:
#         folds_avg_metrics[key] /= len(folds_acc)
#     print(f'AVERAGED OVER FOLDS RESULT : {folds_avg_metrics}')

#     return folds_avg_metrics['val_accuracy'].item()



if __name__ == "__main__":
    main()


