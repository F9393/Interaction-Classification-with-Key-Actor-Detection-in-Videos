import numpy as np

import torch
import torch.nn.functional as F
import torchmetrics

from .datamodules.sbu_datamodule import SBUDataModule

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint

from .models.models import Model1, Model2, Model3, Model4

class KeyActorDetection(pl.LightningModule):
    def __init__(self, CFG):
        super(KeyActorDetection, self).__init__()

        self.CFG = CFG
        if CFG.training.model == 'model1':
            self.model = Model1(**CFG.model1)
        elif CFG.training.model == 'model2':
            self.model = Model2(**CFG.model2)
        elif CFG.training.model == 'model3':
            self.model = Model3(**CFG.model3)
        elif CFG.training.model == 'model4':
            self.model = Model4(**CFG.model4)
        else:
            raise ValueError(f'model "{CFG.training.model}" does not exist! Must be one of "model1", "model2", "model2", "model4"')
            
        self.accuracy = torchmetrics.Accuracy()

    def forward(self, x):
        # used during test time
        out = self.model(x)
        return out
    
    def on_train_start(self):
        self.logger.log_hyperparams(self.CFG)

    def training_step(self, batch, batch_idx):
        *inps, y = batch
        y = y.view(-1,)
        out = self.model(*inps)
        loss = F.cross_entropy(out, y)
        self.log("train_loss",loss,sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        *inps, y = batch
        y = y.view(-1,)
        out = self.model(*inps)
        loss = F.cross_entropy(out, y).item()
        y_pred = torch.argmax(out, axis=1)
        
        #accuracy of rank 0 process (logs called only on rank 0) . Call to self.accuracy() needed to accumulate batch metrics.
        self.log('val_acc_step', self.accuracy(y_pred, y), logger=False)

    def validation_epoch_end(self, val_step_outputs):
        # print(f"acc = {self.accuracy.compute()}")
        self.log('val_acc_epoch', self.accuracy.compute(), logger=False, prog_bar=True)
        self.logger.log_metrics({"val_acc_epoch": self.accuracy.compute().item()}, step = self.trainer.current_epoch)
        self.accuracy.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model._get_parameters(), lr=self.CFG.training.learning_rate)
        return optimizer
        

def train(CFG):
    dm = SBUDataModule(CFG)
    dm.prepare_data()

    for fold_no in CFG.training.folds:
        results = []
        for run_no in range(1,CFG.training.num_runs+1):
            if CFG.deterministic.set:
                seed_everything(CFG.deterministic.seed, workers=True)

            model = KeyActorDetection(CFG)
            dm.setup(fold_no = fold_no)
            mlf_logger = pl_loggers.mlflow.MLFlowLogger(experiment_name=CFG.training.model, run_name = f'fold={fold_no},run={run_no}', save_dir=CFG.training.save_dir)
            checkpoint_callback = ModelCheckpoint(dirpath=None,
                                                monitor='val_acc_epoch',
                                                save_top_k=1,
                                                save_last=True,
                                                save_weights_only=True,
                                                filename='{epoch:02d}-{val_acc_epoch:.4f}',
                                                verbose=False,
                                                mode='max')
        
            trainer = Trainer(
                max_epochs=CFG.training.num_epochs,
                gpus=CFG.gpus,
                precision=32,
                callbacks=[checkpoint_callback],
                logger = mlf_logger,
                weights_summary='top',
                log_every_n_steps=4,
                deterministic=CFG.deterministic.set,
                accelerator = "ddp" if len(CFG.gpus)>1 else None,
            )

            trainer.fit(model, dm)

            best_model_score = checkpoint_callback.best_model_score

            print(f'\nBest model val acc on fold={fold_no},run={run_no} = {best_model_score}')

    return np.mean(results), np.std(results)