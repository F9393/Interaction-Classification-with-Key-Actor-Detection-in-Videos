import torch
from torch.nn.parallel import DistributedDataParallel
import os
import glob

class CheckPointer:
    _last_checkpoint_name = 'last'
    _best_checkpoint_name = 'best'

    def __init__(self,
                 models,
                 optimizer = None,
                 scheduler = None,
                 save_dir = None,
                 best_metrics = {},
                 watch_metric = None,
                 ):
        self.models = models
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.best_metrics = best_metrics
        self.watch_metric = watch_metric

        if self.save_dir:
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

    def get_save_dict(self):
        data = {}

        data['best_metrics'] = self.best_metrics
        data['models'] = []
        for model in self.models:
            if isinstance(model, DistributedDataParallel):
                data['models'].append(model.module.state_dict())
            else:
                data['models'].append(model.state_dict())

        if self.optimizer is not None:
            data["optimizer"] = self.optimizer.state_dict()

        if self.scheduler is not None:
            data["scheduler"] = self.scheduler.state_dict()

        return data

    def save_dict(self, name, save_data):
        """
        saves 'save_dict' with name 'name'
        """
        if not self.save_dir:
            return
        
        if save_data is None:
            save_data = self.get_save_dict()

        save_file = os.path.join(self.save_dir, "{}.pth".format(name))
        print("Saving checkpoint to {}".format(save_file))
        torch.save(save_data, save_file)

    def save_checkpoint(self, current_metrics):
        """
        saves best and last checkpoints
        """
        save_data = self.get_save_dict()

        self.save_dict(name = CheckPointer._last_checkpoint_name, save_data = save_data)
        if current_metrics[self.watch_metric] > self.best_metrics[self.watch_metric]:
            self.best_metrics = current_metrics
            self.save_dict(name = CheckPointer._best_checkpoint_name, save_data = save_data)

            with open(os.path.join(self.save_dir, "best_results.txt"), "w") as f:
                for key,val in self.best_metrics.items():
                    f.write(f"{key} : {val}\n")

    def load_checkpoint(self, load_file = None, checkpoint_type = "best"):
        if load_file is None:
            load_file = self.has_checkpoint(checkpoint_type)
            if not load_file:
                print("No checkpoint found.")
                return {}

        print("Loading checkpoint from {}".format(load_file))

        checkpoint = torch.load(load_file)

        self.best_metrics = checkpoint['best_metrics']
        print(f'current best metrics {self.best_metrics}')

        models = self.models
        for model in models:
            if isinstance(model, DistributedDataParallel):
                model = self.model.module

        for model,ckpt in zip(models,checkpoint['models']):
            model.load_state_dict(ckpt)
        checkpoint.pop("models")
        
        if "optimizer" in checkpoint and self.optimizer:
            print("Loading optimizer from {}".format(load_file))
            self.optimizer.load_state_dict(checkpoint.pop("optimizer"))
        if "scheduler" in checkpoint and self.scheduler:
            print("Loading scheduler from {}".format(load_file))
            self.scheduler.load_state_dict(checkpoint.pop("scheduler"))

        return checkpoint

    def has_checkpoint(self, checkpoint_type):
        if checkpoint_type == "last":
            files = sorted(glob.glob(f'{self.save_dir}/*{CheckPointer._last_checkpoint_name}*'))
            if not files:
                return False
        elif checkpoint_type == "best":
            files = sorted(glob.glob(f'{self.save_dir}/*{CheckPointer._best_checkpoint_name}*'))
            if not files:
                return False
        else:
            print(f"'{checkpoint_type}' is invalid type. Must be one of 'best' or 'last'")
            return False
        return files[-1]
